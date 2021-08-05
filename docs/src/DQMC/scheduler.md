# Update Scheduler

The update scheduler keeps track of and iterates through various Monte Carlo updates. Currently there are two schedulers, `SimpleScheduler` and `AdaptiveScheduler`, and five (full) updates.

!!! note

    This is a DQMC only feature at the moment.

## Schedulers

#### SimpleScheduler

The `SimpleScheduler` keeps track of a sequence of updates and cycles through them as `update(scheduler, mc, model)` is called. It is constructed by simply passing a sequence of updates, i.e. `SimpleScheduler(update1, update2, ...)`.

#### AdaptiveScheduler

The `AdaptiveScheduler` also has a static sequence of updates, but allows some of these updates to be `Adaptive()`. Any `Adaptive()` update in the sequence is replaced with an update from a pool based on their relative weights. These weights are derived from their acceptance rates with some lag.

An adaptive scheduler is created with a sequence of updates and a pool of updates, i.e.

```julia
sequence = (update1, Adaptive())
pool = (update2, update3)
scheduler = AdaptiveScheduler(sequence, pool)
```

Additionally there are a couple of keywords to configure how weights are adjusted.
- `minimum_sampling_rate = 0.01`: This defines the threshold under which the 
sampling rate is set to 0.
- `grace_period = 99`: This sets a minimum number of times an update needs to 
be called before its sampling rate is adjusted. 
- `adaptive_rate = 9.0`: Controls how fast the sampling rate is adjusted to the 
acceptance rate. More means slower.

The adjustments of the sampling rate follow the formula
\begin{equation}
\frac{(adaptive_rate * sampling_rate + \frac{accepted}{total} }{ adaptive_rate + 1 }
\end{equation}

!!! note

    All schedulers wrap their updates in `AcceptanceStatistics`. This wrapper keeps track of the total number of update attempts and the number of accepted updates.


## Updates

Updates are small or even empty structs used to dispatch to different `update` functions. They are assumed to implement `name(::MyUpdate) = "MyUpdate"` and a method `update(::MyUpdate, mc, model)`. 



### Local Updates

Local updates affect one site at one time slice. In order for this to be on a similar scale as global and parallel updates, local updates should come in full sweeps. 

#### LocalSweep

Currently there is only one local update - `LocalSweep([N=1])`. It performs N standard sweeps of local updates, which means two updates per site and time slice each in DQMC. (Two because we go from $\tau = 0$ to $\tau = \beta$ back to $\tau = 0$.) This update returns a float corresponding to its internal acceptance rate $accepted / (2 N_{sites} M_{slices}$



### Global Updates

Global updates affect not just one site at one time slice, but most if not all sites at all time slices. In other words they attempt to adjust the full configuration. We currently have the following global updates.

#### GlobalFlip

The `GlobalFlip()` proposes a flip of the full configuration, i.e. $\pm 1 \to \mp 1$.

#### GlobalShuffle

`GlobalShuffle()` performs a `shuffle(current_configuration)` to generate a new configuration.

#### SpatialShuffle

`SpatialShuffle` shuffles only the spatial part of a configuration. This means that if two sites are swapped, they are swapped the same way for all time slice indices.

#### TemporalShuffle

`TemporalShuffle` shuffles only the imaginary time part of the configuration, meaning that the swap occurs for all lattice indices the same way.

#### Denoise

`Denoise` attempts to align the configuration in lattice space. Specifically, it sets each site to the majority value of its neighbors (including itself with a lower weight). Note that this update assumes configuration values $\pm 1$. Also note that the new configuration is solely based on the old, i.e. it does not take changes of nearby sites into account. Because of this the update may not always push the configuration to a more uniform distribution.

#### DenoiseFlip

`DenoiseFlip` follows the logic of `Denoise`, but flips the resulting configuration.

#### StaggeredDenoise

`StaggeredDenoise` follows the same logic as `Denoise`, but multiplies a factor $\pm 1$ based on the lattice site index. 



### Parallel Updates

Parallel updates communicate between different simulations running on different workers. Note that these simulations must produce configurations of the same size, i.e. have same number of time slices and sites.

#### ReplicaExchange

The `ReplicaExchange(target)` update requires a target worker, and it requires that worker to be in sync. Specifically that means if worker 1 has a `ReplicaExchange(2)` followed by `ReplicaExchange(3)`, worker 2 must have `ReplicaExchange(1)` as its first replica exchange update and worker 3 must have `ReplicaExchange(1)` as its second replica exchange update.

The idea of a replica exchange update is to propose swapping configurations between two simulations. The exchange is based on the product of both of their acceptance probabilities. With $C_i$ the configuration of simulation $i$ and $w_i(C)$ the weight of simulation $i$ with the given configuration the acceptance probability is given by

```math
p = \frac{w_1(C_2)}{w_1(C_1)} \cdot \frac{w_2(C_1)}{w_2(C_2)}
```

#### ReplicaPull

`ReplicaPull()` is an experimental parallel update. Instead of synchronizing with another simulations it pulls a configuration asynchronously and uses that for a global update. This means that there is little waiting on other simulations, but configurations will be duplicated.

This update cycles through a pool of connected workers. This pool can be of any size. Each simulation must make itself available for pulling via `connect(target_workers)`. The `target_workers` should generally be the workers the simulation wants to receive configurations from. When a simulation reaches the end it will automatically `disconnect(target_workers)`.

### MPI Updates

#### MPIReplicaExchange

This is a MPI version of the replica exchange update. In this case `target` is an MPI rank. Much like the normal replica exchange update simulations need to be paired such that they request updates from each other.