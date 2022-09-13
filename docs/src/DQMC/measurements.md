# Measurements

## Construction Measurements

Measurement are created and added to a Simulation via `dqmc[:name] = measurement(dqmc, model)`. Various properties of the measurement can be adjusted via keyword arguments, such as the lattice iteration scheme or the Wicks expanded expectation value. This will be discussed in more detail later.

The currently implemented measurements are the following:

### Greens

The equal time greens function can be measured via `greens_measurement(dqmc, model)`. The measurement will take the Monte Carlo average of `greens(dqmc, model)`.

### Occupation

The per-site occupation $\langle n_i \rangle$ can be measured via `occupation(dqmc, model)`. This will average $1 - G_{ii}$.

### Charge Density

The charge density correlation $\langle \sum_r n(r) n(r+\Delta r) \rangle$ can be measured with `charge_density_correlation(dqmc, model)`. The time integral of that, the charge density susceptibility can be measure with `charge_density_susceptibility(mc, model)`.

Note that you can also pass `kernel = MonteCarlo.reduced_cdc_kernel` to measure just the correlated part, i.e. $\langle \sum_r n(r) n(r+\Delta r) \rangle - \langle n(r) \rangle \langle n(r+\Deltar) \rangle$ instead.

### Magnetization

`magnetization(dqmc, model, dir::Symbol)` measures the per-site x-, y- or z-magnetizations.

### Spin Density

The spin density correlation $\langle S_\gamma S_\gamma \rangle$ in x-, y- or z-direction can be measured with `spin_density_correlation(dqmc, model, dir::Symbol)`. The respective susceptibilities follow from `spin_density_susceptibility(dqmc, model, dir)`.

Like with charge density there are additional kernels that only measure the correlated part of the spin density. Use `kernel = MonteCarlo.reduced_sdc_γ_kernel` with $\gamma \in {x, y, z}$ for the respective direction.

### Pairing

The pairing susceptibility $\Delta_v(s_1, t_1)(\tau) \Delta_v^\dagger(s_2, t_2)(0)$ can be calculated with `pairing_susceptibility(dqmc, model)`. The $\tau = 0$ pairing correlation follows from `pairing_correlation(dqmc, model)`.

### Current-Current Susceptibility

`current_current_susceptibility(dqmc, model)` measures $\langle j_{t_2 - s_2}(s_2, l) j_{t_1 - s_1}(s_1, 0)\rangle$ with $j_{t - s}(s, \tau) = \langle i \sum_\sigma [T_{ts} c_t^\dagger(\tau) c_s(\tau) - T_{st} c_s^\dagger(\tau) c_t(\tau)] \rangle$ where $i^2 = -1$ and $T$ is the hopping matrix.

### Superfluid Density

The superfluid density can be derived from the current-current susceptibility and the Greens function. [MonteCarloAnalysis.jl](https://github.com/ffreyer/MonteCarloAnalysis] provides functionality for that.

### Energies

The energy can be measured with `total_energy(dqmc, model)`. The interacting and hopping parts can be measured independently with `interaction_energy(dqmc, model)` and `kinetic_energy(dqmc, model)`

## General Notes

All measurements are implemented via

```julia
struct DQMCMeasurement{F <: Function, GI, LI, FI, OT, T} <: AbstractMeasurement
    greens_iterator::GI
    lattice_iterator::LI
    flavor_iterator::FI
    kernel::F
    observable::OT
    temp::T
end
```

#### `kernel`

The auxiliary field dependent greens function is readily available at any point in the simulation. As such it is the object on which measurements typically rely on. Using [Wick's theorem](https://en.wikipedia.org/wiki/Wick%27s_theorem) most expectation values can be expressed in terms of greens function elements $G_{ij}(k, l) = \langle c_i(k \Delta\tau) c_j^\dagger(l \Delta\tau)\rangle$ where $i, j$ represent sites and flavors (spins), and $k, l$ represent imaginary time. The `kernel` implements this expanded form. 

For example, the `full_cdc_kernel` implementing $\langle \sum_r n(r) n(r+\Delta r) \rangle$ looks like this:

```julia
@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
    f1, f2 = flv
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    
    id = I[i, j] * I[G0l.k, G0l.l] * I[f1, f2]
    s1 = N * (f1 - 1)
    s2 = N * (f2 - 1)

    # ⟨n_{σ₁}(l) n_{σ₂}(0)⟩ =
    #   ⟨n_{σ₁}(l)⟩ ⟨n_{σ₂}(0)⟩ + 
    #   ⟨c_{σ₁}^†(l) c_{σ₂}(0)⟩ ⟨c_{σ₁}(l) c_{σ₂}^†(0)⟩ =
    return (1 - Gll.val[i+s1, i+s1]) * (1 - G00.val[j+s2, j+s2]) +
            (id - G0l.val[j+s1, i+s2]) * Gl0.val[i+s1, j+s2]
end
```

Here `i, j = ij` are site indices representing $r, r + \Delta r$ coming from the lattice iterator, `G00, G0l, Gl0, Gll = packed_greens` are Greens matrices at different imaginary times coming from the greens iterator, and `f1, f2 = flv` are flavor (spin) indices coming from the flavor iterator. The result of the kernel is the charge density expectation value for a specific set of those indices.

These functions generally have a specialized methods implemented for the different matrix types that are used in DQMC. You can check the source code under "flavors/DQMC/measurements/constructors" for more examples. 

### `greens_iterator`

The `greens_iterator` controls which Greens functions are passed on to the kernel. Internally measurements that use the same `greens_iterator` will be bundled to avoid expensive recalculations. The available iterators include:

* `nothing` specifies that no Greens function is needed
* `Greens()` forwards the equal time greens function `G(0, 0)` (which matches all other equal time greens functions)
* `GreensAt(k, l)` forwards the result of `greens(dqmc, k, l)`, i.e. a greens function at the specific time indices $k, l$.
* `TimeIntegral([recalculate = 2 mc.parameters.safe_mult])` creates an iterator for calculating imaginary time integral of the form $O_i = \int_0^\beta O_(\tau) d\tau$ as $O_i \approx \sum_{l = 0}^{M-1} 0.5 \Delta\tau (O_(l \Delta\tau) + O_((l+1) \Delta\tau))$. In every step this iterator will generate four greens matrices $G(0, 0)$, $G(0, l\Delta\tau)$, $G(l\Delta\tau, 0)$, $G(l\Delta\tau, l\Delta\tau)$. This internally uses `CombinedGreensIterator(mc[; start, stop, recalculate])` to generate these matrices.

### `lattice_iterator`

The `lattice_iterator` controls which combination of site indices are passed to the kernel and how they are further combined before saving the measurements. For example, `EachSitePairByDistance` passes any combination of two sites indices to the kernel and sums up site pairs which have the same distance between them. See the Lattices section for more detail.

### `flavor_iterator`

The `flavor_iterator` similarly specifies which flavor (spin) indices should be iterated. This is primarily an optimization used to pull a flavor sum out of the kernel. Note that this is not always possible/useful, so some measurements may not use this iterator even though multiple flavors are involved. As such this iterator should generally not be adjusted.

### `observable`

The `observable` is the final storage of the measured values. By default this is a `LogBinner` from `BinningAnalysis.jl` but that can be changed. The only hard requirement is that the data structure implements `push!`. If you want to use your own storage structure you can get a zero element from `MonteCarlo._binner_zero_element(dqmc, lattice_iterator, MonteCarlo.geltype(dqmc))` if you need it.

### `temp`

The `temp` field is a temporary storage Array used as a target for summation before pushing the final result of the measurement. It should be initialized with `MonteCarlo._measurement_buffer(dqmc, lattice_iterator, geltype(dqmc))`. Note that this is often but not always the same as the zero element.