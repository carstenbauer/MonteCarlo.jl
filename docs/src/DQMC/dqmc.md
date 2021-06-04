# DQMC

The `DQMC` struct represents both the determinant quantum monte carlo algorithm and the simulation as a whole. Because of that it includes a bunch of options that aren't directly relevant to the algorithm. The minimal default is given by `dqmc = DQMC(model, beta=beta)` and the simulation can then be started with `run!(dqmc)`. Additional keyword arguments include:

* `beta`: The inverse temperature of the simulation.
* `delta_tau = 0.1`: The imaginary time discretization.
* `slices = beta / delta_tau`: The number of imaginary time slices.
* `safe_mult = 10`: The number of save matrix multiplications.
* `thermalization = 100`: The number of thermalization sweeps.
* `sweeps = 100`: The number of measurement sweeps.
* `check_sign_problem = true`: Enables or disables checks for sign problems. (negative or imaginary probabilities)
* `check_propagation_error = true`: Enables or disables checks for time slices propagation errors. (Which may happen if safe_mult or delta_tau is too large.)
* `silent = false`: Enable or disable prints for the checks above.
* `measure_rate = 10`: Sets the frequency of measurements. Every `measure_rate` sweeps a new measurement is taken.
* `print_rate = 10`: Sets the frequency of general information prints (not checks).
* `seed = -1`: Sets a random seed for the simulation. If set to `-1` the seed will be chosen randomly.
* `last_sweep = 0`: Sets the last finished sweep. Used internally for continued simulations.

Beyond this there are a couple of keyword arguments which are more involved and will be discussed in other chapters. These include

* `scheduler = SimpleScheduler(LocalSweep())`: This sets up the sequence of updates performed by the simulation.
* `measurements = Dict{Symbol, AbstractMeasurement}()`: A collection of named measurements that run during the simulation. These are usually added after creating the simulation.
* `thermalization_measurements = Dict{Symbol, AbstractMeasurement}()`: Same as the above, but the measurements run during the thermalization stage. Might be useful to judge convergence or the number of necessary sweeps.
* `recorder = ConfigRecorder`: A recorder for the configurations generated during the simulation.
* `recording_rate = measure_rate`: The rate at which configurations are recorded.

Running a simulation also comes with a bunch of options via keyword arguments - most dealing with saving the simulation. The options for `run(dqmc)` include:

* `verbose = true`: Enables or disables information printing during the runtime.
* `ignore = tuple()`: Measurement keys to ignore during this run (none by default).
* `safe_before::TimeType = now() + Year(100)`: Sets a time stamp before which the simulation will cancel itself and save.
* `safe_every::TimePeriod = Hour(10000)`: Sets a time period interval for regular saves.
* `grace_period::TimePeriod = Minute(5)`: Sets a buffer time period for saving. By default saves at least 5 minutes before the requested time.
* `resumable_filename = "resumable_$(datestring).jld2`: Sets the name the savefile generated from safe_before and safe_every.
* `overwrite = true`: Enables or disables overwriting of existing files. 