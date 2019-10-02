# Measurements

Monte Carlo methods are often used to calculate expectation values related to a complex probability distribution. These expectation values can be written as

\begin{equation}
    \langle X \rangle = \sum_C X(C) p(C) \approx \sum_{C \in MC} X_{C}
\end{equation}

or functions thereof. To compute the former case, we allow for "measurements" of $X(C)$ during the simulation with the `Measurement` interface.

### Setting up Measurements

Measurements can be set when calling `MC()` or `DQMC()` via the keywords `thermalization_measurements::Dict{Symbol, AbstractMeasurement}` (which run only during the thermalization process) and `measurements::Dict{Symbol, AbstractMeasurement}` (which only run during the measurement phase). By default `thermalization_measurements` remain empty and `measurements` are populated with some default measurements for the algorithm and model.

One can also add and remove measurements from an existing simulation using

* `push!(mc, name => measurement[, stage = :ME])`: Adds a `measurement <: AbstractMeasurement` with some `name::Symbol` to the given simulation `mc <: MonteCarloFlavor`. By default, the measurement will be added to the measurement stage (`stage = :ME`). To add it to the thermalization stage, use `stage = :TH`.
* `delete!(mc, name[, stage = :ME])`: Deletes a measurement from a simulation `mc <: MonteCarloFlavor` by `name::Symbol`.
* `delete!(mc, measurement[, stage = :ME])`: Deletes a measurement by type instead of by name.

### Recovering Measurements

* `measurements(mc)`: Returns a nested dictionary containing all measurement of the given simulation `mc <: MonteCarloFlavor`.
* `observables(mc)`: Searches every measurement of the given simulation `mc <: MonteCarloFlavor` for fields `<: MonteCarloObservable.AbstractObservable` and generates a nested dictionary of them.


### Saving and loading measurements

* `save_measurements(mc, filename[; force_overwrite=false, allow_rename=true])`: Saves all measurements in the simulation `mc <: MonteCarloFlavor` to a given `filename`. The resulting file is a HDF5 file following the structure of `observables(mc)`. If `allow_rename = true`, the `filename` will be changed if it already exists. If `force_overwrite = true` and existing `filename` will be overwritten. If both are false, an error occurs when writing to an existing `filename`.
* `load_measurements(filename)`: Loads measurements from a given `filename`. The structure matches `observables(mc)`.


## Building your own Measurements

Every measurement **must** follow a few conventions:

* The measurement must be a (mutable) struct inheriting from `AbstractMeasurement`. (You are however free to add intermediate abstract types.)
* It must implement `measure!(m::MyMeasurement, mc, model, i)`, where `mc <: MonteCarloFlavor`, `model <: Model` and `i` is the current sweep index.
* While technically not required, `MyMeasurement(mc, model)` should be a callable constructor. Otherwise `push!`, `delete!` (by type) and `default_measurements` will not work.

Additionally you may implement

* `default_measurements(mc, model)`: Returns a `Dict{Symbol, AbstractMeasurement}` for a given `mc <: MonteCarloFlavor` and `model <: Model`. This function is called by the constructors for `MC` and `DQMC` to generate measurements. By default and empty dictionary is returned.
* `observables(m::MyMeasurement)`: By default returns all fields `<: MonteCarloObservable.AbstractObservable` of the given measurement `m`.
* `save!(m::MyMeasurement, filename, entryname)`: Saving routine for the given measurement. This method will be called by `save_measurements` with the same `filename`. `entryname` represents the "path" taken to get to the measurement, i.e. "ME/config" for a configuration measurement at the measurement stage. The default implementation will save each observable returned by `observables(m)`.
* `prepare!(m::MyMeasurement, mc, model)`: This method is called just before the first `measure!`. (May get removed)
* `finish!(m::MyMeasurement, mc, model)`: This method is called just after the last `measure!`. (May get removed)


### Example

Every measurement is implemented using this interface. As a simple example, let us look at `ConfigurationMeasurement`. You may check the source code for more examples, e.g. `src/flavors/DQMC/measurements.jl` or `src/models/Ising/measurements.jl`.

The type `ConfigurationMeasurement` is implemented as

```julia
struct ConfigurationMeasurement <: AbstractMeasurement
    obs::Observable
    rate::Int64
    ConfigurationMeasurement(mc, model, rate=1) = new(
        Observable(typeof(mc.conf), "Configurations"), rate
    )
end
```

It includes an Observable `obs`, which keeps track of the measured values and has an additional field `rate` to throttle the number of values saved. It also includes a constructor `ConfigurationMeasurement(mc, model, rate=1)` callable as `ConfigurationMeasurement(mc, model)`.

The method for `measure!` is given by

```julia
function measure!(m::ConfigurationMeasurement, mc, model, i::Int64)
    (i % m.rate == 0) && push!(m.obs, conf(mc))
    nothing
end
```

Whenever it is called it either skips the measurements or records the current configuration `conf(mc)` in its observable `m.obs`.
