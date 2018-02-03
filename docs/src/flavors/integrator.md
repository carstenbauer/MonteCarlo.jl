# Monte Carlo integration (Integrator)

Basic Monte Carlo integration. It can for example be used to integrate a multidimensional [Gaussian function](@ref GaussianFunction).

You can initialize a Monte Carlo integration of a given `model` (function) by
```julia
mc = Integrator(model)
```
where the integration volume is taken as a box.

Allowed keywords are:

* `min_x`: lower integration bounds vector
* `max_x`: upper integration bounds vector
* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `seed`: initialize MC with custom seed

Afterwards, you can run the simulation by
```julia
run!(mc)
```

Note that you can just do another `run!(mc, sweeps=1000)` to continue the simulation.

By default, every Monte Carlo integration stores the energies (function values) and the estimate for the integral as observables. You can for example access the latter via

```julia
mean(mc.obs["integral"])
```

## Examples

```julia
julia> g = GaussianFunction()
GaussianFunction (Mean: [0.0], Std: [1.0])

julia> mc = Integrator(g)
Monte Carlo integration
Function: GaussianFunction (Mean: [0.0], Std: [1.0])
Lower bounds: [-10.0], Upper bounds: [10.0]

julia> run!(mc, thermalization=100000, sweeps=100000, verbose=false);
Integral is 14.146761636413965

```

TODO: Nothing important, just that the result is wrong :)

## Exports

```@autodocs
Modules = [MonteCarlo]
Private = false
Order   = [:function, :type]
Pages = ["Integrator.jl"]
```
