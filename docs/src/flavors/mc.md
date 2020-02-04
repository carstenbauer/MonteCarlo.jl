# Monte Carlo (MC)

This is plain simple Monte Carlo (MC). It can for example be used to simulate the Ising model (see [2D Ising model](@ref)).

You can initialize a Monte Carlo simulation of a given `model` simply through
```julia
mc = MC(model, beta=5.0)
```

Mandatory keywords are:

* `beta`: inverse temperature

Optional keywords are:

* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `global_moves`: wether global moves should be proposed
* `global_rate`: frequency for proposing global moves
* `seed`: initialize MC with custom seed
* `measure_rate`: rate at which measurements are taken
* `print_rate`: rate at which prints happen for `verbose=true`

Afterwards, you can run the simulation by
```julia
run!(mc)
```

Note that you can just do another `run!(mc, sweeps=1000)` to continue the simulation.

## Examples

You can find example simulations of the 2D Ising model under [Getting started](@ref Usage) and here: [2D Ising model](@ref).

## Exports

```@autodocs
Modules = [MonteCarlo]
Private = false
Order   = [:function, :type]
Pages = ["MC.jl"]
```

### Potential extensions

Pull requests are very much welcome!

 * Heat bath (instead of Metropolis) option
