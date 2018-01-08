# Monte Carlo

This is plain simple classical Monte Carlo (MC). It can for example be used to simulate the Ising model (see [2D Ising Model](@ref)).

You can initialize a Monte Carlo simulation of a given `model` simply through
```julia
mc = MC(model)
```

Allowed keywords are:

* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `global_moves`: wether global moves should be proposed
* `global_rate`: frequency for proposing global moves

Afterwards, you can run the simulation by
```julia
run!(mc)
```

Note that you can just do another `run!(mc, sweeps=1000)` to continue the simulation.

## Model interface

Any model that wants to be simulatable by means of MC must implement the following interface, a specification of mandatory and optional fields and methods.
One can exploit multiple dispatch to (if sensible) make a model work with multiple Monte Carlo flavors.

### Mandatory fields

 * `β::Float64`: inverse temperature
 * `l::Lattice`: any [`Lattice`](@ref MonteCarlo.Lattice)

### Mandatory methods

 * [`conftype`](@ref MonteCarlo.conftype): type of a configuration
 * [`energy`](@ref MonteCarlo.energy): energy of configuration
 * [`rand`](@ref MonteCarlo.rand): random configuration
 * [`propose_local`](@ref MonteCarlo.propose_local): propose local move
 * [`accept_local`](@ref MonteCarlo.accept_local): accept a local move
 
Precise signatures can be found here: [Methods: MC](@ref).

### Optional methods

 * [`global_move`](@ref MonteCarlo.global_move): propose and accept or reject a local move
 * [`prepare_observables`](@ref MonteCarlo.prepare_observables): initialize observables
 * [`measure_observables!`](@ref MonteCarlo.measure_observables!): measure observables
 * [`finish_observables!`](@ref MonteCarlo.finish_observables!): finish measurements
 
 Precise signatures can be found here: [Methods: MC](@ref).
 
 ## Potential extensions
 
 Pull requests are very much welcome!
 
 * Heat bath (instead of Metropolis) option
