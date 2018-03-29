# Interface: Monte Carlo (MC)

Any model that wants to be simulated by means of MC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the Monte Carlo flavor [Monte Carlo (MC)](@ref).

Example models: [Ising Model](@ref)

## Mandatory fields

 * `l::Lattice`: any [`Lattice`](@ref Lattices)

## Index of all methods

```@index
Pages = ["MC.md"]
```

## Mandatory methods

```@meta
CurrentModule = MonteCarlo
```

```@docs
conftype(::Type{MC}, m::Model)
```

```@docs
energy(mc::MC, m::Model, conf)
```

```@docs
rand(::MC, ::Model)
```

```@docs
propose_local(mc::MC, m::Model, i::Int, conf, E::Float64)
```

```@docs
accept_local!(mc::MC, m::Model, i::Int, conf, E::Float64, delta_i, delta_E::Float64)
```

## Optional methods

```@docs
global_move(mc::MC, m::Model, conf, E::Float64)
```

```@docs
prepare_observables(mc::MC, m::Model)
```

```@docs
measure_observables!(mc::MC, m::Model, obs::Dict{String,Observable}, conf, E::Float64)
```

```@docs
finish_observables!(mc::MC, m::Model, obs::Dict{String,Observable})
```
