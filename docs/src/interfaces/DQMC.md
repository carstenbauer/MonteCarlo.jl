# Interface: Determinant Quantum Monte Carlo (DQMC)

Any model that wants to be simulated by means of DQMC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the determinant Monte Carlo flavor [Determinant Quantum Monte Carlo (DQMC)](@ref).

Example models: [Attractive Hubbard Model](@ref)

## Mandatory fields

 * `l::AbstractLattice`: any [`AbstractLattice`](@ref Lattices)
 * `flv::Int`: number of distinct fermion flavors (e.g. spins, bands, etc.). The Green's function will have shape `(flv*N, flv*N)`, where `N` is the number of sites. Note that different fermion flavors are often related by symmetry and it's advisable to use this symmetry to work with smaller Green's function matrices. Have a look at the [Attractive Hubbard Model](@ref) as an example where `flv=1` although it has spinful fermions.

## Index of all methods

```@index
Pages = ["DQMC.md"]
```

## Mandatory methods

```@meta
CurrentModule = MonteCarlo
```

```@docs
conftype(::Type{DQMC}, m::Model)
```

```@docs
rand(::DQMC, ::Model)
```

```@docs
hopping_matrix(mc::DQMC, m::Model)
```

```@docs
interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.)
```

```@docs
propose_local(mc::DQMC, m::Model, i::Int, conf, E_boson::Float64)
```

```@docs
accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)
```

## Optional methods

```@docs
greenseltype(::Type{DQMC}, m::Model)
```

```@docs
energy_boson(mc::DQMC, m::Model, conf)
```

```@docs
prepare_observables(mc::DQMC, m::Model)
```

```@docs
measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf)
```

```@docs
finish_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable})
```
