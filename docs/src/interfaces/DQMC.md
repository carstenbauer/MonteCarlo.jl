# Interface: Determinant Quantum Monte Carlo (DQMC)

Any model that wants to be simulated by means of DQMC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the determinant Monte Carlo flavor [Determinant Quantum Monte Carlo (DQMC)](@ref).

## Mandatory fields

 * `l::Lattice`: any [`Lattice`](@ref Lattices)

## Index of all methods

```@index
Pages = ["DQMC.md"]
```

## Mandatory methods

```@autodocs
Modules = [MonteCarlo]
Order   = [:function]
Pages = ["DQMC_mandatory.jl"]
```

## Optional methods

```@autodocs
Modules = [MonteCarlo]
Order   = [:function]
Pages = ["DQMC_optional.jl"]
```

```@docs
MonteCarlo.rand(::DQMC)
```
