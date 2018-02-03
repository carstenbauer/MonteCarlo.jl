# Interface: Monte Carlo (MC)

Any model that wants to be simulated by means of MC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the Monte Carlo flavor [Monte Carlo (MC)](@ref).

## Mandatory fields

 * `l::Lattice`: any [`Lattice`](@ref Lattices)

## Index of all methods

```@index
Pages = ["MC.md"]
```

## Mandatory methods

```@autodocs
Modules = [MonteCarlo]
Order   = [:function]
Pages = ["MC_mandatory.jl"]
```

## Optional methods

```@autodocs
Modules = [MonteCarlo]
Order   = [:function]
Pages = ["MC_optional.jl"]
```
