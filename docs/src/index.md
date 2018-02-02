# Documentation

This is a package for numerically simulating physical systems in Julia. The purpose of this package is to supply efficient Julia implementations of Monte Carlo flavors for the study of physical models of spins, bosons and/or fermions. Examples that ship with the package are

* Ising spin model simulated by Monte Carlo
* Fermionic Hubbard model simulated by variants of determinant quantum Monte Carlo

## Installation

To install the package execute the following command in the REPL:
```julia
Pkg.clone("https://github.com/crstnbr/MonteCarlo.jl")
```

Afterwards, you can use `MonteCarlo.jl` like any other package installed with `Pkg.add()`:
```julia
using MonteCarlo
```

To obtain the latest version of the package just do `Pkg.update()` or specifically `Pkg.update("MonteCarlo")`.
