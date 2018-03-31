# Introduction

**MonteCarlo.jl** is a Julia software library for the simulation of physical models by means of the Markov Chain Monte Carlo technique. The package implements classical and quantum Monte Carlo flavors which can be used to study spin systems, interacting fermions, and boson-fermion mixtures.

### Included models:

* Ising model
* Attractive Hubbard model

### Included Monte Carlo flavors

* Classical Monte Carlo
* Determinant Quantum Monte Carlo (also known as auxiliary field Monte Carlo)

### Included lattices

* Cubic lattices (chain, square, cube, ...)
* Any ALPS lattice

## Installation

MonteCarlo.jl hasn't been released yet. To install the package execute the following command in the Julia REPL:
```julia
Pkg.clone("https://github.com/crstnbr/MonteCarlo.jl")
```

Afterwards, you can use `MonteCarlo.jl` like any other package installed with `Pkg.add()`:
```julia
using MonteCarlo
```

To obtain the latest version of the package just do `Pkg.update()` or specifically `Pkg.update("MonteCarlo")`.
