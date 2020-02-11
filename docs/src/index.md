![logo](assets/logo.png)

# Introduction

**MonteCarlo.jl** is a Julia software library for the simulation of physical models by means of the Markov Chain Monte Carlo technique. The package implements classical and quantum Monte Carlo flavors which can be used to study spin systems, interacting fermions, and boson-fermion mixtures.

!!! warning

    The documentation is out-of-date!

### Included models:

* Ising model
* Attractive Hubbard model

### Included Monte Carlo flavors

* Classical Monte Carlo
* Determinant Quantum Monte Carlo (also known as auxiliary field Monte Carlo)

### Included lattices

* Honeycomb lattice and cubic lattices (chain, square, cube, ...)
* Any ALPS lattice
* Any (https://github.com/janattig/LatticePhysics.jl)[LatticePhysics.jl] lattice

Have a look at the [Showcase](@ref) section to get some inspiration.

## Study your own model

A major idea behind the design of the package is convenient customization. Users should be able to define their own custom physical models (or extend existing ones) and explore their physics through Monte Carlo simulations. We hope that **MonteCarlo.jl** allows the user to put his focus on his physical model rather than tedious implementations of Monte Carlo schemes.

To that end, each (quantum) Monte Carlo flavor has a well-defined model interfaces, i.e. a precise specification of mandatory and optional fields and methods. An example can be found here: [Interface: Monte Carlo (MC)](@ref). Practically, it makes sense to start by taking a look at the implementation of one of the predefined models.

If you implement a custom model that might be worth being added to the collection of supplied models, please consider creating a [pull request](https://github.com/crstnbr/MonteCarlo.jl/pulls)!

## GitHub

**MonteCarlo.jl** is [open-source](https://en.wikipedia.org/wiki/Open-source_software). The source code can be found on [github](https://github.com/crstnbr/MonteCarlo.jl). Criticism and contributions are very much welcome - just [open an issue](https://github.com/crstnbr/MonteCarlo.jl/issues/new). For more details see the [Contribution Guide](Contribution Guide).
