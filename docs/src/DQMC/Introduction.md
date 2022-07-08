# Determinant Quantum Monte Carlo

Determinant Quantum Monte Carlo is a Quantum Monte Carlo algorithm for fermionic Hamiltonians. The general idea is to use the Hubbard-Stranovich transformation to simplify any term with four fermionic operators to two. This comes at the cost of introducing a bosonic field which is sampled by a Monte Carlo procedure.

The minimal working example for a DQMC simulation is the following.

```julia
using MonteCarlo

dqmc = DQMC(HubbardModel(), beta = 1.0)
run(dqmc)
```

This will set up and run a DQMC simulation at inverse temperature $\beta = 1.0$ using an attractive Hubbard model with a two by two square lattice. Of course this example is not very useful. In practice you will want to modify lots of parameters and components of the simulation. We give a brief overview of the different components here. You can also check the examples section for some more involved and realistic examples.

#### Model

The model is a description of the Hamiltonian. Currently MonteCarlo.jl only implements the `HubbardModel` which includes nearest neighbor hoppings, a chemical potential and the Hubbard interaction. One aim of this package is to make it simple to implement different models, though currently only Hubbard interactions are supported.

#### Lattice

MonteCarlo.jl provides its own lattice structure inspired by [LatticePhysics.jl](https://github.com/janattig/LatticePhysics.jl). The structure is generic, meaning that it should be possible to implement any lattice with it. Lattices are currently part of the model and the `HubbardModel` should work with any lattice. 

#### Field

The field defines how the interaction is handled, i.e. what Monte Carlo configurations look like, how the interactive term is interpreted and how updates work. We currently implement Hirsch fields and Gauß-Hermite Quadrature fields, both with a density- and magnetic-channel version. Any of these choices works with any choice of the Hubbard model, though performance and accuracy can vary.

#### Updates

Beyond local updates we provide a selection of global and experimental parallel updates. Like models, this part is designed to be relatively easy to extend. Most global updates just generate a new configuration and call another function to accept or reject it.

The updates themselves are collected in a scheduler. We currently provide two types of them - one which simply iterates through updates and one which adaptively favors updates with high acceptance rates.

#### Configuration Recorder

We provide an interface to record configurations during a MonteCarlo run so that they can later be replayed. This interface also implements functionality for compressing configurations.

#### Measurements

We currently include measurements of the following observables:
- Equal time Greens function (and to a less direct extend time displaced Greens functions)
- occupations
- charge density correlations and susceptibilities
- magnetization in x, y and z direction
- spin density correlations and susceptibilities in x, y and z direction
- pairing correlation and susceptibility for generic symmetries
- current current susceptibilities which are required for the superfluid stiffness (this can also be calculated more directly)
- total and partial energies (quadratic terms and quartic terms separated)

#### DQMC

The `DQMC` struct represents the full simulation. All the components above are collected here. It also contains a bunch paramters, such as the inverse temperature `beta`, the time step discretization `delta_tau`, the number of thermazition and measurements sweeps and more.


## Derivation

If you are interested in the derivation of DQMC you may check [Introduction to Quantum Monte Carlo Simulations for fermionic Systems](https://doi.org/10.1590/S0103-97332003000100003), the book [Quantum Monte Carlo Methods](https://doi.org/10.1017/CBO9780511902581) or [World-line and Determinantal Quantum Monte Carlo Methods for Spins, Phonons and Electrons](https://doi.org/10.1007/978-3-540-74686-7_10). The first reference is most in-line with the implementation of this package.

If you want to go through the source code, compare it and verify for yourself that it is correct there a couple of things that should be pointed out. Most educational sources use the asymmetric two term Suzuki-Trotter decomposition. We use the symmetric three term version for increased accuracy.

```math
B(l) = e^{-\Delta\tau \sum_l T+V(l)} = \prod_j e^{-\Delta\tau T/2} e^{-\Delta\tau V} e^{-\Delta\tau T/2} + \mathcal{O}(\Delta\tau^2)
```

This change is however no trivial as the first or last element of the $B$ matrix/operator needs to be an exponentiated interaction. To get this we use an effective greens function, which cyclically permutes one exponentiation hopping term to the other end of the chain. This adjustment needs to be undone for the actual greens function, which happens in `greens()`.

Another thing worth mentioning is that depending on the choices made at the start of the derivation, matrix products may have different order and indices may vary. The first source should have the same definitions.