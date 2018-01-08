# Custom models

Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, a key idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to extend the package and implement your own physical model. Sometimes examples say more than words, so feel encouraged to have a look at the implementations of the above mentioned models.

## Semantics

Loosely speeking, we define a Model to be a Hamiltonian on a lattice. Therefore, the lattice is part of a model. The motivation for this modeling is that the physics of a system does not only depend on the Hamiltonian but also (sometime drastically) on the underlying lattice. This is for example very obvious for spin systems which due to the lattice might become (geometrically) frustrated and show spin liquids physics. Also, from a technical point of view, lattice information is almost exclusively processed in energy calculations which both relate to the Hamiltonian (and therefore the model).

!!! note

    We will generally use the terminology Hamiltonian, energy and so on. However, this doesn't restrict you from defining your model as an Lagrangian with an action in any way as this just corresponds to a one-to-one mapping of interpretations.

## Mandatory fields and methods

Any concrete model (type), let's call it `MyModel` in the following, must be a subtype of the abstract type `MonteCarlo.Model`. To work with a Monte Carlo flavor, it **must** internally have at least the following fields:

 * `β::Float64`: inverse temperature
 * `l::Lattice`: any [`Lattice`](@ref MonteCarlo.Lattice)

Furthermore it **must** implement the following methods:

 * [`conftype`](@ref MonteCarlo.conftype): type of a configuration
 * [`energy`](@ref MonteCarlo.energy): energy of configuration
 * [`rand`](@ref MonteCarlo.rand): random configuration
 * [`propose_local`](@ref MonteCarlo.propose_local): propose local move
 * [`accept_local`](@ref MonteCarlo.accept_local): accept a local move

 A full list of methods with precise signatures that should be implemented for `MyModel` can be found here: [Methods: Models](@ref).

 ## Lattice requirements

 The Hamiltonian of your model might impose some requirements on the `Lattice` object that you use as it must provide you with enough lattice information.

 It might be educating to look at the structure of the simple [`SquareLattice`](@ref MonteCarlo.SquareLattice) struct.

 ```julia
 mutable struct SquareLattice <: CubicLattice
    L::Int
    sites::Int
    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
    sql::Matrix{Int}
    SquareLattice() = new()
end
```

It only provides access to next nearest neighbors through the arrays `neighs` and `neighs_cartesian`. If your model's Hamiltonian requires higher order neighbor information, because of, let's say, a next next nearest neighbor hopping term, the `SquareLattice` doesn't suffice. You could either extend this Lattice or implement a `NNSquareLattice` for example.