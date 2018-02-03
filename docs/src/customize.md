# Customize

## Custom models

Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, the central idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to extend the package and implement your own physical model (or variations of existing models). You can find the interfaces in the corresponding section of the documentation, for example: [Interface: Monte Carlo (MC)](@ref).

Sometimes examples tell the most, so feel encouraged to have a look at the implementations of the above mentioned models to get a feeling of how to implement your own model.

### General remarks for lattice models

#### Semantics

For lattice models, we define a Model to be a Hamiltonian on a lattice. Therefore, the lattice is part of the model (and not the Monte Carlo flavor). The motivation for this modeling is that the physics of a system does not only depend on the Hamiltonian but also (sometime drastically) on the underlying lattice. This is for example very obvious for spin systems which due to the lattice might become (geometrically) frustrated and show spin liquids physics. Also, from a technical point of view, lattice information is almost exclusively processed in energy calculations which both relate to the Hamiltonian (and therefore the model).

!!! note

    We will generally use the terminology Hamiltonian, energy and so on. However, this doesn't restrict you from defining your model as an Lagrangian with an action in any way as this just corresponds to a one-to-one mapping of interpretations.

### Lattice requirements

The Hamiltonian of your model might impose some requirements on the `Lattice` object that you use as it must provide you with enough lattice information.

It might be educating to look at the structure of the simple `SquareLattice` struct.

```julia
mutable struct SquareLattice <: AbstractCubicLattice
   L::Int
   sites::Int
   neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
   neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
   sql::Matrix{Int}
   SquareLattice() = new()
end
```

It only provides access to next nearest neighbors through the arrays `neighs` and `neighs_cartesian`. If your model's Hamiltonian requires higher order neighbor information, because of, let's say, a next next nearest neighbor hopping term, the `SquareLattice` doesn't suffice. You could either extend this Lattice or implement a `NNSquareLattice` for example.

## Custom lattices

As described in [Custom models](@ref) a lattice is considered to be part of a model. Hence, most of the requirements for fields of a `Lattice` subtype come from potential models (see [Lattice requirements](@ref)). Below you'll find information on which fields are mandatory from a Monte Carlo flavor point of view.

### Mandatory fields

Any concrete lattice type, let's call it `MyLattice` in the following, must be a subtype of the abstract type `MonteCarlo.Lattice`. To work with a Monte Carlo flavor, it **must** internally have at least have the following field,

 * `sites`: number of lattice sites.

However, as already mentioned above depending on the physical model of interest it will typically also have (at least) something like

 * `neighs`: next nearest neighbors,

 as most Hamiltonian will need next nearest neighbor information.

The only reason why such a field isn't generally mandatory is that the Monte Carlo routine doesn't care about the lattice much. Neighbor information is usually only used in the energy (difference) calculation of a particular configuration like done in [`energy`](@ref MonteCarlo.energy) or [`propose_local`](@ref MonteCarlo.propose_local) which both belong to a `Model`.

## Custom Monte Carlo flavors

Coming soon...
