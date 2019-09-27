# Customize

## Custom models

Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, the central idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to extend the package and implement your own physical model (or variations of existing models). You can find the interfaces in the corresponding section of the documentation, for example: [Interface: Monte Carlo (MC)](@ref).

Sometimes examples tell the most, so feel encouraged to have a look at the implementations of the above mentioned models to get a feeling of how to implement your own model.

### General remarks for lattice models

#### Semantics

For lattice models, we define a Model to be a Hamiltonian on a lattice. Therefore, the lattice is part of the model (and not the Monte Carlo flavor). The motivation for this modeling is that the physics of a system does not only depend on the Hamiltonian but also (sometime drastically) on the underlying lattice. This is for example very obvious for spin systems which due to the lattice might become (geometrically) frustrated and show spin liquids physics. Also, from a technical point of view, lattice information is almost exclusively processed in energy calculations which both relate to the Hamiltonian (and therefore the model).

!!! note

    We will generally use the terminology Hamiltonian, energy and so on. However, this doesn't restrict you from defining your model as an Lagrangian with an action in any way as this just corresponds to a one-to-one mapping of interpretations.

### Model Requirements

A model must meat a few requirements given by the Monte Carlo flavor used. `MC` requires the following:

* A method `nsites(m::MyModel)` which returns the number of sites in the underlying lattice.
* A method `rand(::Type{MC}, m::MyModel)` which returns a new random configuration.
* A method `propose_local(mc::MC, m::MyModel, i::Int, conf)` which proposes a local update to a given configuration `conf` at site `i`. This method must return `(ΔE, x)`, i.e. the energy difference and "something" else. This "something" may include additional information useful to you during the update or simply be `nothing`.
* A method `accept_local(mc::MC, m::MyModel, i::Int, conf, x, ΔE)` which finalizes a local update. This includes updating the configuration `conf`. The inputs `x` and `ΔE` correspond to the outputs of `propose_local()`.

And `dqmc` requires:

* A method `nsites(m::MyModel)` which returns the number of sites in the underlying lattice.
* * A method `rand(::Type{DQMC}, m::MyModel, nslices::Int)` which returns a new random configuration.
* A method `propose_local(mc::DQMC, m::MyModel, i::Int, conf)` which proposes a local update to a given configuration `conf` at site `i` and the current time slice. This method must return `(detratio, ΔE_Boson, x)`, i.e. the determinant ratio corresponding to the fermion weight of the update, the boson energy difference giving the bosonic part and "something" else. This "something" may include additional information useful to you during the update or simply be `nothing`.
* A method `accept_local(mc::DQMC, m::MyModel, i::Int, conf, x, detratio, ΔE_Boson)` which finalizes a local update. This includes updating the configuration `conf` and the Greens function `mc.s.greens`. The inputs `x`, `detratio` and `ΔE_Boson` correspond to the outputs of `propose_local()`.
* A method `hopping_matrix(mc::DQMC, m::MyModel)` which returns the hopping matrix of the model (including chemical potential).
* A method `interaction_matrix_exp!(mc::DQMC, m::MyModel, result::Matrix, conf, slice::Int, power::Float64=1.0)` which calculates the exponentiated interaction matrix and saves it to `result`. The chemical potential must not be included here.

For either Monte Carlo flavor `propose_local` and `accept_local` are performance critical. `interaction_matrix_exp!` is also performance critical, however only required for `DQMC`.


## Custom lattices

As described in [Custom models](@ref) a lattice is considered to be part of a model. Hence, most of the requirements of a `Lattice` subtype come from potential models (see [General remarks on lattice model](@ref)). Below you'll find information on the requirements given by the Monte Carlo flavor as well as the implemented models.

### Lattice requirements

Any concrete lattice type, let's call it `MyLattice` in the following, must be a subtype of the abstract type `MonteCarlo.AbstractLattice`. Formally, there are no required methods or fields by the **Monte Carlo flavor**. However, since both flavors require `nsites(model)`, some field or method returning the number of sites of a lattice should exist.

For a lattice to work with the implemented **models**, it must

* implement a method `length(l::MyLattice)` giving the number of lattice sites.
* implement a method `neighbors_lookup_table(l::MyLattice)` returning the neighours lookup table
* either implement methods `_neighbors(::Nothing, l::MyLattice, directed::Val{true})` and `_neighbors(::Nothing, l::MyLattice, site_index::Integer)` or include a field `mylattice.neighs` which carries a neighbors lookup table and implement `has_neighbors_table(::MyLattice) = HasNeighborsTable()`.


The generic checkerboard decomposition `build_checkerboard()` further requires the lattice to

 * either implement a method `_neighbors(::Nothing, l::MyLattice, directed::Val{false})` or include a field `mylattice.bonds` which carries a bonds table and implement `has_bonds_table(::MyLattice) = HasBondsTable()`. The bonds table is a `Matrix{Int}` of size `(nbonds, 3)`, where `bonds[i, 1]` is the source site index, `bonds[i, 2]` the target site index and `bonds[i, 3]` an integer specifying the type of the i-th bond.


## Custom Monte Carlo flavors

Coming soon...
