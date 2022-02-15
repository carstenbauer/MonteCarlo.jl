# Models

In MonteCarlo.jl a `Model` describes a Hamiltonian. It's primary purpose is to collect paramaters from different terms in the Hamiltonian as well as the lattice and generate a hopping matrix for the simulation. We currently provide one model - the Hubbard model.
## Hubbard Model

The  Hubbard model is given by

```math
\mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) - U \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j n_{j},
```

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site Hubbard interaction strength, $\mu$ the chemical potential and $\langle i, j \rangle$ indicates that the sum has to be taken over nearest neighbors. Note that the Hamiltonian is written in particle-hole symmetric form such that $\mu = 0$ corresponds to half-filling.

Our implementation allows for both attractive (positive) and repulsive (negative) $U$. Note that for the repulsive case there is a sign problem for $\mu \ne 0$. The model also works with any lattice, assuming that lattice provides the required functionality. 

You can create a Hubbard model with `HubbardModel()`. Optional keyword arguments include:
- `l::AbstractLattice = choose_lattice(HubbardModel, dims, L)` is the lattice used by the model. 
- `dims::Integer = 2` is the dimensionality of the default lattice (Chain/Square/Cubic)
- `L::Integer = 2` is the linear system size of the default lattice (Chain/Square/Cubic)
- `t::Float64 = 1.0` is the hopping strength.
- `mu::Float64 = 0.0` is the chemical potential. (Must be 0 if U is negative.)
- `U::Float64 = 1.0` is the interaction strength.

## Creating your own Model

To create your own model you will need to inherit from the abstract type `Model`. There is a set of mandatory and optional methods you must/can implement:

#### Mandatory Methods

A custom model needs to implement these methods to function

- `lattice(model)` needs to return a MonteCarlo compatible lattice
- `nflavors(model)` needs to return the number of unique fermion flavors of the hopping matrix. For example, in a two spin model this would return 2 if the hopping matrix is different between spin up and down, or 1 if one sector is a copy of the other. Internally this is used together with `nflavors(field)` to optimize spin/flavor symmetric systems.
- `hopping_matrix(dqmc, model)` needs to generate the hopping matrix, which includes all quadratic terms. (I.e. also the chemical potential.) The hopping matrix should only include as many flavors as necessary. If the hopping matrix contains two copies of the same matrix, one for spin up and one for spin down for example, then it should only return one of these. Expanding it to an appropriate size is handled internally.

#### (Semi-) Optional Methods

These methods aren't strictly necessary to implement, but may boost performance when implemented. It is recommended to provide these if the defaults do not apply.

- `hopping_eltype(model) = Float64` returns the element type of the hopping matrix.
- `hopping_matrix_type(field, model) = Matrix{hopping_eltype(model)}` return the matrix type of the hopping matrix.
- `save_model(file::JLDFile, model, entryname)` should write model information to the given file. It should also save the lattice via `save_lattice` and save a unqiue `tag`. If this is not implemented JLD2 will be asked to save the type as is, which makes it hard to load data when the model type is edited.
- `_load_model(data, ::Val{Symbol(tag)})` loads a model from `data`, which typically is a JLDFile. Note that saved tag is used to dispatch to the correct method.
- `intE_kernel(mc, model, G, ::Val{flv})` should be implemented to enable measurements of the energy from the interactive term as well as the total energy. 
- `choose_field(model)` returns the default field type for a given model. If this is not implemented a field must be passed to `DQMC`.

#### Optional Methods

- `greens_eltype(field, model) = generalized_eltype(interaction_eltype(field), hopping_eltype(model))`
- `greens_matrix_type(field, model) = Matrix{greens_eltype(field, model)}`
* `parameters(m::Model)` should collect the parameters from the model and lattice in a NamedTuple.

Also note that you may need to update the measurement kernels. More information about that on the measurement page.