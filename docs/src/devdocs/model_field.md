# Models and Fields

The model and field of a simulation are two rather tightly connected components. The model represents the raw Hamiltonian of a problem, including an interactive term with four operators. The field controls how this interactive term is represented in a DQMC simulation. As such you may think of it as the implementation of the Hubbard Stratonovich transformation.

## The Model

The model contains the lattice (which perhaps should be moved out) and whatever parameters are relevant to the model, i.e. whatever factors appear in the Hamiltonian.

### DQMC - Model Interface

The model interfaces with the rest of the simulation through a set of functions. These may generally be categorized as mandatory, optional and somewhere in between.

#### Mandatory Methods

Mandatory methods are those every model has to provide for a DQMC simualtion to work.

- `lattice(model)` needs to return a MonteCarlo compatible lattice.
- `total_flavors(model)` is the total number of fermion flavors present in the model. This will control the number of flavors considered in measurements.
- `unique_flavors(model)` is the number of unique fermion flavors present in the hopping matrix of the model. This should return 1 if the hopping matrix does not depend on the flavor index. 
- `hopping_matrix(model)` needs to generate the hopping matrix, which includes all quadratic terms. (I.e. also the chemical potential.) The hopping matrix should only include as many flavors as necessary. If the hopping matrix contains two copies of the same matrix, one for spin up and one for spin down for example, then it should only return one of these. Expanding it to an appropriate size is handled internally.

#### (Semi-) Optional Methods

These methods aren't strictly necessary to implement, but are typically good to have.

- `_save(file::FileLike, entryname, model)` writes model information to the given file. It should also save the lattice via `_save(file, entryname, lattice)` and save a unique `tag`. If this is not implemented JLD2 will be asked to save the type as is, which makes it hard to load data when the model type is edited.
- `_load(data, ::Val{Symbol(tag)})` loads a model from `data`, which typically is a JLDFile. Note that saved tag is used to dispatch to the correct method.
- `intE_kernel(mc, model, idxs, G, ::Val{flv})` should be implemented to enable measurements of the energy from the interactive term as well as the total energy. 
- `choose_field(model)` returns the default field type for a given model. If this is not implemented a field must be passed to `DQMC`.

#### Very Optional Methods

These methods should work as is with their default implementation, or are unnecessary to the simulation. 

- `hopping_eltype(model) = eltype(hopping_matrix(model))` returns the element type of the hopping matrix.
- `hopping_matrix_type(field, model) = typeof(pad_to_nflavors(field, model, hopping_matrix(model)))` return the matrix type of the hopping matrix.
- `greens_eltype(field, model) = generalized_eltype(interaction_eltype(field), hopping_eltype(model))` returns the element type of the greens function. This must be compatible with element types of the hopping matrix and the interaction matrix.
- `greens_matrix_type(field, model) = Matrix{greens_eltype(field, model)}` returns the full type of the greens function. This must be compatible with both the type of the hopping matrix and the interaction matrix.
* `parameters(m::Model)` should collect the parameters from the model and lattice in a NamedTuple. This is a user facing function meant to collect relevant/summarized information on a simulation.

Also note that you may need to update the measurement kernels. More information about that on the measurement page.

## The Field

The field represents the interactive term post Hubbard-Stratonovich transform in DQMC. It generally consists of some constants necessary to calculate local updates propabilities, update the Greens function and calculate the interaction matrix, and the MonteCarlo configuration.

In terms of functionality we can again split the implemented function into a mandatory and optional category

#### Mandatory

The mandatory function are necessary for a simualtion to run. Technically `propose_local!` and `accept_local!` are not absolutely mandatory, as one could run a simulation with only global updates. Doing so is however not a reasonable thing to do.

- `unique_flavors(field)` is the minimum number of flavors the field requires. (Some Hubbard Stratonovich transformations may introduce an asymmetry between flavors which was not in the original Hamiltonian.)
- `interaction_matrix_exp!(field, result, time_slice, power)` writes the interaction matrix corresponding to the current field configuration to the given `result` matrix. `time_slice` defines the relevant imaginary time and `power = -1` is used to identify that the matrix needs to be inverted (otherwise `power = 1`).
- `propose_local!(mc, field, site, time_slice)` calculates the determinant ratio and the bosonic energy associated with an update to the field configuration at some `site` and `time_slice`. If the field configuration has multiple values it can update to, this function should randomly choose one of those values. The function may pass any number of extra arguments as a thrid return value, which will then be forwarded to `accept_local!()`.
- `accept_local!(mc, field, site, time_slice, args...)` updates the configuration and the greens matrix under the assumption that a previously proposed updates has been accepted. `site` and `time_slice` identify the element of the configuration that needs to be updated and `args...` may contain any number of arguments passed on from `propose_local!()`.

The math and temporary variables used in the calculation of local updates are collected in the `FieldCache`. Between different combinations of types and flavors, you should find the optimized update formulas from [Quantum Monte Carlo Methods](https://doi.org/10.1017/CBO9780511902581) here. 

#### Optional

The optional functions presented here should be implemented for the field to be fully compatible and performant. 

- `Base.rand(field)` and `Random.rand!(field)` are overloaded to provide a new field configuration. Mandatory for global updates.
- `compress(field)`, `compressed_conf_type(field)`, `decompres(field, compressed)` and `decompress!(field, compressed)` are implemented to fill out the compression interface for (configuration) recorders. Optional for file size optimization.
- `interaction_eltype(field)` and `interaction_matrix_type(field, model)` give type information on the interaction matrix to the simulation. Since the interaction matrix is set in place, this does not have efficient defaults and thus not implementing this will usually result in bad performance. (Defaults to `Float64` and `Matrix{Float64}`.)
- `energy_boson(field[, conf = field.conf])` computes the bosonic energy associated with a given configuration. This is mandatory for global updates.


## For later consideration

As things are now, the field represents most of the interaction, with only the prefactor being held by the model. The model mainly produces the hopping matrix and contains the lattice. It may make sense to separate these components more, specifically the field as it is the main driver of complexity. 

Example structure:
```julia
# maybe skip this struct entirely?
struct HubbardModel <: Model
    lattice::AbstractLattice # maybe move this to DQMC struct?
    hopping::GenericHoppings
    interaction::HubbardInteraction
end

struct GenericHoppings
    # use bond labels as key to this dict (with some default key fallback)
    parameters::Dict{Symbol, Number}
end

struct HubbardInteraction
    U::Float64
end

struct SomeHubbardField
    interaction::HubbardInteraction

    internals...
end

function field_constructor(parameters, interaction::HubbardInteraction)
    ...
end
```