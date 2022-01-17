################################################################################
### Mandatory Interface
################################################################################

### Model
########################################

"""
    hopping_matrix(mc::DQMC, m::Model)

Calculates the hopping matrix \$T_{i\\sigma, j\\sigma '}\$ where \$i, j\$ are 
site indices and \$\\sigma , \\sigma '\$ are flavor indices (e.g. spin indices). 
The hopping matrix should also contain potential chemical potential terms on the 
diagonal.

A matrix element is the hopping amplitude for a hopping process: \$j,\\sigma ' 
\\rightarrow i,\\sigma\$.

Regarding the order of indices, if `T[i, σ, j, σ']` is your desired 4D hopping 
array, then `reshape(T, (n_sites * n_flavors, :))` is the hopping matrix.
"""
hopping_matrix(mc::DQMC, m::Model) = throw(MethodError(hopping_matrix, (mc, m)))

nflavors(m::Model) = throw(MethodError(nflavors, (m,)))

### Field
########################################

"""
    interaction_matrix_exp!(
        mc::DQMC, m::Model, field::AbstractField, 
        result::AbstractArray, slice::Int, power::Float64 = 1.0
    )

Calculate the, exponentiated interaction matrix 
`exp(- power * delta_tau * V(slice))` and store it in `result::AbstractArray`. 

This only includes terms with 4 operators, i.e. not the chemical potential or 
any hoppings. By default the calculation will be performed by the appropriate 
field type (i.e. by `interaction_matrix_exp!(field, result, slice, power)`)

By default this function will call 
`interaction_matrix_exp!(field, result, slice, power)`.

This is a performance critical method and one might consider efficient in-place 
(in `result`) construction.
"""
@inline function interaction_matrix_exp!(
        mc, model, field, result, slice, power = +1.0
    )
    interaction_matrix_exp!(field, result, slice, power)
end
function interaction_matrix_exp!(f, A, s, p)
    throw(MethodError(interaction_matrix_exp!, (f, A, s, p)))
end

"""
    propose_local(mc::DQMC, m::Model, field::AbstractField, i::Int, slice::Int)

Propose a local move for lattice site `i` at time slice `slice` for a `field` 
holding the current configuration. Returns the Green's function determinant 
ratio, the boson energy difference `ΔE_boson = E_boson_new - E_boson`,
and any extra information `passthrough` that might be useful in `accept_local`.

By default this function will call `propose_local(mc, field, i, slice)`.

See also [`accept_local!`](@ref).
"""
@inline propose_local(mc, m, field, i, slice) = propose_local(mc, field, i, slice)
propose_local(mc, f, i, s) = throw(MethodError(propose_local, (mc, f, i, s)))


"""
    accept_local!(
        mc::DQMC, m::Model, field::AbstractField, i::Int, slice::Int, 
        detratio, ΔE_boson, passthrough
    )

Accept a local move for site `i` at imaginary time slice `slice` of current 
configuration in `field`. Arguments `detratio`, `ΔE_boson` and `passthrough` 
correspond to output of `propose_local` for that local move.

By default this function will call
`accept_local!(mc, field, i, slice, detration, ΔE_boson, passthrough)`

See also [`propose_local`](@ref).
"""
@inline function accept_local!(
        mc, m, field, i, slice, detratio, ΔE_boson, passthrough
    )
    accept_local!(mc, field, i, slice, detratio, ΔE_boson, passthrough)
end
function accept_local!(mc, f, i, s, d, ΔE, pt)
    throw(MethodError(accept_local!, (mc, f, i, s, d, ΔE, pt)))
end

Base.rand(f::AbstractField) = throw(MethodError(rand, (f,)))
Random.rand!(f::AbstractField) = throw(MethodError(rand!, (f,)))

nflavors(f::AbstractField) = throw(MethodError(nflavors, (f,)))

# For ConfigRecorder
compress(f::AbstractField) = throw(MethodError(compress, (f, )))
compressed_conf_type(f::AbstractField) = throw(MethodError(compressed_conf_type, (f, )))
decompress(f::AbstractField, c) = throw(MethodError(decompress, (f, c)))    
decompress!(f::AbstractField, c) = throw(MethodError(decompress!, (f, c)))


################################################################################
### Optional Interface
################################################################################


### Mixed (Field + Model)
########################################

"""
    greens_eltype(field, model)

Returns the type of the elements of the Green's function matrix. Defaults to 
Float64 if both the hopping and interaction matrix contain floats and ComplexF64
otherwise.
"""
function greens_eltype(field::AbstractField, model::Model)
    generalized_eltype(interaction_eltype(field), hopping_eltype(model))
end

"""
    greens_matrix_type(field, model)

Returns the (matrix) type of the greens and most work matrices. Defaults to 
`Matrix{greens_eltype(T, m)}`.
"""
greens_matrix_type(f::AbstractField, m::Model) = Matrix{greens_eltype(f, m)}


### Model
########################################

"""
    hopping_eltype(model)

Returns the type of the elements of the hopping matrix. Defaults to `Float64`.
"""
hopping_eltype(::Model) = Float64

"""
    hopping_matrix_type(field, model)

Returns the (matrix) type of the hopping matrix. Defaults to 
`Matrix{hopping_eltype(model)}`.
"""
hopping_matrix_type(::AbstractField, m::Model) = Matrix{hopping_eltype(m)}


### Field
########################################

"""
    interaction_eltype(model)

Returns the type of the elements of the interaction matrix. Defaults to `Float64`.
"""
interaction_eltype(::AbstractField) = Float64


"""
    interaction_matrix_type(field, model)

Returns the (matrix) type of the interaction matrix. Defaults to 
`Matrix{interaction_eltype(model)}`.
"""
interaction_matrix_type(f::AbstractField, m::Model) = Matrix{interaction_eltype(f)}


"""
    init_interaction_matrix(field::AbstractField, model::Model)

Returns an initial interaction matrix. This only used to allocate a correctly 
sized matrix.

By default this uses the matrix type from `interaction_matrix_type` and uses 
`max(nflavors(field), nflavors(model)) * length(lattice(model))` as the size.
"""
function init_interaction_matrix(f::AbstractField, m::Model)
    flv = max(nflavors(f), nflavors(m))
    N = length(lattice(m))
    FullT = interaction_matrix_type(f, m)

    if FullT <: BlockDiagonal
        MT = matrix_type(interaction_eltype(f)) 
        return BlockDiagonal(MT(undef, N, N), MT(undef, N, N))
    elseif FullT <: Diagonal
        VT = vector_type(interaction_eltype(f))
        Diagonal(VT(undef, flv * N))
    else
        FullT(undef, flv * N, flv * N)
    end
end

"""
    energy_boson(mc::DQMC, model::Model, conf)

Calculate bosonic part (non-Green's function determinant part) of energy for 
configuration `conf` for Model `m`.

This is required for global and parallel updates as well as boson energy 
measurements, but not for local updates. By default calls 
`energy_boson(field(mc), conf)`
"""
energy_boson(mc::DQMC, m::Model, c = nothing) = energy_boson(field(mc), c)
energy_boson(f::AbstractField, c = nothing) = throw(MethodError(energy_boson, (f, c)))


conf(f::AbstractField) = f.conf
conf!(f::AbstractField, c) = conf(f) .= c
temp_conf(f::AbstractField) = f.temp_conf