"""
    HubbardModel(lattice; params...)
    HubbardModel(L, dims; params...)
    HubbardModel(params::Dict)
    HubbardModel(params::NamedTuple)
    HubbardModel(; params...)

Defines an Hubbard model on a given (or derived) `lattice`. If a linear system 
size `L` and dimensionality `dims` is given, the`lattice` will be a Cubic 
lattice of fitting size.

Additional parameters (keyword arguments) include:
* `l::AbstractLattice = lattice`: The lattice the model uses. The keyword 
argument takes precedence over the argument `lattice` as well as `dims` and `L`.
* `U::Float64 = 1.0` is the value of the Hubbard Interaction. A positive `U` 
leads to an attractive Hubbard model, a negative to a repulsive.
* `t::Float64 = 1.0` is the hopping strength.
* `mu::Float64` is the chemical potential. In the repulsive model `mu != 0` 
leads to a sign problem.

The model will select an appropriate Hirsch field based on `U`.
"""
struct HubbardModel{LT <: AbstractLattice} <: Model
    t::Float64
    mu::Float64
    U::Float64
    l::LT
end

@inline function HubbardModel(t::Real, mu::Real, U::Real, l::AbstractLattice)
    HubbardModel(Float64(t), Float64(mu), Float64(U), l)
end

function HubbardModel(; 
        dims = 2, L = 2, l = choose_lattice(HubbardModel, dims, L), 
        U = 1.0, mu = 0.0, t = 1.0
    )
    if (U < 0.0) && (mu != 0.0)
        @warn("A repulsive Hubbard model with chemical potential µ = $mu will have a sign problem!")
    end
    HubbardModel(t, mu, U, l)
end
HubbardModel(params::Dict{Symbol}) = HubbardModel(; params...)
HubbardModel(params::NamedTuple) = HubbardModel(; params...)
HubbardModel(lattice::AbstractLattice; kwargs...) = HubbardModel(l = lattice; kwargs...)
HubbardModel(L, dims; kwargs...) = HubbardModel(dims = dims, L = L; kwargs...)

HubbardModelAttractive(args...; kwargs...) = HubbardModel(args...; kwargs...)
function HubbardModelRepulsive(args...; kwargs...)
    d = Dict(kwargs)
    d[:U] = - get(d, :U, 1.0)
    HubbardModel(args...; d...)
end


function choose_lattice(::Type{<: HubbardModel}, dims, L)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

# cosmetics
import Base.summary
import Base.show
function Base.summary(model::HubbardModel)
    (model.U < 0 ? "repulsive" : "attractive") * " Hubbard model"
end
function Base.show(io::IO, model::HubbardModel)
    println(io, (model.U < 0 ? "repulsive" : "attractive") * " Hubbard model")
    println(io, "\tParameters t = $(model.t), µ = $(model.mu), U = $(model.U)")
    print(io, "\t$(nameof(typeof(model.l))) with $(length(model.l)) sites")
end
Base.show(io::IO, m::MIME"text/plain", model::HubbardModel) = print(io, model)

# Convenience
@inline parameters(m::HubbardModel) = (N = length(m.l), t = m.t, U = m.U, mu = m.mu)
choose_field(m::HubbardModel) = m.U < 0.0 ? MagneticHirschField : DensityHirschField

# implement DQMC interface:
@inline lattice(m::HubbardModel) = m.l
nflavors(::HubbardModel) = 1

hopping_eltype(model::HubbardModel) = typeof(model.t)
function hopping_matrix_type(field::AbstractField, model::HubbardModel)
    flv = nflavors(field, model)
    T = hopping_eltype(model)
    MT = matrix_type(T)
    if flv == 1
        return MT
    else
        return BlockDiagonal{T, flv, MT}
    end
end


"""
    hopping_matrix(model)

Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(m::HubbardModel)
    N = length(m.l)
    T = diagm(0 => fill(-m.mu, N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(m.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -m.t
        end
    end

    return T
end

function save_model(file::JLDFile, m::HubbardModel, entryname::String = "Model")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "HubbardModel")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")

    nothing
end

# compat
function _load_model(data, ::Val{:HubbardModel})
    l = _load(data["l"], to_tag(data["l"]))
    HubbardModel(data["t"], data["mu"], data["U"], l)
end
_load_model(data, ::Val{:HubbardModelAttractive}) = _load_model(data, Val(:HubbardModel))
function _load_model(data, ::Val{:HubbardModelRepulsive})
    l = _load(data["l"], to_tag(data["l"]))
    HubbardModel(data["t"], 0.0, -data["U"], l)
end
_load_model(data, ::Val{:AttractiveGHQHubbardModel}) = _load_model(data, Val(:HubbardModelAttractive))
_load_model(data, ::Val{:RepulsiveGHQHubbardModel}) = _load_model(data, Val(:HubbardModelRepulsive))
field_hint(m, ::Val) = choose_field(m)
field_hint(m, ::Val{:AttractiveGHQHubbardModel}) = MagneticGHQField
field_hint(m, ::Val{:RepulsiveGHQHubbardModel}) = MagneticGHQField

function intE_kernel(mc, model::HubbardModel, G::GreensMatrix, ::Val{1})
    # ⟨U (n↑ - 1/2)(n↓ - 1/2)⟩ = ... 
    # = U [G↑↑ G↓↓ - G↓↑ G↑↓ - 0.5 G↑↑ - 0.5 G↓↓ + G↑↓ + 0.25]
    # = U [(G↑↑ - 1/2)(G↓↓ - 1/2) + G↑↓(1 + G↑↓)]
    # with up-up = down-down and up-down = 0
    - model.U * sum((diag(G.val) .- 0.5).^2)
end
# Technically this only applies to BlockDiagonal
function intE_kernel(mc, model::HubbardModel, G::GreensMatrix{Float64}, ::Val{2})
    - model.U * sum((diag(G.val.blocks[1]) .- 0.5) .* (diag(G.val.blocks[2]) .- 0.5))
end
function intE_kernel(mc, model::HubbardModel, G::GreensMatrix{ComplexF64}, ::Val{2})
    N = length(lattice(model))
    output = 0.0 + 0.0im
    for i in 1:N
        output -= (G.val[i, i] .- 0.5) .* (G.val[N+i, N+i] .- 0.5)
    end
    output * model.U
end