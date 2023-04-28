mutable struct DQMCStack{
        GreensElType <: Number, 
        HoppingElType <: Number, 
        GreensMatType <: AbstractArray{GreensElType},
        HoppingMatBaseType <: AbstractArray{HoppingElType},
        HoppingMatType <: AbstractArray{HoppingElType},
        InteractionMatType <: AbstractArray,
        FieldCacheType <: AbstractFieldCache
    } <: AbstractDQMCStack

    field_cache::FieldCacheType

    u_stack::Vector{GreensMatType}
    d_stack::Vector{Vector{Float64}}
    t_stack::Vector{GreensMatType}

    Ul::GreensMatType
    Ur::GreensMatType
    Dl::Vector{Float64}
    Dr::Vector{Float64}
    Tl::GreensMatType
    Tr::GreensMatType
    pivot::Vector{Int64}
    tempv::Vector{GreensElType}
    tempvf::Vector{Float64}

    greens::GreensMatType
    greens_temp::GreensMatType
    complex_greens_temp::Matrix{ComplexF64}

    tmp1::GreensMatType
    tmp2::GreensMatType

    ranges::Vector{UnitRange{Int}}
    n_elements::Int
    # running internally over 0:mc.parameters.slices+1, where 0 and 
    # mc.parameters.slices+1 are artifcial to prepare next sweep direction.
    current_slice::Int 
    # current index into ranges
    current_range::Int
    direction::Int

    # preallocated, reused arrays
    curr_U::GreensMatType
    eV::InteractionMatType

    # hopping matrices (mu included)
    hopping_matrix::HoppingMatBaseType
    hopping_matrix_exp::HoppingMatType
    hopping_matrix_exp_inv::HoppingMatType
    hopping_matrix_exp_squared::HoppingMatType
    hopping_matrix_exp_inv_squared::HoppingMatType



    function DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}(field_cache::FCT) where {
            GET<:Number, HET<:Number, GMT<:AbstractArray{GET}, 
            HMBT<:AbstractArray{HET}, HMT<:AbstractArray{HET}, 
            IMT<:AbstractArray, FCT<:AbstractFieldCache
        }
        @assert isconcretetype(FCT);
        @assert isconcretetype(GET);
        @assert isconcretetype(HET);
        @assert isconcretetype(GMT);
        @assert isconcretetype(HMBT);
        @assert isconcretetype(HMT);
        @assert isconcretetype(IMT);
        @assert eltype(GMT) == GET;
        @assert eltype(HMT) == HET;
        stack = new{GET, HET, GMT, HMBT, HMT, IMT, FCT}()
        stack.field_cache = field_cache
        return stack
    end
end

# type helpers
geltype(::DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}) where {GET, HET, GMT, HMBT, HMT, IMT} = GET
heltype(::DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}) where {GET, HET, GMT, HMBT, HMT, IMT} = HET
gmattype(::DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}) where {GET, HET, GMT, HMBT, HMT, IMT} = GMT
hmattype(::DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}) where {GET, HET, GMT, HMBT, HMT, IMT} = HMT
imattype(::DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}) where {GET, HET, GMT, HMBT, HMT, IMT} = IMT

geltype(mc::DQMC) = geltype(mc.stack)
heltype(mc::DQMC) = heltype(mc.stack)
gmattype(mc::DQMC) = gmattype(mc.stack)
hmattype(mc::DQMC) = hmattype(mc.stack)
imattype(mc::DQMC) = imattype(mc.stack)


# Would be cool to have a function that automatically determines the type but
# I don't want to spend a lot of time on figuring that one out.
# function greens_matrix_type(field::AbstractField, model::Model)
#     IMT = interaction_matrix_type(field, model)
#     HMT = hopping_matrix_type(field, model)
#     ...
# end

function to_checkerboard_type(::Type{Matrix{T}}) where {T <: Union{ComplexF64, Float64}}
    CheckerboardDecomposed{T}
end
function to_checkerboard_type(::Type{BlockDiagonal{T, N, MT}}) where {T, N, MT}
    BlockDiagonal{T, N, to_checkerboard_type(MT)}
end
function to_checkerboard_type(::Type{CMat64})
    StructArray{
        ComplexF64, 2, 
        NamedTuple{(:re, :im), Tuple{CheckerboardDecomposed{Float64}, CheckerboardDecomposed{Float64}}}, 
        Int64
    }
end


function DQMCStack(field::AbstractField, model::Model, checkerboard::Bool)
    # Why do we need eltypes?
    HET = hopping_eltype(model)
    GET = greens_eltype(field, model)
    
    IMT = interaction_matrix_type(field, model)
    HMBT = hopping_matrix_type(field, model)
    GMT = greens_matrix_type(field, model)
    
    if checkerboard
        # Matrix -> CheckerboardDecomposed
        # BlockDiagonal{Matrix} -> BlockDiagonal{CheckerboardDecomposed}
        # StructArray -> ... dunno yet
        HMT = to_checkerboard_type(HMBT)
    else
        HMT = Hermitian{HET, HMBT}
    end

    DQMCStack{GET, HET, GMT, HMBT, HMT, IMT}(FieldCache(field, model))
end



################################################################################
### Stack Initialization
################################################################################



# T, V and their exponential version should be hermitian, because the 
# Hamiltonian should be hermitian. We should allow float errors though.
function is_approximately_hermitian(M; atol = 0.0, rtol = sqrt(eps(maximum(abs.(M)))))
    for i in 1:size(M, 1), j in size(M, 2)
        if !isapprox(M[i, j], conj(M[j, i]), atol=atol, rtol=rtol)
            return false
        end
    end
    return true
end


function generate_chunks(length, max_chunk_size)
    N_chunks = cld(length, max_chunk_size)
    step = length / N_chunks
    [round(Int, (i-1) * step) + 1 : round(Int, i * step) for i in 1:N_chunks]
end

function initialize_stack(mc::DQMC, ::DQMCStack)
    GreensElType = geltype(mc)
    GreensMatType = gmattype(mc)
    N = length(lattice(mc))
    flv = unique_flavors(mc)

    # Generate safe multiplication chunks
    # - every chunk must have ≤ safe_mult elements
    # - chunks should have close to equal size
    mc.stack.ranges = generate_chunks(mc.parameters.slices, mc.parameters.safe_mult)
    @assert first(mc.stack.ranges[1]) == 1
    @assert last(mc.stack.ranges[end]) == mc.parameters.slices
    mc.stack.n_elements = length(mc.stack.ranges) + 1

    mc.stack.u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.stack.n_elements]
    mc.stack.d_stack = [zeros(Float64, flv*N) for _ in 1:mc.stack.n_elements]
    mc.stack.t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.stack.n_elements]

    mc.stack.greens = GreensMatType(undef, flv*N, flv*N)
    mc.stack.greens_temp = GreensMatType(undef, flv*N, flv*N)
    # TODO rework measurements to work well with StructArrays and remove this
    if (GreensMatType <: BlockDiagonal{ComplexF64, N, CMat64} where N) || (GreensMatType <: CMat64)
        mc.stack.complex_greens_temp = Matrix(mc.stack.greens)
    end

    # used in calculate_greens
    # do not change in slice_matrices.jl or interaction_matrix_exp!
    mc.stack.Ul = GreensMatType(I, flv*N, flv*N)
    mc.stack.Ur = GreensMatType(I, flv*N, flv*N)
    mc.stack.Tl = GreensMatType(I, flv*N, flv*N)
    mc.stack.Tr = GreensMatType(I, flv*N, flv*N)
    mc.stack.Dl = ones(Float64, flv*N)
    mc.stack.Dr = ones(Float64, flv*N)
    # can be changed anywhere
    mc.stack.pivot = Vector{Int64}(undef, flv*N)
    mc.stack.tempv = Vector{GreensElType}(undef, flv*N)
    mc.stack.tempvf = Vector{Float64}(undef, flv*N)

    # can be changed anywhere
    mc.stack.tmp1 = GreensMatType(undef, flv*N, flv*N)
    mc.stack.tmp2 = GreensMatType(undef, flv*N, flv*N)

    mc.stack.curr_U = GreensMatType(undef, flv*N, flv*N)
    mc.stack.eV = init_interaction_matrix(field(mc), model(mc))

    nothing
end

# hopping
function init_hopping_matrices(mc::DQMC, m::Model)
    dtau = mc.parameters.delta_tau
    T = pad_to_unique_flavors(mc, hopping_matrix(m))

    if !is_approximately_hermitian(T)
        error(
            "The hopping matrix from `hopping_matrix(::DQMC, ::$(typeof(m)))`" *
            " is not approximately Hermitian. Since the Hamiltonian is a" * 
            " Hermitian operator, the hopping matrix should also be Hermitian." *
            " It is recommended that you verify your hopping matrix - are you" *
            " including all bonds (reverse bonds?) and conjugating complex" * 
            " hoppings? If so it might be worth it to explicitly make your" *
            " hopping matrix Hermitian via `0.5 * (M + M')`."
        )
    elseif !ishermitian(T)
        @warn(
            "The hopping matrix from `hopping_matrix(::DQMC, ::$(typeof(m)))`" *
            " is not exactly Hermitian. It might be worth it to explicitly" *
            " make your hopping matrix Hermitian via `0.5 * (M + M')`."
        )
    end

    # Assuming T is Hermitian we have e^T/2 e^T/2 e^V = e^T e^V as e^aA e^bA = e^(a+b)A
    mc.stack.hopping_matrix = T
    if !mc.parameters.checkerboard
        # greens
        mc.stack.hopping_matrix_exp = Hermitian(fallback_exp(-0.5 * dtau * T))
        mc.stack.hopping_matrix_exp_inv = Hermitian(fallback_exp(+0.5 * dtau * T))
        # slice matrix multiplications
        mc.stack.hopping_matrix_exp_squared = Hermitian(fallback_exp(-dtau * T))
        mc.stack.hopping_matrix_exp_inv_squared = Hermitian(fallback_exp(+dtau * T))
    else
        l = lattice(mc)
        mc.stack.hopping_matrix_exp = CheckerboardDecomposed(T, l, -0.5 * dtau)
        mc.stack.hopping_matrix_exp_inv = CheckerboardDecomposed(T, l, +0.5 * dtau)
        mc.stack.hopping_matrix_exp_squared = CheckerboardDecomposed(T, l, -dtau)
        mc.stack.hopping_matrix_exp_inv_squared = CheckerboardDecomposed(T, l, +dtau)
    end

    nothing
end


"""
    build_stack(mc::DQMC)

Build slice matrix stack from scratch.
"""
@bm function build_stack(mc::DQMC, ::DQMCStack)
    copyto!(mc.stack.u_stack[1], I)
    mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
    copyto!(mc.stack.t_stack[1], I)

    @inbounds for i in 1:length(mc.stack.ranges)
        add_slice_sequence_left(mc, i)
    end

    mc.stack.current_slice = mc.parameters.slices + 1
    # strictly this should be +1 but we special based on current_slice anyway
    mc.stack.current_range = length(mc.stack.ranges)
    mc.stack.direction = -1

    # Calculate valid greens function
    copyto!(mc.stack.Ul, mc.stack.u_stack[end])
    copyto!(mc.stack.Dl, mc.stack.d_stack[end])
    copyto!(mc.stack.Tl, mc.stack.t_stack[end])
    copyto!(mc.stack.Ur, I)
    mc.stack.Dr .= one(eltype(mc.stack.Dr))
    copyto!(mc.stack.Tr, I)
    calculate_greens(mc)

    nothing
end


@bm function reverse_build_stack(mc::DQMC, ::DQMCStack)
    copyto!(mc.stack.u_stack[end], I)
    mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
    copyto!(mc.stack.t_stack[end], I)

    @inbounds for i in length(mc.stack.ranges):-1:1
        add_slice_sequence_right(mc, i)
    end

    mc.stack.current_slice = 0
    # strictly this should be 0 but we special based on current_slice anyway
    mc.stack.current_range = 1
    mc.stack.direction = 1

    # Calculate valid greens function
    copyto!(mc.stack.Ul, I)
    mc.stack.Dl .= one(eltype(mc.stack.Dl))
    copyto!(mc.stack.Tl, I)
    copyto!(mc.stack.Ur, mc.stack.u_stack[1])
    copyto!(mc.stack.Dr, mc.stack.d_stack[1])
    copyto!(mc.stack.Tr, mc.stack.t_stack[1])
    calculate_greens(mc)

    nothing
end


################################################################################
### Slice matrix stack manipulations/updates
################################################################################



# Multiplications of Trotter decomposed e^H(τ) and a matrix M (typically the 
# result of a matrix product chain)
@bm function multiply_slice_matrix_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # eT^2 eV M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, mc.stack.eV, M)
    vmul!(M, mc.stack.hopping_matrix_exp_squared, mc.stack.tmp1, mc.stack.tmp2)
    nothing
end

@bm function multiply_slice_matrix_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # M eT^2 eV
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, M, mc.stack.hopping_matrix_exp_squared, mc.stack.tmp2)
    vmul!(M, mc.stack.tmp1, mc.stack.eV)
    nothing
end

@bm function multiply_slice_matrix_inv_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # M * eV^-1 eT2^-1 
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, -1.0)
    vmul!(mc.stack.tmp1, M, mc.stack.eV)
    vmul!(M, mc.stack.tmp1, mc.stack.hopping_matrix_exp_inv_squared, mc.stack.tmp2)
    nothing
end

@bm function multiply_slice_matrix_inv_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # eV^-1 eT2^-1 M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, -1.0)
    vmul!(mc.stack.tmp1, mc.stack.hopping_matrix_exp_inv_squared, M, mc.stack.tmp2)
    vmul!(M, mc.stack.eV, mc.stack.tmp1)
    nothing
end

@bm function multiply_daggered_slice_matrix_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # adjoint(eT^2 eV) M = eV' eT2' M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, adjoint(mc.stack.hopping_matrix_exp_squared), M, mc.stack.tmp2)
    vmul!(M, adjoint(mc.stack.eV), mc.stack.tmp1)
    nothing
end


"""
    add_slice_sequence_left(mc::DQMC, idx)

Computes the next `mc.parameters.safe_mult` slice matrix products from the current `idx`
and writes them to `idx+1`. The index `idx` does not refer to the slice index, 
but `mc.parameters.safe_mult` times the slice index.
"""
@bm function add_slice_sequence_left(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.stack.ranges[idx])
        for slice in mc.stack.ranges[idx]
            multiply_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx]))
        udt_AVX_pivot!(
            mc.stack.u_stack[idx + 1], mc.stack.d_stack[idx + 1], mc.stack.tmp1, 
            mc.stack.pivot, mc.stack.tempv
        )
        vmul!(mc.stack.t_stack[idx + 1], mc.stack.tmp1, mc.stack.t_stack[idx])
    end
end

"""
    add_slice_sequence_right(mc::DQMC, idx)

Computes the next `mc.parameters.safe_mult` slice matrix products from the current 
`idx+1` and writes them to `idx`. The index `idx` does not refer to the slice 
index, but `mc.parameters.safe_mult` times the slice index.
"""
@bm function add_slice_sequence_right(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx + 1])

        for slice in reverse(mc.stack.ranges[idx])
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx + 1]))
        udt_AVX_pivot!(
            mc.stack.u_stack[idx], mc.stack.d_stack[idx], mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
        )
        vmul!(mc.stack.t_stack[idx], mc.stack.tmp1, mc.stack.t_stack[idx + 1])
    end
end



################################################################################
### Green's function calculation
################################################################################



"""
    calculate_greens_AVX!(Ul, Dl, Tl, Ur, Dr, Tr, G[, pivot, temp])

Calculates the effective Greens function matrix `G` from two UDT decompositions
`Ul, Dl, Tl` and `Ur, Dr, Tr`. Additionally a `pivot` vector can be given. Note
that all inputs will be overwritten.

The UDT should follow from a set of slice_matrix multiplications, such that
`Ur, Dr, Tr = udt(B(slice)' ⋯ B(M)')` and `Ul, Dl, Tl = udt(B(slice-1) ⋯ B(1))`.
The computed Greens function is then given as `G = inv(I + Ul Dl Tl Tr Dr Ur)`
and computed here.

`Ul, Tl, Ur, Tr, G` should be square matrices, `Dl, Dr` real Vectors (from 
Diagonal matrices), `pivot` an integer Vector and `temp` a Vector with the same
element type as the matrices.
"""
@bm function calculate_greens_AVX!(
        Ul, Dl, Tl, Ur, Dr, Tr, G::AbstractArray{T},
        pivot = Vector{Int64}(undef, length(Dl)),
        temp = Vector{T}(undef, length(Dl))
    ) where T
    # @bm "B1" begin
        # Used: Ul, Dl, Tl, Ur, Dr, Tr
        # TODO: [I + Ul Dl Tl Tr^† Dr Ur^†]^-1
        # Compute: Dl * ((Tl * Tr) * Dr) -> Tr * Dr * G   (UDT)
        vmul!(G, Tl, adjoint(Tr))
        vmul!(Tr, G, Diagonal(Dr))
        vmul!(G, Diagonal(Dl), Tr)
        udt_AVX_pivot!(Tr, Dr, G, pivot, temp, Val(false)) # Dl available
    # end

    # @bm "B2" begin
        # Used: Ul, Ur, G, Tr, Dr  (Ul, Ur, Tr unitary (inv = adjoint))
        # TODO: [I + Ul Tr Dr G Ur^†]^-1
        #     = [(Ul Tr) ((Ul Tr)^-1 (G Ur^†) + Dr) (G Ur)]^-1
        #     = Ur G^-1 [(Ul Tr)^† Ur G^-1 + Dr]^-1 (Ul Tr)^†
        # Compute: Ul Tr -> Tl
        #          (Ur G^-1) -> Ur
        #          ((Ul Tr)^† Ur G^-1) -> Tr
        vmul!(Tl, Ul, Tr)
        rdivp!(Ur, G, Ul, pivot) # requires unpivoted udt decompostion (Val(false))
        vmul!(Tr, adjoint(Tl), Ur)
    # end

    # @bm "B3" begin
        # Used: Tl, Ur, Tr, Dr
        # TODO: Ur [Tr + Dr]^-1 Tl^† -> Ur [Tr]^-1 Tl^†
        rvadd!(Tr, Diagonal(Dr))
    # end

    # @bm "B4" begin
        # Used: Ur, Tr, Tl
        # TODO: Ur [Tr]^-1 Tl^† -> Ur [Ul Dr Tr]^-1 Tl^† 
        #    -> Ur Tr^-1 Dr^-1 Ul^† Tl^† -> Ur Tr^-1 Dr^-1 (Tl Ul)^†
        # Compute: Ur Tr^-1 -> Ur,  Tl Ul -> Tr
        udt_AVX_pivot!(Ul, Dr, Tr, pivot, temp, Val(false)) # Dl available
        rdivp!(Ur, Tr, G, pivot) # requires unpivoted udt decompostion (false)
        vmul!(Tr, Tl, Ul)
    # end

    # @bm "B5" begin
        vinv!(Dl, Dr)
    # end

    # @bm "B6" begin
        # Used: Ur, Tr, Dl, Ul, Tl
        # TODO: (Ur Dl) Tr^† -> G
        vmul!(Ul, Ur, Diagonal(Dl))
        vmul!(G, Ul, adjoint(Tr))
    # end
end

"""
    calculate_greens(mc::DQMC)

Computes the effective greens function from the current state of the stack and
saves the result to `mc.stack.greens`.

This assumes the `mc.stack.Ul, mc.stack.Dl, mc.stack.Tl = udt(B(slice-1) ⋯ B(1))` and
`mc.stack.Ur, mc.stack.Dr, mc.stack.Tr = udt(B(slice)' ⋯ B(M)')`. 

This should only used internally.
"""
@bm function calculate_greens(mc::DQMC, output::AbstractMatrix = mc.stack.greens)
    calculate_greens_AVX!(
        mc.stack.Ul, mc.stack.Dl, mc.stack.Tl,
        mc.stack.Ur, mc.stack.Dr, mc.stack.Tr,
        output, mc.stack.pivot, mc.stack.tempv
    )
    output
end

"""
    calculate_greens(mc::DQMC, slice[, output=mc.stack.greens, safe_mult])

Compute the effective equal-time greens function from scratch at a given `slice`.

This does not invalidate the stack, but it does overwrite `mc.stack.greens`.
"""
@bm function calculate_greens(
        mc::DQMC, slice, output = mc.stack.greens, 
        field = field(mc), safe_mult = mc.parameters.safe_mult
    )
    copyto!(mc.stack.curr_U, I)
    copyto!(mc.stack.Ur, I)
    mc.stack.Dr .= one(eltype(mc.stack.Dr))
    copyto!(mc.stack.Tr, I)

    # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
    if slice+1 <= mc.parameters.slices
        start = slice+1
        stop = mc.parameters.slices
        for k in reverse(start:stop)
            if mod(k,safe_mult) == 0
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, field)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dr))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dr, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tr)
                vmul!(mc.stack.Tr, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, field)
            end
        end
        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dr))
        udt_AVX_pivot!(mc.stack.Ur, mc.stack.Dr, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
        copyto!(mc.stack.tmp2, mc.stack.Tr)
        vmul!(mc.stack.Tr, mc.stack.tmp1, mc.stack.tmp2)
    end


    copyto!(mc.stack.curr_U, I)
    copyto!(mc.stack.Ul, I)
    mc.stack.Dl .= one(eltype(mc.stack.Dl))
    copyto!(mc.stack.Tl, I)

    # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
    if slice >= 1
        start = 1
        stop = slice
        for k in start:stop
            if mod(k,safe_mult) == 0
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, field)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dl))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dl, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tl)
                vmul!(mc.stack.Tl, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, field)
            end
        end
        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dl))
        udt_AVX_pivot!(mc.stack.Ul, mc.stack.Dl, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
        copyto!(mc.stack.tmp2, mc.stack.Tl)
        vmul!(mc.stack.Tl, mc.stack.tmp1, mc.stack.tmp2)
    end

    return calculate_greens(mc, output)
end



################################################################################
### Stack update
################################################################################



# Green's function propagation
@inline @bm function wrap_greens!(mc::DQMC, gf, curr_slice::Int, direction::Int)
    if direction == -1
        multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
        multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
    else
        multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
        multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
    end
    nothing
end

@bm function propagate(mc::DQMC)
    @debug(
        '[' * lpad(mc.stack.current_slice, 3, ' ') * " -> " * 
        rpad(mc.stack.current_slice + mc.stack.direction, 3, ' ') * 
        ", " * mc.stack.direction == +1 ? '+' : '-', "] "
    )

    # Advance according to direction
    mc.stack.current_slice += mc.stack.direction

    @inbounds if mc.stack.direction == 1
        if mc.stack.current_slice == 1
            @debug("init direction, clearing 1 to I")
            copyto!(mc.stack.u_stack[1], I)
            mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
            copyto!(mc.stack.t_stack[1], I)

        elseif mc.stack.current_slice-1 == last(mc.stack.ranges[mc.stack.current_range])
            idx = mc.stack.current_range
            @debug("Stabilize: decompose into $idx -> $(idx+1)")

            copyto!(mc.stack.Ur, mc.stack.u_stack[idx+1])
            copyto!(mc.stack.Dr, mc.stack.d_stack[idx+1])
            copyto!(mc.stack.Tr, mc.stack.t_stack[idx+1])
            add_slice_sequence_left(mc, idx)
            copyto!(mc.stack.Ul, mc.stack.u_stack[idx+1])
            copyto!(mc.stack.Dl, mc.stack.d_stack[idx+1])
            copyto!(mc.stack.Tl, mc.stack.t_stack[idx+1])

            if mc.parameters.check_propagation_error
                copyto!(mc.stack.greens_temp, mc.stack.greens)
            end

            # Should this be mc.stack.greens_temp?
            # If so, shouldn't this only run w/ mc.parameters.all_checks = true?
            wrap_greens!(mc, mc.stack.greens_temp, mc.stack.current_slice - 1, 1)

            calculate_greens(mc) # greens_{slice we are propagating to}

            if mc.parameters.check_propagation_error
                # OPT: could probably be optimized through explicit loop
                greensdiff = maximum(abs.(mc.stack.greens_temp - mc.stack.greens)) 
                if greensdiff > 1e-7
                    push!(mc.analysis.propagation_error, greensdiff)
                    mc.parameters.silent || @printf(
                        "->%d \t+1 Propagation instability\t %.1e\n", 
                        mc.stack.current_slice, greensdiff
                    )
                end
            end

            if mc.stack.current_range == length(mc.stack.ranges)
                # We are going from M -> M+1. Switch direction.
                @assert mc.stack.current_slice == mc.parameters.slices + 1
                mc.stack.direction = -1
                propagate(mc)
            else
                mc.stack.current_range += 1
            end

        else
            @debug("standard wrap")
            # Wrapping (we already advanced current_slice but wrap according to previous)
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice-1, 1)
        end

    else # REVERSE
        if mc.stack.current_slice == mc.parameters.slices
            @debug("init direction, clearing end to I")
            copyto!(mc.stack.u_stack[end], I)
            mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
            copyto!(mc.stack.t_stack[end], I)

            # wrap to greens_{mc.parameters.slices}
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)

        elseif mc.stack.current_slice+1 == first(mc.stack.ranges[mc.stack.current_range])
            idx = mc.stack.current_range
            @debug("Stabilize: decompose into $(idx+1) -> $idx")

            copyto!(mc.stack.Ul, mc.stack.u_stack[idx])
            copyto!(mc.stack.Dl, mc.stack.d_stack[idx])
            copyto!(mc.stack.Tl, mc.stack.t_stack[idx])
            add_slice_sequence_right(mc, idx)
            copyto!(mc.stack.Ur, mc.stack.u_stack[idx])
            copyto!(mc.stack.Dr, mc.stack.d_stack[idx])
            copyto!(mc.stack.Tr, mc.stack.t_stack[idx])

            if mc.parameters.check_propagation_error
                copyto!(mc.stack.greens_temp, mc.stack.greens)
            end

            calculate_greens(mc)

            if mc.parameters.check_propagation_error
                # OPT: could probably be optimized through explicit loop
                greensdiff = maximum(abs.(mc.stack.greens_temp - mc.stack.greens)) 
                if greensdiff > 1e-7
                    push!(mc.analysis.propagation_error, greensdiff)
                    mc.parameters.silent || @printf(
                        "->%d \t-1 Propagation instability\t %.1e\n", 
                        mc.stack.current_slice, greensdiff
                    )
                end
            end

            if mc.stack.current_range == 1
                # We are going from 1 -> 0. Switch direction.
                @assert mc.stack.current_slice == 0
                mc.stack.direction = +1
                propagate(mc)
            else
                # We'd be undoing this wrap in forward 1 if we always applied it
                wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)
                mc.stack.current_range -= 1
            end

        else
            @debug("standard wrap")
            # Wrapping (we already advanced current_slice but wrap according to previous)
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice+1, -1)
        end
    end

    nothing
end
