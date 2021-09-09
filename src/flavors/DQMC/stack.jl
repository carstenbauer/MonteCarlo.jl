mutable struct DQMCStack{
        GreensElType <: Number, 
        HoppingElType <: Number, 
        GreensMatType <: AbstractArray{GreensElType},
        HoppingMatType <: AbstractArray{HoppingElType},
        InteractionMatType <: AbstractArray
    } <: AbstractDQMCStack

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

    tmp1::GreensMatType
    tmp2::GreensMatType

    ranges::Array{UnitRange, 1}
    n_elements::Int
    current_slice::Int # running internally over 0:mc.parameters.slices+1, where 0 and mc.parameters.slices+1 are artifcial to prepare next sweep direction.
    direction::Int

    # # -------- Global update backup
    # gb_u_stack::Array{GreensElType, 3}
    # gb_d_stack::Matrix{Float64}
    # gb_t_stack::Array{GreensElType, 3}

    # gb_greens::GreensMatType
    # gb_log_det::Float64

    # gb_conf::Array{Float64, 3}
    # # --------


    # preallocated, reused arrays
    curr_U::GreensMatType
    eV::InteractionMatType

    # hopping matrices (mu included)
    hopping_matrix::HoppingMatType
    hopping_matrix_exp::HoppingMatType
    hopping_matrix_exp_inv::HoppingMatType
    hopping_matrix_exp_squared::HoppingMatType
    hopping_matrix_exp_inv_squared::HoppingMatType

    # checkerboard hopping matrices
    checkerboard::Matrix{Int} # src, trg, bondid
    groups::Vector{UnitRange}
    n_groups::Int
    chkr_hop_half::Vector{SparseMatrixCSC{HoppingElType, Int64}}
    chkr_hop_half_inv::Vector{SparseMatrixCSC{HoppingElType, Int64}}
    chkr_hop_half_dagger::Vector{SparseMatrixCSC{HoppingElType, Int64}}
    chkr_hop::Vector{SparseMatrixCSC{HoppingElType, Int64}} # without prefactor 0.5 in matrix exponentials
    chkr_hop_inv::Vector{SparseMatrixCSC{HoppingElType, Int64}}
    chkr_hop_dagger::Vector{SparseMatrixCSC{HoppingElType, Int64}}
    chkr_mu_half::SparseMatrixCSC{HoppingElType, Int64}
    chkr_mu_half_inv::SparseMatrixCSC{HoppingElType, Int64}
    chkr_mu::SparseMatrixCSC{HoppingElType, Int64}
    chkr_mu_inv::SparseMatrixCSC{HoppingElType, Int64}


    function DQMCStack{GET, HET, GMT, HMT, IMT}() where {
            GET<:Number, HET<:Number, 
            GMT<:AbstractArray{GET}, HMT<:AbstractArray{HET}, IMT<:AbstractArray
        }
        @assert isconcretetype(GET);
        @assert isconcretetype(HET);
        @assert isconcretetype(GMT);
        @assert isconcretetype(HMT);
        @assert isconcretetype(IMT);
        @assert eltype(GMT) == GET;
        @assert eltype(HMT) == HET;
        new{GET, HET, GMT, HMT, IMT}()
    end
end

# type helpers
geltype(::DQMCStack{GET, HET, GMT, HMT, IMT}) where {GET, HET, GMT, HMT, IMT} = GET
heltype(::DQMCStack{GET, HET, GMT, HMT, IMT}) where {GET, HET, GMT, HMT, IMT} = HET
gmattype(::DQMCStack{GET, HET, GMT, HMT, IMT}) where {GET, HET, GMT, HMT, IMT} = GMT
hmattype(::DQMCStack{GET, HET, GMT, HMT, IMT}) where {GET, HET, GMT, HMT, IMT} = HMT
imattype(::DQMCStack{GET, HET, GMT, HMT, IMT}) where {GET, HET, GMT, HMT, IMT} = IMT

geltype(mc::DQMC) = geltype(mc.stack)
heltype(mc::DQMC) = heltype(mc.stack)
gmattype(mc::DQMC) = gmattype(mc.stack)
hmattype(mc::DQMC) = hmattype(mc.stack)
imattype(mc::DQMC) = imattype(mc.stack)



################################################################################
### Stack Initialization
################################################################################



function initialize_stack(mc::DQMC, ::DQMCStack)
    GreensElType = geltype(mc)
    GreensMatType = gmattype(mc)
    HoppingElType = heltype(mc)
    N = length(lattice(mc))
    flv = nflavors(mc.model)

    mc.stack.n_elements = convert(Int, mc.parameters.slices / mc.parameters.safe_mult) + 1

    mc.stack.u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.stack.n_elements]
    mc.stack.d_stack = [zeros(Float64, flv*N) for _ in 1:mc.stack.n_elements]
    mc.stack.t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.stack.n_elements]

    mc.stack.greens = GreensMatType(undef, flv*N, flv*N)
    mc.stack.greens_temp = GreensMatType(undef, flv*N, flv*N)

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


    # # Global update backup
    # mc.stack.gb_u_stack = zero(mc.stack.u_stack)
    # mc.stack.gb_d_stack = zero(mc.stack.d_stack)
    # mc.stack.gb_t_stack = zero(mc.stack.t_stack)
    # mc.stack.gb_greens = zero(mc.stack.greens)
    # mc.stack.gb_log_det = 0.
    # mc.stack.gb_conf = zero(mc.conf)

    mc.stack.ranges = UnitRange[]

    for i in 1:mc.stack.n_elements - 1
        push!(mc.stack.ranges, 1 + (i - 1) * mc.parameters.safe_mult:i * mc.parameters.safe_mult)
    end

    mc.stack.curr_U = GreensMatType(undef, flv*N, flv*N)
    mc.stack.eV = init_interaction_matrix(mc.model)

    nothing
end

# hopping
function init_hopping_matrices(mc::DQMC{M,CB}, m::Model) where {M, CB<:Checkerboard}
    init_hopping_matrix_exp(mc, m)
    CB <: CheckerboardTrue && init_checkerboard_matrices(mc, m)
    nothing
end
function init_hopping_matrix_exp(mc::DQMC, m::Model)
    N = length(lattice(m))
    flv = nflavors(m)
    dtau = mc.parameters.delta_tau

    T = hopping_matrix(mc, m)
    size(T) == (flv*N, flv*N) || error("Hopping matrix should have size "*
                                "$((flv*N, flv*N)) but has size $(size(T)) .")
    mc.stack.hopping_matrix = T
    mc.stack.hopping_matrix_exp = exp(-0.5 * dtau * T)
    mc.stack.hopping_matrix_exp_inv = exp(0.5 * dtau * T)
    mc.stack.hopping_matrix_exp_squared = mc.stack.hopping_matrix_exp * mc.stack.hopping_matrix_exp
    mc.stack.hopping_matrix_exp_inv_squared = mc.stack.hopping_matrix_exp_inv * mc.stack.hopping_matrix_exp_inv
    nothing
end

# checkerboard
rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)
function init_checkerboard_matrices(mc::DQMC, m::Model)
    s = mc.stack
    l = lattice(m)
    flv = nflavors(m)
    H = heltype(mc)
    N = length(l)
    dtau = mc.parameters.delta_tau
    mu = m.mu

    s.checkerboard, s.groups, s.n_groups = build_checkerboard(l)
    n_grps = s.n_groups
    cb = s.checkerboard

    T = reshape(hopping_matrix(mc, m), (N, flv, N, flv))

    s.chkr_hop_half = Vector{SparseMatrixCSC{H, Int}}(undef, n_grps)
    s.chkr_hop_half_inv = Vector{SparseMatrixCSC{H, Int}}(undef, n_grps)
    s.chkr_hop = Vector{SparseMatrixCSC{H, Int}}(undef, n_grps)
    s.chkr_hop_inv = Vector{SparseMatrixCSC{H, Int}}(undef, n_grps)

    for (g, gr) in enumerate(s.groups)
        Tg = zeros(H, N, flv, N, flv)
        for i in gr
            src, trg = cb[1:2,i]
            for f1 in 1:flv, f2 in 1:flv
                Tg[trg, f1, src, f2] = T[trg, f1, src, f2]
            end
        end

        Tgg = reshape(Tg, (N*flv, N*flv))
        s.chkr_hop_half[g] = sparse(rem_eff_zeros!(exp(-0.5 * dtau * Tgg)))
        s.chkr_hop_half_inv[g] = sparse(rem_eff_zeros!(exp(0.5 * dtau * Tgg)))
        s.chkr_hop[g] = sparse(rem_eff_zeros!(exp(- dtau * Tgg)))
        s.chkr_hop_inv[g] = sparse(rem_eff_zeros!(exp(dtau * Tgg)))
    end

    s.chkr_hop_half_dagger = adjoint.(s.chkr_hop_half)
    s.chkr_hop_dagger = adjoint.(s.chkr_hop)

    mus = diag(reshape(T, (N*flv, N*flv)))
    s.chkr_mu_half = spdiagm(0 => exp.(-0.5 * dtau * mus))
    s.chkr_mu_half_inv = spdiagm(0 => exp.(0.5 * dtau * mus))
    s.chkr_mu = spdiagm(0 => exp.(-dtau * mus))
    s.chkr_mu_inv = spdiagm(0 => exp.(dtau * mus))

    # hop_mat_exp_chkr = foldl(*,s.chkr_hop_half) * sqrt.(s.chkr_mu)
    # r = effreldiff(s.hopping_matrix_exp,hop_mat_exp_chkr)
    # r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
    # println("Checkerboard - Exact ≈ ", round(maximum(absdiff(s.hopping_matrix_exp,hop_mat_exp_chkr)), 4))
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
    mc.stack.direction = -1

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
    mc.stack.direction = 1

    nothing
end


################################################################################
### Slice matrix stack manipulations/updates
################################################################################



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
        mc::DQMC, slice::Int, output::AbstractMatrix = mc.stack.greens, 
        conf::AbstractArray = mc.conf, safe_mult::Int = mc.parameters.safe_mult
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
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dr))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dr, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tr)
                vmul!(mc.stack.Tr, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
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
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dl))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dl, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tl)
                vmul!(mc.stack.Tl, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
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
    @inbounds if mc.stack.direction == 1
        if mod(mc.stack.current_slice, mc.parameters.safe_mult) == 0
            mc.stack.current_slice +=1 # slice we are going to
            if mc.stack.current_slice == 1
                copyto!(mc.stack.Ur, mc.stack.u_stack[1])
                copyto!(mc.stack.Dr, mc.stack.d_stack[1])
                copyto!(mc.stack.Tr, mc.stack.t_stack[1])
                copyto!(mc.stack.u_stack[1], I)
                mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
                copyto!(mc.stack.t_stack[1], I)
                copyto!(mc.stack.Ul, mc.stack.u_stack[1])
                copyto!(mc.stack.Dl, mc.stack.d_stack[1])
                copyto!(mc.stack.Tl, mc.stack.t_stack[1])

                calculate_greens(mc) # greens_1 ( === greens_{m+1} )

            elseif 1 < mc.stack.current_slice <= mc.parameters.slices
                idx = Int((mc.stack.current_slice - 1)/mc.parameters.safe_mult)

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

            else # we are going to mc.parameters.slices+1
                idx = mc.stack.n_elements - 1
                add_slice_sequence_left(mc, idx)
                mc.stack.direction = -1
                mc.stack.current_slice = mc.parameters.slices+1 # redundant
                propagate(mc)
            end

        else
            # Wrapping
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice, 1)
            mc.stack.current_slice += 1
        end

    else # mc.stack.direction == -1
        if mod(mc.stack.current_slice-1, mc.parameters.safe_mult) == 0
            mc.stack.current_slice -= 1 # slice we are going to
            if mc.stack.current_slice == mc.parameters.slices
                copyto!(mc.stack.Ul, mc.stack.u_stack[end])
                copyto!(mc.stack.Dl, mc.stack.d_stack[end])
                copyto!(mc.stack.Tl, mc.stack.t_stack[end])
                copyto!(mc.stack.u_stack[end], I)
                mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
                copyto!(mc.stack.t_stack[end], I)
                copyto!(mc.stack.Ur, mc.stack.u_stack[end])
                copyto!(mc.stack.Dr, mc.stack.d_stack[end])
                copyto!(mc.stack.Tr, mc.stack.t_stack[end])

                calculate_greens(mc) # greens_{mc.parameters.slices+1} === greens_1

                # wrap to greens_{mc.parameters.slices}
                wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)

            elseif 0 < mc.stack.current_slice < mc.parameters.slices
                idx = Int(mc.stack.current_slice / mc.parameters.safe_mult) + 1

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

                wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice + 1, -1)

            else # we are going to 0
                idx = 1
                add_slice_sequence_right(mc, idx)
                mc.stack.direction = 1
                mc.stack.current_slice = 0 # redundant
                propagate(mc)
            end

        else
            # Wrapping
            wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice, -1)
            mc.stack.current_slice -= 1
        end
    end
    nothing
end
