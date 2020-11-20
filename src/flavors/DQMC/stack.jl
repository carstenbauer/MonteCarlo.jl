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

    greens::GreensMatType
    greens_temp::GreensMatType

    tmp1::GreensMatType
    tmp2::GreensMatType

    ranges::Array{UnitRange, 1}
    n_elements::Int
    current_slice::Int # running internally over 0:mc.p.slices+1, where 0 and mc.p.slices+1 are artifcial to prepare next sweep direction.
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

geltype(mc::DQMC) = geltype(mc.s)
heltype(mc::DQMC) = heltype(mc.s)
gmattype(mc::DQMC) = gmattype(mc.s)
hmattype(mc::DQMC) = hmattype(mc.s)
imattype(mc::DQMC) = imattype(mc.s)



################################################################################
### Stack Initialization
################################################################################



function initialize_stack(mc::DQMC, ::DQMCStack)
    GreensElType = geltype(mc)
    GreensMatType = gmattype(mc)
    HoppingElType = heltype(mc)
    N = length(lattice(mc))
    flv = nflavors(mc.model)

    mc.s.n_elements = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

    mc.s.u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.s.n_elements]
    mc.s.d_stack = [zeros(Float64, flv*N) for _ in 1:mc.s.n_elements]
    mc.s.t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:mc.s.n_elements]

    mc.s.greens = GreensMatType(undef, flv*N, flv*N)
    mc.s.greens_temp = GreensMatType(undef, flv*N, flv*N)

    # used in calculate_greens
    # do not change in slice_matrices.jl or interaction_matrix_exp!
    mc.s.Ul = GreensMatType(I, flv*N, flv*N)
    mc.s.Ur = GreensMatType(I, flv*N, flv*N)
    mc.s.Tl = GreensMatType(I, flv*N, flv*N)
    mc.s.Tr = GreensMatType(I, flv*N, flv*N)
    mc.s.Dl = ones(Float64, flv*N)
    mc.s.Dr = ones(Float64, flv*N)
    # can be changed anywhere
    mc.s.pivot = Vector{Int64}(undef, flv*N)
    mc.s.tempv = Vector{GreensElType}(undef, flv*N)

    # can be changed anywhere
    mc.s.tmp1 = GreensMatType(undef, flv*N, flv*N)
    mc.s.tmp2 = GreensMatType(undef, flv*N, flv*N)


    # # Global update backup
    # mc.s.gb_u_stack = zero(mc.s.u_stack)
    # mc.s.gb_d_stack = zero(mc.s.d_stack)
    # mc.s.gb_t_stack = zero(mc.s.t_stack)
    # mc.s.gb_greens = zero(mc.s.greens)
    # mc.s.gb_log_det = 0.
    # mc.s.gb_conf = zero(mc.conf)

    mc.s.ranges = UnitRange[]

    for i in 1:mc.s.n_elements - 1
        push!(mc.s.ranges, 1 + (i - 1) * mc.p.safe_mult:i * mc.p.safe_mult)
    end

    mc.s.curr_U = GreensMatType(undef, flv*N, flv*N)
    mc.s.eV = init_interaction_matrix(mc.model)

    # mc.s.hopping_matrix_exp = zeros(HoppingElType, flv*N, flv*N)
    # mc.s.hopping_matrix_exp_inv = zeros(HoppingElType, flv*N, flv*N)
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
    dtau = mc.p.delta_tau

    T = hopping_matrix(mc, m)
    size(T) == (flv*N, flv*N) || error("Hopping matrix should have size "*
                                "$((flv*N, flv*N)) but has size $(size(T)) .")
    mc.s.hopping_matrix = T
    mc.s.hopping_matrix_exp = exp(-0.5 * dtau * T)
    mc.s.hopping_matrix_exp_inv = exp(0.5 * dtau * T)
    mc.s.hopping_matrix_exp_squared = mc.s.hopping_matrix_exp * mc.s.hopping_matrix_exp
    mc.s.hopping_matrix_exp_inv_squared = mc.s.hopping_matrix_exp_inv * mc.s.hopping_matrix_exp_inv
    nothing
end

# checkerboard
rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)
function init_checkerboard_matrices(mc::DQMC, m::Model)
    s = mc.s
    l = lattice(m)
    flv = nflavors(m)
    H = heltype(mc)
    N = length(l)
    dtau = mc.p.delta_tau
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
function build_stack(mc::DQMC, ::DQMCStack)
    copyto!(mc.s.u_stack[1], I)
    mc.s.d_stack[1] .= one(eltype(mc.s.d_stack[1]))
    copyto!(mc.s.t_stack[1], I)

    @inbounds for i in 1:length(mc.s.ranges)
        add_slice_sequence_left(mc, i)
    end

    mc.s.current_slice = mc.p.slices + 1
    mc.s.direction = -1

    nothing
end



################################################################################
### Slice matrix stack manipulations/updates
################################################################################



"""
    add_slice_sequence_left(mc::DQMC, idx)

Computes the next `mc.p.safe_mult` slice matrix products from the current `idx`
and writes them to `idx+1`. The index `idx` does not refer to the slice index, 
but `mc.p.safe_mult` times the slice index.
"""
@bm function add_slice_sequence_left(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.s.curr_U, mc.s.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
        for slice in mc.s.ranges[idx]
            multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
        end

        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.d_stack[idx]))
        udt_AVX_pivot!(
            mc.s.u_stack[idx + 1], mc.s.d_stack[idx + 1], mc.s.tmp1, 
            mc.s.pivot, mc.s.tempv
        )
        vmul!(mc.s.t_stack[idx + 1], mc.s.tmp1, mc.s.t_stack[idx])
    end
end

"""
    add_slice_sequence_right(mc::DQMC, idx)

Computes the next `mc.p.safe_mult` slice matrix products from the current 
`idx+1` and writes them to `idx`. The index `idx` does not refer to the slice 
index, but `mc.p.safe_mult` times the slice index.
"""
@bm function add_slice_sequence_right(mc::DQMC, idx::Int)
    @inbounds begin
        copyto!(mc.s.curr_U, mc.s.u_stack[idx + 1])

        for slice in reverse(mc.s.ranges[idx])
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
        end

        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.d_stack[idx + 1]))
        udt_AVX_pivot!(
            mc.s.u_stack[idx], mc.s.d_stack[idx], mc.s.tmp1, mc.s.pivot, mc.s.tempv
        )
        vmul!(mc.s.t_stack[idx], mc.s.tmp1, mc.s.t_stack[idx + 1])
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
        @avx for i in eachindex(Dr)
            Dl[i] = 1.0 / Dr[i]
        end
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
saves the result to `mc.s.greens`.

This assumes the `mc.s.Ul, mc.s.Dl, mc.s.Tl = udt(B(slice-1) ⋯ B(1))` and
`mc.s.Ur, mc.s.Dr, mc.s.Tr = udt(B(slice)' ⋯ B(M)')`. 

This should only used internally.
"""
@bm function calculate_greens(mc::DQMC, output::AbstractMatrix = mc.s.greens)
    calculate_greens_AVX!(
        mc.s.Ul, mc.s.Dl, mc.s.Tl,
        mc.s.Ur, mc.s.Dr, mc.s.Tr,
        output, mc.s.pivot, mc.s.tempv
    )
    output
end

"""
    calculate_greens(mc::DQMC, slice[, output=mc.s.greens, safe_mult])

Compute the effective equal-time greens function from scratch at a given `slice`.

This does not invalidate the stack, but it does overwrite `mc.s.greens`.
"""
@bm function calculate_greens(
        mc::DQMC, slice::Int, output::AbstractMatrix = mc.s.greens, 
        safe_mult::Int = mc.p.safe_mult
    )
    copyto!(mc.s.curr_U, I)
    copyto!(mc.s.Ur, I)
    mc.s.Dr .= one(eltype(mc.s.Dr))
    copyto!(mc.s.Tr, I)

    # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
    if slice+1 <= mc.p.slices
        start = slice+1
        stop = mc.p.slices
        for k in reverse(start:stop)
            if mod(k,safe_mult) == 0
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U)
                vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dr))
                udt_AVX_pivot!(mc.s.curr_U, mc.s.Dr, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
                copyto!(mc.s.tmp2, mc.s.Tr)
                vmul!(mc.s.Tr, mc.s.tmp1, mc.s.tmp2)
            else
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U)
            end
        end
        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dr))
        udt_AVX_pivot!(mc.s.Ur, mc.s.Dr, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
        copyto!(mc.s.tmp2, mc.s.Tr)
        vmul!(mc.s.Tr, mc.s.tmp1, mc.s.tmp2)
    end


    copyto!(mc.s.curr_U, I)
    copyto!(mc.s.Ul, I)
    mc.s.Dl .= one(eltype(mc.s.Dl))
    copyto!(mc.s.Tl, I)

    # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
    if slice >= 1
        start = 1
        stop = slice
        for k in start:stop
            if mod(k,safe_mult) == 0
                multiply_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U)
                vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dl))
                udt_AVX_pivot!(mc.s.curr_U, mc.s.Dl, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
                copyto!(mc.s.tmp2, mc.s.Tl)
                vmul!(mc.s.Tl, mc.s.tmp1, mc.s.tmp2)
            else
                multiply_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U)
            end
        end
        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dl))
        udt_AVX_pivot!(mc.s.Ul, mc.s.Dl, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
        copyto!(mc.s.tmp2, mc.s.Tl)
        vmul!(mc.s.Tl, mc.s.tmp1, mc.s.tmp2)
    end

    return calculate_greens(mc, output)
end



################################################################################
### Stack update
################################################################################



# Green's function propagation
@inline @bm function wrap_greens!(mc::DQMC, gf, curr_slice::Int, direction::Int)
    if direction == -1
        # @info "by applying B_$(curr_slice-1)^-1 G B_$(curr_slice-1)"
        multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
        multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
    else
        # @info "by applying B_$(curr_slice) G B_$(curr_slice)^-1"
        multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
        multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
    end
    nothing
end

@bm function propagate(mc::DQMC)
    @inbounds if mc.s.direction == 1
        if mod(mc.s.current_slice, mc.p.safe_mult) == 0
            mc.s.current_slice +=1 # slice we are going to
            if mc.s.current_slice == 1
                copyto!(mc.s.Ur, mc.s.u_stack[1])
                copyto!(mc.s.Dr, mc.s.d_stack[1])
                copyto!(mc.s.Tr, mc.s.t_stack[1])
                copyto!(mc.s.u_stack[1], I)
                mc.s.d_stack[1] .= one(eltype(mc.s.d_stack[1]))
                copyto!(mc.s.t_stack[1], I)
                copyto!(mc.s.Ul, mc.s.u_stack[1])
                copyto!(mc.s.Dl, mc.s.d_stack[1])
                copyto!(mc.s.Tl, mc.s.t_stack[1])

                calculate_greens(mc) # greens_1 ( === greens_{m+1} )

            elseif 1 < mc.s.current_slice <= mc.p.slices
                idx = Int((mc.s.current_slice - 1)/mc.p.safe_mult)

                copyto!(mc.s.Ur, mc.s.u_stack[idx+1])
                copyto!(mc.s.Dr, mc.s.d_stack[idx+1])
                copyto!(mc.s.Tr, mc.s.t_stack[idx+1])
                add_slice_sequence_left(mc, idx)
                copyto!(mc.s.Ul, mc.s.u_stack[idx+1])
                copyto!(mc.s.Dl, mc.s.d_stack[idx+1])
                copyto!(mc.s.Tl, mc.s.t_stack[idx+1])

                if mc.p.check_propagation_error
                    copyto!(mc.s.greens_temp, mc.s.greens)
                end

                # Should this be mc.s.greens_temp?
                # If so, shouldn't this only run w/ mc.p.all_checks = true?
                wrap_greens!(mc, mc.s.greens_temp, mc.s.current_slice - 1, 1)

                calculate_greens(mc) # greens_{slice we are propagating to}

                if mc.p.check_propagation_error
                    # OPT: could probably be optimized through explicit loop
                    greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) 
                    if greensdiff > 1e-7
                        push!(mc.a.propagation_error, greensdiff)
                        mc.p.silent || @printf(
                            "->%d \t+1 Propagation instability\t %.1e\n", 
                            mc.s.current_slice, greensdiff
                        )
                    end
                end

            else # we are going to mc.p.slices+1
                idx = mc.s.n_elements - 1
                add_slice_sequence_left(mc, idx)
                mc.s.direction = -1
                mc.s.current_slice = mc.p.slices+1 # redundant
                propagate(mc)
            end

        else
            # Wrapping
            wrap_greens!(mc, mc.s.greens, mc.s.current_slice, 1)
            mc.s.current_slice += 1
        end

    else # mc.s.direction == -1
        if mod(mc.s.current_slice-1, mc.p.safe_mult) == 0
            mc.s.current_slice -= 1 # slice we are going to
            if mc.s.current_slice == mc.p.slices
                copyto!(mc.s.Ul, mc.s.u_stack[end])
                copyto!(mc.s.Dl, mc.s.d_stack[end])
                copyto!(mc.s.Tl, mc.s.t_stack[end])
                copyto!(mc.s.u_stack[end], I)
                mc.s.d_stack[end] .= one(eltype(mc.s.d_stack[end]))
                copyto!(mc.s.t_stack[end], I)
                copyto!(mc.s.Ur, mc.s.u_stack[end])
                copyto!(mc.s.Dr, mc.s.d_stack[end])
                copyto!(mc.s.Tr, mc.s.t_stack[end])

                calculate_greens(mc) # greens_{mc.p.slices+1} === greens_1

                # wrap to greens_{mc.p.slices}
                wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

            elseif 0 < mc.s.current_slice < mc.p.slices
                idx = Int(mc.s.current_slice / mc.p.safe_mult) + 1

                copyto!(mc.s.Ul, mc.s.u_stack[idx])
                copyto!(mc.s.Dl, mc.s.d_stack[idx])
                copyto!(mc.s.Tl, mc.s.t_stack[idx])
                add_slice_sequence_right(mc, idx)
                copyto!(mc.s.Ur, mc.s.u_stack[idx])
                copyto!(mc.s.Dr, mc.s.d_stack[idx])
                copyto!(mc.s.Tr, mc.s.t_stack[idx])

                if mc.p.check_propagation_error
                    copyto!(mc.s.greens_temp, mc.s.greens)
                end

                calculate_greens(mc)

                if mc.p.check_propagation_error
                    # OPT: could probably be optimized through explicit loop
                    greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) 
                    if greensdiff > 1e-7
                        push!(mc.a.propagation_error, greensdiff)
                        mc.p.silent || @printf(
                            "->%d \t-1 Propagation instability\t %.1e\n", 
                            mc.s.current_slice, greensdiff
                        )
                    end
                end

                wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

            else # we are going to 0
                idx = 1
                add_slice_sequence_right(mc, idx)
                mc.s.direction = 1
                mc.s.current_slice = 0 # redundant
                propagate(mc)
            end

        else
            # Wrapping
            wrap_greens!(mc, mc.s.greens, mc.s.current_slice, -1)
            mc.s.current_slice -= 1
        end
    end
    nothing
end
