mutable struct UnequalTimeStack{GT} <: AbstractDQMCStack
    forward_idx::Int64
    forward_u_stack::Array{GT, 3}
    forward_d_stack::Matrix{Float64}
    forward_t_stack::Array{GT, 3}
    
    backward_idx::Int64
    backward_u_stack::Array{GT, 3}
    backward_d_stack::Matrix{Float64}
    backward_t_stack::Array{GT, 3}
    
    inv_done::Vector{Bool}
    inv_u_stack::Array{GT, 3}
    inv_d_stack::Matrix{Float64}
    inv_t_stack::Array{GT, 3}

    # greens_base::Matrix{GT}
    greens::Matrix{GT}
    U::Matrix{GT}
    D::Vector{Float64}
    T::Matrix{GT}

    # sweep prolly
    last_update::Int64
end

function UnequalTimeStack(mc)
    GreensEltype =  MonteCarlo.geltype(mc)
    HoppingEltype = MonteCarlo.heltype(mc)
    N = length(mc.model.l)
    flv = mc.model.flv

    s = UnequalTimeStack(
        1,
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),
        zeros(Float64, flv*N, mc.s.n_elements),
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),

        length(mc.s.ranges),
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),
        zeros(Float64, flv*N, mc.s.n_elements),
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),

        fill(false, length(mc.s.ranges)),
        zeros(GreensEltype, flv*N, flv*N, length(mc.s.ranges)),
        zeros(Float64, flv*N, length(mc.s.ranges)),
        zeros(GreensEltype, flv*N, flv*N, length(mc.s.ranges)),

        # Matrix{GreensEltype}(I, flv*N, flv*N),
        Matrix{GreensEltype}(I, flv*N, flv*N),
        Matrix{GreensEltype}(undef, flv*N, flv*N),
        Vector{Float64}(undef, flv*N),
        Matrix{GreensEltype}(undef, flv*N, flv*N),
        -1
    )

    # maybe skip identities?
    @views copyto!(s.forward_u_stack[:, :, 1], I)
    @views s.forward_d_stack[:, 1] .= 1
    @views copyto!(s.forward_t_stack[:, :, 1], I)

    @views copyto!(s.backward_u_stack[:, :, end], I)
    @views s.backward_d_stack[:, end] .= 1
    @views copyto!(s.backward_t_stack[:, :, end], I)

    mc.ut_stack = s
end

@bm function build_stack(mc::DQMC, s::UnequalTimeStack)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if mc.last_sweep == s.last_update && all(s.inv_done) && 
        (s.forward_idx == length(mc.s.ranges)+1) && (s.backward_idx == 1)
        return nothing
    end

    # forward
    @bm "forward build" begin
        @inbounds for idx in 1:length(mc.s.ranges)
            copyto!(mc.s.curr_U, s.forward_u_stack[:, :, idx])
            for slice in mc.s.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            @views rvmul!(mc.s.curr_U, Diagonal(s.forward_d_stack[:, idx]))
            @views udt_AVX_pivot!(
                s.forward_u_stack[:, :, idx+1], s.forward_d_stack[:, idx+1], 
                mc.s.curr_U, mc.s.pivot, mc.s.tempv
            )
            @views vmul!(s.forward_t_stack[:, :, idx+1], mc.s.curr_U, s.forward_t_stack[:, :, idx])
        end
    end

    # backward
    @bm "backward build" begin
        @inbounds for idx in length(mc.s.ranges):-1:1
            copyto!(mc.s.curr_U, s.backward_u_stack[:, :, idx + 1])
            for slice in reverse(mc.s.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            @views rvmul!(mc.s.curr_U, Diagonal(s.backward_d_stack[:, idx + 1]))
            @views udt_AVX_pivot!(
                s.backward_u_stack[:, :, idx], s.backward_d_stack[:, idx], 
                mc.s.curr_U, mc.s.pivot, mc.s.tempv
            )
            @views vmul!(s.backward_t_stack[:, :, idx], mc.s.curr_U, s.backward_t_stack[:, :, idx + 1])
        end
    end

    # inverse
    @bm "inverse build" begin
        @inbounds for idx in 1:length(mc.s.ranges)
            @views copyto!(s.inv_t_stack[:, :, idx], I)
            for slice in reverse(mc.s.ranges[idx])
                @views multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[:, :, idx])
            end
            @views udt_AVX_pivot!(
                s.inv_u_stack[:, :, idx], s.inv_d_stack[:, idx], s.inv_t_stack[:, :, idx], 
                mc.s.pivot, mc.s.tempv
            )
        end
    end

    s.inv_done .= true
    s.forward_idx = length(mc.s.ranges)+1
    s.backward_idx = 1
    s.last_update = mc.last_sweep

    nothing
end

@bm function lazy_build_forward!(mc::DQMC, s::UnequalTimeStack, upto)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # forward
    @bm "forward build" begin
        @inbounds for idx in s.forward_idx:upto-1
            copyto!(mc.s.curr_U, s.forward_u_stack[:, :, idx])
            for slice in mc.s.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            @views rvmul!(mc.s.curr_U, Diagonal(s.forward_d_stack[:, idx]))
            @views udt_AVX_pivot!(
                s.forward_u_stack[:, :, idx+1], s.forward_d_stack[:, idx+1], 
                mc.s.curr_U, mc.s.pivot, mc.s.tempv
            )
            @views vmul!(s.forward_t_stack[:, :, idx+1], mc.s.curr_U, s.forward_t_stack[:, :, idx])
        end
    end

    s.forward_idx = upto
    nothing
end

@bm function lazy_build_backward!(mc::DQMC, s::UnequalTimeStack, downto)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # backward
    @bm "backward build" begin
        @inbounds for idx in s.backward_idx-1:-1:downto
            copyto!(mc.s.curr_U, s.backward_u_stack[:, :, idx + 1])
            for slice in reverse(mc.s.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            end
            @views rvmul!(mc.s.curr_U, Diagonal(s.backward_d_stack[:, idx + 1]))
            @views udt_AVX_pivot!(
                s.backward_u_stack[:, :, idx], s.backward_d_stack[:, idx], 
                mc.s.curr_U, mc.s.pivot, mc.s.tempv
            )
            @views vmul!(s.backward_t_stack[:, :, idx], mc.s.curr_U, s.backward_t_stack[:, :, idx + 1])
        end
    end

    s.backward_idx = downto
end

@bm function lazy_build_inv!(mc::DQMC, s::UnequalTimeStack, from, to)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.s.ranges)+1
    end

    # inverse
    @bm "inverse build" begin
        @inbounds for idx in from:to #length(mc.s.ranges)
            s.inv_done[idx] && continue
            @views copyto!(s.inv_t_stack[:, :, idx], I)
            for slice in reverse(mc.s.ranges[idx])
                @views multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[:, :, idx])
            end
            @views udt_AVX_pivot!(
                s.inv_u_stack[:, :, idx], s.inv_d_stack[:, idx], s.inv_t_stack[:, :, idx], 
                mc.s.pivot, mc.s.tempv
            )
        end
    end

    nothing
end

"""
    greens(mc::DQMC, k, l)

Calculates the unequal-time Greens function 
`greens(mc, k, l) = G(k <- l) = G(kΔτ <- lΔτ)` where `nslices(mc) ≥ k > l ≥ 0`.

Note that `G(0, 0) = G(nslices, nslices) = G(β, β) = G(0, 0)`.
"""
greens(mc::DQMC, slice1::Int64, slice2::Int64) = greens!(mc, slice1, slice2)
@bm function greens!(mc::DQMC, slice1::Int64, slice2::Int64)
    @assert 0 ≤ slice1 ≤ mc.p.slices
    @assert 0 ≤ slice2 ≤ mc.p.slices
    @assert slice2 ≤ slice1
    s = mc.ut_stack
    # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
    # Complete build (do this, or lazy builds for each)
    # build_stack(mc, mc.ut_stack) 

    # forward = [1:((i-1)*mc.p.safe_mult) for i in 1:mc.s.n_elements]
    # backward = [mc.p.slices:-1:(1+(i-1)*mc.p.safe_mult) for i in 1:mc.s.n_elements]
    # _inv = [((i-1)*mc.p.safe_mult)+1:(i*mc.p.safe_mult) for i in eachindex(mc.s.ranges)]

    # @info forward
    # @info backward
    # @info _inv

    # k ≥ l or slice1 ≥ slice2
    # B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    inv_slices = Int64[]
    @bm "inverse pre-computed" begin
        lower = div(slice2+1 + mc.p.safe_mult - 2, mc.p.safe_mult) + 1
        upper = div(slice1, mc.p.safe_mult)
        lazy_build_inv!(mc, s, lower, upper) # only if build_stack is commented out
        copyto!(s.U, I)
        s.D .= 1
        copyto!(s.T, I)
        for idx in lower:upper
            # () is operation order, [] marks UDT decomposition
            # U [(D (T stack_U)) stack_D] stack_T
            # (U [U') D (T'] stack_T)
            # U D T
            @views vmul!(mc.s.curr_U, s.T, s.inv_u_stack[:, :, idx])
            lvmul!(Diagonal(s.D), mc.s.curr_U)
            @views rvmul!(mc.s.curr_U, Diagonal(s.inv_d_stack[:, idx]))
            udt_AVX_pivot!(mc.s.tmp1, s.D, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
            @views vmul!(s.T, mc.s.curr_U, s.inv_t_stack[:, :, idx])
            vmul!(mc.s.curr_U, s.U, mc.s.tmp1)
            copyto!(s.U, mc.s.curr_U)
            # append!(inv_slices, _inv[idx])
        end
    end
  
    @bm "inverse fine tuning" begin
        lower_slice = (lower-1) * mc.p.safe_mult + 1
        upper_slice = upper * mc.p.safe_mult

        for slice in min(lower_slice-1, slice1) : -1 : slice2+1
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.U)
            # inv_slices = [slice, inv_slices...]
        end
        for slice in max(upper_slice+1, min(lower_slice-1, slice1)+1) : slice1
            multiply_slice_matrix_inv_right!(mc, mc.model, slice, s.T)
            # push!(inv_slices, slice)
        end
        # U D T = B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    end
    # @info "               inv: $(inv_slices)"

    @bm "forward B" begin
        # B(slice1, 1) = Ul Dl Tl
        idx = div(slice2-1, mc.p.safe_mult) # 0 based index into stack
        lazy_build_forward!(mc, s, idx+1) # only if build_stack is commented out
        # forward_slices = collect(forward[idx+1])
        copyto!(mc.s.curr_U, s.forward_u_stack[:, :, idx+1])
        # @info "$slice2 || $(idx+1) || $(mc.p.safe_mult * idx + 1) : $(slice2)"
        for slice in mc.p.safe_mult * idx + 1 : slice2
            multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            # push!(forward_slices, slice)
        end
        @views rvmul!(mc.s.curr_U, Diagonal(s.forward_d_stack[:, idx+1]))
        @views udt_AVX_pivot!(mc.s.Ul, mc.s.Dl, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
        @views vmul!(mc.s.Tl, mc.s.curr_U, s.forward_t_stack[:, :, idx+1])
        # @info "               forward: $(reverse(forward_slices)) {$(idx+1)}"
    end

    @bm "backward B" begin
        # B(N, slice2) = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
        idx = div.(slice1 + mc.p.safe_mult - 1, mc.p.safe_mult) # 0 based index into stack
        lazy_build_backward!(mc, s, idx+1) # only if build_stack is commented out
        # backward_slices = collect(backward[idx+1])
        copyto!(mc.s.curr_U, s.backward_u_stack[:, :, idx+1])
        # @info "$slice1 || $(idx+1) || $(mc.p.safe_mult * idx) : -1 : $(slice1+1)"
        for slice in mc.p.safe_mult * idx : -1 : slice1+1
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
            # push!(backward_slices, slice)
        end
        @views rvmul!(mc.s.curr_U, Diagonal(s.backward_d_stack[:, idx+1]))
        @views udt_AVX_pivot!(mc.s.Ur, mc.s.Dr, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
        @views vmul!(mc.s.Tr, mc.s.curr_U, s.backward_t_stack[:, :, idx + 1])
        # @info "               backward: $(backward_slices) {$(idx+1)}"
    end

    @bm "compute G" begin
        # [B_{l+1}^-1 B_{l+2}^-1 ⋯ B_k^-1 + B_l ⋯ B_1 B_N ⋯ B_{k+1}]^-1
        # [U D T + Ul Dl Tl Tr^† Dr Ur^†]^-1
        @bm "B1" begin
            vmul!(s.greens, mc.s.Tl, adjoint(mc.s.Tr))
            rvmul!(s.greens, Diagonal(mc.s.Dr))
            lvmul!(Diagonal(mc.s.Dl), s.greens)
        end
        @bm "udt" begin
            udt_AVX_pivot!(mc.s.Tr, mc.s.Dr, s.greens, mc.s.pivot, mc.s.tempv)
        end
        # [U D T + Ul Tr Dr G Ur^†]^-1
        @bm "B2" begin
            vmul!(mc.s.Tl, mc.s.Ul, mc.s.Tr)
            vmul!(mc.s.Ul, s.greens, adjoint(mc.s.Ur))
            # [U D T + Tl Dr Ul]^-1  U is not unitary, Tl is
            copyto!(mc.s.Tr, mc.s.Ul)
        end
        @bm "inv" begin
            LinearAlgebra.inv!(RecursiveFactorization.lu!(mc.s.Tr, mc.s.pivot))
        end
        # free: U D T Tl Dr Ul Tr  | G Dl Ur
        @bm "B3" begin
            vmul!(mc.s.Ur, adjoint(mc.s.Tl), s.U)
            vmul!(s.greens, s.T, mc.s.Tr)
            # [Tl (Tl^† U D T Ul^-1 + Dr) Ul]^-1 = [Tl (Ur D G + Dr) Ul]^-1
            rvmul!(mc.s.Ur, Diagonal(s.D))
            vmul!(s.T, mc.s.Ur, s.greens)
            # [Tl (T + Dr) Ul]^-1  =  [Tl (Tl^† U D T Ul^-1 + Dr) Ul]^-1
            @avx for i in 1:length(mc.s.Dr)
                s.T[i, i] = s.T[i, i] + mc.s.Dr[i]
            end
        end
        # [Tl T Ul]^-1
        @bm "udt" begin
            udt_AVX_pivot!(s.U, s.D, s.T, mc.s.pivot, mc.s.tempv)
        end
        @bm "B4" begin
            vmul!(mc.s.Ur, mc.s.Tl, s.U)
            vmul!(mc.s.Tr, s.T, mc.s.Ul)
        end
        # [Ur D Tr]^-1 with U unitary
        @bm "inv" begin
            LinearAlgebra.inv!(RecursiveFactorization.lu!(mc.s.Tr, mc.s.pivot))
        end
        @bm "B5" begin
            @avx for i in eachindex(mc.s.Dr)
                mc.s.Dr[i] = 1.0 / s.D[i]
            end
            rvmul!(mc.s.Tr, Diagonal(mc.s.Dr))
            vmul!(s.greens, mc.s.Tr, adjoint(mc.s.Ur))
        end
        # Tr^-1 D^-1 Ur^†
    end

    s.greens
end
