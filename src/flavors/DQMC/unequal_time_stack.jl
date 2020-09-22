# We could technically wait with allocating the stacks until they are actually
# needed. Gk0 doesn't need the forward stack, for example (though it does use it atm)

# We can probably also use the normal stack here...
# - If we assume calls to greens() to happen at a current_slice = 1 the normal
#   stack should either be the forward or backward stack (need to check). So we
#   could drop one of the stacks here completely.
# - More generally, we could copy data from any current_slice instead of
#   recalculating.

mutable struct UnequalTimeStack{GT} <: AbstractDQMCStack
    # B_{n*safe_mult} ... B_1
    forward_idx::Int64
    forward_u_stack::Array{GT, 3}
    forward_d_stack::Matrix{Float64}
    forward_t_stack::Array{GT, 3}
    
    # B_{n*safe_mult+1}^† ... B_M^†
    backward_idx::Int64
    backward_u_stack::Array{GT, 3}
    backward_d_stack::Matrix{Float64}
    backward_t_stack::Array{GT, 3}
    
    # B_{n*safe_mult+1}^-1 .. B_{(n+1)*safe_mult}^-1
    inv_done::Vector{Bool}
    inv_u_stack::Array{GT, 3}
    inv_d_stack::Matrix{Float64}
    inv_t_stack::Array{GT, 3}

    # Greens construction
    greens::Matrix{GT}
    U::Matrix{GT}
    D::Vector{Float64}
    T::Matrix{GT}

    # To avoid recalculating
    last_update::Int64
    last_slices::Tuple{Int64, Int64}
end

function Base.sizeof(::Type{UnequalTimeStack}, mc::DQMC)
    GreensEltype =  MonteCarlo.geltype(mc)
    N = length(mc.model.l)
    flv = mc.model.flv
    blocks = mc.s.n_elements

    sizeof(GreensEltype) * flv*N * flv*N * blocks * 4 +         # forward+backward U, T
    sizeof(GreensEltype) * flv*N * blocks * 2 +                 # forward+backward D
    sizeof(GreensEltype) * flv*N * flv*N * (blocks-1) * 2 +     # inv U, T
    sizeof(GreensEltype) * flv*N * (blocks-1) +                 # inv Dr
    5*8 + sizeof(Bool) * (blocks-1) +                           # book-keeping/skipping
    sizeof(GreensEltype) * flv*N * flv*N * 3 +                  # greens, U, Tr
    sizeof(GreensEltype) * flv*N                                # D
end

function UnequalTimeStack(mc)
    GreensEltype =  MonteCarlo.geltype(mc)
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
        -1, (-1, -1)
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

########################################
# Stack building (full & lazy)
########################################

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



########################################
# Calculating greens
########################################



"""
    greens(mc::DQMC, k, l)

Calculates the unequal-time Greens function at slice `k` and `l`, i.e.
`G(k <- l) = G(kΔτ <- lΔτ) = ⟨cᵢ(k⋅Δτ)cⱼ(l⋅Δτ)^†⟩` where `nslices(mc) ≥ k > l ≥ 0`.

Note that `G(0, 0) = G(nslices, nslices) = G(β, β) = G(0, 0)`.

This method requires the `UnequalTimeStack` to be available. The stack 
variables `Ul`, `Dl`, `Tl`, `Ur`, `Dr`, `Tr`, `curr_U` and `tmp1` will
be overwritten (but are avaible as `output` and `temp`).
"""
@bm greens(mc::DQMC, slice1::Int64, slice2::Int64) = copy(_greens!(mc, slice1, slice2))
"""
    greens!(mc::DQMC, k, l[; output=mc.s.greens_temp, temp=mc.s.tmp1])
"""
@bm function greens!(
        mc::DQMC, slice1::Int64, slice2::Int64; 
        output = mc.s.greens_temp, temp = mc.s.tmp1
    ) 
    _greens!(mc, slice1, slice2, output, temp)
end
"""
    _greens!(mc::DQMC, k, l, output=mc.s.greens_temp, temp=mc.s.tmp1)
"""
function _greens!(
        mc::DQMC, slice1::Int64, slice2::Int64,
        output = mc.s.greens_temp, temp = mc.s.tmp1
    )
    calculate_greens(mc, slice1, slice2)
    _greens!(mc, output, mc.ut_stack.greens, temp)
end
@bm function calculate_greens(mc::DQMC, slice1::Int64, slice2::Int64)
    @assert 0 ≤ slice1 ≤ mc.p.slices
    @assert 0 ≤ slice2 ≤ mc.p.slices
    @assert slice2 ≤ slice1
    s = mc.ut_stack
    if s.last_slices != (slice1, slice2)
        s.last_slices = (slice1, slice2)
        calculate_greens_full!(mc, s, slice1, slice2)
    end
    s.greens
end


@bm function calculate_greens_full!(mc, s, slice1, slice2)
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
    # inv_slices = Int64[]
    @bm "inverse pre-computed" begin
        lower = div(slice2+1 + mc.p.safe_mult - 2, mc.p.safe_mult) + 1
        upper = div(slice1, mc.p.safe_mult)
        # @info "$lower:$upper"
        lazy_build_inv!(mc, s, lower, upper) # only if build_stack is commented out
        copyto!(s.U, I)
        s.D .= 1
        copyto!(s.T, I)
        # UDT combining style
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
        if min(lower_slice-1, slice1) ≥ slice2+1
            vmul!(mc.s.curr_U, s.U, Diagonal(s.D))
            udt_AVX_pivot!(s.U, s.D, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
            vmul!(mc.s.tmp1, mc.s.curr_U, s.T)
            copyto!(s.T, mc.s.tmp1)
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

    # evaluated = "G_{$slice1, $slice2} = ["
    # evaluated *= join(("B_$(i)⁻¹" for i in inv_slices), "⋅") * " + "
    # evaluated *= join(("B_$i" for i in vcat(forward_slices, backward_slices)), "⋅") * "]⁻¹"
    # println(evaluated)

    @bm "compute G" begin
        # [B_{l+1}^-1 B_{l+2}^-1 ⋯ B_k^-1 + B_l ⋯ B_1 B_N ⋯ B_{k+1}]^-1
        # [U D T + Ul Dl Tl Tr^† Dr Ur^†]^-1
        @bm "B1" begin
            vmul!(s.greens, mc.s.Tl, adjoint(mc.s.Tr))
            rvmul!(s.greens, Diagonal(mc.s.Dr))
            lvmul!(Diagonal(mc.s.Dl), s.greens)
        end
        @bm "udt" begin
            # udt_AVX_pivot!(mc.s.Tr, mc.s.Dr, s.greens, mc.s.pivot, mc.s.tempv)
            udt_AVX_pivot!(mc.s.Tr, mc.s.Dr, s.greens, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [U D T + Ul Tr Dr G Ur^†]^-1
        @bm "B2" begin
            vmul!(mc.s.Tl, mc.s.Ul, mc.s.Tr)
            # vmul!(mc.s.Ul, s.greens, adjoint(mc.s.Ur))
            copyto!(mc.s.Tr, mc.s.Ur)
            rdivp!(mc.s.Tr, s.greens, mc.s.Ul, mc.s.pivot)
            # [U D T + Tl Dr Ul]^-1  Ul is not unitary, Tl is
            # copyto!(mc.s.Tr, mc.s.Ul)
        end
        # used:         U D T Tl Dr Ul
        # available:    Ur Dl Tr greens tmp1 tmp2
        # [U (D T Ul^-1 + U^† Tl Dr) Ul]^-1
        # [U D_max (D_min T Ul^-1 1/Dr_max + 1/D_max U^† Tl Dr_min) Dr_max Ul]^-1
        @bm "B3" begin
            # 1/D_max U^† Tl Dr_min
            vmul!(mc.s.Ur, adjoint(s.U), mc.s.Tl)
            mc.s.Dl .= 1.0 ./ max.(1.0, s.D)
            lvmul!(Diagonal(mc.s.Dl), mc.s.Ur)
            mc.s.Dl .= min.(1.0, mc.s.Dr)
            rvmul!(mc.s.Ur, Diagonal(mc.s.Dl))
        end
        # @bm "inv" begin
        #     # Tr = Ul^-1
        #     copyto!(mc.s.Tr, mc.s.Ul)
        #     LinearAlgebra.inv!(RecursiveFactorization.lu!(mc.s.Tr, mc.s.pivot))
        # end
        @bm "B4" begin
            # D_min T Ul^-1 1/Dr_max = D_min T Tr 1/Dr_max
            vmul!(mc.s.Tl, s.T, mc.s.Tr)
            mc.s.Dl .= min.(1.0, s.D)
            lvmul!(Diagonal(mc.s.Dl), mc.s.Tl)
            mc.s.Dl .= 1.0 ./ max.(1.0, mc.s.Dr)
            rvmul!(mc.s.Tl, Diagonal(mc.s.Dl))
        end
        # [U D_max (Ur + Tl) Dr_max Ul]^-1
        @bm "sum, UDT" begin
            mc.s.Tl .+= mc.s.Ur
            # udt_AVX_pivot!(mc.s.Ur, mc.s.Dl, mc.s.Tl, mc.s.pivot, mc.s.tempv)
            udt_AVX_pivot!(mc.s.Ur, mc.s.Dl, mc.s.Tl, mc.s.pivot, mc.s.tempv, Val(false))
        end
        # [U D_max Ur Dl Tl Dr_max Ul]^-1 # Ul is not unitary
        # Ul^-1 ((((Dr_max^-1 Tl^-1) Dl^-1) Ur^†) D_max^-1) U^† # Ul^-1 = Tr
        @bm "B5" begin
            # LinearAlgebra.inv!(RecursiveFactorization.lu!(mc.s.Tl, mc.s.pivot))
            # This order seems to be fine
            mc.s.Dr .= 1.0 ./ max.(1.0, mc.s.Dr)
            rdivp!(mc.s.Ul, mc.s.Dr, mc.s.Tl, mc.s.pivot)
            @avx for i in eachindex(mc.s.Dl)
                mc.s.Dl[i] = 1.0 / mc.s.Dl[i]
            end
            rvmul!(mc.s.Ul, Diagonal(mc.s.Dl))
            vmul!(s.greens, mc.s.Ul, adjoint(mc.s.Ur))
            mc.s.Dl .= 1.0 ./ max.(1.0, s.D)
            rvmul!(s.greens, Diagonal(mc.s.Dl))
        end
        @bm "UDT" begin
            udt_AVX_pivot!(mc.s.Ur, s.D, s.greens, mc.s.pivot, mc.s.tempv)
        end
        @bm "B6" begin
            vmul!(mc.s.Ul, mc.s.Tr, mc.s.Ur)
            vmul!(mc.s.Tl, s.greens, adjoint(s.U))
            rvmul!(mc.s.Ul, Diagonal(s.D))
            vmul!(s.greens, mc.s.Ul, mc.s.Tl)
        end
    end

    s.greens
end



################################################################################
### Iterators
################################################################################



# The method above provides the unequal-time Greens function in a stable, 
# general but expensive way. The idea of iterators is to provide what is needed
# more quickly, without the UnequalTimeStack and without having to worry about
# managing stack variables.

# Currently we have:
# GreensIterator
#   - attempts to be general (but isn't yet)
#   - requires UnequalTimeStack
#   - should have high accuracy, but still faster than manual
# CombinedGreensIterator
#   - does just l = 0, k = 0:M-1, but returns G_kl and G_ll
#   - does not require UnequalTimeStack
#   - same accuracy as GreensIterator

abstract type AbstractGreensIterator end
init!(::AbstractGreensIterator) = nothing
"""
    verify(iterator::AbstractGreensIterator[, maxerror=1e-6])

Returns true if the given `iterator` is accurate up to the given `maxerror`.

See also: (@ref)[`accuracy`]
"""
verify(it::AbstractGreensIterator, maxerror=1e-6) = maximum(accuracy(it)) < maxerror


# Maybe split into multiple types?
struct GreensIterator{slice1, slice2, T <: DQMC} <: AbstractGreensIterator
    mc::T
    recalculate::Int64
end

"""
    GreensIterator(mc::DQMC[, ks=Colon(), ls=0, recalculate=4mc.p.safe_mult])

Creates an Iterator which calculates all `[G[k <- l] for l in ls for k in ks]`
efficiently. Currently only allows `ks = :, ls::Integer`. 

For stability it is necessary to fully recompute `G[k <- l]` every so often. one
can adjust the frequency of recalculations via `recalculate`. To estimate the 
resulting accuracy, one may use `accuracy(GreensIterator(...))`.

This iterator requires the `UnequalTimeStack`. Iteration overwrites the 
`DQMCStack` variables `curr_U`, `Ul`, `Dl`, `Tl`, `Ur`, `Dr`, `Tr`, `tmp1` and 
`tmp2`, with `curr_U` acting as the output. For the iteration to remain valid, 
the UnequalTimeStack must not be overwritten. As such:
- `greens!(mc)` can be called and remains valid
- `greens!(mc, slice)` can be called and remains valid
- `greens!(mc, k, l)` will break iteration but remains valid (call before iterating)
"""
function GreensIterator(mc::T, slice1=Colon(), slice2=0, recalculate=4mc.p.safe_mult) where {T <: DQMC}
    GreensIterator{slice1, slice2, T}(mc, recalculate)
end
init!(it::GreensIterator) = it.mc.ut_stack = UnequalTimeStack(it.mc)
Base.length(it::GreensIterator{:, i}) where {i} = it.mc.p.slices + 1 - i

# Slower, versatile version:
function Base.iterate(it::GreensIterator{:, i}) where {i}
    s = it.mc.ut_stack
    calculate_greens_full!(it.mc, s, i, i)
    copyto!(s.T, s.greens)
    udt_AVX_pivot!(s.U, s.D, s.T, it.mc.s.pivot, it.mc.s.tempv)
    G = _greens!(it.mc, it.mc.s.curr_U, s.greens, it.mc.s.Ur)
    return (G, (i+1, i))
end
function Base.iterate(it::GreensIterator{:}, state)
    s = it.mc.ut_stack
    k, l = state
    if k > it.mc.p.slices
        return nothing
    elseif k % it.recalculate == 0
        # Recalculate
        calculate_greens_full!(it.mc, s, k, l) # writes s.greens
        G = _greens!(it.mc, it.mc.s.curr_U, s.greens, it.mc.s.Tl)
        copyto!(s.T, s.greens)
        udt_AVX_pivot!(s.U, s.D, s.T, it.mc.s.pivot, it.mc.s.tempv)
        return (G, (k+1, l))
    elseif k % it.mc.p.safe_mult == 0
        # Stabilization
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.s.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.s.tmp1, it.mc.s.curr_U, s.T)
        udt_AVX_pivot!(s.U, s.D, it.mc.s.curr_U, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(it.mc.s.tmp2, it.mc.s.curr_U, s.T)
        copyto!(s.T, it.mc.s.tmp2)
        G = _greens!(it.mc, it.mc.s.curr_U, it.mc.s.tmp1, it.mc.s.tmp2)
        return (G, (k+1, l))
    else
        # Quick advance
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.s.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.s.tmp1, it.mc.s.curr_U, s.T)
        G = _greens!(it.mc, it.mc.s.curr_U, it.mc.s.tmp1, it.mc.s.Ur)
        s.last_slices = (-1, -1) # for safety
        return (G, (k+1, l))
    end
end
"""
    accuracy(iterator::AbstractGreensIterator)

Compares values from the given iterator to more verbose computations, returning 
the maximum differences for each. This can be used to check numerical stability.
"""
function accuracy(iter::GreensIterator{:, l}) where {l}
    mc = iter.mc
    Gk0s = [deepcopy(greens(mc, k, l)) for k in l:nslices(mc)]
    [maximum(abs.(Gk0s[i] .- G)) for (i, G) in enumerate(iter)]
end



"""
    CombinedGreensIterator(mc::DQMC, recalculate=4mc.p.safe_mult)

Returns an iterator which iterates `[(G[k, 0], G[k, k]) for k in 0:nslices-1]`. 
Does a full recalculation of `G[k, 0]` and `G[k, k]` if `k % recalculate == 0`.


This iterator requires the `UnequalTimeStack` and uses `U` and `T` as outputs. 
For correct iteration it requires the `DQMCStack` variable `Ul`, `Dl`, `Tl`, 
`Ur`, `Dr` and `Tr` to remain unchanged. Further, the stack variables `curr_U`,
`tmp1` and `tmp2` are overwritten. As such
- `greens!(mc)` will break iteration but remains valid (call before iterating)
- `greens!(mc, slice)` will break iteration but remains valid
- `greens!(mc, k, l)` will break iteration but remains valid
"""
struct CombinedGreensIterator{T <: DQMC} <: AbstractGreensIterator
    mc::T
    recalculate::Int64
end

function CombinedGreensIterator(mc::T, recalculate=4mc.p.safe_mult) where T
    CombinedGreensIterator{high_precision, T}(mc, recalculate)
end
function init!(it::CombinedGreensIterator)
    if it.recalculate * it.mc.p.safe_mult < nslices(it.mc)
        it.mc.ut_stack = UnequalTimeStack(it.mc)
    end
end

Base.length(it::CombinedGreensIterator) = nslices(it.mc)

# Fast specialized version
function Base.iterate(it::CombinedGreensIterator)
    # Measurements take place at current_slice = 1 <> l = τ = 0
    @assert current_slice(it.mc) == 1
    s = it.mc.s
    copyto!(s.Tl, s.greens)
    Gkl = _greens!(it.mc, it.mc.ut_stack.U, s.Tl, s.curr_U)
    Gkk = copyto!(it.mc.ut_stack.T, it.mc.ut_stack.U)
    udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, it.mc.s.pivot, it.mc.s.tempv)
    copyto!(s.Ur, s.Ul)
    copyto!(s.Dr, s.Dl)
    copyto!(s.Tr, s.Tl)
    return ((Gkl, Gkk), 1)
end
function Base.iterate(it::CombinedGreensIterator, k)
    # k is 1-based
    s = it.mc.s
    if k >= it.mc.p.slices
        return nothing
    elseif k % it.recalculate == 0 
        # Recalculation will overwrite Ul, Dl, Tl, Ur, Dr, Tr, curr_U, tmp1, tmp2
        copyto!(s.tmp2, calculate_greens_full!(it.mc, it.mc.ut_stack, k, k))
        copyto!(s.Tl, calculate_greens_full!(it.mc, it.mc.ut_stack, k, 0))
        copyto!(s.Tr, s.tmp2)
    
        Gkk = _greens!(it.mc, it.mc.ut_stack.T, s.tmp2, s.curr_U)
        udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, it.mc.s.pivot, it.mc.s.tempv)

        Gkl = _greens!(it.mc, it.mc.ut_stack.U, s.Tl, s.curr_U)
        udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, it.mc.s.pivot, it.mc.s.tempv)
        return ((Gkl, Gkk), k+1)
    elseif k % it.mc.p.safe_mult == 0
        # Stabilization
        # Gkl
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ul) # writes to tmp1 and tmp2
        vmul!(s.tmp1, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp2, s.tmp1, s.Tl)
        udt_AVX_pivot!(s.Ul, s.Dl, s.tmp1, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(s.curr_U, s.tmp1, s.Tl)
        copyto!(s.Tl, s.curr_U)
        Gkl = _greens!(it.mc, it.mc.ut_stack.U, s.tmp2, s.curr_U)
                
        # Gkk
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ur)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, k, s.Tr)
        vmul!(s.tmp2, s.Ur, Diagonal(s.Dr))
        vmul!(s.tmp1, s.tmp2, s.Tr)
        udt_AVX_pivot!(s.curr_U, s.Dr, s.tmp2, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(s.Ur, s.tmp2, s.Tr)
        vmul!(s.Tr, Diagonal(s.Dr), s.Ur)
        udt_AVX_pivot!(s.tmp2, s.Dr, s.Tr, it.mc.s.pivot, it.mc.s.tempv)
        vmul!(s.Ur, s.curr_U, s.tmp2)
        Gkk = _greens!(it.mc, it.mc.ut_stack.T, s.tmp1, s.curr_U)

        return ((Gkl, Gkk), k+1)
    else
        # Quick advance
        # Gkl
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ul)
        vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp1, s.curr_U, s.Tl)
        Gkl = _greens!(it.mc, it.mc.ut_stack.U, s.tmp1, s.curr_U)

        # Gkk
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ur)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, k, s.Tr)
        vmul!(s.curr_U, s.Ur, Diagonal(s.Dr))
        vmul!(s.tmp1, s.curr_U, s.Tr)
        Gkk = _greens!(it.mc, it.mc.ut_stack.T, s.tmp1, s.curr_U)
        return ((Gkl, Gkk), k+1)
    end
end

function accuracy(iter::CombinedGreensIterator)
    mc = iter.mc
    Gk0s = [deepcopy(greens(mc, k, 0)) for k in 0:nslices(mc)-1]
    [maximum(abs.(Gk0s[i] .- G)) for (i, G) in enumerate(iter)]
end

################################################################################

# TO-DONE
# - greens at arbitrary times with reasonably high precision
# - iterators for efficient and save computation thereof

# TODO
# - measurements basically
#   - implement some kind of Integrator
#   - add a preparation step (or maybe in constructor?) that merges Integrators
#   - Integrator runs through iterator
#   - also figure out how to add masks to this, I guess?

# Essentially we should end up with
# function integrate!(...)
#     for (Gk0, Gkk) in iterator
#         for m in measurements
#             # maybe also pass G = G00? that should be kept alive, I think
#             kernel!(m, Gk0, Gkk, ...)
#         end
#     end
# end



################################################################################

# 1
# TODO
# make slice indices match!
# this file uses τ = l * Δτ
# Carsten used: τ = (l-1) * Δτ





# # TODO: check this, integrate this
# struct EqualTimeIterator{T <: DQMC}
#     mc::T
# end


# function EqualTimeIterator(mc::T, range=0:N-1) where {T <: DQMC}
#     EqualTimeIterator{T}(mc)
# end

# function Base.iterate(it::EqualTimeIterator)
#     # 2x Faster fuck yea!
#     # Avoids building the ut_stack
#     # Measurements take place at slice = nslices = 0
#     s = it.mc.s
#     copyto!(s.Tl, s.greens)
#     G = _greens!(it.mc, s.Ur, s.Tl, s.Tr) # just be careful here
#     udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, it.mc.s.pivot, it.mc.s.tempv)
#     return (G, 1)
# end
# function Base.iterate(it::GreensIterator{:, 0}, state)
#     s = it.mc.s
#     k = state
#     if k ≥ it.mc.p.slices
#         return nothing
#     elseif k % it.mc.p.safe_mult == 0
#         # Stabilization
#         # vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
#         # vmul!(s.Ur, s.curr_U, s.Tl)
#         # udt_AVX_pivot!(s.Tr, s.Dl, s.curr_U, it.mc.s.pivot, it.mc.s.tempv)
#         # vmul!(s.Ul, s.Ul, s.Tr)
#         # vmul!(s.Tr, s.curr_U, s.Tl)
#         # copyto!(s.Tl, s.Tr)
#         # G = _greens!(it.mc, s.curr_u, s.Ur, s.Tr)

#         # Shouldn't these be here?
#         multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ul)
#         multiply_slice_matrix_inv_rightt!(it.mc, it.mc.model, k, s.Tl)
#         vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
#         vmul!(s.Tr, s.curr_U, s.Tl)
#         G = _greens!(it.mc, s.curr_u, s.Tr, s.Ur)
#         udt_AVX_pivot!(s.Ul, s.Dl, s.Tr, it.mc.s.pivot, it.mc.s.tempv)
#         copyto!(s.Tl, s.Tr)
#         return (G, k+1)
#     else
#         # Quick advance
#         multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.Ul)
#         multiply_slice_matrix_inv_rightt!(it.mc, it.mc.model, k, s.Tl)
#         vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
#         vmul!(s.Ur, s.curr_U, s.Tl)
#         G = _greens!(it.mc, s.curr_U, s.Ur, s.Tr)
#         return (G, k+1)
#     end
# end





# # TODO
# # - iterate (0, 0) -> (M-1, 0)  DONE
# # - synchronize         CONCEPT done?
# #   - define Integrator w/ a GreensIterator
# #   - measurements use Integrator
# #   - integrater depends on measurement kernels like similar to _measurement_kernels

# # Sketch:

# function measure!(m, ...)
#     result = integrate!(m.Integrator, m, time_kernel)
#     mask_kernel!(m, m.mask, result, mask_kernel)
#     push!(m.obs, result)
# end



# # more planning
# kernel!(IntegrationStep{range, range}(k, l), measurements, Gtau0, G0, Gtau, args...)
# # or IterationStep?
# kernel!(MaskIndex{Masktype}(i, j, dir maybe))

# synchronize!(measurements) -> remake everything with identical Iterators


# # -------

# for (write_idxs, read_idxs) in mask
#     for (tau, Greens) in enumerate(it)
#         kernel!()
#     end
# end

# # becomes
# for step::MaskStep in mask
#     kernel!(step, measurement)
# end

# kernel!(::MaskStep) = for step::TimeStep in iter; kernel!(step, ...) end

# # this wouldn't synchronize easily though, because TimeStep is hidden in kernel!, no?

# # Just combine them 4Head
# # a bunch of setup
# # like calculate Greens I-G, make it avaible in mask step somehow
# for mask_step::MaskStep in mask
#     for time_step::TimeStep in enumerate(it)
#         # injected after the fact
#         for measurement in values(mc.measurements)
#             kernel!(mask_step, time_step, measurement, args...)
#         end
#     end
# end

# # but that requires skipping default measure!

# function measure!(m::Measurement)
#     compute!(m)
#     push!(m.obs, m.temp)
# end
# function kernel!(measurement) end
# function compute!(m)
#     needs_computation && compute_all!()
#     nothing
# end

# # We will also need G(l, l) btw

