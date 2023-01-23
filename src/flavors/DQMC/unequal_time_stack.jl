mutable struct UnequalTimeStack{GT<:Number, GMT<:AbstractArray{GT}} <: AbstractDQMCStack
    # B_{n*safe_mult} ... B_1
    forward_idx::Int64
    forward_u_stack::Vector{GMT}
    forward_d_stack::Vector{Vector{Float64}}
    forward_t_stack::Vector{GMT}
    
    # B_{n*safe_mult+1}^† ... B_M^†
    backward_idx::Int64
    backward_u_stack::Vector{GMT}
    backward_d_stack::Vector{Vector{Float64}}
    backward_t_stack::Vector{GMT}
    
    # B_{n*safe_mult+1}^-1 .. B_{(n+1)*safe_mult}^-1
    inv_done::Vector{Bool}
    inv_u_stack::Vector{GMT}
    inv_d_stack::Vector{Vector{Float64}}
    inv_t_stack::Vector{GMT}

    # Greens construction
    greens::GMT
    tmp::GMT
    U::GMT
    D::Vector{Float64}
    T::GMT

    # temps for complex Greens
    # TODO rework measurements to work well with StructArrays and remove this
    complex_greens_temp1::Matrix{ComplexF64}
    complex_greens_temp2::Matrix{ComplexF64}
    complex_greens_temp3::Matrix{ComplexF64}

    # To avoid recalculating
    last_update::Int64
    last_slices::Tuple{Int64, Int64}

    function UnequalTimeStack{GT, GMT}() where {GT<:Number, GMT<:AbstractArray{GT}}
        @assert isconcretetype(GT);
        @assert isconcretetype(GMT);
        new{GT, GMT}()
    end
end

# TODO
# function Base.sizeof(::Type{UnequalTimeStack}, mc::DQMC)
#     GreensEltype =  MonteCarlo.geltype(mc)
#     N = length(mc.model.l)
#     flv = mc.model.flv
#     blocks = mc.stack.n_elements

#     sizeof(GreensEltype) * flv*N * flv*N * blocks * 4 +         # forward+backward U, T
#     sizeof(GreensEltype) * flv*N * blocks * 2 +                 # forward+backward D
#     sizeof(GreensEltype) * flv*N * flv*N * (blocks-1) * 2 +     # inv U, T
#     sizeof(GreensEltype) * flv*N * (blocks-1) +                 # inv Dr
#     5*8 + sizeof(Bool) * (blocks-1) +                           # book-keeping/skipping
#     sizeof(GreensEltype) * flv*N * flv*N * 3 +                  # greens, U, Tr
#     sizeof(GreensEltype) * flv*N                                # D
# end

function initialize_stack(mc::DQMC, s::UnequalTimeStack)
    GreensElType  = geltype(mc)
    GreensMatType = gmattype(mc)
    N = length(lattice(mc))
    flv = unique_flavors(mc)
    # mirror stack
    M = mc.stack.n_elements

    # B_{n*safe_mult} ... B_1
    s.forward_idx = 1
    s.forward_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    s.forward_d_stack = [zeros(Float64, flv*N) for _ in 1:M]
    s.forward_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    
    # B_{n*safe_mult+1}^† ... B_M^†
    s.backward_idx = M-1
    s.backward_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    s.backward_d_stack = [zeros(Float64, flv*N) for _ in 1:M]
    s.backward_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M]
    
    # B_{n*safe_mult+1}^-1 .. B_{(n+1)*safe_mult}^-1
    s.inv_done = fill(false, M-1)
    s.inv_u_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M-1]
    s.inv_d_stack = [zeros(Float64, flv*N) for _ in 1:M-1]
    s.inv_t_stack = [GreensMatType(undef, flv*N, flv*N) for _ in 1:M-1]

    # Greens construction
    s.greens = GreensMatType(undef, flv*N, flv*N)
    s.tmp = GreensMatType(undef, flv*N, flv*N)
    s.U = GreensMatType(undef, flv*N, flv*N)
    s.D = zeros(Float64, flv*N)
    s.T = GreensMatType(undef, flv*N, flv*N)

    # TODO rework measurements to work well with StructArrays and remove this
    if (GreensMatType <: BlockDiagonal{ComplexF64, N, CMat64} where N) || (GreensMatType <: CMat64)
        s.complex_greens_temp1 = Matrix(s.greens)
        s.complex_greens_temp2 = Matrix(s.greens)
        s.complex_greens_temp3 = Matrix(s.greens)
    end

    # To avoid recalculating
    s.last_update = -1
    s.last_slices = (-1, -1)

    # maybe skip identities?
    copyto!(s.forward_u_stack[1], I)
    s.forward_d_stack[1] .= 1
    copyto!(s.forward_t_stack[1], I)

    copyto!(s.backward_u_stack[end], I)
    s.backward_d_stack[end] .= 1
    copyto!(s.backward_t_stack[end], I)

    s
end



################################################################################
### Stack building (full & lazy)
################################################################################



@bm function build_stack(mc::DQMC, s::UnequalTimeStack)
    # forward
    @bm "forward build" begin
        @inbounds for idx in 1:length(mc.stack.ranges)
            copyto!(mc.stack.curr_U, s.forward_u_stack[idx])
            for slice in mc.stack.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
            end
            vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(s.forward_d_stack[idx]))
            udt_AVX_pivot!(
                s.forward_u_stack[idx+1], s.forward_d_stack[idx+1], 
                mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
            )
            vmul!(s.forward_t_stack[idx+1], mc.stack.tmp1, s.forward_t_stack[idx])
        end
    end

    # backward
    @bm "backward build" begin
        @inbounds for idx in length(mc.stack.ranges):-1:1
            copyto!(mc.stack.curr_U, s.backward_u_stack[idx + 1])
            for slice in reverse(mc.stack.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
            end
            vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(s.backward_d_stack[idx + 1]))
            udt_AVX_pivot!(
                s.backward_u_stack[idx], s.backward_d_stack[idx], 
                mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
            )
            vmul!(s.backward_t_stack[idx], mc.stack.tmp1, s.backward_t_stack[idx+1])
        end
    end

    # inverse
    # TODO should this multiply to U's instead?
    @bm "inverse build" begin
        @inbounds for idx in 1:length(mc.stack.ranges)
            copyto!(s.inv_t_stack[idx], I)
            for slice in reverse(mc.stack.ranges[idx])
                multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[idx])
            end
            udt_AVX_pivot!(
                s.inv_u_stack[idx], s.inv_d_stack[idx], s.inv_t_stack[idx], 
                mc.stack.pivot, mc.stack.tempv
            )
        end
    end

    s.inv_done .= true
    s.forward_idx = length(mc.stack.ranges)+1
    s.backward_idx = 1
    s.last_update = mc.last_sweep
    s.last_slices = (-1, -1)

    nothing
end

@bm function lazy_build_stack(mc::DQMC, s::UnequalTimeStack)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if mc.last_sweep == s.last_update && all(s.inv_done) && 
        (s.forward_idx == length(mc.stack.ranges)+1) && (s.backward_idx == 1)
        return nothing
    end

    build_stack(mc, s)
    
    nothing
end

@bm function lazy_build_forward!(mc::DQMC, s::UnequalTimeStack, upto)
    # stack = [0, Δτ, 2Δτ, ..., β]
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.stack.ranges)+1
    end

    # @assert 1 <= s.forward_idx
    # @assert upto-1 <= length(mc.stack.ranges)

    # forward
    @bm "forward build" begin
        @inbounds for idx in s.forward_idx:upto-1
            copyto!(mc.stack.curr_U, s.forward_u_stack[idx])
            for slice in mc.stack.ranges[idx]
                multiply_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
            end
            vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(s.forward_d_stack[idx]))
            udt_AVX_pivot!(
                s.forward_u_stack[idx+1], s.forward_d_stack[idx+1], 
                mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
            )
            vmul!(s.forward_t_stack[idx+1], mc.stack.tmp1, s.forward_t_stack[idx])
        end
    end

    s.forward_idx = max(upto, s.forward_idx)
    nothing
end

@bm function lazy_build_backward!(mc::DQMC, s::UnequalTimeStack, downto)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.stack.ranges)+1
    end

    # @assert s.backward_idx-1 <= length(mc.stack.ranges)
    # @assert 1 <= downto

    # backward
    @bm "backward build" begin
        @inbounds for idx in s.backward_idx-1:-1:downto
            copyto!(mc.stack.curr_U, s.backward_u_stack[idx + 1])
            for slice in reverse(mc.stack.ranges[idx])
                multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
            end
            vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(s.backward_d_stack[idx + 1]))
            udt_AVX_pivot!(
                s.backward_u_stack[idx], s.backward_d_stack[idx], 
                mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv
            )
            vmul!(s.backward_t_stack[idx], mc.stack.tmp1, s.backward_t_stack[idx+1])
        end
    end

    s.backward_idx = min(downto, s.backward_idx)
end

@bm function lazy_build_inv!(mc::DQMC, s::UnequalTimeStack, from, to)
    if s.last_update != mc.last_sweep
        s.last_update = mc.last_sweep
        s.inv_done .= false
        s.forward_idx = 1
        s.backward_idx = length(mc.stack.ranges)+1
    end

    # @assert 1 <= from
    # @assert to <= length(mc.stack.ranges)

    # inverse
    @bm "inverse build" begin
        @inbounds for idx in from:to
            s.inv_done[idx] && continue
            s.inv_done[idx] = true
            copyto!(s.inv_t_stack[idx], I)
            for slice in reverse(mc.stack.ranges[idx])
                multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_t_stack[idx])
            end
            udt_AVX_pivot!(
                s.inv_u_stack[idx], s.inv_d_stack[idx], s.inv_t_stack[idx], 
                mc.stack.pivot, mc.stack.tempv
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
    greens!(mc::DQMC, k, l[; output=mc.stack.greens_temp, temp=mc.stack.tmp1])
"""
@bm function greens!(
        mc::DQMC, slice1::Int64, slice2::Int64; 
        output = mc.stack.greens_temp, temp = mc.stack.tmp1
    ) 
    _greens!(mc, slice1, slice2, output, temp)
end
"""
    _greens!(mc::DQMC, k, l, output=mc.stack.greens_temp, temp=mc.stack.tmp1)
"""
function _greens!(
        mc::DQMC, slice1::Int64, slice2::Int64,
        output = mc.stack.greens_temp, temp = mc.stack.tmp1
    )
    calculate_greens(mc, slice1, slice2)
    G = _greens!(mc, output, mc.ut_stack.greens, temp)
    return GreensMatrix(slice1, slice2, G)
end
@bm function calculate_greens(mc::DQMC, slice1::Int64, slice2::Int64)
    @assert 0 ≤ slice1 ≤ mc.parameters.slices
    @assert 0 ≤ slice2 ≤ mc.parameters.slices
    s = mc.ut_stack
    if (s.last_slices != (slice1, slice2)) || (s.last_update != mc.last_sweep)
        s.last_slices = (slice1, slice2)
        if slice1 ≥ slice2
            calculate_greens_full1!(mc, s, slice1, slice2)
        else
            calculate_greens_full2!(mc, s, slice1, slice2)
        end
    end
    s.greens
end





# Notes on ranges:
# Let's say we are looking for the first full block in a product chain 
# i i+1 ... j-1 j. We can find this with _find_range_with_value(mc, i - 1) + 1
# - if i is the first element of a block ... i-1 | i ... then 
#   _find_range_with_value(mc, i-1) is the block just before that
# - if i is not the first element, then i-1 is in the same block. Thus 
#   _find_range_with_value(mc, i-1) == _find_range_with_value(mc, i) and the 
#   first full block is _find_range_with_value(mc, i-1) + 1
# Similarly for last full block we can check _find_range_with_value(mc, i+1) - 1
function _find_range_with_value(mc, val)
    # This returns the index of the range in mc.stack.ranges which contains val.
    # If val is below or above all of the ranges it returns 0 or length(ranges)+1
    ranges = mc.stack.ranges
    if val < 1
        return 0
    elseif val > last(last(ranges))
        return length(ranges) + 1
    else
        # this might actually be exact?
        estimate = clamp(
            trunc(Int, length(mc.stack.ranges) * (val-1) / (last(last(ranges)) - 1) + 1), 
            1, length(ranges)
        )
        low = first(ranges[estimate])
        high = last(ranges[estimate])
        while 1 <= estimate <= length(ranges)
            if low <= val <= high
                return estimate
            elseif val < low
                estimate -= 1
                high = low - 1
                low = first(ranges[estimate])
            else
                estimate += 1
                low = high + 1
                high = last(ranges[estimate])
            end
        end
        error("Failed to find slice index in ranges. Oh no.")
    end
end

# low = slice2, high = slice1
"""
    compute_inverse_udt_block!(mc, s::UnequalTimeStack, low, high[, ...])

Computes a inverse UDT block `U D T = B_{low+1}^-1 B_{low+2}^-1 ⋯ B_high^-1`.

Default extra args are: (in order)
* `U = s.U`
* `D = s.D`
* `T = s.T`
* `tmp1 = mc.stack.tmp1`
* `tmp2 = mc.stack.tmp2`
"""
function compute_inverse_udt_block!(
        mc, s, low, high,
        U = s.U, D = s.D, T = s.T, tmp1 = mc.stack.tmp1, tmp2 = mc.stack.tmp2
    )
    # @assert low ≤ high
    # B_{low+1}^-1 B_{low+2}^-1 ⋯ B_{high-1}^-1 B_high^-1

    # Combine pre-computed blocks
    @bm "inverse pre-computed" begin
        # lower = div(low+1 + mc.parameters.safe_mult - 2, mc.parameters.safe_mult) + 1
        lower = _find_range_with_value(mc, low) + 1 
        # upper = div(high, mc.parameters.safe_mult)
        upper = _find_range_with_value(mc, high+1) - 1

        lazy_build_inv!(mc, s, lower, upper)
        copyto!(U, I)
        D .= 1
        copyto!(T, I)

        # UDT combining style
        for idx in lower:upper
            # () is operation order, [] marks UDT decomposition
            # U [(D (T stack_U)) stack_D] stack_T
            # (U [U') D (T'] stack_T)
            # U D T
            vmul!(tmp1, T, s.inv_u_stack[idx])
            vmul!(tmp2, Diagonal(D), tmp1)
            vmul!(tmp1, tmp2, Diagonal(s.inv_d_stack[idx]))
            udt_AVX_pivot!(tmp2, D, tmp1, mc.stack.pivot, mc.stack.tempv)
            vmul!(T, tmp1, s.inv_t_stack[idx])
            vmul!(tmp1, U, tmp2)
            copyto!(U, tmp1)
        end
    end

    # remaining multiplications to reach specified bounds
    @bm "inverse fine tuning" begin
        # lower_slice = (lower-1) * mc.parameters.safe_mult + 1
        lower_slice = if lower <= length(mc.stack.ranges)
            first(mc.stack.ranges[lower]) else last(last(mc.stack.ranges)) + 1
        end
        # upper_slice = upper * mc.parameters.safe_mult
        upper_slice = upper > 0 ? last(mc.stack.ranges[upper]) : 0


        for slice in min(lower_slice-1, high) : -1 : low+1
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, U)
        end
        if min(lower_slice-1, high) ≥ low+1
            vmul!(tmp1, U, Diagonal(D))
            udt_AVX_pivot!(U, D, tmp1, mc.stack.pivot, mc.stack.tempv)
            vmul!(tmp2, tmp1, T)
            copyto!(T, tmp2)
        end
        for slice in max(upper_slice+1, min(lower_slice-1, high)+1) : high
            multiply_slice_matrix_inv_right!(mc, mc.model, slice, T)
        end
        # U D T = B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    end
    return nothing
end


"""
    compute_forward_udt_block!(mc, s::UnequalTimeStack, slice[, ...])

Computes a UDT block `U D T = B_slice B_{slice-1} ⋯ B_1`.

Default extra args are: (in order)
* `U = mc.stack.Ul`
* `D = mc.stack.Dl`
* `T = mc.stack.Tl`
* `tmp = mc.stack.tmp1`
"""
function compute_forward_udt_block!(
        mc, s, slice, apply_pivot = Val(true),
        U = mc.stack.Ul, D = mc.stack.Dl, T = mc.stack.Tl, tmp = mc.stack.tmp1,
    )
    # B(slice, 1) = B_slice B_{slice-1} ⋯ B_1 = Ul Dl Tl
    @bm "forward B" begin
        # idx = div(slice-1, mc.parameters.safe_mult)
        idx = max(0, _find_range_with_value(mc, slice) - 1)
        
        lazy_build_forward!(mc, s, idx+1)
        copyto!(T, s.forward_u_stack[idx+1])
        target = idx > 0 ? last(mc.stack.ranges[idx]) + 1 : 1

        for l in target : slice
            multiply_slice_matrix_left!(mc, mc.model, l, T)
        end
        vmul!(tmp, T, Diagonal(s.forward_d_stack[idx+1]))
        udt_AVX_pivot!(U, D, tmp, mc.stack.pivot, mc.stack.tempv, apply_pivot)
        vmul!(T, tmp, s.forward_t_stack[idx+1])
    end
    return nothing
end


"""
    compute_backward_udt_block!(mc, s::UnequalTimeStack, slice[, ...])

Computes a UDT block `(U D T)^† = T^† D U^† = B_N B_{N-1} ⋯ B_slice`.

Default extra args are: (in order)
* `U = mc.stack.Ur`
* `D = mc.stack.Dr`
* `T = mc.stack.Tr`
* `tmp = mc.stack.tmp1`
"""
function compute_backward_udt_block!(
        mc, s, slice,
        U = mc.stack.Ur, D = mc.stack.Dr, T = mc.stack.Tr, tmp = mc.stack.tmp1
    )
    # B(N, slice) = B_N B_{N-1} ⋯ B_{slice+1} = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    @bm "backward B" begin
        # idx = div.(slice + mc.parameters.safe_mult - 1, mc.parameters.safe_mult)
        idx = _find_range_with_value(mc, slice) + 1

        lazy_build_backward!(mc, s, idx)
        copyto!(U, s.backward_u_stack[idx])
        target = if idx <= length(mc.stack.ranges)
            first(mc.stack.ranges[idx]) - 1 else last(last(mc.stack.ranges))
        end
        
        for l in target : -1 : slice+1
            multiply_daggered_slice_matrix_left!(mc, mc.model, l, U)
        end
        vmul!(tmp, U, Diagonal(s.backward_d_stack[idx]))
        udt_AVX_pivot!(U, D, tmp, mc.stack.pivot, mc.stack.tempv)
        vmul!(T, tmp, s.backward_t_stack[idx])
    end
    return nothing
end



@bm function calculate_greens_full1!(mc, s, slice1, slice2)
    # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
    # @assert slice1 ≥ slice2

    # k ≥ l or slice1 ≥ slice2
    # B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    compute_inverse_udt_block!(mc, s, slice2, slice1) # low high
    
    # B(slice2, 1) = Ul Dl Tl
    compute_forward_udt_block!(mc, s, slice2)
    
    # B(N, slice1) = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    compute_backward_udt_block!(mc, s, slice1)
    
    
    
    # U D T remains valid here
    @bm "compute G" begin
        # [B_{l+1}^-1 B_{l+2}^-1 ⋯ B_k^-1 + B_l ⋯ B_1 B_N ⋯ B_{k+1}]^-1
        # [U D T + Ul (Dl Tl Tr^† Dr) Ur^†]^-1
        # @bm "B1" begin
            vmul!(s.greens, mc.stack.Tl, adjoint(mc.stack.Tr))
            vmul!(mc.stack.tmp1, s.greens, Diagonal(mc.stack.Dr))
            vmul!(s.greens, Diagonal(mc.stack.Dl), mc.stack.tmp1)
        # end
        # [U D T + Ul (G) Ur^†]^-1
        # @bm "udt" begin
            udt_AVX_pivot!(mc.stack.Tr, mc.stack.Dr, s.greens, mc.stack.pivot, mc.stack.tempv, Val(false))
        # end
        # [U D T + (Ul Tr) Dr (G Ur^†)]^-1
        # @bm "B2" begin
            vmul!(mc.stack.Tl, mc.stack.Ul, mc.stack.Tr)
            # (G Ur^†) = (Ur / G)^-1
            # Ur := Ur / G
            rdivp!(mc.stack.Ur, s.greens, mc.stack.Ul, mc.stack.pivot) 
        # end
        # [U D T + Tl Dr Ur^-1]^-1
        # [U (D T Ur + U^† Tl Dr) Ur^-1]^-1
        # [U D_max (D_min T Ur 1/Dr_max + 1/D_max U^† Tl Dr_min) Dr_max Ur^-1]^-1
        # @bm "B3" begin
            # 1/D_max U^† Tl Dr_min
            vmul!(mc.stack.Tr, adjoint(s.U), mc.stack.Tl)
            vmaxinv!(mc.stack.Dl, s.D) # mc.stack.Dl .= 1.0 ./ max.(1.0, s.D)
            vmul!(mc.stack.tmp1, Diagonal(mc.stack.Dl), mc.stack.Tr)
            vmin!(mc.stack.Dl, mc.stack.Dr) # mc.stack.Dl .= min.(1.0, mc.stack.Dr)
            vmul!(mc.stack.Tr, mc.stack.tmp1, Diagonal(mc.stack.Dl))
        # end
        # [U D_max (D_min T Ur 1/Dr_max + Tr) Dr_max Ur^-1]^-1
        # @bm "B4" begin
            # D_min T Ur 1/Dr_max
            vmul!(mc.stack.Tl, s.T, mc.stack.Ur)
            vmin!(mc.stack.Dl, s.D) # mc.stack.Dl .= min.(1.0, s.D)
            vmul!(mc.stack.tmp1, Diagonal(mc.stack.Dl), mc.stack.Tl)
            vmaxinv!(mc.stack.Dl, mc.stack.Dr) # mc.stack.Dl .= 1.0 ./ max.(1.0, mc.stack.Dr)
            vmul!(mc.stack.Tl, mc.stack.tmp1, Diagonal(mc.stack.Dl))
        # end
        # [U D_max (Tl + Tr) Dr_max Ur^-1]^-1
        # @bm "sum, UDT" begin
            rvadd!(mc.stack.Tl, mc.stack.Tr)
            udt_AVX_pivot!(mc.stack.Tr, mc.stack.Dl, mc.stack.Tl, mc.stack.pivot, mc.stack.tempv, Val(false))
        # end
        # [U D_max (Tr Dl Tl) Dr_max Ur^-1]^-1
        # Ur 1/Dr_max Tl^-1 1/Dl Tr^† D_max U^†
        # @bm "B5" begin
            # [[((1/Dr_max) / Tl) 1/Dl] Tr^†] D_max
            vmaxinv!(mc.stack.Dr, mc.stack.Dr) # mc.stack.Dr .= 1.0 ./ max.(1.0, mc.stack.Dr)
            copyto!(mc.stack.Ul, Diagonal(mc.stack.Dr))
            rdivp!(mc.stack.Ul, mc.stack.Tl, mc.stack.tmp1, mc.stack.pivot)
            vinv!(mc.stack.Dl) # mc.stack.Dl[i] .= 1.0 ./ mc.stack.Dl[i]
            vmul!(mc.stack.tmp1, mc.stack.Ul, Diagonal(mc.stack.Dl))
            vmul!(mc.stack.Ul, mc.stack.tmp1, adjoint(mc.stack.Tr))
            vmaxinv!(mc.stack.Dl, s.D) # mc.stack.Dl .= 1.0 ./ max.(1.0, s.D)
            vmul!(s.greens, mc.stack.Ul, Diagonal(mc.stack.Dl))
        # end
        # Ur G U^†
        # @bm "B6" begin
            vmul!(mc.stack.Tr, s.greens, adjoint(s.U))
            vmul!(s.greens, mc.stack.Ur, mc.stack.Tr)
        # end
    end

    s.greens
end


@bm function calculate_greens_full2!(mc, s, slice1, slice2)
    # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
    # @assert slice1 ≤ slice2

    # k ≤ l or slice1 ≤ slice2
    # B_{k+1}^-1 B_{k+2}^-1 ⋯ B_{l-1}^-1 B_l^-1
    compute_inverse_udt_block!(mc, s, slice1, slice2) # low high

    # B(slice1, 1) = B_k ⋯ B_1 = Ul Dl Tl
    compute_forward_udt_block!(mc, s, slice1)
    
    # B(M, slice2) = B_M ⋯ B_slice2+1 = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
    compute_backward_udt_block!(mc, s, slice2)


    # U D T remains valid here
    @bm "compute G" begin
        # [B_{l} B_{l-1} ⋯ B_{k+1} + (B_k ⋯ B_1 B_M ⋯ B_{l+1})^-1]^-1
        # [T^-1 D^-1 U^† + (Ul Dl Tl Tr^† Dr Ur^†)^-1]^-1
        # [T^-1 D^-1 U^† + Ur (Dl Tl Tr^† Dr)^-1 Ul^†]^-1
        # @bm "B1" begin
            vmul!(s.greens, mc.stack.Tl, adjoint(mc.stack.Tr))
            vmul!(mc.stack.tmp1, Diagonal(mc.stack.Dl), s.greens)
            vmul!(s.greens, mc.stack.tmp1, Diagonal(mc.stack.Dr))
        # end
        # [T^-1 D^-1 U^† + Ur G^-1 Ul^†]^-1
        # @bm "udt" begin
            udt_AVX_pivot!(mc.stack.Tr, mc.stack.Dr, s.greens, mc.stack.pivot, mc.stack.tempv, Val(false))
        # end
        # [T^-1 D^-1 U^† + Ur G^-1 Dr^-1 Tr^† Ul^†]^-1
        # [T^-1 D_min^-1 (D_max^-1 U^† (Ul Tr) Dr_min + D_min T Ur G^-1 Dr_max^-1) Dr_min^-1 (Ul Tr)^†]^-1
        # @bm "B2" begin
            # D_max^-1 U^† (Ul Tr) Dr_min
            vmul!(mc.stack.Tl, mc.stack.Ul, mc.stack.Tr) # keep me alive
            vmul!(mc.stack.Ul, adjoint(s.U), mc.stack.Tl)
            vmaxinv!(mc.stack.Dl, s.D)
            vmul!(s.U, Diagonal(mc.stack.Dl), mc.stack.Ul)
            vmin!(mc.stack.Dl, mc.stack.Dr)
            vmul!(mc.stack.Ul, s.U, Diagonal(mc.stack.Dl))
        # end
        # [T^-1 D_min^-1 (Ul + D_min T Ur G^-1 Dr_max^-1) Dr_min^-1 Tl^†]^-1
        # @bm "B3" begin
            # D_min T Ur G^-1 Dr_max^-1
            vmul!(s.U, s.T, mc.stack.Ur)
            rdivp!(s.U, s.greens, mc.stack.Ur, mc.stack.pivot)
            vmin!(mc.stack.Dl, s.D)
            vmul!(mc.stack.Ur, Diagonal(mc.stack.Dl), s.U)
            vmaxinv!(mc.stack.Dl, mc.stack.Dr)
            vmul!(mc.stack.Tr, mc.stack.Ur, Diagonal(mc.stack.Dl))
        # end
        # [T^-1 D_min^-1 (Ul + Tr) Dr_min^-1 Tl^†]^-1
        # @bm "sum, udt" begin
            rvadd!(mc.stack.Tr, mc.stack.Ul)
            udt_AVX_pivot!(mc.stack.Ul, mc.stack.Dl, mc.stack.Tr, mc.stack.pivot, mc.stack.tempv, Val(false))
        # end
        # [T^-1 D_min^-1 Ul Dl Tr Dr_min^-1 Tl^-1]^-1
        # Tl ({[(Dr_min / Tr) Dl^-1] Ul^†} D_min) T
        # @bm "B4" begin
            vmin!(mc.stack.Dr, mc.stack.Dr)
            copyto!(s.U, Diagonal(mc.stack.Dr))
            rdivp!(s.U, mc.stack.Tr, mc.stack.Ur, mc.stack.pivot)
            vinv!(mc.stack.Dl)
            vmul!(mc.stack.Ur, s.U, Diagonal(mc.stack.Dl))
            vmul!(s.U, mc.stack.Ur, adjoint(mc.stack.Ul))
            vmin!(s.D, s.D)
            vmul!(mc.stack.Ur, s.U, Diagonal(s.D))
        # end
        # Tl Ur T
        # @bm "B6" begin
            vmul!(mc.stack.Tr, mc.stack.Ur, s.T)
            vmul!(s.greens, mc.stack.Tl, mc.stack.Tr)
            rmul!(s.greens, -1.0)
        # end
    end
    
    s.greens
end
