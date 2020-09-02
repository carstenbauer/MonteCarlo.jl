mutable struct UnequalTimeStack{GT} <: AbstractDQMCStack
    forward_u_stack::Array{GT, 3}
    forward_d_stack::Matrix{Float64}
    forward_t_stack::Array{GT, 3}
    
    backward_u_stack::Array{GT, 3}
    backward_d_stack::Matrix{Float64}
    backward_t_stack::Array{GT, 3}
    
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
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),
        zeros(Float64, flv*N, mc.s.n_elements),
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),

        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),
        zeros(Float64, flv*N, mc.s.n_elements),
        zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements),

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

function build_stack(mc::DQMC, s::UnequalTimeStack)
    mc.last_sweep == s.last_update && return nothing

    # stack = [0, Δτ, 2Δτ, ..., β]

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
            @views copyto!(s.inv_u_stack[:, :, idx], I)
            for slice in reverse(mc.s.ranges[idx])
                @views multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.inv_u_stack[:, :, idx])
            end
            @views udt_AVX_pivot!(
                s.inv_u_stack[:, :, idx], s.inv_d_stack[:, idx], s.inv_u_stack[:, :, idx], 
                mc.s.pivot, mc.s.tempv
            )
        end
    end

    s.last_update = mc.last_sweep

    nothing
end

greens(mc::DQMC, slice1::Int64, slice2::Int64) = greens!(mc, slice1, slice2)
@bm function greens!(mc::DQMC, slice1::Int64, slice2::Int64)
    @assert slice1 >= slice2
    s = mc.ut_stack
    # stack = [0, Δτ, 2Δτ, ..., β]
    #       = [0, safe_mult, 2safe_mult, ... N]
    @bm "build" begin 
        build_stack(mc, mc.ut_stack) 
    end

    # k ≥ l or slice1 ≥ slice2
    # B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
    @bm "inverse pre-computed" begin
        lower = div(slice2+1 + mc.p.safe_mult - 2, mc.p.safe_mult) + 1
        upper = div(slice1, mc.p.safe_mult)
        # @info "UDTs from $lower to $upper"
        copyto!(s.U, I)
        s.D .= 1
        copyto!(s.T, I)
        for idx in lower:upper
            # () is operation order, [] marks UDT decomposition
            # U [(D (T stack_T)) stack_D] stack_T
            # (U [U') D (T'] stack_T)
            # U D T
            @views vmul!(mc.s.curr_U, s.T, s.inv_u_stack[:, :, idx])
            lvmul!(Diagonal(s.D), mc.s.curr_U)
            @views rvmul!(mc.s.curr_U, Diagonal(s.inv_d_stack[:, idx]))
            udt_AVX_pivot!(mc.s.tmp1, s.D, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
            @views vmul!(s.T, mc.s.curr_U, s.inv_t_stack[:, :, idx])
            vmul!(mc.s.curr_U, s.U, mc.s.tmp1)
            copyto!(s.U, mc.s.curr_U)
        end
    end
  
    @bm "inverse fine tuning" begin
        lower_slice = (lower-1) * mc.p.safe_mult + 1
        upper_slice = upper * mc.p.safe_mult
        # @info "UDT borders: $lower_slice to $upper_slice"
        # @info "left pad: $(slice2+1) to $(min(lower_slice-1, slice1))"
        # @info "right pad: $(max(upper_slice+1, min(lower_slice-1, slice1)+1)) to $slice1"

        for slice in slice2+1 : min(lower_slice-1, slice1)
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, s.U)
        end
        for slice in max(upper_slice+1, min(lower_slice-1, slice1)+1) : slice1
            multiply_slice_matrix_inv_right!(mc, mc.model, slice, s.T)
        end
        # U D T = B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
        # @info s.U * Diagonal(s.D) * s.T ≈ I
    end

    @bm "forward B" begin
        # B(slice1, 1) = Ul Dl Tl
        idx = div(slice1-1, mc.p.safe_mult) # 0 based index into stack
        copyto!(mc.s.curr_U, s.forward_u_stack[:, :, idx+1])
        # @info "$slice1 || $(idx+1) || $(mc.p.safe_mult * idx + 1) : $(slice1-1)"
        for slice in mc.p.safe_mult * idx + 1 : slice1-1
            multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
        end
        @views rvmul!(mc.s.curr_U, Diagonal(s.forward_d_stack[:, idx+1]))
        @views udt_AVX_pivot!(mc.s.Ul, mc.s.Dl, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
        @views vmul!(mc.s.Tl, mc.s.curr_U, s.forward_t_stack[:, :, idx+1])
    end

    @bm "backward B" begin
        # B(N, slice2) = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
        idx = div.(slice2 + mc.p.safe_mult -2, mc.p.safe_mult) # 0 based index into stack
        copyto!(mc.s.curr_U, s.backward_u_stack[:, :, idx+1])
        # @info "$slice2 || $(idx+1) || $(mc.p.safe_mult * idx) : -1 : $(slice2-1)"
        for slice in mc.p.safe_mult * idx : -1 : slice2
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
        end
        @views rvmul!(mc.s.curr_U, Diagonal(s.backward_d_stack[:, idx+1]))
        @views udt_AVX_pivot!(mc.s.Ur, mc.s.Dr, mc.s.curr_U, mc.s.pivot, mc.s.tempv)
        @views vmul!(mc.s.Tr, mc.s.curr_U, s.backward_t_stack[:, :, idx + 1])
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



################################################################################
### Measurements
################################################################################


abstract type UnequalTimeMeasurement <: AbstractMeasurement end

struct UTGreensMeasurement{OT} <: UnequalTimeMeasurement
    obs::OT
    l1::Int64
    l2::Int64
end

"""
    UTGreensMeasurement(mc, model[; capacity, τ1, τ2, slice1 = τ1/Δτ, slice2 = τ2/Δτ])

G(τ2 ← τ1)
"""
function UTGreensMeasurement(
        mc::DQMC, model; 
        capacity=_default_capacity(mc),
        τ1 = 0.5mc.p.beta, slice1 = τ1 / mc.p.delta_tau,
        τ2 = 0.0 slice2 = τ2 / mc.p.delta_tau
    )
    l1 = round(Int64, slice1)
    l2 = round(Int64, slice2)
    abs(l1 - slice1) > 1e-6 && throw(ArgumentError(
        "slice1 = τ1 / mc.p.delta_tau should be an integer, but is $slice1."
    ))
    abs(l2 - slice2) > 1e-6 && throw(ArgumentError(
        "slice2 = τ2 / mc.p.delta_tau should be an integer, but is $slice2."
    ))
    

    T = greenseltype(DQMC, model)
    N = model.flv * length(lattice(model))
    o = LightObservable(
        LogBinner(zeros(T, (N, N)), capacity=capacity),
        "Unequal-times Green's function",
        "Observables.jld",
        "G"
    )
    UTGreensMeasurement{typeof(o)}(o, l1, l2)
end
@bm function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens!(mc, m.l2, m.l1))
end


# TODO
# - figure out how to intialize UnequalTimeStack nicely
#   - needs to work from `resume!(filename)`, from `DQMC()` and `push!(dqmc, m)`
# - test DQMC stack
# - add unequal time Measurements
# - test those too
