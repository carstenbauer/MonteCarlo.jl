include("linalg/old_linalg.jl")
# Just in case
using MonteCarlo, LinearAlgebra, SparseArrays
using StableDQMC

# Calculate Ul, Dl, Tl =B(stop) ... B(start)
"""
Calculate effective(!) Green's function (direct, i.e. without stack) using QR DECOMPOSITION
"""
function calculate_slice_matrix_chain(mc::DQMC, start::Int, stop::Int, safe_mult::Int=mc.parameters.safe_mult)
    @assert 0 < start <= mc.parameters.slices
    @assert 0 < stop <= mc.parameters.slices
    @assert start <= stop

    flv = MonteCarlo.unique_flavors(MonteCarlo.field(mc))
    N = length(lattice(mc.model))
    GreensType = MonteCarlo.geltype(mc)

    U = Matrix{GreensType}(I, flv*N, flv*N)
    D = ones(Float64, flv*N)
    T = Matrix{GreensType}(I, flv*N, flv*N)
    Tnew = Matrix{GreensType}(I, flv*N, flv*N)

    svs = zeros(flv*N,length(start:stop))
    svc = 1
    for k in start:stop
        if mod(k,safe_mult) == 0
            MonteCarlo.multiply_slice_matrix_left!(mc, mc.model, k, U)
            U *= spdiagm(0 => D)
            U, D, Tnew = decompose_udt(U)
            T = Tnew * T
            svs[:,svc] = log.(D)
            svc += 1
        else
            MonteCarlo.multiply_slice_matrix_left!(mc, mc.model, k, U)
        end
    end
    U *= spdiagm(0 => D)
    U, D, Tnew = decompose_udt(U)
    T = Tnew * T
    return (U,D,T,svs)
end

# Calculate (Ur, Dr, Tr)' = B(stop) ... B(start) => Ur,Dr, Tr = B(start)' ... B(stop)'
function calculate_slice_matrix_chain_dagger(mc::DQMC, start::Int, stop::Int, safe_mult::Int=mc.parameters.safe_mult)
    @assert 0 < start <= mc.parameters.slices
    @assert 0 < stop <= mc.parameters.slices
    @assert start <= stop

    flv = MonteCarlo.unique_flavors(MonteCarlo.field(mc))
    N = length(lattice(mc.model))
    GreensType = MonteCarlo.geltype(mc)

    U = Matrix{GreensType}(I, flv*N, flv*N)
    D = ones(Float64, flv*N)
    T = Matrix{GreensType}(I, flv*N, flv*N)
    Tnew = Matrix{GreensType}(I, flv*N, flv*N)

    svs = zeros(flv*N,length(start:stop))
    svc = 1
    for k in reverse(start:stop)
        if mod(k,safe_mult) == 0
            MonteCarlo.multiply_daggered_slice_matrix_left!(mc, mc.model, k, U)
            U *= spdiagm(0 => D)
            U, D, Tnew = decompose_udt(U)
            T = Tnew * T
            svs[:,svc] = log.(D)
            svc += 1
        else
            MonteCarlo.multiply_daggered_slice_matrix_left!(mc, mc.model, k, U)
        end
    end
    U *= spdiagm(0 => D)
    U, D, Tnew = decompose_udt(U)
    T = Tnew * T
    return (U,D,T,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its singular values in a stable manner
function calculate_greens_and_logdet(mc::DQMC, slice::Int, safe_mult::Int=mc.parameters.safe_mult)
    GreensType = MonteCarlo.geltype(mc)
    flv = MonteCarlo.unique_flavors(MonteCarlo.field(mc))
    N = length(lattice(mc.model))

    # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
    if slice+1 <= mc.parameters.slices
        Ur, Dr, Tr = calculate_slice_matrix_chain_dagger(mc,slice+1,mc.parameters.slices, safe_mult)
    else
        Ur = Matrix{GreensType}(I, flv * N, flv * N)
        Dr = ones(Float64, flv * N)
        Tr = Matrix{GreensType}(I, flv * N, flv * N)
    end

    # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
    if slice >= 1
        Ul, Dl, Tl = calculate_slice_matrix_chain(mc,1,slice, safe_mult)
    else
        Ul = Matrix{GreensType}(I, flv * N, flv * N)
        Dl = ones(Float64, flv * N)
        Tl = Matrix{GreensType}(I, flv * N, flv * N)
    end

    tmp = Tl * adjoint(Tr)
    U, D, T = decompose_udt(Diagonal(Dl) * tmp * Diagonal(Dr))
    U = Ul * U
    T *= adjoint(Ur)

    u, d, t = decompose_udt(adjoint(U) * inv(T) + Diagonal(D))

    T = inv(t * T)
    U *= u
    U = adjoint(U)
    d = 1. ./ d

    ldet = real(log(complex(det(U))) + sum(log.(d)) + log(complex(det(T))))

    return T * Diagonal(d) * U, ldet
end

struct GreensCalculator{T}
    U::Matrix{T}
    Uinv::Matrix{T}
    vals::Vector{T}
    beta::Float64
    delta_tau::Float64
end

function analytic_greens(mc::DQMC)
    @assert mc.model.U == 0
    T = Matrix(mc.stack.hopping_matrix)
    vals, U = eigen(T)
    return GreensCalculator(U, inv(U), vals, mc.parameters.beta, mc.parameters.delta_tau)
end

(gc::GreensCalculator)() = gc(0.0, 0.0)
(gc::GreensCalculator)(tau) = gc(tau, tau)
(gc::GreensCalculator)(k::Integer, l::Integer) = gc(_tau(gc, k), _tau(gc, l))
_tau(gc::GreensCalculator, k::Integer) = gc.delta_tau * k
_tau(gc::GreensCalculator, tau::Float64) = tau

function (gc::GreensCalculator)(tau1::Float64, tau2::Float64)
    # Following dos Santos, not Quantum Monte Carlo Methods.
    # G = [I + e^{-βT}]⁻¹
    # G = [I + e^{-βUDU'}]⁻¹ = [I + U e^{-βD} U']⁻¹ = U [I + e^{-βD}]⁻¹ U'
    # Losing precision of the I or small e^{-βD} entries should be irrelevant
    # since the larger value will dominate the result after inversion.
    # Regardless for higher precision one could use
    # 1 / (1 + x) = 1 - x + x² - x³ + x⁴              for x << 1
    # 1 / (1 + x) = 1/x - (1/x)² + (1/x)³ - (1/x)⁴    for x >> 1
    # 1 / (1 + x) (as is)                             for x ~ 1
    # x = e^(log(x)) may also be useful.
    U = gc.U
    Uinv = gc.Uinv
    vals = gc.vals
    beta = gc.beta

    if !((0 <= tau1 <= beta) && (0 <= tau2 <= beta))
        error("Bad interval. 0 ≤ $tau1, $tau2 ≤ $beta not given.")
    end

    # equal time
    if tau1 == tau2
        D = Diagonal(1 ./ (1 .+ exp.(-beta * vals)))
        return U * D * Uinv # technically U doesn't need to be unitary
    elseif tau1 > tau2
        # unequal time tau1 > tau2
        # G(τ1, τ2) = [e^{(τ1 - τ2)T} + e^{-τ2 T} e^{-(β - τ1) T}]⁻¹
        vals_inv  = exp.((tau1 - tau2) * vals)
        vals_low  = exp.(-tau2 * vals)
        vals_high = exp.(-(beta - tau1) * vals)
        return U * Diagonal(1 ./ (vals_inv .+ vals_low .* vals_high)) * Uinv
    else
        error("TODO")
        # Same as above oops.
        # # G(τ1, τ2) = e^{(τ1 - τ2)T} - [e^{(τ1 - τ2)T} + e^{-τ2 T} e^{-(β - τ1) T}]⁻¹
        # vals_inv  = exp.((tau1 - tau2) * vals)
        # vals_diff = exp.(-(tau1 - tau2) * vals)
        # vals_low  = exp.(-tau1 * vals)
        # vals_high = exp.(-(beta - tau2) * vals)
        # return U * Diagonal(vals_inv - 1 ./ (vals_diff .+ vals_low .* vals_high)) * Uinv
    end
end