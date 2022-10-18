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
