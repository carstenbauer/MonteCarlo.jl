# these are just shorthads
using MonteCarlo: FVec64, FMat64, CVec64, CMat64

@testset "Field Cache & Interaction Matrix" begin
    T2 = NTuple{2}
    
    mc = DQMC(HubbardModelAttractive(8, 1), field = DensityHirschField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{Float64, Float64, FVec64, Float64}
    @test MonteCarlo.interaction_eltype(mc.field) == Float64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{Float64, FVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)

    mc = DQMC(HubbardModelRepulsive(8, 1), field = DensityHirschField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{ComplexF64, ComplexF64, CVec64, ComplexF64}
    @test MonteCarlo.interaction_eltype(mc.field) == ComplexF64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{ComplexF64, CVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)
    
    mc = DQMC(HubbardModelAttractive(8, 1), field = MagneticHirschField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{CVec64, CVec64, T2{CVec64}, ComplexF64}
    @test MonteCarlo.interaction_eltype(mc.field) == ComplexF64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{ComplexF64, CVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)

    mc = DQMC(HubbardModelRepulsive(8, 1), field = MagneticHirschField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{FVec64, FVec64, T2{FVec64}, Float64}
    @test MonteCarlo.interaction_eltype(mc.field) == Float64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{Float64, FVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)
    
    mc = DQMC(HubbardModelAttractive(8, 1), field = MagneticGHQField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{CVec64, CVec64, T2{CVec64}, ComplexF64}
    @test MonteCarlo.interaction_eltype(mc.field) == ComplexF64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{ComplexF64, CVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)

    mc = DQMC(HubbardModelRepulsive(8, 1), field = MagneticGHQField, beta = 1.0)
    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{FVec64, FVec64, T2{FVec64}, Float64}
    @test MonteCarlo.interaction_eltype(mc.field) == Float64
    @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{Float64, FVec64}
    @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa typeof(mc.stack.eV)
end

@testset "Lookup tables" begin
    param = MonteCarlo.DQMCParameters(beta = 1.0)
    model = HubbardModelRepulsive(2, 2)
    field = MagneticGHQField(param, model)

    @test field.α == sqrt(-0.5 * 0.1 * model.U)
    
    # See ALF Documentation for the formulas of η and γ
    @test field.γ[1] ≈ Float64(BigFloat(1) - sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[2] ≈ Float64(BigFloat(1) + sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[3] ≈ Float64(BigFloat(1) + sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[4] ≈ Float64(BigFloat(1) - sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    
    @test field.η[1] ≈ Float64(- sqrt(BigFloat(2) * (BigFloat(3) + sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[2] ≈ Float64(- sqrt(BigFloat(2) * (BigFloat(3) - sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[3] ≈ Float64(+ sqrt(BigFloat(2) * (BigFloat(3) - sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[4] ≈ Float64(+ sqrt(BigFloat(2) * (BigFloat(3) + sqrt(BigFloat(6)))))  rtol=1e-15

    @test field.choices[1, :] == Int8[2, 3, 4]
    @test field.choices[2, :] == Int8[1, 3, 4]
    @test field.choices[3, :] == Int8[1, 2, 4]
    @test field.choices[4, :] == Int8[1, 2, 3]
end

@testset "Interface" begin
    fields = (MagneticHirschField, DensityHirschField, MagneticGHQField)

    for field_type in fields
        mc = DQMC(HubbardModelRepulsive(2, 2), beta = 1.0, field = field_type)
        f = mc.field

        @test MonteCarlo.maybe_to_float(ComplexF64(1, 0)) === 1.0
        @test MonteCarlo.maybe_to_float(ComplexF64(0, 1)) === ComplexF64(0, 1)

        @test MonteCarlo.field(mc) == f
        @test MonteCarlo.conf(f) == f.conf
        @test MonteCarlo.temp_conf(f) == f.temp_conf
        @test MonteCarlo.length(f) == length(f.conf)
        
        c = rand(f)
        @test MonteCarlo.conf(f) != c
        MonteCarlo.conf!(f, c)
        @test MonteCarlo.conf(f) == c
        rand!(f)
        @test MonteCarlo.conf(f) != c

        compressed = MonteCarlo.compress(f)
        @test compressed isa BitArray
        @test MonteCarlo.compressed_conf_type(f) <: BitArray
        @test MonteCarlo.decompress(f, compressed) == MonteCarlo.conf(f)
        c = MonteCarlo.conf(f); rand!(f)
        MonteCarlo.decompress!(f, compressed)
        @test MonteCarlo.conf(f) == c

        @test MonteCarlo.nflavors(f) == (2 - (f isa DensityHirschField))
        E = f isa DensityHirschField ? f.α * sum(f.conf) : 0.0
        @test MonteCarlo.energy_boson(f) == E
    end
end

using StructArrays, LinearAlgebra
using MonteCarlo, Test

@testset "Linear Algebra" begin
    i = 3; N = 4; C = ComplexF64; SA = StructArray

    G = rand(N, N)
    cache = MonteCarlo.StandardFieldCache(rand(), rand(), rand(), rand(N), rand(N), rand())
    cache.R = 1 + cache.Δ * (1 - G[i, i])
    cache.detratio = cache.R * cache.R

    invRΔ = cache.Δ / cache.R
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)
    @test invRΔ ≈ cache.invRΔ rtol = 1e-14

    IG = (I - Matrix(G))[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)
    # @test cache.IG ≈ IG rtol = 1e-14

    g = invRΔ * Matrix(G)[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)
    # @test g ≈ cache.G rtol = 1e-14

    Q = Matrix(G) - IG * g
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)
    @test G ≈ Q rtol = 1e-14


    G = SA(rand(ComplexF64, N, N))
    cache = MonteCarlo.StandardFieldCache(rand(C), rand(C), rand(C), SA(rand(C, N)), SA(rand(C, N)), rand(C))
    cache.R = 1 + cache.Δ * (1 - G[i, i])
    cache.detratio = cache.R * cache.R

    invRΔ = cache.Δ / cache.R
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)
    @test invRΔ ≈ cache.invRΔ rtol = 1e-14

    IG = (I - Matrix(G))[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)
    # @test cache.IG ≈ IG rtol = 1e-14

    g = invRΔ * Matrix(G)[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)
    # @test g ≈ cache.G rtol = 1e-14

    Q = Matrix(G) - IG * g
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)
    @test G ≈ Q rtol = 1e-14


    G = rand(2N, 2N)
    cache = MonteCarlo.StandardFieldCache(rand(2), rand(2, 2), rand(2, 2), rand(2N, 2), rand(2N, 2), rand())
    cache.R .= I + Diagonal(cache.Δ) * (I - G[i:N:end, i:N:end])
    cache.detratio = det(cache.R)

    invRΔ = inv(cache.R) * Diagonal(cache.Δ)
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)
    @test invRΔ ≈ cache.invRΔ rtol = 1e-14

    IG = (I - Matrix(G))[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)
    # @test cache.IG ≈ IG rtol = 1e-14

    g = invRΔ * Matrix(G)[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)
    # @test g ≈ cache.G rtol = 1e-14

    Q = Matrix(G) - IG * g
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)
    @test G ≈ Q rtol = 1e-14


    G = MonteCarlo.BlockDiagonal(rand(N, N), rand(N, N))
    cache = MonteCarlo.StandardFieldCache(rand(2), rand(2), rand(2), (rand(N), rand(N)), (rand(N), rand(N)), rand())
    cache.R[1] = 1 + cache.Δ[1] * (1 - G[i, i])
    cache.R[2] = 1 + cache.Δ[2] * (1 - G[i+N, i+N])
    cache.detratio = cache.R[1] * cache.R[2]

    invRΔ = cache.Δ ./ cache.R
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)
    @test invRΔ ≈ cache.invRΔ rtol = 1e-14

    IG = (I - Matrix(G))[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)
    # @test cache.IG ≈ IG rtol = 1e-14

    g = Diagonal(invRΔ) * Matrix(G)[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)
    # @test g ≈ cache.G rtol = 1e-14

    Q = Matrix(G) - IG * g
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)
    @test G ≈ Q rtol = 1e-14


    G = MonteCarlo.BlockDiagonal(SA(rand(ComplexF64, N, N)), SA(rand(ComplexF64, N, N)))
    cache = MonteCarlo.StandardFieldCache(SA(rand(C, 2)), SA(rand(C, 2)), SA(rand(C, 2)), (SA(rand(C, N)), SA(rand(C, N))), (SA(rand(C, N)), SA(rand(C, N))), rand(C))
    cache.R[1] = 1 + cache.Δ[1] * (1 - G[i, i])
    cache.R[2] = 1 + cache.Δ[2] * (1 - G[i+N, i+N])
    cache.detratio = cache.R[1] * cache.R[2]

    invRΔ = cache.Δ ./ cache.R
    MonteCarlo.vldiv22!(cache, cache.R, cache.Δ)
    @test invRΔ ≈ cache.invRΔ rtol = 1e-14

    IG = (I - Matrix(G))[:, i:N:end]
    MonteCarlo.vsub!(cache.IG, I, G, i, N)
    # @test cache.IG ≈ IG rtol = 1e-14

    g = Diagonal(invRΔ) * Matrix(G)[i:N:end, :]
    MonteCarlo.vmul!(cache.G, cache.invRΔ, G, i, N)
    # @test g ≈ cache.G rtol = 1e-14

    Q = Matrix(G) - IG * g
    MonteCarlo.vsubkron!(G, cache.IG, cache.G)
    @test G ≈ Q rtol = 1e-14
end

@testset "Sign Problem in field - model Combinations" begin
    models = (HubbardModelAttractive(8, 1), HubbardModelRepulsive(8, 1))
    fields = (MagneticHirschField, DensityHirschField, MagneticGHQField)

    for m in models, field in fields
        mc = DQMC(m, field = field, beta=3.1, thermalization=50, sweeps = 50, print_rate = 100)
        run!(mc, verbose=false)
        
        @testset "$(nameof(field)) $(nameof(typeof(m)))" begin
            @test mc.analysis.imaginary_probability.count == 0
            @test mc.analysis.negative_probability.count == 0
            @test mc.analysis.propagation_error.count == 0
        end
    end
end