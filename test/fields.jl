# these are just shorthads
using MonteCarlo: FVec64, FMat64, CVec64, CMat64, BlockDiagonal

@testset "Typing (Field Cache, hopping, interaction & greens matrix" begin
    T2 = NTuple{2}
    fields = (DensityHirschField, MagneticHirschField, DensityGHQField, MagneticGHQField)

    for U in (+1.0, -1.0)
        for (i, field) in enumerate(fields)

            @testset "U = $U field = $field" begin
                mc = DQMC(HubbardModel(U = U), field = field, beta = 1.0)
                MonteCarlo.init!(mc)

                # missmatch between U and Density/Magnetic results in complex interaction
                if (i + (U == +1.0)) % 2 == 0 # (U == 1 && Density) || (U == -1 && Magnetic)
                    T = Float64; VT = FVec64; mT = FMat64
                else
                    T = ComplexF64; VT = CVec64; mT = CMat64
                end
                @test MonteCarlo.interaction_eltype(mc.field) == T
                @test MonteCarlo.interaction_matrix_type(mc.field, mc.model) == Diagonal{T, VT}
                @test MonteCarlo.init_interaction_matrix(mc.field, mc.model) isa Diagonal{T, VT}
                @test mc.stack.eV isa Diagonal{T, VT}

                # hopping are always real and 1 flavor. They should get dublicated 
                # if interaction requires two flavors (magnetic)
                @test MonteCarlo.nflavors(mc.model) == 1
                @test MonteCarlo.hopping_eltype(mc.model) == Float64
                
                # Greens matrix takes eltype from interaction because that might be complex
                @test MonteCarlo.greens_eltype(mc.field, mc.model) == T

                # 1 flavor for Density, 2 for Magnetic in interaction/total
                @test MonteCarlo.nflavors(mc.field) == 2 - (i % 2)
                @test MonteCarlo.nflavors(mc.field, mc.model) == 2 - (i % 2)
                
                x = rand(4, 4)

                if i % 2 == 1
                    @test MonteCarlo.hopping_matrix_type(mc.field, mc.model) == FMat64
                    @test mc.stack.hopping_matrix isa FMat64

                    @test MonteCarlo.pad_to_nflavors(mc.field, mc.model, x) == x

                    @test MonteCarlo.greens_matrix_type(mc.field, mc.model) == mT
                    @test mc.stack.greens isa mT

                    # single flavor -> single value of greens eltype
                    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{T, T, VT, T}
                else
                    @test MonteCarlo.hopping_matrix_type(mc.field, mc.model) == BlockDiagonal{Float64, 2, FMat64}
                    @test mc.stack.hopping_matrix isa BlockDiagonal{Float64, 2, FMat64}

                    @test MonteCarlo.pad_to_nflavors(mc.field, mc.model, x) == BlockDiagonal(x, x)

                    @test MonteCarlo.greens_matrix_type(mc.field, mc.model) == BlockDiagonal{T, 2, mT}
                    @test mc.stack.greens isa BlockDiagonal{T, 2, mT}

                    # two flavor -> matrix types of greens eltypes
                    @test mc.stack.field_cache isa MonteCarlo.StandardFieldCache{VT, VT, T2{VT}, T}
                end
            end

        end
    end

end

@testset "Lookup tables" begin
    param = MonteCarlo.DQMCParameters(beta = 1.0)
    model = HubbardModel()
    field = MagneticGHQField(param, model)

    @test field.α == sqrt(-0.5 * 0.1 * ComplexF64(model.U))
    
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
        mc = DQMC(HubbardModel(2, 2), beta = 1.0, field = field_type)
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
    models = (HubbardModel(8, 1, U = 1.0), HubbardModel(8, 1, U = -1.0))
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