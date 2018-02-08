using MonteCarlo, MonteCarloObservable
using Base.Test

@testset "All Tests" begin
    @testset "Models" begin
        @testset "Ising Model" begin
            # constructors
            m = IsingModel(L=8);
            @test m.L == 8 && m.dims == 2
            @test typeof(m) == MonteCarlo.IsingModel{MonteCarlo.SquareLattice}
            m = IsingModel(dims=1, L=10);
            @test typeof(m) == MonteCarlo.IsingModel{MonteCarlo.Chain}
            @test m.L == 10 && m.dims == 1
            d = Dict{String,Any}(Pair{String,Any}("dims", 3),Pair{String,Any}("L", 3))
            m = IsingModel(d)
            @test typeof(m) == MonteCarlo.IsingModel{MonteCarlo.CubicLattice{Array{Int64,3}}}
            @test m.L == 3 && m.dims == 3

            # energy, general
            @test MonteCarlo.energy(MC(m), m, reshape(Int8.(1:27), (3,3,3))) == 355.0
            # energy, square lattice
            m = IsingModel()
            mc = MC(m)
            conf = reshape(Int8.(1:64), (8,8))
            conff = deepcopy(conf)
            @test MonteCarlo.energy(mc, m, conf) == 288.0

            # rand, conftype
            srand(123)
            @test MonteCarlo.rand(mc, m) == Int8[-1 -1 1 -1 1 1 1 -1; -1 1 1 -1 -1 1 1 1; -1 1 1 1 1 1 -1 1; -1 1 1 1 -1 -1 1 1; 1 1 1 -1 1 -1 -1 -1; 1 -1 -1 -1 1 1 -1 1; -1 -1 1 1 1 1 -1 1; 1 -1 1 -1 -1 -1 1 1]
            @test MonteCarlo.conftype(m) == Array{Int8, m.dims}

            # propose, accept
            @test MonteCarlo.propose_local(mc, m, 13, conff, 288.0) == (1352.0, 2)
            @test conf == conff
            @test MonteCarlo.accept_local!(mc, m, 13, conff, 288.0, 2, 1352.0) == nothing
            conff[13] *= -1
            @test conf == conff

            # observables
            obs = MonteCarlo.prepare_observables(mc, m);
            @test typeof(obs) == Dict{String,MonteCarloObservable.Observable}
            @test MonteCarlo.measure_observables!(mc, m, obs, conf, 288.0) == nothing
            @test MonteCarlo.finish_observables!(mc, m, obs) == nothing
        end
    end

    @testset "Examples" begin
        @testset "Ising Model + MC" begin
            srand(123)
            m = IsingModel(dims=2, L=8);
            mc = MC(m, beta=0.35);
            run!(mc, sweeps=1000, thermalization=10, verbose=false);
            m = mc.obs["m"] # magnetization

            @test isapprox(0.398, round(mean(m),3))
            @test isapprox(0.014, round(std(m), 3))
            @test typeof(observables(mc)) == Dict{String, String}
        end
    end
end
