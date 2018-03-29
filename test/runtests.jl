using MonteCarlo, MonteCarloObservable
using Base.Test

@testset "All Tests" begin
    @testset "Models" begin
        include("modeltests_IsingModel.jl")
        include("modeltests_HubbardModelAttractive.jl")
    end

    @testset "Flavors" begin
        # include("flavortests_MC.jl")
        include("flavortests_DQMC.jl")
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
