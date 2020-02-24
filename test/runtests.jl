using MonteCarlo, MonteCarloObservable
using Test
using Random
using MonteCarlo: @bm, TimerOutputs


@testset "All Tests" begin
    @testset "Utilities" begin
        @bm function test1(x, y)
            sleep(x+y)
        end
        @bm test2(x, y) = sleep(x+y)
        function test3(x, y)
            TimerOutputs.@timeit_debug "test1" begin sleep(x+y) end
        end
        test4(x, y) = TimerOutputs.@timeit_debug "test2" begin sleep(x+y) end

        x = code_lowered(test1, Tuple{Float64, Float64})[1]
        y = code_lowered(test3, Tuple{Float64, Float64})[1]
        @test x.code == y.code

        x = code_lowered(test2, Tuple{Float64, Float64})[1]
        y = code_lowered(test4, Tuple{Float64, Float64})[1]
        @test x.code == y.code

        @test !MonteCarlo.timeit_debug_enabled()
        enable_benchmarks()
        @test MonteCarlo.timeit_debug_enabled()
        disable_benchmarks()
        @test !MonteCarlo.timeit_debug_enabled()
    end

    @testset "Lattices" begin
        include("lattices.jl")
    end

    @testset "Models" begin
        include("modeltests_IsingModel.jl")
        include("modeltests_HubbardModelAttractive.jl")
    end

    @testset "Flavors" begin
        # include("flavortests_MC.jl")
        include("flavortests_DQMC.jl")
    end

    @testset "Measurements" begin
        include("measurements.jl")
    end

    @testset "Intergration tests" begin
        include("integration_tests.jl")
    end

    @testset "Exact Diagonalization" begin
        include("ED/ED_tests.jl")
    end
end
