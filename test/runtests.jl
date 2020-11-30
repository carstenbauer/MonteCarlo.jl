using MonteCarlo, MonteCarloObservable, StableDQMC
using Test
using LinearAlgebra, Random, Dates
using MonteCarlo: @bm, TimerOutputs

# In case some test failed and left behind a .jld file
for f in readdir()
    if endswith(f, ".jld") || endswith(f, "jld2")
        @warn "Removing $f"
        rm(f)
    end
end

macro benchmark_test(name, code)
    TimerOutputs.timer_expr(MonteCarlo, true, name, code)
end

@testset "All Tests" begin
    @testset "Utilities" begin
        @bm function test1(x, y)
            sleep(x+y)
        end
        @bm test2(x, y) = sleep(x+y)
        # @eval MonteCarlo ... will put the body of test3, test4 in the wrong scope
        # eval(timer_expr(...)) doesn't work
        function test3(x, y)
            @benchmark_test "test1(::Any, ::Any)" begin sleep(x+y) end
        end
        test4(x, y) = @benchmark_test "test2(::Any, ::Any)" begin sleep(x+y) end

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
        include("modeltests_HubbardModel.jl")
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

    @testset "File IO" begin
        include("FileIO.jl")
    end
end