using MonteCarlo, MonteCarloObservable
using Test
using Random
using MonteCarlo: @bm, TimerOutputs

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
            @benchmark_test "test1" begin sleep(x+y) end
        end
        test4(x, y) = @benchmark_test "test2" begin sleep(x+y) end

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

using MonteCarlo
using MonteCarlo: @bm
using TimerOutputs
@bm function test1(x, y)
    sleep(x+y)
end
@bm test2(x, y) = sleep(x+y)
macro benchmark_test(name, code)
    TimerOutputs.timer_expr(MonteCarlo, true, name, code)
end
function test3(x, y)
    @benchmark_test "test1" begin sleep(x+y) end
end
test4(x, y) = @benchmark_test "test2" begin sleep(x+y) end


x = code_lowered(test1, Tuple{Float64, Float64})[1]
y = code_lowered(test3, Tuple{Float64, Float64})[1]
@test x.code == y.code

x = code_lowered(test2, Tuple{Float64, Float64})[1]
y = code_lowered(test4, Tuple{Float64, Float64})[1]
@test x.code == y.code
