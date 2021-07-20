@time using MonteCarlo, MonteCarloObservable, StableDQMC
#  3.567540 seconds (7.22 M allocations: 469.081 MiB, 4.08% gc time, 0.34% compilation time)
using Test
using LinearAlgebra, Random, Dates
using MonteCarlo: @bm, TimerOutputs

struct DummyModel <: MonteCarlo.Model 
    lattice
end
DummyModel() = DummyModel(SquareLattice(2))

# check elementwise, not matrix norm
function check(A::Array, B::Array, atol, rtol=atol)
    for (x, y) in zip(A, B)
        if !isapprox(x, y, atol=atol, rtol=rtol)
            @info "$x ≉ $y "
            return false
        end
    end
    true
end
function check(x::Number, y::Number, atol, rtol=atol)
    result = isapprox(x, y, atol=atol, rtol=rtol)
    result || @info "$x ≉ $y"
    result
end

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
    println("Utilities")
    @time @testset "Utilities" begin
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
    # 0.473232 seconds (778.10 k allocations: 47.050 MiB, 2.72% gc time, 99.50% compilation time)

    println("Lattices")
    @time @testset "Lattices" begin
        include("lattices.jl")
    end
    # 26.765963 seconds (50.23 M allocations: 2.407 GiB, 3.35% gc time, 58.25% compilation time)

    println("Model")
    @time @testset "Models" begin
        include("modeltests_IsingModel.jl")
        include("modeltests_HubbardModel.jl")
    end
    # 11.418479 seconds (29.34 M allocations: 1.301 GiB, 4.78% gc time, 7.60% compilation time)

    println("DQMC")
    @time @testset "Flavors" begin
        # include("flavortests_MC.jl")
        include("flavortests_DQMC.jl")
    end
    # 61.075442 seconds (107.31 M allocations: 6.180 GiB, 4.44% gc time, 5.18% compilation time)

    println("Scheduler")
    @time @testset "Scheduler & (DQMC) Updates" begin
        include("updates.jl")
    end
    # 12.962446 seconds (17.87 M allocations: 1017.561 MiB, 2.51% gc time, 7.02% compilation time)

    println("Measurement")
    @time @testset "Measurements" begin
        include("measurements.jl")
    end
    # 13.129428 seconds (21.85 M allocations: 1.240 GiB, 4.31% gc time, 2.01% compilation time)

    println("Integration")
    @time @testset "Integration tests" begin
        include("integration_tests.jl")
    end
    # 20.755311 seconds (39.02 M allocations: 2.359 GiB, 3.11% gc time, 12.65% compilation time)

    println("ED")
    @time @testset "Exact Diagonalization" begin
        include("ED/ED_tests.jl")
    end
    # 61.415264 seconds (49.97 M allocations: 12.824 GiB, 2.79% gc time, 10.31% compilation time)

    println("File IO")
    @time @testset "File IO" begin
        include("FileIO.jl")
    end
    # 81.479393 seconds (120.31 M allocations: 5.953 GiB, 2.20% gc time, 5.36% compilation time)
end