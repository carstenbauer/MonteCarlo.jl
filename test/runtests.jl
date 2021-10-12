using MonteCarlo, MonteCarloObservable, StableDQMC
using Test
using LinearAlgebra, Random, Dates
using MonteCarlo: @bm, TimerOutputs

using MonteCarlo: vmul!, lvmul!, rvmul!, rdivp!, udt_AVX_pivot!, rvadd!, vsub!
using MonteCarlo: vmin!, vmininv!, vmax!, vmaxinv!, vinv!
using MonteCarlo: BlockDiagonal, CMat64, CVec64, StructArray

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
    # println("Utilities")
    # @time @testset "Utilities" begin
    #     @bm function test1(x, y)
    #         sleep(x+y)
    #     end
    #     @bm test2(x, y) = sleep(x+y)
    #     # @eval MonteCarlo ... will put the body of test3, test4 in the wrong scope
    #     # eval(timer_expr(...)) doesn't work
    #     function test3(x, y)
    #         @benchmark_test "test1(::Any, ::Any)" begin sleep(x+y) end
    #     end
    #     test4(x, y) = @benchmark_test "test2(::Any, ::Any)" begin sleep(x+y) end

    #     x = code_lowered(test1, Tuple{Float64, Float64})[1]
    #     y = code_lowered(test3, Tuple{Float64, Float64})[1]
    #     @test x.code == y.code

    #     x = code_lowered(test2, Tuple{Float64, Float64})[1]
    #     y = code_lowered(test4, Tuple{Float64, Float64})[1]
    #     @test x.code == y.code

    #     @test !MonteCarlo.timeit_debug_enabled()
    #     enable_benchmarks()
    #     @test MonteCarlo.timeit_debug_enabled()
    #     disable_benchmarks()
    #     @test !MonteCarlo.timeit_debug_enabled()
    # end
    # # 0.473232 seconds (778.10 k allocations: 47.050 MiB, 2.72% gc time, 99.50% compilation time)

    # println("Lattices")
    # @time @testset "Lattices" begin
    #     include("lattices.jl")
    # end
    # # 22.329261 seconds (51.70 M allocations: 2.487 GiB, 3.09% gc time, 63.07% compilation time)

    # println("Model")
    # @time @testset "Models" begin
    #     include("modeltests_IsingModel.jl")
    #     include("modeltests_HubbardModel.jl")
    # end
    # # 11.418479 seconds (29.34 M allocations: 1.301 GiB, 4.78% gc time, 7.60% compilation time)

    println("Linear Algebra")
    @time @testset "Linear Algebra" begin
        @time include("linalg.jl")
    end
    # 72.325652 seconds (119.03 M allocations: 5.922 GiB, 3.30% gc time, 33.24% compilation time)

    # println("Slice Matrices")
    # @time @testset "Slice Matrices" begin
    #     include("slice_matrices.jl")
    # end
    # # 5.212298 seconds (7.85 M allocations: 424.303 MiB, 1.66% gc time, 96.86% compilation time)

    # println("DQMC")
    # @time @testset "Flavors" begin
    #     include("flavortests_DQMC.jl")
    # end
    # # 27.297888 seconds (55.24 M allocations: 3.149 GiB, 2.60% gc time, 2.67% compilation time)

    # println("Scheduler")
    # @time @testset "Scheduler & (DQMC) Updates" begin
    #     include("updates.jl")
    # end
    # # 17.197345 seconds (33.06 M allocations: 1.846 GiB, 2.53% gc time, 4.69% compilation time)

    # println("Measurement")
    # @time @testset "Measurements" begin
    #     include("measurements.jl")
    # end
    # # 20.091780 seconds (37.56 M allocations: 1.877 GiB, 2.46% gc time, 1.43% compilation time)

    # println("Integration")
    # @time @testset "Integration tests" begin
    #     include("integration_tests.jl")
    # end
    # # 11.359702 seconds (19.67 M allocations: 1.132 GiB, 2.00% gc time, 28.81% compilation time)

    # println("ED")
    # @time @testset "Exact Diagonalization" begin
    #     include("ED/ED_tests.jl")
    # end
    # # 71.714791 seconds (52.37 M allocations: 14.521 GiB, 1.82% gc time, 4.57% compilation time)

    # println("File IO")
    # @time @testset "File IO" begin
    #     include("FileIO.jl")
    # end
    # # 72.591676 seconds (86.55 M allocations: 4.172 GiB, 1.27% gc time, 9.25% compilation time)
end