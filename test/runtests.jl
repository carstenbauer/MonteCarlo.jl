using MonteCarlo, StableDQMC
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
struct DummyField <: MonteCarlo.AbstractField end

# check elementwise, not matrix norm
function check(A::Array, B::Array, atol, rtol)
    for (x, y) in zip(A, B)
        if !isapprox(x, y, atol=atol, rtol=rtol)
            @info "$x ≉ $y "
            return false
        end
    end
    true
end
function check(x::Number, y::Number, atol, rtol)
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
    # 0.318543 seconds (390.04 k allocations: 21.698 MiB, 99.14% compilation time)

    println("Lattices")
    @time @testset "Lattices" begin
        include("lattices.jl")
    end
    # 20.743133 seconds (48.56 M allocations: 2.088 GiB, 3.00% gc time, 69.97% compilation time)

    println("Model")
    @time @testset "Models" begin
        include("modeltests_IsingModel.jl")
        include("modeltests_HubbardModel.jl")
    end
    # 2.326224 seconds (5.23 M allocations: 297.223 MiB, 3.35% gc time, 98.42% compilation time)

    println("Linear Algebra")
    @time @testset "Linear Algebra" begin
        include("linalg.jl")
    end
    # 70.500653 seconds (113.03 M allocations: 5.533 GiB, 3.65% gc time, 53.82% compilation time)

    println("Slice Matrices")
    @time @testset "Slice Matrices" begin
        include("slice_matrices.jl")
    end
    # 6.358704 seconds (10.47 M allocations: 531.320 MiB, 1.75% gc time, 99.56% compilation time)

    println("DQMC")
    @time @testset "Flavors" begin
        include("flavortests_DQMC.jl")
    end
    # 30.060617 seconds (42.91 M allocations: 2.579 GiB, 2.19% gc time, 70.84% compilation time)

    println("Fields")
    @time @testset "Fields" begin
        include("fields.jl")
    end
    # 31.349194 seconds (38.40 M allocations: 1.924 GiB, 1.51% gc time, 98.22% compilation time)

    println("Scheduler")
    @time @testset "Scheduler & (DQMC) Updates" begin
        include("updates.jl")
    end
    # 17.825457 seconds (28.21 M allocations: 1.522 GiB, 2.02% gc time, 95.81% compilation time)

    println("Measurement")
    @time @testset "Measurements" begin
        include("measurements.jl")
    end
    # 7.075271 seconds (9.16 M allocations: 505.699 MiB, 2.00% gc time, 98.01% compilation time)

    println("Integration")
    @time @testset "Integration tests" begin
        include("integration_tests.jl")
    end
    # 0.731501 seconds (748.57 k allocations: 41.296 MiB, 84.61% compilation time)

    println("ED")
    @time @testset "Exact Diagonalization" begin
        include("ED/ED_tests.jl")
    end
    # 143.648447 seconds (71.89 M allocations: 14.621 GiB, 1.40% gc time, 27.53% compilation time)

    println("File IO")
    @time @testset "File IO" begin
        include("FileIO.jl")
    end
    # 70.228255 seconds (65.91 M allocations: 3.273 GiB, 1.34% gc time, 39.71% compilation time)
end