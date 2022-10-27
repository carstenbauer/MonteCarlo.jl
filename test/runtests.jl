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

@testset "All Tests" showtiming = true begin
    # @testset "Utilities" begin
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
    
    # @testset "Lattices" begin
    #     include("lattices.jl")
    # end

    # @testset "Models" begin
    #     include("modeltests_IsingModel.jl")
    #     include("modeltests_HubbardModel.jl")
    # end

    # @testset "Linear Algebra" begin
    #     include("linalg.jl")
    # end

    # @testset "Slice Matrices" begin
    #     include("slice_matrices.jl")
    # end

    @testset "Flavors" begin
        # include("flavortests_DQMC.jl")
        include("DQMC/checkerboard.jl")
    end

    # @testset "Fields" begin
    #     include("fields.jl")
    # end

    # @testset "Scheduler & (DQMC) Updates" begin
    #     include("updates.jl")
    # end

    # @testset "Measurements" begin
    #     include("measurements.jl")
    #     include("DQMC/measurements.jl")
    # end

    # @testset "Integration tests" begin
    #     include("integration_tests.jl")
    # end

    # @testset "Exact Diagonalization" begin
    #     include("ED/ED_tests.jl")
    # end

    # @testset "File IO" begin
    #     include("FileIO.jl")
    # end
end