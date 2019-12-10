using MonteCarlo, MonteCarloObservable
using Test
using Random, Dates

# In case some test failed and left behind a .jld file
for f in readdir()
    if endswith(f, ".jld")
        @warn "Removing $f"
        rm(f)
    end
end

@testset "All Tests" begin
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

    @testset "File IO" begin
        include("FileIO.jl")
    end
end
