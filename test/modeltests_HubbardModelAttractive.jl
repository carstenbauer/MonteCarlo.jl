@testset "Hubbard Model Attractive" begin
    # constructors
    m = HubbardModelAttractive(dims=1, L=8);
    @test m.L == 8 && m.dims == 1
    @test typeof(m) == MonteCarlo.HubbardModelAttractive{MonteCarlo.Chain}
    d = Dict{String,Any}(Pair{String,Any}("dims", 2),Pair{String,Any}("L", 3))
    m = HubbardModelAttractive(d)
    @test typeof(m) == MonteCarlo.HubbardModelAttractive{MonteCarlo.SquareLattice}
    @test m.L == 3 && m.dims == 2
end
