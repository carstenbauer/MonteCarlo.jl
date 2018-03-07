@testset "DQMC" begin
    m = HubbardModel(dims=1, L=8);

    # constructors
    dqmc = DQMC(m; beta=5.0)
    @test m.L == 8 && m.dims == 1
    @test typeof(m) == MonteCarlo.HubbardModel{MonteCarlo.Chain}
    d = Dict{String,Any}(Pair{String,Any}("dims", 2),Pair{String,Any}("L", 3))
    m = HubbardModel(d)
    @test typeof(m) == MonteCarlo.HubbardModel{MonteCarlo.SquareLattice}
    @test m.L == 3 && m.dims == 2
end
