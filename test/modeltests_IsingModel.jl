@testset "Ising Model" begin
    IsingSpin = MonteCarlo.IsingSpin

    # constructors
    m = IsingModel(L=8,dims=2);
    @test m.L == 8 && m.dims == 2
    @test ndims(m) == 2
    @test length(lattice(m)) == 64
    @test typeof(m) == IsingModel{SquareLattice}
    m = IsingModel(dims=1, L=10);
    @test typeof(m) == IsingModel{Chain}
    @test m.L == 10 && m.dims == 1
    @test length(lattice(m)) == 10
    @test ndims(m) == 1
    d = Dict(:dims=>3, :L=>3)
    m = IsingModel(d)
    @test typeof(m) == IsingModel{CubicLattice{Array{Int64,3}}}
    @test m.L == 3 && m.dims == 3

    # energy, general
    @test MonteCarlo.energy(MC(m, beta=1), m, reshape(IsingSpin.(1:27), (3,3,3))) == -18333.0
    # energy, square lattice
    m = IsingModel(dims=2, L=8)
    mc = MC(m, beta=1)
    conf = reshape(IsingSpin.(1:64), (8,8))
    conff = deepcopy(conf)
    @test MonteCarlo.energy(mc, m, conf) == -164320.0

    # rand, conftype
    Random.seed!(123)
    _conf = if VERSION.major == 1 && VERSION.minor == 5
        IsingSpin[1 -1 1 1 -1 -1 1 -1; 1 -1 1 -1 1 1 -1 -1; -1 -1 -1 -1 1 -1 1 1; 1 1 1 1 -1 1 -1 -1; -1 1 1 1 1 -1 -1 1; 1 1 -1 -1 -1 -1 1 -1; 1 -1 1 -1 -1 -1 1 -1; 1 1 -1 1 -1 -1 1 -1]
    else # assuming v1.6
        IsingSpin[1 -1 -1 1 1 -1 1 1; -1 1 1 1 1 -1 1 -1; -1 -1 -1 -1 1 1 1 -1; -1 1 1 -1 -1 -1 -1 -1; 1 -1 1 1 1 1 -1 1; 1 -1 -1 1 -1 -1 -1 1; -1 1 -1 -1 -1 1 -1 1; 1 -1 -1 1 -1 1 -1 1]
    end
    @test rand(MC, m) == _conf

    # propose, accept
    @test MonteCarlo.propose_local(mc, m, 13, conff) == (1352.0, nothing)
    @test conf == conff
    @test MonteCarlo.accept_local!(mc, m, 13, conff, 1352.0, 2) === nothing
    conff[13] *= -1
    @test conf == conff
end
