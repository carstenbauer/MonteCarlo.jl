@testset "Hubbard Model" begin
    @test HubbardModel <: Model

    # constructors
    m = HubbardModel(8, 1)
    @test length(m.l) == 8
    @test typeof(m) == HubbardModel{Lattice{1}}
        
    l = SquareLattice(5)
    m = HubbardModel(l = l);
    @test length(m.l) == 25
    @test typeof(m) == HubbardModel{Lattice{2}}

    m = HubbardModel(l, t = 0.5);
    @test length(m.l) == 25
    @test m.t == 0.5
    @test typeof(m) == HubbardModel{Lattice{2}}
        
    d = Dict(:l => l, :U => HubbardModel == HubbardModelAttractive ? 0.5 : -0.5)
    m = HubbardModel(d)
    @test typeof(m) == HubbardModel{Lattice{2}}
    @test m.U == (HubbardModel == HubbardModelAttractive ? 0.5 : -0.5)
end
