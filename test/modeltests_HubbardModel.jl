@testset "Hubbard Model" begin

    # Type hierarchy
    @test HubbardModelAttractive <: HubbardModel
    @test HubbardModelRepulsive <: HubbardModel

    # constructors
    @test HubbardModel(1, 1, U =  1.0) isa HubbardModelRepulsive
    @test HubbardModel(1, 1, U = -1.0) isa HubbardModelAttractive

    for T in (HubbardModelAttractive, HubbardModelRepulsive)
        m = T(8, 1)
        @test length(m.l) == 8
        @test typeof(m) == T{Chain}
        
        l = SquareLattice(5)
        m = T(l = l);
        @test length(m.l) == 25
        @test typeof(m) == T{SquareLattice}

        m = T(l, t = 0.5);
        @test length(m.l) == 25
        @test m.t == 0.5
        @test typeof(m) == T{SquareLattice}
        
        d = Dict(:l => l, :U => 0.5)
        m = T(d)
        @test typeof(m) == T{SquareLattice}
        @test m.U == 0.5
    end
end
