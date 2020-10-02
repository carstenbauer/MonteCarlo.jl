@testset "Hubbard Model" begin

    # Type hierarchy
    @test HubbardModelAttractive <: HubbardModel
    @test HubbardModelRepulsive <: HubbardModel

    # constructors
    @test HubbardModel(U =  1.0, dims=1, L=1) isa HubbardModelRepulsive
    @test HubbardModel(U = -1.0, dims=1, L=1) isa HubbardModelAttractive

    for T in (HubbardModelAttractive, HubbardModelRepulsive)
        m = T(dims=1, L=8);
        @test m.L == 8 && m.dims == 1
        @test typeof(m) == T{Chain}
        
        d = Dict(:dims=>2,:L=>3)
        m = T(d)
        @test typeof(m) == T{SquareLattice}
        @test m.L == 3 && m.dims == 2
    end
end
