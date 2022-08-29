using MonteCarlo

oldpath = pwd()
cd(joinpath(pkgdir(MonteCarlo), "test/assets"))

struct TestModel <: MonteCarlo.Model
    l::AbstractLattice
    U::Float64
    x::Int64
    y::String
end

Base.rand(::Type{DQMC}, ::TestModel, n::Int64) = Base.rand(4, n)
MonteCarlo.choose_field(::TestModel) = DensityHirschField
MonteCarlo.lattice(m::TestModel) = m.l
MonteCarlo.unique_flavors(::TestModel) = 1
MonteCarlo.hopping_matrix(::TestModel) = ones(4, 4)

mc = DQMC(TestModel(SquareLattice(2), 1.0, 7, "foo"), beta=1.0, recorder=Discarder())
MonteCarlo.save("dummy_in.jld2", mc, overwrite=true)

cd(oldpath)