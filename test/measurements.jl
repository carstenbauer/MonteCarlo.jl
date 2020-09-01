AbstractMeasurement = MonteCarlo.AbstractMeasurement

IsingMeasurement = MonteCarlo.IsingMeasurement
IsingEnergyMeasurement = MonteCarlo.IsingEnergyMeasurement
IsingMagnetizationMeasurement = MonteCarlo.IsingMagnetizationMeasurement

GreensMeasurement = MonteCarlo.GreensMeasurement
BosonEnergyMeasurement = MonteCarlo.BosonEnergyMeasurement

AbstractObservable = MonteCarloObservable.AbstractObservable

struct DummyModel <: MonteCarlo.Model end
struct DummyMeasurement <: AbstractMeasurement end

@testset "Type Hierarchy" begin
    @test IsingEnergyMeasurement <: IsingMeasurement
    @test IsingMagnetizationMeasurement <: IsingMeasurement
    @test IsingMeasurement <: AbstractMeasurement
    @test GreensMeasurement <: AbstractMeasurement
    @test BosonEnergyMeasurement <: AbstractMeasurement
end

@testset "Interface" begin
    m = DummyMeasurement()
    dummy_model = DummyModel()
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    @test nothing == MonteCarlo.prepare!(m, mc, model)
    @test_throws MethodError MonteCarlo.measure!(m, mc, model, 1)
    @test nothing == MonteCarlo.finish!(m, mc, model)
    @test MonteCarlo.default_measurements(mc, dummy_model) == Dict{Symbol, AbstractMeasurement}()
end

@testset "Checking defaults" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    defaults = MonteCarlo.default_measurements(mc, model)
    @test !isempty(defaults)

    @test haskey(defaults, :Magn) && defaults[:Magn] isa IsingMagnetizationMeasurement
    obs = observables(defaults[:Magn])
    @test haskey(obs, "Magnetization per site") && obs["Magnetization per site"] isa AbstractObservable
    @test haskey(obs, "Magnetic susceptibility") && obs["Magnetic susceptibility"] isa AbstractObservable
    @test haskey(obs, "Total magnetization") && obs["Total magnetization"] isa AbstractObservable
    @test haskey(obs, "Total magnetization squared") && obs["Total magnetization squared"] isa AbstractObservable

    @test haskey(defaults, :Energy) && defaults[:Energy] isa IsingEnergyMeasurement
    obs = observables(defaults[:Energy])
    @test haskey(obs, "Energy per site") && obs["Energy per site"] isa AbstractObservable
    @test haskey(obs, "Specific heat") && obs["Specific heat"] isa AbstractObservable
    @test haskey(obs, "Total energy") && obs["Total energy"] isa AbstractObservable
    @test haskey(obs, "Total energy squared") && obs["Total energy squared"] isa AbstractObservable


    @test isempty(mc.thermalization_measurements)
    @test !isempty(mc.measurements)
    @test haskey(mc.measurements, :Magn) && mc.measurements[:Magn] isa IsingMagnetizationMeasurement
    @test haskey(mc.measurements, :Energy) && mc.measurements[:Energy] isa IsingEnergyMeasurement

    model = HubbardModelAttractive(dims=2, L=2)
    mc = DQMC(model, beta=1.0)

    defaults = MonteCarlo.default_measurements(mc, model)
    @test !isempty(defaults)
    @test haskey(defaults, :Greens) && defaults[:Greens] isa GreensMeasurement
    obs = observables(defaults[:Greens])
    @test haskey(obs, "Equal-times Green's function") && obs["Equal-times Green's function"] isa AbstractObservable

    @test haskey(defaults, :BosonEnergy) && defaults[:BosonEnergy] isa BosonEnergyMeasurement
    obs = observables(defaults[:BosonEnergy])
    @test haskey(obs, "Bosonic Energy") && obs["Bosonic Energy"] isa AbstractObservable

    @test isempty(mc.thermalization_measurements)
    @test !isempty(mc.measurements)
    @test haskey(mc.measurements, :Greens) && mc.measurements[:Greens] isa GreensMeasurement
    @test haskey(mc.measurements, :BosonEnergy) && mc.measurements[:BosonEnergy] isa BosonEnergyMeasurement
end

@testset "Retrieving Measurements" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    ms = MonteCarlo.measurements(mc)
    @test mc.thermalization_measurements == ms[:TH]
    @test mc.measurements == ms[:ME]

    obs = MonteCarlo.observables(mc)
    @test keys(obs[:TH]) == keys(ms[:TH])
    @test keys(obs[:ME]) == keys(ms[:ME])

    @test haskey(obs[:ME][:Energy], "Total energy")
    @test typeof(obs[:ME][:Energy]["Total energy"]) <: AbstractObservable
    @test haskey(obs[:ME][:Energy], "Total energy squared")
    @test typeof(obs[:ME][:Energy]["Total energy squared"]) <: AbstractObservable
    @test haskey(obs[:ME][:Energy], "Energy per site")
    @test typeof(obs[:ME][:Energy]["Energy per site"]) <: AbstractObservable
    @test haskey(obs[:ME][:Energy], "Specific heat")
    @test typeof(obs[:ME][:Energy]["Specific heat"]) <: AbstractObservable

    @test haskey(obs[:ME][:Magn], "Total magnetization")
    @test typeof(obs[:ME][:Magn]["Total magnetization"]) <: AbstractObservable
    @test haskey(obs[:ME][:Magn], "Total magnetization squared")
    @test typeof(obs[:ME][:Magn]["Total magnetization squared"]) <: AbstractObservable
    @test haskey(obs[:ME][:Magn], "Magnetization per site")
    @test typeof(obs[:ME][:Magn]["Magnetization per site"]) <: AbstractObservable
    @test haskey(obs[:ME][:Magn], "Magnetic susceptibility")
    @test typeof(obs[:ME][:Magn]["Magnetic susceptibility"]) <: AbstractObservable
end

@testset "Interacting with Measurements" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    @test length(mc.measurements) == 2
    delete!(mc, AbstractMeasurement)
    @test isempty(mc.measurements)

    @test_throws ErrorException push!(mc, :E => Int64)
    @test_throws MethodError MonteCarlo.unsafe_push!(mc, :E => 1.0)
    @test_throws ErrorException push!(mc, :E => IsingEnergyMeasurement, :bad_stage)
    @test_throws ErrorException MonteCarlo.unsafe_push!(mc, :E => IsingEnergyMeasurement(mc, model), :bad_stage)
    @test_throws ErrorException delete!(mc, :E, :bad_stage)
    @test_throws ErrorException delete!(mc, IsingEnergyMeasurement, :bad_stage)

    push!(mc, :E => IsingEnergyMeasurement)
    @test haskey(mc.measurements, :E) && mc.measurements[:E] isa IsingEnergyMeasurement
    delete!(mc, :E)
    @test !haskey(mc.measurements, :E)

    MonteCarlo.unsafe_push!(mc, :Magn => IsingMagnetizationMeasurement(mc, model))
    @test haskey(mc.measurements, :Magn) && mc.measurements[:Magn] isa IsingMagnetizationMeasurement
    delete!(mc, IsingMagnetizationMeasurement)
    @test !haskey(mc.measurements, :Magn)

    @test isempty(mc.thermalization_measurements)
    push!(mc, :E => IsingEnergyMeasurement, :TH)
    @test haskey(mc.thermalization_measurements, :E) &&
        mc.thermalization_measurements[:E] isa IsingEnergyMeasurement
    delete!(mc, :E, :TH)

    MonteCarlo.unsafe_push!(mc, :E => IsingEnergyMeasurement(mc, model), :TH)
    @test haskey(mc.thermalization_measurements, :E) &&
        mc.thermalization_measurements[:E] isa IsingEnergyMeasurement
    delete!(mc, IsingEnergyMeasurement, :TH)
    @test !haskey(mc.thermalization_measurements, :TH)
end

@testset "Saving and Loading" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)
    run!(mc, thermalization=10, sweeps=10, verbose=false)
    push!(mc, :E => IsingEnergyMeasurement, :TH)

    meas = measurements(mc)
    MonteCarlo.save_measurements("testfile.jld", mc, force_overwrite=true)
    _meas = MonteCarlo.load_measurements("testfile.jld")
    for (k, v) in meas
        for (k2, v2) in v
            for f in fieldnames(typeof(v2))
                @test getfield(v2, f) == getfield(_meas[k][k2], f)
            end
        end
    end
    rm("testfile.jld")
end
