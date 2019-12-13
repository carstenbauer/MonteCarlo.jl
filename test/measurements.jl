AbstractMeasurement = MonteCarlo.AbstractMeasurement
ConfigurationMeasurement = MonteCarlo.ConfigurationMeasurement

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

    @test haskey(defaults, :conf) && defaults[:conf] isa ConfigurationMeasurement
    obs = observables(defaults[:conf])
    @test haskey(obs, "Configurations") && obs["Configurations"] isa AbstractObservable

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
    @test haskey(mc.measurements, :conf) && mc.measurements[:conf] isa ConfigurationMeasurement
    @test haskey(mc.measurements, :Magn) && mc.measurements[:Magn] isa IsingMagnetizationMeasurement
    @test haskey(mc.measurements, :Energy) && mc.measurements[:Energy] isa IsingEnergyMeasurement

    model = HubbardModelAttractive(dims=2, L=2)
    mc = DQMC(model, beta=1.0)

    defaults = MonteCarlo.default_measurements(mc, model)
    @test !isempty(defaults)
    @test haskey(defaults, :conf) && defaults[:conf] isa ConfigurationMeasurement

    @test haskey(defaults, :Greens) && defaults[:Greens] isa GreensMeasurement
    obs = observables(defaults[:Greens])
    @test haskey(obs, "Equal-times Green's function") && obs["Equal-times Green's function"] isa AbstractObservable

    @test haskey(defaults, :BosonEnergy) && defaults[:BosonEnergy] isa BosonEnergyMeasurement
    obs = observables(defaults[:BosonEnergy])
    @test haskey(obs, "Bosonic Energy") && obs["Bosonic Energy"] isa AbstractObservable

    @test isempty(mc.thermalization_measurements)
    @test !isempty(mc.measurements)
    @test haskey(mc.measurements, :conf) && mc.measurements[:conf] isa ConfigurationMeasurement
    @test haskey(mc.measurements, :Greens) && mc.measurements[:Greens] isa GreensMeasurement
    @test haskey(mc.measurements, :BosonEnergy) && mc.measurements[:BosonEnergy] isa BosonEnergyMeasurement

    @test measurements(mc) == mc.measurements
    @test measurements(mc, :ME) == mc.measurements
    @test measurements(mc, :TH) == mc.thermalization_measurements
    @test measurements(mc, :ALL)[:ME] == mc.measurements
    @test measurements(mc, :ALL)[:TH] == mc.thermalization_measurements

    _obs = observables(mc, :all)
    @test observables(mc, :ME) == _obs[:ME]
    @test observables(mc, :TH) == _obs[:TH]
end

@testset "Retrieving Measurements" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    @test mc.thermalization_measurements == MonteCarlo.measurements(mc, :TH)
    @test mc.measurements == MonteCarlo.measurements(mc)

    obs = MonteCarlo.observables(mc, :all)
    @test keys(obs[:TH]) == keys(MonteCarlo.measurements(mc, :TH))
    @test keys(obs[:ME]) == keys(MonteCarlo.measurements(mc))

    @test haskey(obs[:ME][:conf], "Configurations")
    @test typeof(obs[:ME][:conf]["Configurations"]) <: AbstractObservable

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

    @test length(mc.measurements) == 3
    delete!(mc, AbstractMeasurement)
    @test isempty(mc.measurements)

    @test_throws ErrorException push!(mc, :conf => Int64)
    @test_throws MethodError MonteCarlo.unsafe_push!(mc, :conf => 1.0)
    @test_throws ErrorException push!(mc, :conf => ConfigurationMeasurement, :bad_stage)
    @test_throws ErrorException MonteCarlo.unsafe_push!(mc, :conf => ConfigurationMeasurement(mc, model), :bad_stage)
    @test_throws ErrorException delete!(mc, :conf, :bad_stage)
    @test_throws ErrorException delete!(mc, ConfigurationMeasurement, :bad_stage)

    push!(mc, :conf => ConfigurationMeasurement)
    @test haskey(mc.measurements, :conf) && mc.measurements[:conf] isa ConfigurationMeasurement
    delete!(mc, :conf)
    @test !haskey(mc.measurements, :conf)

    MonteCarlo.unsafe_push!(mc, :Magn => IsingMagnetizationMeasurement(mc, model))
    @test haskey(mc.measurements, :Magn) && mc.measurements[:Magn] isa IsingMagnetizationMeasurement
    delete!(mc, IsingMagnetizationMeasurement)
    @test !haskey(mc.measurements, :Magn)

    @test isempty(mc.thermalization_measurements)
    push!(mc, :conf => ConfigurationMeasurement, :TH)
    @test haskey(mc.thermalization_measurements, :conf) &&
        mc.thermalization_measurements[:conf] isa ConfigurationMeasurement
    delete!(mc, :conf, :TH)

    MonteCarlo.unsafe_push!(mc, :conf => ConfigurationMeasurement(mc, model), :TH)
    @test haskey(mc.thermalization_measurements, :conf) &&
        mc.thermalization_measurements[:conf] isa ConfigurationMeasurement
    delete!(mc, ConfigurationMeasurement, :TH)
    @test !haskey(mc.thermalization_measurements, :TH)
end

@testset "Saving and Loading" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)
    run!(mc, thermalization=10, sweeps=10, verbose=false)
    push!(mc, :conf => ConfigurationMeasurement, :TH)

    obs = observables(mc, :all)
    save_measurements!(mc, "test.jld", force_overwrite=true)
    _obs = load_measurements("test.jld")
    @test obs == _obs
    rm("test.jld")
end
