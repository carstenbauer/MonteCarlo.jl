IsingMeasurement = MonteCarlo.IsingMeasurement

struct DummyMeasurement <: AbstractMeasurement end

@testset "Type Hierarchy" begin
    @test IsingEnergyMeasurement <: IsingMeasurement
    @test IsingMagnetizationMeasurement <: IsingMeasurement
    @test IsingMeasurement <: AbstractMeasurement
    @test MonteCarlo.DQMCMeasurement <: AbstractMeasurement
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
    @test haskey(obs, "m")   && obs["m"]   isa FullBinner
    @test haskey(obs, "chi") && obs["chi"] isa FullBinner
    @test haskey(obs, "M")   && obs["M"]   isa FullBinner
    @test haskey(obs, "M2")  && obs["M2"]  isa FullBinner

    @test haskey(defaults, :Energy) && defaults[:Energy] isa IsingEnergyMeasurement
    obs = observables(defaults[:Energy])
    @test haskey(obs, "e")  && obs["e"] isa FullBinner
    @test haskey(obs, "C")  && obs["C"] isa FullBinner
    @test haskey(obs, "E")  && obs["E"] isa FullBinner
    @test haskey(obs, "E2") && obs["E2"] isa FullBinner


    @test isempty(mc.thermalization_measurements)
    @test !isempty(mc.measurements)
    @test haskey(mc.measurements, :Magn) && mc.measurements[:Magn] isa IsingMagnetizationMeasurement
    @test haskey(mc.measurements, :Energy) && mc.measurements[:Energy] isa IsingEnergyMeasurement

    model = HubbardModel(2, 2)
    mc = DQMC(model, beta=1.0)

    @test isempty(mc.measurements)
    @test isempty(mc.thermalization_measurements)
end

@testset "Retrieving Measurements" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)

    @test mc.thermalization_measurements == measurements(mc, :TH)
    @test mc.measurements == measurements(mc)

    obs = observables(mc, :all)
    @test keys(obs[:TH]) == keys(measurements(mc, :TH))
    @test keys(obs[:ME]) == keys(measurements(mc))

    @test haskey(obs[:ME][:Energy], "E")
    @test typeof(obs[:ME][:Energy]["E"]) <: FullBinner
    @test haskey(obs[:ME][:Energy], "E2")
    @test typeof(obs[:ME][:Energy]["E2"]) <: FullBinner
    @test haskey(obs[:ME][:Energy], "e")
    @test typeof(obs[:ME][:Energy]["e"]) <: FullBinner
    @test haskey(obs[:ME][:Energy], "C")
    @test typeof(obs[:ME][:Energy]["C"]) <: FullBinner

    @test haskey(obs[:ME][:Magn], "M")
    @test typeof(obs[:ME][:Magn]["M"]) <: FullBinner
    @test haskey(obs[:ME][:Magn], "M2")
    @test typeof(obs[:ME][:Magn]["M2"]) <: FullBinner
    @test haskey(obs[:ME][:Magn], "m")
    @test typeof(obs[:ME][:Magn]["m"]) <: FullBinner
    @test haskey(obs[:ME][:Magn], "chi")
    @test typeof(obs[:ME][:Magn]["chi"]) <: FullBinner
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

@testset "Statistics of measurements" begin
    m = IsingModel(dims=2, L=4);
    mc = MC(m, beta=1.0);
    run!(mc, sweeps=100, thermalization=0, verbose=false);

    ms = measurements(mc)[:Energy]
    obs = observables(mc)[:Energy]

    @test mean(ms)["E"] == mean(obs["E"])
    @test var(ms)["E"] == var(obs["E"])
    @test std_error(ms)["E"] == std_error(obs["E"])
    # This wont work because we're not using LightObservables
    # @test tau(ms)["E"] == tau(obs["E"])
end