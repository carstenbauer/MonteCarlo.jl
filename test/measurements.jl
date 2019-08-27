let
    AbstractMeasurement = MonteCarlo.AbstractMeasurement
    ConfigurationMeasurement = MonteCarlo.ConfigurationMeasurement

    IsingMeasurement = MonteCarlo.IsingMeasurement
    IsingEnergyMeasurement = MonteCarlo.IsingEnergyMeasurement
    IsingMagnetizationMeasurement = MonteCarlo.IsingMagnetizationMeasurement

    HubbardMeasurement = MonteCarlo.HubbardMeasurement
    GreensMeasurement = MonteCarlo.GreensMeasurement
    BosonEnergyMeasurement = MonteCarlo.BosonEnergyMeasurement

    AbstractObservable = MonteCarloObservable.AbstractObservable

    @testset "Type Hierarchy" begin
        @test IsingEnergyMeasurement <: IsingMeasurement
        @test IsingMagnetizationMeasurement <: IsingMeasurement
        @test IsingMeasurement <: AbstractMeasurement

        @test GreensMeasurement <: HubbardMeasurement
        @test BosonEnergyMeasurement <: HubbardMeasurement
        @test HubbardMeasurement <: AbstractMeasurement
    end

    @testset "Checking defaults" begin
        model = IsingModel(dims=2, L=2)
        mc = MC(model, beta=1.0)

        defaults = MonteCarlo.default_measurements(mc, model)
        @test !isempty(defaults)
        @test haskey(defaults, :conf) && defaults[:conf] isa ConfigurationMeasurement
        @test haskey(defaults, :Magn) && defaults[:Magn] isa IsingMagnetizationMeasurement
        @test haskey(defaults, :Energy) && defaults[:Energy] isa IsingEnergyMeasurement

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
        @test haskey(defaults, :BosonEnergy) && defaults[:BosonEnergy] isa BosonEnergyMeasurement

        @test isempty(mc.thermalization_measurements)
        @test !isempty(mc.measurements)
        @test haskey(mc.measurements, :conf) && mc.measurements[:conf] isa ConfigurationMeasurement
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
        # TODO doesn"t work?
        @test_throws ErrorException push!(mc, :conf => Int64)
        @test_throws MethodError MonteCarlo.unsafe_push!(mc, :conf => 1.0)

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
        @test !haskey(mc.thermalization_measurements, :TH)
    end
end
