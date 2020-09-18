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

@testset "Statistics of measurements" begin
    m = IsingModel(dims=2, L=4);
    mc = MC(m, beta=1.0);
    run!(mc, sweeps=100, thermalization=0, verbose=false);

    ms = measurements(mc)[:Energy]
    obs = observables(mc)[:Energy]

    @test mean(ms)["Total energy"] == mean(obs["Total energy"])
    @test var(ms)["Total energy"] == var(obs["Total energy"])
    @test std_error(ms)["Total energy"] == std_error(obs["Total energy"])
    # This wont work because we're not using LightObservables
    # @test tau(ms)["Total energy"] == tau(obs["Total energy"])
end

@testset "Saving and Loading" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=1.0)
    run!(mc, thermalization=10, sweeps=10, verbose=false)
    push!(mc, :E => IsingEnergyMeasurement, :TH)

    meas = measurements(mc, :all)
    MonteCarlo.save_measurements("testfile.jld", mc, overwrite=true)
    _meas = MonteCarlo.load("testfile.jld")
    for (k, v) in meas
        for (k2, v2) in v
            for f in fieldnames(typeof(v2))
                @test getfield(v2, f) == getfield(_meas[k][k2], f)
            end
        end
    end
    rm("testfile.jld")
end

function calc_measured_greens(mc::DQMC, G::Matrix)
    eThalfminus = mc.s.hopping_matrix_exp
    eThalfplus = mc.s.hopping_matrix_exp_inv

    eThalfplus * G * eThalfminus
end

@testset "Measured Greens function" begin
    m = HubbardModelAttractive(dims=2, L=8, mu=0.5)
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.build_stack(mc)
    MonteCarlo.propagate(mc)

    # greens(mc) matches expected output
    @test greens(mc) ≈ calc_measured_greens(mc, mc.s.greens)

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.s.greens, mc.s.current_slice - k, -1)
    end
    # greens(mc) matches expected output
    @test greens(mc) ≈ calc_measured_greens(mc, mc.s.greens)
end

@testset "Uniform Fourier" begin
    A = rand(64, 64)
    @test uniform_fourier(A, 64) == sum(A) / 64
    @test uniform_fourier(A, 10) == sum(A) / 10

    m = HubbardModelAttractive(dims=2, L=8)
    mc = DQMC(m, beta=5.0)
    @test uniform_fourier(A, mc) == sum(A) / 64

    mask = MonteCarlo.DistanceMask(MonteCarlo.lattice(m))
    MonteCarlo.unsafe_push!(mc, :CDC => MonteCarlo.ChargeDensityCorrelationMeasurement(mc, m, mask=mask))
    MonteCarlo.unsafe_push!(mc, :SDC => MonteCarlo.SpinDensityCorrelationMeasurement(mc, m, mask=mask))
    MonteCarlo.unsafe_push!(mc, :PC => MonteCarlo.PairingCorrelationMeasurement(mc, m, mask=mask))
    run!(mc, verbose=false)
    measured = measurements(mc)

    @test uniform_fourier(measured[:CDC]) isa MonteCarlo.UniformFourierWrapped
    @test_throws MethodError uniform_fourier(measured[:SDC])
    @test uniform_fourier(measured[:SDC], :x) isa MonteCarlo.UniformFourierWrapped
    @test uniform_fourier(measured[:SDC].y) isa MonteCarlo.UniformFourierWrapped
    @test uniform_fourier(measured[:PC]) isa MonteCarlo.UniformFourierWrapped

    @test mean(uniform_fourier(measured[:CDC])) == sum(mean(measured[:CDC])) / 64
    @test var(uniform_fourier(measured[:SDC], :x)) == sum(var(measured[:SDC].x)) / 64
    @test std_error(uniform_fourier(measured[:SDC].z)) == sum(std_error(measured[:SDC].z)) / 64
    @test tau(uniform_fourier(measured[:PC])) == maximum(tau(measured[:PC]))
end
