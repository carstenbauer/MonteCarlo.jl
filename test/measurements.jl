IsingMeasurement = MonteCarlo.IsingMeasurement
AbstractObservable = MonteCarloObservable.AbstractObservable

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
    _meas = load("testfile.jld")
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
    eThalfminus = mc.stack.hopping_matrix_exp
    eThalfplus = mc.stack.hopping_matrix_exp_inv

    eThalfplus * G * eThalfminus
end

@testset "Measured Greens function" begin
    m = HubbardModel(8, 2, mu=0.5)
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.init!(mc)
    MonteCarlo.build_stack(mc, mc.stack)
    MonteCarlo.propagate(mc)

    # greens(mc) matches expected output
    @test greens(mc).val ≈ calc_measured_greens(mc, mc.stack.greens)

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice - k, -1)
    end
    # greens(mc) matches expected output
    @test greens(mc).val ≈ calc_measured_greens(mc, mc.stack.greens)
end

@testset "DQMC Measurement constructors" begin
    for m1 in (HubbardModel(4, 2), HubbardModel(4, 2, U = -1.0))
        mc = DQMC(m1, beta=1.0, safe_mult=1)

        # Greens
        m = greens_measurement(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator === nothing
        @test m.kernel == MonteCarlo.greens_kernel
        @test m.observable isa LogBinner{Matrix{Float64}}
        @test m.temp === nothing

        # Occupation
        m = occupation(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator == EachSiteAndFlavor()
        @test m.kernel == MonteCarlo.occupation_kernel
        @test m.observable isa LogBinner{Vector{Float64}}
        @test m.temp isa Vector{Float64}

        for time in (:equal, :unequal)
            # Charge densities
            if time == :equal
                m = charge_density_correlation(mc, m1)
                @test m.greens_iterator == Greens()
            else
                m = charge_density_susceptibility(mc, m1)
                @test m.greens_iterator == TimeIntegral(mc)
            end
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.lattice_iterator == EachSitePairByDistance()
            @test m.kernel == MonteCarlo.cdc_kernel
            @test m.observable isa LogBinner{Vector{Float64}}
            @test m.temp isa Vector{Float64}

            # Spin densities
            for dir in (:x, :y, :z)
                if time == :equal
                    m = spin_density_correlation(mc, m1, dir)
                    @test m.greens_iterator == Greens()
                else
                    m = spin_density_susceptibility(mc, m1, dir)
                    @test m.greens_iterator == TimeIntegral(mc)
                end
                @test m isa MonteCarlo.DQMCMeasurement
                @test m.lattice_iterator == EachSitePairByDistance()
                @test m.kernel == Core.eval(MonteCarlo, Symbol(:sdc_, dir, :_kernel))
                @test m.observable isa LogBinner{Vector{Float64}}
                @test m.temp isa Vector{Float64}
            end

            # pairings
            if time == :equal
                m = pairing_correlation(mc, m1)
                @test m.greens_iterator == Greens()
            else
                m = pairing_susceptibility(mc, m1)
                @test m.greens_iterator == TimeIntegral(mc)
            end
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.lattice_iterator == EachLocalQuadByDistance(1:5)
            @test m.kernel == MonteCarlo.pc_combined_kernel
            @test m.observable isa LogBinner{Array{Float64, 3}}
            @test m.temp isa Array{Float64, 3}
        end

        # Magnetizations
        for dir in (:x, :y, :z)
            m = magnetization(mc, m1, dir)
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.greens_iterator == Greens()
            @test m.lattice_iterator == EachSite()
            @test m.kernel == Core.eval(MonteCarlo, Symbol(:m, dir, :_kernel))
            @test m.observable isa LogBinner{Vector{Float64}}
            @test m.temp isa Vector{Float64}
        end

        # Current Current susceptibility
        m = current_current_susceptibility(mc, m1, lattice_iterator = EachLocalQuadBySyncedDistance(2:5))
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == TimeIntegral(mc)
        @test m.lattice_iterator == EachLocalQuadBySyncedDistance(2:5)
        @test m.kernel == MonteCarlo.cc_kernel
        @test m.observable isa LogBinner{Matrix{Float64}}
        @test m.temp isa Matrix{Float64}
    end



    # m = 
    # @test m isa MonteCarlo.DQMCMeasurement
    # @test m.greens_iterator == 
    # @test m.lattice_iterator == 
    # @test m.kernel == MonteCarlo.
    # @test m.observable isa LogBinner{Vector{Float64}}
    # @test m.temp isa 
end

# @testset "Uniform Fourier" begin
#     A = rand(64, 64)
#     @test uniform_fourier(A, 64) == sum(A) / 64
#     @test uniform_fourier(A, 10) == sum(A) / 10

#     m = HubbardModel(8, 2)
#     mc = DQMC(m, beta=5.0)
#     @test uniform_fourier(A, mc) == sum(A) / 64

#     mask = MonteCarlo.DistanceMask(MonteCarlo.lattice(m))
#     MonteCarlo.unsafe_push!(mc, :CDC => ChargeDensityCorrelationMeasurement(mc, m, mask=mask))
#     MonteCarlo.unsafe_push!(mc, :SDC => SpinDensityCorrelationMeasurement(mc, m, mask=mask))
#     MonteCarlo.unsafe_push!(mc, :PC => PairingCorrelationMeasurement(mc, m, mask=mask))
#     run!(mc, verbose=false)
#     measured = measurements(mc)

#     @test uniform_fourier(measured[:CDC]) isa MonteCarlo.UniformFourierWrapped
#     @test_throws MethodError uniform_fourier(measured[:SDC])
#     @test uniform_fourier(measured[:SDC], :x) isa MonteCarlo.UniformFourierWrapped
#     @test uniform_fourier(measured[:SDC].y) isa MonteCarlo.UniformFourierWrapped
#     @test uniform_fourier(measured[:PC]) isa MonteCarlo.UniformFourierWrapped

#     @test mean(uniform_fourier(measured[:CDC])) == sum(mean(measured[:CDC])) / 64
#     @test var(uniform_fourier(measured[:SDC], :x)) == sum(var(measured[:SDC].x)) / 64
#     @test std_error(uniform_fourier(measured[:SDC].z)) == sum(std_error(measured[:SDC].z)) / 64
#     @test tau(uniform_fourier(measured[:PC])) == maximum(tau(measured[:PC]))
# end
