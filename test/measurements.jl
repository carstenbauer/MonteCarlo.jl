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

        fi = if MonteCarlo.unique_flavors(mc) == 2
            [(1, 1), (1, 2), (2, 1), (2, 2)]
        else
            [(1, 1),]
        end

        # Greens
        m = greens_measurement(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator === nothing
        @test m.flavor_iterator === nothing
        @test m.kernel == MonteCarlo.greens_kernel
        @test m.observable isa LogBinner{Matrix{Float64}}
        @test m.temp === nothing

        # Occupation
        m = occupation(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator === nothing
        @test m.flavor_iterator === nothing
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
            @test m.flavor_iterator == fi
            @test m.kernel == MonteCarlo.full_cdc_kernel
            @test m.observable isa LogBinner{Array{Float64, 3}}
            @test m.temp isa Array{Float64, 3}

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
                @test m.kernel == Core.eval(MonteCarlo, Symbol(:full_sdc_, dir, :_kernel))
                @test m.observable isa LogBinner{Array{Float64, 3}}
                @test m.temp isa Array{Float64, 3}
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
            @test m.flavor_iterator == 2
            @test m.kernel == MonteCarlo.pc_combined_kernel
            @test m.observable isa LogBinner{Array{Float64, 5}}
            @test m.temp isa Array{Float64, 5}
        end

        # Magnetizations
        for dir in (:x, :y, :z)
            m = magnetization(mc, m1, dir)
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.greens_iterator == Greens()
            @test m.lattice_iterator == EachSite()
            @test m.flavor_iterator == 2
            @test m.kernel == Core.eval(MonteCarlo, Symbol(:m, dir, :_kernel))
            @test m.observable isa LogBinner{Vector{Float64}}
            @test m.temp isa Vector{Float64}
        end

        # Current Current susceptibility
        m = current_current_susceptibility(mc, m1, lattice_iterator = EachLocalQuadBySyncedDistance(2:5))
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == TimeIntegral(mc)
        @test m.lattice_iterator == EachLocalQuadBySyncedDistance(2:5)
        @test m.flavor_iterator == fi
        @test m.kernel == MonteCarlo.cc_kernel
        @test m.observable isa LogBinner{Array{Float64, 4}}
        @test m.temp isa Array{Float64, 4}
    end

    m = HubbardModel(4, 2)
    mc = DQMC(m1, beta=1.0, safe_mult=1)
    add_default_measurements!(mc)

    @test !haskey(mc, :occ) # skipped with :G
    @test !haskey(mc, :Mx) # skipped with :G
    @test !haskey(mc, :My) # skipped with :G
    @test !haskey(mc, :Mz) # skipped with :G
    @test haskey(mc, :G) && (mc[:G].kernel == MonteCarlo.greens_kernel)
    @test haskey(mc, :K) && (mc[:K].kernel == MonteCarlo.kinetic_energy_kernel)
    @test haskey(mc, :V) && (mc[:V].kernel == MonteCarlo.interaction_energy_kernel)
    @test !haskey(mc, :E) # skipped with :K, :V

    @test haskey(mc, :CDC)  && (mc[:CDC].kernel == MonteCarlo.full_cdc_kernel)
    @test haskey(mc, :PC)   && (mc[:PC].kernel == MonteCarlo.pc_kernel)
    @test haskey(mc, :SDCx) && (mc[:SDCx].kernel == MonteCarlo.full_sdc_x_kernel)
    @test haskey(mc, :SDCy) && (mc[:SDCy].kernel == MonteCarlo.full_sdc_y_kernel)
    @test haskey(mc, :SDCz) && (mc[:SDCz].kernel == MonteCarlo.full_sdc_z_kernel)

    @test haskey(mc, :CDS)  && (mc[:CDS].kernel == MonteCarlo.full_cdc_kernel)
    @test haskey(mc, :PS)   && (mc[:PS].kernel == MonteCarlo.pc_kernel)
    @test haskey(mc, :SDSx) && (mc[:SDSx].kernel == MonteCarlo.full_sdc_x_kernel)
    @test haskey(mc, :SDSy) && (mc[:SDSy].kernel == MonteCarlo.full_sdc_y_kernel)
    @test haskey(mc, :SDSz) && (mc[:SDSz].kernel == MonteCarlo.full_sdc_z_kernel)
    @test haskey(mc, :CCS)  && (mc[:CCS].kernel == MonteCarlo.cc_kernel)
end