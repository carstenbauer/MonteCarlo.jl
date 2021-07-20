function test_mc(mc, x)
    # Check if loaded/replayed mc matches original
    for f in fieldnames(typeof(mc.p))
        @test getfield(mc.p, f) == getfield(x.p, f)
    end
    @test mc.conf == x.conf
    @test mc.model.L == x.model.L
    @test mc.model.dims == x.model.dims
    for f in fieldnames(typeof(mc.model.l))
        @test getfield(mc.model.l, f) == getfield(x.model.l, f)
    end
    # @test mc.model.neighs == x.model.neighs
    @test mc.model.energy[] == x.model.energy[]
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            r = getfield(v, f) == getfield(x.thermalization_measurements[k], f)
            r != true && @info "Check failed for $k -> $f"
            @test r
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            r = getfield(v, f) == getfield(x.measurements[k], f)
            r != true && @info "Check failed for $k -> $f"
            @test r
        end
    end
    nothing
end

println("MC")
@time @testset "MC" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=0.66, thermalization=33, sweeps=123, recorder=ConfigRecorder)
    @time run!(mc, verbose=false)
    save("testfile.jld2", mc)
    x = load("testfile.jld2")
    rm("testfile.jld2")
    test_mc(mc, x)

    x.measurements = MonteCarlo.default_measurements(mc, model) 
    x.last_sweep = 0
    @time replay!(x, verbose=false)
    test_mc(mc, x)

    # Test resume

    # Run for 1s with known RNG
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(
        model, beta=1.0, 
        thermalization = 0, sweeps=10_000_000, 
        measure_rate=10_000, recorder=ConfigRecorder
    )
    @time state = run!(
        mc, verbose = false,
        safe_before = now() + Second(1),
        grace_period = Millisecond(0),
        resumable_filename = "resumable_testfile.jld"
    )

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    @info mc.last_sweep
    @time mc, state = resume!(
        "resumable_testfile.jld",
        verbose = false,
        safe_before = now() + Second(10),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "resumable_testfile.jld"
    )
    @info mc.last_sweep

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test isfile("resumable_testfile.jld")

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(
        model, beta=1.0, 
        thermalization=0, sweeps=10_000length(cs), 
        measure_rate=10_000, recorder=ConfigRecorder
    )
    @time state = run!(mc, verbose = false)
    @test mc.configs.configs == cs.configs
    @test mc.configs.rate == cs.rate
    rm("resumable_testfile.jld")
end


function test_dqmc(mc, x)
    for f in fieldnames(typeof(mc.parameters))
        @test getfield(mc.parameters, f) == getfield(x.parameters, f)
    end
    # @test mc.conf == x.conf
    @test mc.model.mu == x.model.mu
    @test mc.model.t == x.model.t
    @test mc.model.U == x.model.U
    for f in fieldnames(typeof(mc.model.l))
        @test getfield(mc.model.l, f) == getfield(x.model.l, f)
    end
    @test mc.model.flv == x.model.flv
    @test mc.scheduler == x.scheduler
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            r = if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                getfield(v, f).B == getfield(x.measurements[k], f).B
            else
                getfield(v, f) == getfield(x.measurements[k], f)
            end
            r != true && @info "Check failed for $k -> $f"
            @test r
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            v isa MonteCarlo.DQMCMeasurement && f == :temp && continue
            v isa MonteCarlo.DQMCMeasurement && f == :kernel && continue
            r = if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                # TODO: implement ≈ for LightObservable, LogBinner, etc
                r = true
                a = getfield(v, f)
                b = getfield(x.measurements[k], f)
                for i in eachindex(getfield(v, f).B.compressors)
                    r = r && (a.B.compressors[i].value ≈ b.B.compressors[i].value)
                    r = r && (a.B.compressors[i].switch ≈ b.B.compressors[i].switch)
                end
                r = r && (a.B.x_sum ≈ b.B.x_sum)
                r = r && (a.B.x2_sum ≈ b.B.x2_sum)
                r = r && (a.B.count ≈ b.B.count)
            elseif getfield(v, f) isa LogBinner
                r = true
                a = getfield(v, f)
                b = getfield(x.measurements[k], f)
                for i in eachindex(a.compressors)
                    r = r && (a.compressors[i].value ≈ b.compressors[i].value)
                    r = r && (a.compressors[i].switch ≈ b.compressors[i].switch)
                end
                r = r && (a.x_sum ≈ b.x_sum)
                r = r && (a.x2_sum ≈ b.x2_sum)
                r = r && (a.count ≈ b.count)
            else
                getfield(v, f) == getfield(x.measurements[k], f)
            end
            r != true && @info "Check failed for $k -> $f"
            @test r
        end
    end
    nothing
end

for file in readdir()
    (endswith(file, "jld") || endswith(file, "jld2")) && rm(file)
end

println("DQMC")
@time @testset "DQMC" begin
    model = HubbardModelAttractive(4, 2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=21, sweeps=117, measure_rate = 1)
    mc[:CDC] = charge_density_correlation(mc, model)
    t = time()
    run!(mc, verbose=false)
    t = time() - t
    @info t

    save("testfile.jld", mc)
    x = load("testfile.jld")
    rm("testfile.jld")
    @test mc.conf == x.conf

    # Repeat these tests once with x being replayed rather than loaded
    test_dqmc(mc, x)    

    # Check everything again with x being a replayed simulation
    x[:CDC] = charge_density_correlation(x, model)
    x.last_sweep = 0
    @time replay!(x, verbose=false)
    test_dqmc(mc, x)
    

    # Test resume

    # Run for 1s with known RNG
    Random.seed!(123)
    model = HubbardModelAttractive(2, 2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=0, sweeps=10_000_000, measure_rate=100)
    mc[:CDC] = charge_density_correlation(mc, model)

    @time state = run!(
        mc, verbose = false,
        safe_before = now() + Second(2),
        grace_period = Millisecond(0),
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == false
    cs = deepcopy(mc.recorder)
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    @time mc, state = resume!(
        "resumable_testfile.jld2",
        verbose = false,
        safe_before = now() + Second(10),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == false
    cs = deepcopy(mc.recorder)
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test isfile("resumable_testfile.jld2")

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = HubbardModelAttractive(2, 2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=0, sweeps=100length(cs), measure_rate=100)
    @time state = run!(mc, verbose = false)
    @test mc.recorder.configs == cs.configs
    @test mc.recorder.rate == cs.rate
    rm("resumable_testfile.jld2")
end