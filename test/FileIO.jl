const p = "temp_dir"
isdir(p) || mkdir(p)

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

@testset "MC" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=0.66, thermalization=33, sweeps=123, recorder=MonteCarlo.ConfigRecorder)
    run!(mc, verbose=false)
    MonteCarlo.save("$p/testfile.jld2", mc)
    x = MonteCarlo.load("$p/testfile.jld2")
    rm("$p/testfile.jld2")
    test_mc(mc, x)

    x.measurements = MonteCarlo.default_measurements(mc, model) 
    x.last_sweep = 0
    replay!(x, verbose=false)
    test_mc(mc, x)

    # Test resume

    # Run for 1s with known RNG
    rm.(joinpath.(p, readdir(p)))
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(model, beta=1.0, sweeps=10_000_000, measure_rate=10_000, recorder=MonteCarlo.ConfigRecorder)
    state = run!(
        mc, verbose = false,
        safe_before = now() + Second(1),
        grace_period = Millisecond(0),
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "$p/resumable_testfile.jld",
        verbose = false,
        safe_before = now() + Second(12),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test length(readdir(p)) == 1

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(model, beta=1.0, sweeps=10_000length(cs), measure_rate=10_000, recorder=MonteCarlo.ConfigRecorder)
    state = run!(mc, verbose = false)
    @test mc.configs.configs == cs.configs
    @test mc.configs.rate == cs.rate
end


rm.(joinpath.(p, readdir(p)))

function test_dqmc(mc, x)
    for f in fieldnames(typeof(mc.p))
        @test getfield(mc.p, f) == getfield(x.p, f)
    end
    @test mc.conf == x.conf
    @test mc.model.dims == x.model.dims
    @test mc.model.L == x.model.L
    @test mc.model.mu == x.model.mu
    @test mc.model.t == x.model.t
    @test mc.model.U == x.model.U
    for f in fieldnames(typeof(mc.model.l))
        @test getfield(mc.model.l, f) == getfield(x.model.l, f)
    end
    @test mc.model.flv == x.model.flv
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
            else
                getfield(v, f) == getfield(x.measurements[k], f)
            end
            r != true && @info "Check failed for $k -> $f"
            @test r
        end
    end
    nothing
end

@testset "DQMC" begin
    model = HubbardModelAttractive(dims=2, L=4, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=21, sweeps=117, measure_rate = 1)
    t = time()
    run!(mc, verbose=false)
    t = time() - t
    MonteCarlo.save("$p/testfile.jld", mc)
    x = MonteCarlo.load("$p/testfile.jld")
    rm("$p/testfile.jld")

    # Repeat these tests once with x being replayed rather than loaded
    test_dqmc(mc, x)    

    # Check everything again with x being a replayed simulation
    x.measurements = MonteCarlo.default_measurements(mc, model) 
    x.last_sweep = 0
    replay!(x, verbose=false)
    test_dqmc(mc, x)
    

    # Test resume

    # Run for 1s with known RNG
    rm.(joinpath.(p, readdir(p)))
    Random.seed!(123)
    model = HubbardModelAttractive(dims=2, L=2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, sweeps=10_000_000, measure_rate=100)

    state = run!(
        mc, verbose = false,
        safe_before = now() + Second(1),
        grace_period = Millisecond(0),
        resumable_filename = "$p/resumable_testfile.jld2"
    )

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "$p/resumable_testfile.jld2",
        verbose = false,
        safe_before = now() + Second(8),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld2"
    )

    @test state == false
    cs = deepcopy(mc.configs)
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test length(readdir(p)) == 1

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = HubbardModelAttractive(dims=2, L=2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, sweeps=100length(cs), measure_rate=100)
    state = run!(mc, verbose = false)
    @test mc.configs.configs == cs.configs
    @test mc.configs.rate == cs.rate
end

isdir(p) && rm(p, recursive=true)
