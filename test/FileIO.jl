const p = "temp_dir"
isdir(p) || mkdir(p)

@testset "MC" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=0.66, thermalization=33, sweeps=123)
    run!(mc, verbose=false)
    MonteCarlo.save("$p/testfile.jld", mc)
    x = MonteCarlo.load("$p/testfile.jld")
    rm("$p/testfile.jld")

    # Repeat these tests once with x being replayed rather than loaded
    replay_done = false
    @label all_checks

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
    @test mc.model.neighs == x.model.neighs
    @test mc.model.energy[] == x.model.energy[]
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            @test getfield(v, f) == getfield(x.thermalization_measurements[k], f)
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            @test getfield(v, f) == getfield(x.measurements[k], f)
        end
    end

    # Check everything again with x being a replayed simulation
    if !replay_done
        replay!(x)
        replay_done = true
        @goto all_checks
    end

    # Test resume

    # Run for 1s with known RNG
    rm.(joinpath.(p, readdir(p)))
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(model, beta=1.0, sweeps=10_000_000, measure_rate=1000)
    state = run!(
        mc, verbose = false,
        safe_before = now() + Second(1),
        grace_period = Millisecond(0),
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1 "No measurements have been taken. Test with more time!"
    L = length(ts)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "$p/resumable_testfile.jld",
        verbose = false,
        safe_before = now() + Second(8),
        grace_period = Millisecond(0),
        force_overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) - L > 1 "No new measurements have been taken. Test with more time!"
    @test length(readdir(p)) == 1

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(model, beta=1.0, sweeps=1000length(ts), measure_rate=1000)
    state = run!(mc, verbose = false)
    @test timeseries(mc.measurements[:conf].obs) == ts
end


rm.(joinpath.(p, readdir(p)))

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
    replay_done = false
    @label all_checks

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
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                @test getfield(v, f).B == getfield(x.measurements[k], f).B
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                @test getfield(v, f).B == getfield(x.measurements[k], f).B
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end

    # Check everything again with x being a replayed simulation
    if !replay_done
        replay!(x)
        replay_done = true
        @goto all_checks
    end


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
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1 "No measurements have been taken. Test with more time!"
    L = length(ts)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "$p/resumable_testfile.jld",
        verbose = false,
        safe_before = now() + Second(8),
        grace_period = Millisecond(0),
        force_overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld"
    )

    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) - L > 1 "No new measurements have been taken. Test with more time!"
    @test length(readdir(p)) == 1

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = HubbardModelAttractive(dims=2, L=2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, sweeps=100length(ts), measure_rate=100)
    state = run!(mc, verbose = false)
    @test timeseries(mc.measurements[:conf].obs) == ts
end

isdir(p) && rm(p, recursive=true)
