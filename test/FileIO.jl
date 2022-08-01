@time @testset "BufferedConfigRecorder" begin
    @testset "Runtime File IO" begin
        isfile("testfile.confs") && rm("testfile.confs")
        r = MonteCarlo.BufferedConfigRecorder{Matrix{Float64}}("testfile.confs", 2, 20)
    
        # Initialization
        @test r.filename == MonteCarlo.FilePath(false, "testfile.confs", "testfile.confs")
        @test length(r.buffer) == 20
        @test eltype(r.buffer) == Matrix{Float64}
        @test r.rate == 2
        @test r.idx == 1
        @test r.chunk == 1
        @test r.total_length == 0
        @test r.save_idx == -1
        @test !isdefined(r.buffer, 1)

        # Basic utility functions
        @test length(r) == r.total_length
        @test lastindex(r) == r.total_length
        @test isempty(r)

        # push to empty memory buffer
        c = rand(4, 4)
        MonteCarlo._push!(r, c)

        @test r.idx == 2
        @test r.chunk == 1
        @test r.total_length == 1
        @test r.save_idx == -1
        @test r.buffer[1] == c
        @test !isdefined(r.buffer, 2)

        @test length(r) == r.total_length
        @test lastindex(r) == r.total_length
        @test !isempty(r)
        @test r[1] == c

        # fill memory buffer - no save yet
        for _ in 2:20
            MonteCarlo._push!(r, c)
        end

        @test r.idx == 21
        @test r.chunk == 1
        @test r.total_length == 20
        @test r.save_idx == -1
        @test r.buffer[20] == c
        @test r[20] == c

        # overflow push - should save and reset buffer
        c2 = rand(4, 4)
        MonteCarlo._push!(r, c2)

        @test r.idx == 2
        @test r.chunk == 2
        @test r.total_length == 21
        @test r.save_idx == 20
        @test r.buffer[1] == c2
        @test r[21] == c2

        # query last chunk - should load and mark idx for reload
        @test r[20] == c
        @test r[1] == c
        @test r.idx == -1
        @test r.chunk == 1
        @test r.total_length == 21
        @test r.save_idx == 21

        # push to wrong chunk - should load correct chunk
        c3 = rand(4, 4)
        MonteCarlo._push!(r, c3)

        @test r.idx == 3
        @test r.chunk == 2
        @test r.total_length == 22
        @test r.save_idx == 21
        @test r.buffer[1] == c2 # this would fail if chunk not loaded
        @test r.buffer[2] == c3
        @test r.buffer[3] == c

        # test merge!
        c4 = rand(4, 4)
        cs = ConfigRecorder{Matrix{Float64}}([c4 for _ in 1:17], 10)
        merge!(r, cs)

        @test r.idx == 20
        @test r.chunk == 2
        @test r.total_length == 39
        @test r.save_idx == 21
        @test r.buffer[1] == c2   # old
        @test r.buffer[2] == c3   # old
        @test r.buffer[3] == c4   # new
        @test r.buffer[19] == c4  # new
        @test r.buffer[20] == c   # to be overwritten
        rm("testfile.confs")
    end

    dir1 = randstring(16)
    dir2 = randstring(16)
    mkdir(dir1)
    mkdir(dir2)

    @testset "Parent related file IO (moving, renaming, replacing)" begin
        m = HubbardModel(2, 2, U = 1.0)
        recorder = BufferedConfigRecorder(DensityHirschField, RelativePath("testfile.confs"))
        link_id = recorder.link_id
        mc = DQMC(m, beta=1.0, recorder = recorder)
        mc.field.conf .= rand(mc.field)
        push!(recorder, mc.field, 0)

        # BCR should not save on construction
        @test recorder.filename.absolute_path == joinpath(pwd(), "testfile.confs")
        @test !isfile("testfile.confs")

        # parent save should trigger save
        MonteCarlo.save("$dir1/testfile.jld2", mc)
        @test isfile("$dir1/testfile.confs")
        @test isfile("$dir1/testfile.jld2")
        @test !isfile("testfile.confs")

        # both files should have same link_id
        MonteCarlo.JLD2.jldopen("$dir1/testfile.confs", "r") do file
            @test file["link_id"] == link_id
        end
        MonteCarlo.JLD2.jldopen("$dir1/testfile.jld2", "r") do file
            @test file["MC"]["configs"]["link_id"] == link_id
        end

        # check if data loaded is correct
        x = MonteCarlo.load("$dir1/testfile.jld2")
        @test x.recorder.filename == recorder.filename
        @test x.recorder.link_id == link_id
        @test recorder[1] == x.recorder[1]

        function move_load_check(recorder, filename_should_match, source_exists = false)
            mv("$dir1/testfile.jld2", "$dir2/testfile.jld2")
            x = MonteCarlo.load("$dir2/testfile.jld2")
            @test filename_should_match == (x.recorder.filename.relative_path == recorder.filename.relative_path)
            @test x.recorder.filename.absolute_path == joinpath(pwd(), dir2, x.recorder.filename.relative_path)
            @test x.recorder.link_id == link_id
            @test recorder[1] == x.recorder[1]
            filename = x.recorder.filename.relative_path
            @test isfile("$dir2/$filename")
            @test source_exists == isfile("$dir1/testfile.confs")
            @test length(readdir(dir2)) == 3 - filename_should_match
            mv("$dir2/testfile.jld2", "$dir1/testfile.jld2", force = true)
            mv("$dir2/$filename", "$dir1/testfile.confs", force = true)
            nothing
        end

        # Move parent only - recorder should move on load
        move_load_check(recorder, true)

        # Move both - recorder should adjust filename
        mv("$dir1/testfile.confs", "$dir2/testfile.confs")
        move_load_check(recorder, true)

        # Move parent, dublicate BCS file
        # file should be recognized as the same and both should stay
        cp("$dir1/testfile.confs", "$dir2/testfile.confs")
        move_load_check(recorder, true, true)

        # Move parent, create collision
        # file should be renamed
        open(f -> write(f, "Test"), "$dir2/testfile.confs", "w")
        try 
            move_load_check(recorder, false)
        catch e
            @warn "Windows please" exception = e
        end

        # save uses the same function to move/replace/rename so it's not really
        # necessary to test this specifically
    end

    try
        rm(dir1, recursive = true, force = true)
        rm(dir2, recursive = true, force = true)
    catch e
        @warn "Windows please" exception = e
    end
end


@testset "BinningAnalysis" begin
    import JLD2
    using BinningAnalysis
    isfile("testfile.jld2") && rm("testfile.jld2")

    # stable Variance
    LB = LogBinner(rand(1000), accumulator = BinningAnalysis.Variance)
    file = JLD2.jldopen("testfile.jld2", "w")
    MonteCarlo._save(file, "binner", LB)
    close(file)

    loaded = MonteCarlo._load(MonteCarlo.FileData(JLD2.load("testfile.jld2"), "")["binner"])
    rm("testfile.jld2")
    @test LB == loaded

    # fast variance
    LB = LogBinner(rand(1000), accumulator = BinningAnalysis.FastVariance)
    file = JLD2.jldopen("testfile.jld2", "w")
    MonteCarlo._save(file, "binner", LB)
    close(file)

    loaded = MonteCarlo._load(MonteCarlo.FileData(JLD2.load("testfile.jld2"), "")["binner"])
    rm("testfile.jld2")
    @test LB == loaded

    # FullBinner
    FB = FullBinner(rand(1000))
    file = JLD2.jldopen("testfile.jld2", "w")
    MonteCarlo._save(file, "binner", FB)
    close(file)

    loaded = MonteCarlo._load(MonteCarlo.FileData(JLD2.load("testfile.jld2"), "")["binner"])
    rm("testfile.jld2")
    @test FB == loaded
end



@time @testset "MC" begin

    function test_mc(mc, x)
        # Check if loaded/replayed mc matches original
        for f in fieldnames(typeof(mc.p))
            @test getfield(mc.p, f) == getfield(x.p, f)
        end
        @test mc.conf == x.conf
        @test mc.model.L == x.model.L
        @test mc.model.dims == x.model.dims
        @test mc.model.l == x.model.l
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

    model = IsingModel(dims=2, L=2)
    mc = MC(
        model, beta = 0.66, thermalization = 33, sweeps = 123, 
        recorder = ConfigRecorder(MC, IsingModel, 1)
    )
    run!(mc, verbose=false)
    save("testfile.jld2", mc)
    x = load("testfile.jld2")
    rm("testfile.jld2")
    test_mc(mc, x)

    x.measurements = MonteCarlo.default_measurements(mc, model) 
    x.last_sweep = 0
    replay!(x, verbose=false)
    test_mc(mc, x)

    # Test resume

    # Run for 1s with known RNG
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(
        model, beta=1.0, thermalization = 0, sweeps=10_000_000, 
        measure_rate=10_000, recorder=ConfigRecorder(MC, IsingModel, 10_000)
    )
    state = run!(
        mc, verbose = false,
        safe_before = now() + Second(2),
        grace_period = Millisecond(0),
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == MonteCarlo.CANCELLED_TIME_LIMIT
    cs = deepcopy(mc.configs)
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "resumable_testfile.jld2",
        verbose = false,
        safe_before = now() + Second(10),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == MonteCarlo.CANCELLED_TIME_LIMIT
    cs = deepcopy(mc.configs)
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test isfile("resumable_testfile.jld2")

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = IsingModel(dims=2, L=10)
    mc = MC(
        model, beta=1.0, 
        thermalization=0, sweeps=10_000length(cs), 
        measure_rate=10_000, recorder=ConfigRecorder(MC, IsingModel, 10_000)
    )
    state = run!(mc, verbose = false)
    @test state = MonteCarlo.SUCCESS
    @test mc.configs.configs == cs.configs
    @test mc.configs.rate == cs.rate
    rm("resumable_testfile.jld2")
end

for file in readdir()
    if endswith(file, "jld2") || endswith(file, ".confs")
        rm(file)
    end
end



@time @testset "DQMC" begin

    function test_dqmc(mc, x)
        for f in fieldnames(typeof(mc.parameters))
            @test getfield(mc.parameters, f) == getfield(x.parameters, f)
        end
        @test mc.field.conf == x.field.conf
        @test mc.model.mu == x.model.mu
        @test mc.model.t == x.model.t
        @test mc.model.U == x.model.U
        @test mc.model.l == x.model.l
        @test MonteCarlo.nflavors(mc.field) == MonteCarlo.nflavors(x.field)
        @test mc.scheduler == x.scheduler
        for (k, v) in mc.thermalization_measurements
            for f in fieldnames(typeof(v))
                r = if f == :observable
                    getfield(v, f) ≈ getfield(x.measurements[k], f)
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
                r = if f == :observable
                    getfield(v, f) ≈ getfield(x.measurements[k], f)
                else
                    getfield(v, f) == getfield(x.measurements[k], f)
                end
                r != true && @info "Check failed for $k -> $f"
                @test r
            end
        end
        nothing
    end

    isfile("testfile.confs") && rm("testfile.confs")
    model = HubbardModel(4, 2, t = 1.7, U = 2.5)
    mc = DQMC(
        model, beta = 1.0, thermalization = 21, sweeps = 117, measure_rate = 1, 
        recorder = BufferedConfigRecorder(DensityHirschField, "testfile.confs", rate = 1)
    )
    mc[:CDC] = charge_density_correlation(mc, model)
    run!(mc, verbose=false)

    save("testfile.jld2", mc)
    x = load("testfile.jld2")
    rm("testfile.jld2")
    @test mc.field.conf == x.field.conf

    # Repeat these tests once with x being replayed rather than loaded
    test_dqmc(mc, x)    

    # Check everything again with x being a replayed simulation
    x[:CDC] = charge_density_correlation(x, model)
    x.last_sweep = 0
    replay!(x, verbose=false)
    test_dqmc(mc, x)
    isfile("testfile.confs") && rm("testfile.confs")
    

    # Test resume

    # Run for 1s with known RNG
    Random.seed!(123)
    model = HubbardModel(2, 2, t = 1.7, U = 2.5)
    mc = DQMC(
        model, beta = 1.0, thermalization = 0, sweeps = 10_000_000, measure_rate = 100,
        recorder = BufferedConfigRecorder(DensityHirschField, "testfile.confs", rate = 100)
    )
    mc[:CDC] = charge_density_correlation(mc, model)

    state = run!(
        mc, verbose = false,
        safe_before = now() + Second(2),
        grace_period = Millisecond(0),
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == MonteCarlo.CANCELLED_TIME_LIMIT
    cs = [mc.recorder[i] for i in 1:length(mc.recorder)]
    @assert length(cs) > 1 "No measurements have been taken. Test with more time!"
    L = length(cs)

    # Test whether safe file gets overwritten correctly
    mc, state = resume!(
        "resumable_testfile.jld2",
        verbose = false,
        safe_before = now() + Second(10),
        grace_period = Millisecond(0),
        overwrite = true,
        resumable_filename = "resumable_testfile.jld2"
    )

    @test state == MonteCarlo.CANCELLED_TIME_LIMIT
    cs = [mc.recorder[i] for i in 1:length(mc.recorder)]
    @assert length(cs) - L > 1 "No new measurements have been taken. Test with more time!"
    @test isfile("resumable_testfile.jld2")
    isfile("testfile.confs") && rm("testfile.confs")

    # Test whether data from resumed simulation is correct
    Random.seed!(123)
    model = HubbardModel(2, 2, t = 1.7, U = 2.5)
    mc = DQMC(
        model, beta = 1.0, thermalization = 0, sweeps = 100length(cs), measure_rate = 100,
        recorder = BufferedConfigRecorder(DensityHirschField, "testfile.confs", rate = 100)
    )
    state = run!(mc, verbose = false)
    matches = true
    for i in 1:length(mc.recorder)
        matches = matches && (mc.recorder[i] == cs[i])
    end
    @test matches
    @test MonteCarlo.SUCCESS
    rm("resumable_testfile.jld2")
    isfile("testfile.confs") && rm("testfile.confs")
end


function is_file_content_equal(file1, file2)
    data1 = open(readavailable, file1, "r")
    data2 = open(readavailable, file2, "r")
    if length(data1) != length(data2)
        return false
    end
    return all((a == b for (a, b) in zip(data1, data2)))
end

@testset "DummyModel" begin
    # See assets/generate.jl
    cp("assets/dummy_in.jld2", "dummy_in.jld2", force = true)
    mc = MonteCarlo.load("dummy_in.jld2")
    @test mc.model isa MonteCarlo.DummyModel
    @test mc.model.data["x"] == 7
    @test mc.model.data["y"] == "foo"

    @test_throws ErrorException MonteCarlo.save("dummy_in.jld2", mc, overwrite=true)
    @test isfile("dummy_in.jld2")
    @test is_file_content_equal("dummy_in.jld2", "assets/dummy_in.jld2")
    
    rm("dummy_in.jld2")
end