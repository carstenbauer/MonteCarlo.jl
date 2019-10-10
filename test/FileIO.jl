@testset "MC" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=0.66, thermalization=33, sweeps=123)
    run!(mc, verbose=false)
    MonteCarlo.save("testfile.jld", mc)
    x = MonteCarlo.load("testfile.jld")

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
    rm("testfile.jld")
end

@testset "DQMC" begin
    model = HubbardModelAttractive(dims=2, L=2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=21, sweeps=117)
    run!(mc, verbose=false)
    MonteCarlo.save("testfile.jld", mc)
    x = MonteCarlo.load("testfile.jld")

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
    @test mc.model.neighs == x.model.neighs
    @test mc.model.flv == x.model.flv
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LogBinner in BinningAnalysis
                # implement == for LightObservable in MonteCarloObservable
                @test true
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LogBinner in BinningAnalysis
                # implement == for LightObservable in MonteCarloObservable
                @test true
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end
    rm("testfile.jld")
end
