@testset "Update Type Tree" begin
    @test MonteCarlo.AbstractLocalUpdate <: MonteCarlo.AbstractUpdate
    @test MonteCarlo.AbstractGlobalUpdate <: MonteCarlo.AbstractUpdate
    @test MonteCarlo.Adaptive <: MonteCarlo.AbstractUpdate

    @test LocalSweep <: MonteCarlo.AbstractLocalUpdate

    @test MonteCarlo.AbstractParallelUpdate <: MonteCarlo.AbstractGlobalUpdate
    @test GlobalShuffle <: MonteCarlo.AbstractGlobalUpdate
    @test GlobalFlip <: MonteCarlo.AbstractGlobalUpdate

    @test ReplicaExchange <: MonteCarlo.AbstractParallelUpdate
    @test ReplicaPull <: MonteCarlo.AbstractParallelUpdate
end

@testset "Scheduler" begin
    model = HubbardModelAttractive(2,2)

    # These do not advance sweeps and should therefore error
    @test_throws ErrorException SimpleScheduler()
    @test_throws ErrorException SimpleScheduler(Adaptive())
    @test_throws ErrorException SimpleScheduler(GlobalFlip())

    @test_throws MethodError AdaptiveScheduler()
    @test_throws ErrorException AdaptiveScheduler(tuple(), tuple())
    @test_throws ErrorException AdaptiveScheduler((Adaptive(),), tuple())
    @test_throws ErrorException AdaptiveScheduler((GlobalFlip(),), tuple())

    # Check constructor
    schedulers = (
        SimpleScheduler(LocalSweep(), GlobalFlip(), LocalSweep(2)),
        AdaptiveScheduler((LocalSweep(), GlobalFlip(), LocalSweep(2)), tuple()),
    )
    for scheduler in schedulers
        Random.seed!(123)
        @test scheduler.sequence == MonteCarlo.AcceptanceStatistics.(
            (LocalSweep(), GlobalFlip(), LocalSweep(), LocalSweep())
        )
        @test scheduler.idx == 0

        mc = DQMC(model, beta=1.0, scheduler = scheduler)
        MonteCarlo.reverse_build_stack(mc, mc.stack)
        MonteCarlo.propagate(mc)

        @test mc.scheduler.idx == 0
        @test mc.last_sweep == 0

        accepted = MonteCarlo.update(mc.scheduler, mc, mc.model)
        @test mc.scheduler.sequence[1].total == 1
        @test mc.scheduler.sequence[1].accepted == accepted
        @test mc.scheduler.idx == 1
        @test mc.last_sweep == 1
        
        accepted = MonteCarlo.update(mc.scheduler, mc, mc.model)
        @test mc.scheduler.sequence[2].total == 1
        @test mc.scheduler.sequence[2].accepted == accepted
        @test mc.scheduler.idx == 2
        @test mc.last_sweep == 1

        accepted = MonteCarlo.update(mc.scheduler, mc, mc.model)
        @test mc.scheduler.sequence[3].total == 1
        @test mc.scheduler.sequence[3].accepted == accepted
        @test mc.scheduler.idx == 3
        @test mc.last_sweep == 2

        for _ in 1:300
            MonteCarlo.update(mc.scheduler, mc, mc.model)
        end

        # io = IOBuffer()
        # MonteCarlo.show_statistics(io, mc.scheduler)
        # @test String(take!(io)) == "Update statistics (since start):\n\tLocalSweep            93.1% accepted   (211 / 227)\n\tGlobalFlip           100.0% accepted   ( 76 /  76)\n\t--------------------------------------------------\n\tTotal                 94.8% accepted   (287 / 303)\n"
    end
end

@testset "AdaptiveScheduler" begin
    struct BadUpdate <: MonteCarlo.AbstractLocalUpdate end
    MonteCarlo.name(::BadUpdate) = "BadUpdate"
    MonteCarlo.update(::BadUpdate, args...) = 0.0

    struct GoodUpdate <: MonteCarlo.AbstractLocalUpdate end
    MonteCarlo.name(::GoodUpdate) = "GoodUpdate"
    MonteCarlo.update(::GoodUpdate, args...) = 1.0

    scheduler = AdaptiveScheduler(
        (BadUpdate(), Adaptive()), 
        (GoodUpdate(), BadUpdate())
    )

    # check defaults
    @test scheduler.grace_period == 99
    @test scheduler.minimum_sampling_rate == 0.01
    @test scheduler.adaptive_rate == 9.0

    model = HubbardModelAttractive(2,2)
    mc = DQMC(model, beta=1.0, scheduler = scheduler)

    # Checks without adaptive corrections
    # should be save to run, 1 / 2^50 chance of already adapting
    for _ in 1:200
        MonteCarlo.update(scheduler, mc, model)
    end

    @test scheduler.sampling_rates == [0.5, 0.5, 1e-10]
    @test scheduler.sequence[1].accepted == 0.0
    @test scheduler.sequence[1].total == 100
    # no adaptive stuff yet, both updates should be picked at 50%
    @test scheduler.adaptive_pool[1].accepted == scheduler.adaptive_pool[1].total
    @test 30 <= scheduler.adaptive_pool[1].total <= 70
    @test scheduler.adaptive_pool[2].accepted == 0
    @test 30 <= scheduler.adaptive_pool[2].total <= 70

    # Checks with adaptive corrections
    for _ in 1:800
        MonteCarlo.update(scheduler, mc, model)
    end
    @test scheduler.sampling_rates[1] > 0.95
    @test scheduler.sampling_rates[2] < 0.05
    @test scheduler.sampling_rates[3] == 1e-10
    @test scheduler.sequence[1].accepted == 0.0
    @test scheduler.sequence[1].total == 500

    # With 0 acceptance rate we should have the hard limit for total:
    i_min = ceil(Int, log(
        scheduler.adaptive_rate / (scheduler.adaptive_rate+1), 
        scheduler.minimum_sampling_rate / 0.5
    )) + 99 + 1 # +1 to be save
    @test scheduler.adaptive_pool[1].accepted == scheduler.adaptive_pool[1].total
    @test scheduler.adaptive_pool[1].total > 500 - i_min
    @test scheduler.adaptive_pool[2].accepted == 0
    @test scheduler.adaptive_pool[2].total < i_min
end


using MonteCarlo: conf, current_slice, nslices

@testset "Global update" begin
    models = (HubbardModelAttractive(2,2,mu=0.5), HubbardModelRepulsive(2,2))
    for model in models
        @testset "$(typeof(model))" begin
            mc1 = DQMC(model, beta=2.0)
            mc2 = DQMC(model, beta=2.0)
            
            # Verify probabilities and greens in global updates
            # This is the backbone check for all global and parallel updates
            for _ in 1:10
                # re-initialize everything with a random conf
                mc1.conf .= rand(DQMC, model, nslices(mc1))
                mc2.conf .= mc1.conf
                for mc in (mc1, mc2)
                    MonteCarlo.init!(mc)
                    MonteCarlo.reverse_build_stack(mc, mc.stack)
                    MonteCarlo.propagate(mc)
                end

                # global update
                temp_conf = shuffle(deepcopy(conf(mc1)))
                detratio, ΔE_boson, passthrough = MonteCarlo.propose_global_from_conf(mc1, model, temp_conf)
                global_p = exp(- ΔE_boson) * detratio
                MonteCarlo.accept_global!(mc1, model, temp_conf, passthrough)

                # global update through successive local updates
                local_p = 1.0
                for t in 1:nslices(mc2)
                    for i in 1:length(lattice(mc2))
                        if mc2.conf[i, current_slice(mc2)] != temp_conf[i, current_slice(mc2)]
                            detratio, ΔE_boson, passthrough = MonteCarlo.propose_local(
                                mc2, model, i, current_slice(mc2), conf(mc2)
                            )
                            local_p *= real(exp(- ΔE_boson) * detratio)
                            MonteCarlo.accept_local!(
                                mc2, model, i, current_slice(mc2), conf(mc2), detratio, ΔE_boson, passthrough
                            )
                        end
                    end
                    MonteCarlo.propagate(mc2)
                end

                # Move to correct time slice
                for t in 1:nslices(mc2)
                    MonteCarlo.propagate(mc2)
                end

                # Verify
                @test local_p ≈ global_p
                @test mc1.conf == mc2.conf
                @test current_slice(mc1) == current_slice(mc2)
                @test mc1.stack.greens ≈ mc2.stack.greens
            end
        end
    end

    # Check config adjustments for global updates
    model = HubbardModelAttractive(2,2)
    mc = DQMC(model, beta=1.0)
    MonteCarlo.init!(mc)
    MonteCarlo.reverse_build_stack(mc, mc.stack)
    MonteCarlo.propagate(mc)
    c = deepcopy(conf(mc))
    MonteCarlo.update(GlobalFlip(), mc, model)
    @test mc.temp_conf == -c

    # make this unlikely to fail randomly via big conf size
    model = HubbardModelAttractive(8,2) 
    mc = DQMC(model, beta=10.0)
    MonteCarlo.init!(mc)
    MonteCarlo.reverse_build_stack(mc, mc.stack)
    MonteCarlo.propagate(mc)
    c = deepcopy(conf(mc))
    MonteCarlo.update(GlobalShuffle(), mc, model)
    @test sum(mc.temp_conf) == sum(c)
    @test mc.temp_conf != c
end

@testset "Utilities/Printing" begin
    @test MonteCarlo.name(NoUpdate()) == "NoUpdate"
    @test MonteCarlo.name(LocalSweep()) == "LocalSweep"
    @test MonteCarlo.name(GlobalFlip()) == "GlobalFlip"
    @test MonteCarlo.name(GlobalShuffle()) == "GlobalShuffle"
    @test MonteCarlo.name(ReplicaExchange(1)) == "ReplicaExchange"
    @test MonteCarlo.name(ReplicaPull()) == "ReplicaPull"
    @test MonteCarlo.name(Adaptive()) == "Adaptive"

    # Idk if I want to keep these methods...
    model = HubbardModelAttractive(2,2) 
    mc = DQMC(model, beta=1.0)
    for T in (LocalSweep, GlobalFlip, GlobalShuffle, NoUpdate, ReplicaPull)
        @test T(mc, model) == T()
    end
    @test ReplicaExchange(mc, model, 1) == ReplicaExchange(1)
end