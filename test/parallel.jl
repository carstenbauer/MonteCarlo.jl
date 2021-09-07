using Distributed

addprocs(2)

@everywhere using MonteCarlo, Distributed

@everywhere function simulate(mu)
    # Connect other workers
    pool = filter(x -> x != myid(), workers())
    println(myid(), " => ", pool)
    connect(pool, wait = true)

    println("Creating model")
    model = HubbardModelAttractive(4, 2, U = 1.0, mu = mu)
    
    println("Creating scheduler")
    # scheduler = SimpleScheduler(
    #     LocalSweep(3), GlobalFlip(),
    #     LocalSweep(3), GlobalShuffle(),
    #     LocalSweep(3), ReplicaExchange(pool[1]),
    #     LocalSweep(3), GlobalFlip(),
    #     LocalSweep(3), GlobalShuffle(),
    #     LocalSweep(3), ReplicaPull()
    # )
    scheduler = AdaptiveScheduler(
        (
            LocalSweep(3), Adaptive(), 
            LocalSweep(3), ReplicaPull(), 
            LocalSweep(3), Adaptive(), 
            LocalSweep(3), ReplicaExchange(pool[1])
        ),
        (GlobalFlip(), GlobalShuffle())
    )

    println("Creating dqmc")
    dqmc = DQMC(
        model, 
        beta=5.0, recorder=Discarder(), scheduler = scheduler,
        thermalization=10000, sweeps=10000
    )

    dqmc[:G]    = greens_measurement(dqmc, model)
    dqmc[:E]    = total_energy(dqmc, model)
    dqmc[:Occs] = occupation(dqmc, model)
    dqmc[:CDC]  = charge_density_correlation(dqmc, model)
    dqmc[:Mx]   = magnetization(dqmc, model, :x)
    dqmc[:My]   = magnetization(dqmc, model, :y)
    dqmc[:Mz]   = magnetization(dqmc, model, :z)
    dqmc[:SDCx] = spin_density_correlation(dqmc, model, :x)
    dqmc[:SDCy] = spin_density_correlation(dqmc, model, :y)
    dqmc[:SDCz] = spin_density_correlation(dqmc, model, :z)
    dqmc[:PC]   = pairing_correlation(dqmc, model, K=4)

    println("Running Simulation")
    run!(dqmc, verbose=!true)

    dqmc
end

begin
    @everywhere MonteCarlo.enable_benchmarks()
    @time dqmcs = pmap(simulate, [1.0, 0.0])
    @everywhere MonteCarlo.reset_timer!()
    @time dqmcs = pmap(simulate, [1.0, 0.0])
    MonteCarlo.show_statistics(stdout, dqmcs[1].scheduler)
    MonteCarlo.show_statistics(stdout, dqmcs[2].scheduler)
end

TOs = pmap(_ -> MonteCarlo.TimerOutputs.DEFAULT_TIMER, workers());