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
    scheduler = SimpleScheduler(
        DQMC, model, 
        GlobalFlip(),
        GlobalShuffle(),
        ReplicaExchange(pool[1]),
        GlobalFlip(),
        GlobalShuffle(),
        ReplicaPull()
    )

    println("Creating dqmc")
    dqmc = DQMC(
        model, 
        beta=5.0, recorder=Discarder, scheduler = scheduler,
        thermalization=1000, sweeps=1000
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

@everywhere MonteCarlo.enable_benchmarks()
@everywhere MonteCarlo.reset_timer!()

begin
    dqmcs = pmap(simulate, [1.0, 0.5])
    MonteCarlo.show_statistics(dqmcs[1].scheduler)
    MonteCarlo.show_statistics(dqmcs[2].scheduler)
end

TOs = pmap(_ -> MonteCarlo.TimerOutputs.DEFAULT_TIMER, workers())