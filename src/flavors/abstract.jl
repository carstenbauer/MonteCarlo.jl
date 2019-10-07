# abstract monte carlo flavor definition
"""
Abstract definition of a Monte Carlo flavor.
"""
abstract type MonteCarloFlavor end

# MonteCarloFlavor interface: mandatory
"""
    init!(mc)

Initialize the Monte Carlo simulation. Has an alias function `reset!`.
"""
init!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `init!`!")

"""
    run!(mc)

Run the Monte Carlo Simulation.
"""
run!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `run!`!")






# general functions
"""
    reset!(mc::MonteCarloFlavor)

Resets the Monte Carlo simulation `mc`.
Previously set parameters will be retained.
"""
function reset!(mc::MonteCarloFlavor)
    th_meas = Dict{Symbol, AbstractMeasurement}([
        k => typeof(v)(mc, mc.model) for (k, v) in mc.thermalization_measurements
    ])

    meas = Dict{Symbol, AbstractMeasurement}([
        k => typeof(v)(mc, mc.model) for (k, v) in mc.measurements
    ])

    init!(mc, thermalization_measurements=th_meas, measurements=meas)
end
