# abstract monte carlo flavor definition
"""
Abstract definition of a Monte Carlo flavor.

A concrete monte carlo flavor must implement the following methods:

    - `init!(mc)`: initialize the simulation without overriding parameters (will also automatically be available as `reset!`)
    - `run!(mc)`: run the simulation
"""
abstract type MonteCarloFlavor end

"""
    reset!(mc::MonteCarloFlavor)

Resets the Monte Carlo simulation `mc`.
Previously set parameters will be retained.
"""
reset!(mc::MonteCarloFlavor) = init!(mc) # convenience mapping