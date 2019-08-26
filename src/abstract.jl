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






# abstract lattice definition
"""
Abstract definition of a lattice.
"""
abstract type Lattice end


# Lattice interface: mandatory
# TODO: This needs to be updated. There must be a general way to access sites and bonds.
"""
    nsites(l::Lattice)

Number of lattice sites.
"""
nsites(l::Lattice) = error("Lattice $(typeof(l)) doesn't implement `nsites`.")

# Typically, you also want to implement

#     - `neighbors_lookup_table(lattice)`: return a neighbors matrix where
#                                         row = neighbors and col = siteidx.


"""
Abstract cubic lattice.

- 1D -> Chain
- 2D -> SquareLattice
- ND -> NCubeLattice
"""
abstract type AbstractCubicLattice <: Lattice end







# abstract model definition
"""
Abstract model.
"""
abstract type Model end


# Model interface: mandatory
"""
    nsites(m::Model)

Number of lattice sites of the given model.
"""
nsites(m::Model) = error("Model $(typeof(m)) doesn't implement `nsites`!")






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
