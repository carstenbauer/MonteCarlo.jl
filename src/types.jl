abstract type MonteCarloMethod end
# abstract DQMC <: MonteCarloMethod
# abstract ContinuousMC <: MonteCarloMethod


abstract type Model end

abstract type Lattice end
abstract type CubicLattice <: Lattice end

# abstract type Parameters end
