# abstract lattice definition
"""
Abstract definition of a lattice.
Necessary fields depend on Monte Carlo flavor.
However, any concrete Lattice type should have at least the following fields:

    - `sites`: number of lattice sites
    - `neighs::Matrix{Int}`: neighbor matrix (row = neighbors, col = siteidx)
"""
abstract type Lattice end

"""
Abstract cubic lattice.

- 1D -> Chain
- 2D -> SquareLattice
- ND -> NCubeLattice
"""
abstract type CubicLattice <: Lattice end
