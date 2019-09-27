# Lattices

The package ships with a couple of standard lattices

| Type                        | Description                       |
| --------------------------- |:---------------------------------:|
| MonteCarlo.Chain            | 1-dimensional chain               |
| MonteCarlo.SquareLattice    | 2-dimensional square lattice      |
| MonteCarlo.CubicLattice     | D-dimensional cubic lattice       |
| MonteCarlo.HoneycombLattice | 2-dimensional honeycomb lattice   |

Each lattice includes nearest neighbor coupling, with the honeycomb lattice also including next-nearest neighbor couplings.

The package also provides routines to load the following common lattice formats

| Type                      | Description                   |
| ------------------------- |:-----------------------------:|
| MonteCarlo.ALPSLattice    | [ALPS simple lattice graph](http://alps.comp-phys.org/mediawiki/index.php/Tutorials:LatticeHOWTO:SimpleGraphs) (XML file)  |

Lattices from [LatticePhysics](https://github.com/janattig/LatticePhysics.jl) can also be used via the `LatPhysLattice()` wrapper.

## Didn't find your desired lattice?

Just implement your own lattice for later use in a model of choice. See [Custom lattices](@ref).
