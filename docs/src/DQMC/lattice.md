# Lattices

Lattices are a generic component for both quantum and classical Monte Carlo simulations. At this point they are part of the model. We currently offer a couple of directly implemented lattices, an interface for loading ALPS lattices and an interface to [LatticePhysics.jl](https://github.com/janattig/LatticePhysics.jl).

#### Available Lattices

We currently have the following lattices. Unless mentioned, they all include nearest neighbor bonds.

* `Chain(N)`: A one dimensional chain lattice with `N` sites.
* `SquareLattice(L)`: A two dimensional square lattice with `L²` sites.
* `CubicLattice(D, L)`: A `D` dimensional cubic lattice with `L^D` sites.
* `TriangularLattice(L[; Lx = L, Ly = L])`: A two dimensional triangular lattice with `Lx * Ly` sites. This lattice also includes next nearest neighbor bonds.

Additionally we have `ALPSLattice(file)` which loads an ALPS lattice from an xml file and `LatPhysLattice(lattice)` which wraps a generic lattice from [LatticePhysics.jl](https://github.com/janattig/LatticePhysics.jl).

#### Implementing your own Lattice

If you want to implement your own lattice you need to implement a couple of things for compatability. Your lattice should inherit from `MonteCarlo.AbstractLattice`. It should implement a method `length(lattice)` returning the total number of sites. 
The more complex lattice iterators require a method `positions(lattice)` returning the positions of each site in matching order, and a method `lattice_vectors(lattice)` returning D vectors pointing from one end of the lattice to the other along nearest neighbor directions, where D is the dimensionality of the lattice. 
If you are using the default models you will also need to implement some way to get nearest neighbor directions. You have two options here - either implement some traits and fields or implement the getter function directly. For the first option you need to implement: 
* the field `neighs::Matrix{Int}` with `target = neighs[neighbor_idx, source]` and `has_neighbors_table(lattice) = true`
* the field `bonds::Matrix{Int}` where `(source, target, type) = bonds[total_bond_idx, :]` and `has_bonds_table(lattice) = true`
For the second option you need to write your own `neighbors(lattice, directed)` where `directed = Val{true}` returns forward and backwards facing bonds and `directed = Val{false}` considers those the same, returning only one of them.

#### Lattice Iterators

Lattice Iterators are to some degree a backend component. They specify and often cache a way to iterate through the lattice. They are mainly used for DQMC measurements, which frequently require specific pairing of sites. There are currently three abstract subtypes of lattice iterators, each with multiple concrete types. Each concrete iterator can be created via `iterator(dqmc, model)`.

First we have `DirectLatticeIterator`. These iterators return just site indices, e.g. `(source_index, target_index)`. The concrete implementations include:
* `EachSiteAndFlavor` iterates all indices from 1 to `length(lattice) * nflavors(model)`
* `EachSite` iterates all indices from 1 to `length(lattice)`
* `OnSite` also iterates from `1:length(lattice)`, however returns two indices `(i, i)` at each step
* `EachSitePair` iterates through all possible pairs `(i, j)` where both i and j run from 1 to `length(lattice)`.

Next we have `DeferredLatticeIterator`. These iterators return some meta information with each site index, for example a directional index. They are used to do partial summation. The concrete implementations include:
* `EachSitePairByDistance` which iterates the same range as `EachSitePair` but returns `(dir_idx, i, j)` at each step.
* `EachLocalQuadByDistance{K}` iterates through four sets `(1, length(lattice))`, returning `(combined_dir_idx, src1, trg1, src2, trg2)` at each step. Here the directional index relates to three directional indices `(dir_idx, dir_idx1, dir_idx2)` representing the vectors between `src1` and `src2`, `src1` and `trg1`, and `src2` and `trg2` respectifly. `K` restricts the included number of bonds between `src1` and `trg1` (and `src2` and `trg2`). Note that an on-site connection is also counted here - i.e. to include four nearest neighbors you mustr set `K = 5`.
* `EachLocalQuadBySyncedDistance{K}` does the same thing as `EachLocalQuadByDistance{K}` with the additional of synchronizing the direction between `(src1, trg1)` and `(src2, trg2)`.
Note that you can get the directions matching the indices from `directions(lattice/model/dqmc)`.

Lastly we have `LatticeIterationWrapper{LI <: LatticeIterator}`. Generally results from `DeferredLatticeIterator`s will be saved in a vector, where `values[dir_idx]` is the sum of all values with the same directional index. The wrappers are there to further process what happens to these values before they are saved. The parametric type of them specifies the iteration procedure that is used. The concrete implementations are:
* `Sum{LI}` tells the measurement to sum up all values before saving.
* `ApplySymmetry{LI}(symmetries...)` tells the measurement to do sum directional indices past the first with some weights given as `symmetries`. For example you may use `ApplySymmetry{EachLocalQuadByDistance}([1], [0, 1, 1, 1, 1])` to generate s-wave and extended s-wave summations for square lattices.