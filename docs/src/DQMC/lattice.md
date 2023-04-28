# Lattices

A lattice as we implement it describes a collection of sites in space, connected by bonds.


## Interface

There are a few functions defined for lattices which you may find useful. 

- `position(lattice)` returns an iterator which produces all site position of the lattice. When collected, this will create a D+1 dimensional array for a D dimensional lattice, where the first index represents the basis and the following represent the extend along the different lattice vectors.
- `bonds(lattice[, directed = Val(false)])` returns an iterator which produces all bonds of the lattice. If `directed = Val(false)` bonds will be assumed to be directionless, meaning that only one of $1 \to 2$ and $2 \to 1$ will be returned. If `directed = Val(true)` both bonds will be generated.
- `bond_open(lattice[, directed = false)` returns an iterator which filters out periodic bonds which might be useful for plotting.
- `bonds(lattice, site::Int)` returns an iterator which produces bonds starting from `site`.
- `lattice_vectors(lattice)` returns the lattice vectors of the lattice. (These are the vectors $v_i$ in $R = v_1 i_1 + v_2 i_2 + v_3 i_3$, i.e.e the vectors that generate the periodic Bravais lattice.)
- `reciprocal_vectors(lattice)` returns the Fourier transformed lattice vectors.
- `length(lattice)` returns the total number of sites in the lattice.
- `size(lattice)` returns the size of the lattice starting with the number of basis sites. 
- `eachindex(lattice)` returns a `1:length(lattice)`
- `Bravais(lattice)` returns a wrapped lattice for which the above methods ignore the basis. For example `positions(Bravais(lattice))` will return an iterator of all Bravais lattice positions.


## Implementing your own lattice

### LatticePhysics.jl

[LatticePhysics.jl](https://github.com/janattig/LatticePhysics.jl) already has a lot of common lattices implemented, which can be converted to MonteCarlo.jl lattices. For this you simple need to call `MonteCarlo.Lattice(lattice_physics_lattice)`. Note that the reverse is also implemented as `LatPhysBase.Lattice(mc_lattice)`.

### MonteCarlo.jl

If neither MonteCarlo.jl nor LatticePhysics.jl implements the lattice you need, you can implement your own through a custom constructor. This process is very similar between both libraries. Let us take the implementation for the `Honeycomb` lattice as an example:

```julia
function Honeycomb(Lx, Ly = Lx)
    uc = UnitCell(
        # Name
        "Honeycomb",
        # lattice vectors
        (Float64[sqrt(3.0)/2, -0.5], Float64[sqrt(3.0)/2, +0.5]),
        # basis
        [Float64[0.0, 0.0], Float64[1/sqrt(3.0), 0.0]],
        # bonds
        [
            Bond(1, 2, (0, 0)), Bond(1, 2, (-1, 0)), Bond(1, 2, (0, -1)),
            Bond(2, 1, (0, 0)), Bond(2, 1, ( 1, 0)), Bond(2, 1, (0,  1)),
        ]
    )

    return Lattice(uc, (Lx, Ly))
end
```

As you can see the main task here is to create a fitting unit cell. In order, the unit cell constructors takes the following arguments.

1. The name of the lattice. This is used for printing and might be useful if you wish to restrict a model to a specific lattice.
2. The basis of the lattice. (The positions of sites within a unit cell.)
3. The lattice vectors, i.e. the vectors that generate the periodic Bravais lattice.
4. The bonds of the lattice. Each bond contains three values: The basis site the bond starts at, the basis site it ends at and the Bravais shift which allows a bond to connect to neighboring unit cell. 
   
Note that bonds also have an integer label which can be used to differentiate them later. Note as well that the `UnitCell` constructor will generate missing bonds $b \to a$ if $a \to b$ exists.


## Lattice Iterators

Lattice Iterators are to some degree a backend component. They specify a way to iterate through the lattice, and are mainly used for DQMC measurements which frequently require specific pairings of sites. The iterators fall into three categories:

### DirectLatticeIterator

First we have `DirectLatticeIterator`. These iterators return just site indices, e.g. `(source_index, target_index)`. The concrete implementations include:

* `EachSiteAndFlavor` iterates all indices from 1 to `length(lattice) * nflavors(mc)`
* `EachSite` iterates through `eachindex(lattice)`
* `EachSitePair` iterates through all possible pairs `(i, j)` where both i and j iterate `eachindex(lattice)`.


### DeferredLatticeIterator

The second category inherit from `DeferredLatticeIterator`. Iterators in this category produce two sets of indices, one which is used to access lattice sites and one which is used to access some output array. In this category we have:

* `OnSite` which also `eachindex(lattice)`, but returns three indices `(i, i, i)` at each step.
* `EachSitePairByDistance` which iterates the same range as `EachSitePair` but returns `(combined_idx, i, j)` at each step. The `combined_idx` combines the index of source basis site, the target basis and an index corresponding to a Bravais lattice direction. The output array is generally assumed to be three dimensional, matching those indices.
* `EachLocalQuadByDistance(directions)` iterators through combinations of four sites $s^\prime \leftarrow s \rightarrow t \rightarrow t^\prime$. The given `directions` are directional indices used to derive $s^\prime$ and $t^\prime$ from the current source and target site $s$ and $t$. The full output of this iterator is `(combined_idx, s, t, s', t')`, where `combined_idx` includes the basis index of $s$, $t$, the direction $s \to t$ as well as the indices into `directions` for $s \to s^\prime$ and $t \to t^\prime$.
* `EachLocalQuadBySyncedDistance(directions)` is the same as `EachLocalQuadByDistance` with the restriction that the directions $s \to s^\prime$ and $t \to t^\prime$ are the same. The `combined_idx` does contain an index $t \to t^\prime$ as result.


### Wrappers


The last category are wrappers around lattice iterators. They are used either to adjust the summation and further compress the output array, or to dispatch to different methods during measurement. The following are currently available:

* `Sum(iter)` sets the output index to 1. This has the effect of summing all site combinations.