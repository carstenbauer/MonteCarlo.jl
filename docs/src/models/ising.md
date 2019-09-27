# Ising Model

## Hamiltonian
The famous Hamiltonian of the Ising model is given by

\begin{align}
\mathcal{H} = -\sum_{\langle i,j \rangle} \sigma_i \sigma_j ,
\end{align}

where $\langle i, j \rangle$ indicates that the sum has to be taken over nearest neighbors.

## Creating an Ising model
You can create an Ising model as follows,
```julia
model = IsingModel(dims=2, L=8)
```

Mandatory keywords are:

* `dims::Int`: dimensionality of the lattice
* `L::Int`: linear system size

Optional keywords are:

* `l<:AbstractLattice`: any lattice (if none is given a cubic lattice is generated based on L and dims, i.e. dims = 1 gives a chain, dims = 2 gives a  square lattice, etc.)

## Supported Monte Carlo flavors

 * [Monte Carlo (MC)](@ref) (Have a look at the examples section below)

## Examples

You can find example simulations of the 2D Ising model under [Getting started](@ref Usage) and here: [2D Ising model](@ref).

## Exports

```@autodocs
Modules = [MonteCarlo]
Private = false
Order   = [:function, :type]
Pages = ["IsingModel.jl"]
```

## Analytic results

### Square lattice (2D)

The model can be solved exactly by transfer matrix method ([Onsager solution](https://en.wikipedia.org/wiki/Ising_model#Onsager's_exact_solution)). This gives the following results.

Critical temperature: $T_c = \frac{2}{\ln{1+\sqrt{2}}}$

Magnetization (per site): $m = \left(1-\left[\sinh 2\beta \right]^{-4}\right)^{\frac {1}{8}}$

## Potential extensions

Pull requests are very much welcome!

* Arbitrary dimensions
* Magnetic field
* Maybe explicit $J$ instead of implicit $J=1$
* Non-cubic lattices (just add `lattice::Lattice` keyword)
