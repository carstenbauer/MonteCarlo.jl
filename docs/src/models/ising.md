# IsingModel

## Hamiltonian
The famous Hamiltonian of the Ising model is given by

\begin{align}
\mathcal{H} = -\sum_{\langle i,j \rangle} \sigma_i \sigma_j ,
\end{align}

where $ \langle i, j \rangle $ indicates that the sum has to be taken over nearest neighbors.

## Creating an Ising model
You can create an Ising model as follows,
```julia
model = IsingModel(; dims::Int=2, L::Int=8)
```

The following parameters can be set via keyword arguments:

* `dims`: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)
* `L`: linear system size

!!! note

    So far only `dims=2` is supported. Feel free to extend the model and create a pull request!



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

Critical temperature: $ T_c = \frac{2}{\ln{1+\sqrt{2}}} $

Magnetization (per site): $ m = \left(1-\left[\sinh 2\beta \right]^{-4}\right)^{\frac {1}{8}} $

### Potential extensions

Pull requests are very much welcome!

* Arbitrary dimensions
* Magnetic field
* Maybe explicit $J$ instead of implicit $J=1$
* Non-cubic lattices (just add `lattice::Lattice` keyword)
