# Models

In MonteCarlo.jl a `Model` describes a Hamiltonian. It's primary purpose is to collect paramaters from different terms in the Hamiltonian as well as the lattice and generate a hopping matrix for the simulation. We currently provide one model - the Hubbard model.
## Hubbard Model

The  Hubbard model is given by

```math
\mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) - U \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j n_{j},
```

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site Hubbard interaction strength, $\mu$ the chemical potential and $\langle i, j \rangle$ indicates that the sum has to be taken over nearest neighbors. Note that the Hamiltonian is written in particle-hole symmetric form such that $\mu = 0$ corresponds to half-filling.

Our implementation allows for both attractive (positive) and repulsive (negative) $U$. Note that for the repulsive case there is a sign problem for $\mu \ne 0$. The model also works with any lattice, assuming that lattice provides the required functionality. 

You can create a Hubbard model with `HubbardModel()`. Optional keyword arguments include:
- `l::AbstractLattice = choose_lattice(HubbardModel, dims, L)` is the lattice used by the model. 
- `dims::Integer = 2` is the dimensionality of the default lattice (Chain/Square/Cubic)
- `L::Integer = 2` is the linear system size of the default lattice (Chain/Square/Cubic)
- `t::Float64 = 1.0` is the hopping strength.
- `mu::Float64 = 0.0` is the chemical potential. (Must be 0 if U is negative.)
- `U::Float64 = 1.0` is the interaction strength.