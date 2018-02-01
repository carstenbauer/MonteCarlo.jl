# Hubbard model

## Hamiltonian
The Hamiltonian of the repulsive Hubbard model is given by 

\begin{align}
\mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) + U \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j\left( n_{j\uparrow} + n_{j\downarrow} \right),
\end{align}

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site repulsive interaction strength, $\mu$ the chemical potential and $ \langle i, j \rangle $ indicates that the sum has to be taken over nearest neighbors. Note that (1) is the Hubbard model in particle-hole symmetric form which has the nice property that $\mu = 0$ corresponds to half-filling.

## Constructor
You can create a Hubbard model instance as follows,
```julia
model = HubbardModel(; dims::Int=2, L::Int=8, beta::Float64=1.0)
```

The following parameters can be set via keyword arguments:

* `dims`: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)
* `L`: linear system size
* `beta`: inverse temperature
