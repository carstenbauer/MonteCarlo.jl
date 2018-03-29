# Hubbard model

## Hamiltonian
The Hamiltonian of the attractive (negative $U$) Hubbard model reads

\begin{align}
\mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) - |U| \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j n_{j},
\end{align}

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site repulsive interaction strength, $\mu$ the chemical potential and $ \langle i, j \rangle $ indicates that the sum has to be taken over nearest neighbors. Note that (1) is written in particle-hole symmetric form such that $\mu = 0$ corresponds to half-filling.

## DQMC formulation

We decouple the onsite electron-electron interaction via Hirsch transformation, i.e. a discrete Hubbard-Stratonovich transformation in the density/charge channel.

The hopping and interaction matrix of the model read

$$ T_{ij} = \begin{cases} -t & \textrm{if $i$ and $j$ are nearest neighbors,} \\ 0 & \textrm{otherwise,} \end{cases} $$

$$ T_{ij} = \begin{cases} -t & 1 \\ 0 & 2 \end{cases} $$


\begin{align}
V_{ij}(l) &= \delta_{ij} V_i(l), \\
V_i(l) &= - \frac{1}{\Delta \tau} \lambda s_i(l) - \mu (-1)^i.
\end{align}

## Constructor
You can create an attractive Hubbard model instance as follows,
```julia
model = HubbardModelAttractive(dims=1, L=8)
```

The following parameters can be set via keyword arguments:

* `dims::Int`: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)
* `L::Int`: linear system size
* `t::Float64 = 1.0`: hopping energy
* `U::Float64 = 1.0`: onsite interaction strength, "Hubbard $U$"
* `mu::Float64 = 0.0`: chemical potential

## Potential extensions

Pull requests are very much welcome!

* Arbitrary lattices (so far only cubic lattices supported)