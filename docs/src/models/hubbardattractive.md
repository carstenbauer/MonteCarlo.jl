# Attractive Hubbard Model

## Hamiltonian
The Hamiltonian of the attractive (negative $U$) Hubbard model reads

\begin{align}
\mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) - |U| \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j n_{j},
\end{align}

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site repulsive interaction strength, $\mu$ the chemical potential and $ \langle i, j \rangle $ indicates that the sum has to be taken over nearest neighbors. Note that (1) is written in particle-hole symmetric form such that $\mu = 0$ corresponds to half-filling.

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

## Supported Monte Carlo flavors

 * [Determinant Quantum Monte Carlo (DQMC)](@ref), see details below

## DQMC formulation

We decouple the onsite electron-electron interaction by performing a Hirsch transformation, i.e. a discrete Hubbard-Stratonovich transformation in the density/charge channel,

\begin{align}
e^{|U|\Delta \tau \left( n_{i\uparrow} - \frac{1}{2} \right) \left(n_{i\downarrow} - \frac{1}{2} \right)} = \frac{1}{2} e^{-|U|\Delta \tau /4} \sum_{s=\pm 1} \prod_{\sigma=\pm 1} e^{s\lambda (n_{i\sigma}-\frac{1}{2})}.
\end{align}

The interaction matrix of the model then reads

\begin{align}
V_{ij}(l) &= \delta_{ij} V_i(l), \\\\
V_i(l) &= - \frac{1}{\Delta \tau} \lambda s_i(l).
\end{align}

For completeness, the hopping matrix is
\begin{align}
T_{ij} &= \begin{cases} -t & \text{if i and j are nearest neighbors,} \\\\
-\mu & \text{if i == j,} \\\\
0 & \text{otherwise.} \end{cases}
\end{align}

As neither $T$ nor $V$ depend on spin, neither does the equal-times Green's function. We can therefore restrict our computations to one spin flavor (`flv=1`) and benefit from operating with smaller matrices.

## Potential extensions

Pull requests are very much welcome!

* Arbitrary lattices (so far only cubic lattices supported)