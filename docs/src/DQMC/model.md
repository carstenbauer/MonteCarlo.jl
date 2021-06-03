# Models

The `Model` is an abstract type whose concrete versions should implement lightweight description of a Hamiltonian. There are two Models implemented for DQMC - the attractive and repulsive Hubbard model.

## Attractive Hubbard Model

The attractive (negative $U$) Hubbard model is given by

\begin{align}
    \mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) - |U| \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right) - \mu\sum_j n_{j},
\end{align}

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site attractive interaction strength, $\mu$ the chemical potential and $\langle i, j \rangle$ indicates that the sum has to be taken over nearest neighbors. Note that (1) is written in particle-hole symmetric form such that $\mu = 0$ corresponds to half-filling.

The parameters $U = 1.0$, $t = 1.0$ and $\mu = 0$ as well as the lattice are saved in the model. They can be specified via keyword arguments when calling `HubbardModelAttractive()`.

## Repulsive Hubbard Model

The repulsive (positive $U$) Hubbard model is given by

\begin{align}
    \mathcal{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) + |U| \sum_j \left( n_{j\uparrow} - \frac{1}{2} \right) \left( n_{j\downarrow} - \frac{1}{2} \right),
\end{align}

where $\sigma$ denotes spin, $t$ is the hopping amplitude, $U$ the on-site repulsive interaction strength and $\langle i, j \rangle$ indicates that the sum has to be taken over nearest neighbors. This model does not include a chemical potential due to the sign problem.

The parameters $U = 1.0$ and $t = 1.0$ as well as the lattice are saved in the model. They can be specified via keyword arguments when calling `HubbardModelRepulsive()`.

## Creating your own Model

To create your own model you will need to inherit from `Model`. There is a set of mandatory and optional methods you must/can implement:

#### Mandatory Methods

* `rand(::Type{DQMC}, m::Model, nslices)` should return a full new random configuration, where `nslices` is the number of imaginary time slices.
* `nflavors(m::Model)` should return the number of fermion flavors used by the model. This controls the size of most simulation matrices, e.g. the greens matrix, hopping and interaction matrices, etc.
* `hopping_matrix(dqmc::DQMC, m::Model)` should return the hopping matrix of the model. This includes all terms that are not connected to a bosonic field.
* `interaction_matrix_exp!(dqmc::DQMC, m::Model, result::AbstractArray, conf, slice, power=1.0)` should calculate $exp(- power \cdot delta_tau \cdot V_{slice})$ for the current `conf` at time slice `slice` and save it to `result`. Note that this method is performance critical.
* `propose_local(dqmc::DQMC; m::Model, i, slice, conf)` should propose a local update at site `i` and the current tiem slice. It should calculate the determinant ratio and bosonic energy difference (which maybe 0 for some models) and return `determinant_ratio, bosonic_energy, passthrough` where `passthrough` are a tuple of varibles you may want to use in `accept_local!`. Note this is also a performance critical method.
* `accept_local!(mc::DQMC, m::MOdel, i, slice, conf, detratio, ΔE_boson, passthrough)` should update the greens function when the proposed change is accepted. Specifically this means updating `dqmc.stack.greens`. This is a performance critical method.

#### (Semi-) Optional Methods

* `greenseltype(::Type{DQMC}, m::Model) = ComplexF64` sets the element type of the greens function. If your greens function is real this function should be implemented for better performance.
* `hoppingeltype(::Type{DQMC}, m::Modle) = Float64` sets the expected element type of the hopping matrix. Adjust this if it is wrong.
* `interaction_matrix_type(::Type{DQMC}, m::Model) = Matrix{greenseltype(DQMC, m)}` Sets the matrix type of the interaction matrix. For the Hubbard model this is `Diagonal{Float64}`.
* `hopping_matrix_type(::Type{DQMC}, m::Model) = Matrix{hoppingeltype(DQMC, m)}` Sets the matrix type of the hopping matrix. If there is a more efficient way to represent your matrix you may use this to change to that representation. The repulsive Hubbard model uses `BlockDiagonal` matrices, for example, which are implemented by MonteCarlo.jl
* `greens_matrix_type(::Type{DQMC}, m::Model) = Matrix{greenseltype(DQMC, m)}` Sets the matrix type of the greens matrix. The repulsive Hubbard model also uses `BlockDiagonal` here.
* `init_interaction_matrix(m::Model)` initiatilizes the interaction matrix. If you are using a custom matrix type you must implement this. Note that the values are irrelevant.
* `energy_boson(dqmc::DQMC, m::Model, conf)` should be implemented if you want to measure energies in your simulation.
* `parameters(m::Model)` should collect the parameters from the model and lattice in a NamedTuple. This is purely utility.

Also note that you may need to update the measurement kernels. More information about that on the measurement page.