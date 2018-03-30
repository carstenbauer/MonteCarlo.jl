# Determinant Quantum Monte Carlo (DQMC)

This is determinant quantum Monte Carlo (MC) also known auxiliary field quantum Monte Carlo. It can be used to simulate interacting fermion systems, here the auxiliary boson arises from a Hubbard Stratonovich transformation, or fermions which are naturally coupled to bosons. An example is the [Attractive Hubbard Model](@ref).

You can initialize a determinant quantum Monte Carlo simulation of a given `model` simply through
```julia
dqmc = DQMC(model, beta=5.0)
```

Mandatory keywords are:

* `beta`: inverse temperature

Allowed keywords are:

* `delta_tau::Float64 = 0.1`: imaginary time step size
* `safe_mult::Int = 10`: stabilize Green's function calculations every `safe_mult` step (How many slice matrices can be multiplied until singular value information is lost due to numerical unaccuracy?)
* `checkerboard::Float64 = false`: use [Checkerboard decomposition](@ref) (faster)
* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `seed`: initialize DQMC with custom seed
* `all_checks::Bool = true`: turn off to suppress some numerical checks


Afterwards, you can run the simulation by
```julia
run!(dqmc)
```

## Technical details

### Symmetric Suzuki-Trotter decomposition

We use the symmetric version of the Suzuki-Trotter decomposition, i.e.

\begin{align}
e^{-\Delta\tau \sum_l T+V(l)} = \prod_j e^{-\Delta\tau T/2} e^{-\Delta\tau V} e^{-\Delta\tau T/2} + \mathcal{O}(\Delta\tau^2)
\end{align}

where $T$ is the hopping matrix and $V(l)$ is the interaction matrix with $l$ an imaginary time slice index indicating an auxiliary field dependence.

With the imaginary time slice matrices $B_l = e^{-\Delta\tau T/2} e^{-\Delta\tau V(l)} e^{-\Delta\tau T/2}$ the equal-time Green's function is $G = \left( 1 + B_M \cdots B_1 \right)^{-1}$.

### Checkerboard decomposition

We provide a general algorithm to construct the "checkerboard" split up of a generic `Lattice`. The only requirement is that the `Lattice` has the following two fields,

* `n_bonds::Int`: total number of bonds (lattice graph edges)
* `bonds::Matrix{Int}`: bond matrix of shape `(n_bonds, 3)`. Rows correspond to bonds and columns indicate source site, target site, and bond type in this order.

Of course, one can also manually construct a (more efficient) checkerboard split up by overloading the following function for the specific `Lattice` subtype.

```@docs
MonteCarlo.build_checkerboard(l::MonteCarlo.Lattice)
```

### Effective slice matrices and Green's function

Combining the symmetric Suzuki-Trotter and checkerboard decomposition we can write (assuming two checkerboard groups $a$ and $b$)

\begin{align}
e^{-\Delta\tau \sum_l T+V(l)} &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} \\\\
&\times \left( \prod_j e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a} e^{-\Delta\tau T_b/2} e^{-\Delta\tau V} \right) e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2} + \mathcal{O}(\Delta\tau^2)
\end{align}

For performance resons we internally work with effective imaginary time slice matrices

$$B_l^{\text{eff}} = e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a} e^{-\Delta\tau T_b/2} e^{-\Delta\tau V}$$

instead of the original $B_l$s above.

!!! warning

    Note that one consequence is that the field `dqmc.s.greens` isn't the actual Green's function but an effective one defined by
    \begin{align}
    G &= \left( 1 + B_M \cdots B_1 \right)^{-1} \\\\
    &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} \left( 1 + B^{\text{eff}}_M \cdots B^{\text{eff}}_1 \right)^{-1} e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2} \\\\
    &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} G^{\text{eff}} e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2}
    \end{align}

    To obtain the actual equal-times Green's function, for example for measuring, use `greens(dqmc::DQMC)`. Note that although $G\overset{!}{=}G^\text{eff}$} one can readily show that $\det G = \det G^{\text{eff}}$ holds and the Metropolis acceptance is not affected by switching to the effective matrices.

## Exports

```@autodocs
Modules = [MonteCarlo]
Private = false
Order   = [:function, :type]
Pages = ["DQMC.jl"]
```

## Potential extensions

Pull requests are very much welcome!

 * todo
