# Determinant Quantum Monte Carlo (DQMC)

Determinant quantum Monte Carlo (DQMC) is also known as auxiliary field quantum Monte Carlo. It can be used to simulate interacting fermion systems. The auxiliary field may arise from a Hubbard Stratonovich transformation, or is simply a part of the given model. An example of the former is the [Attractive Hubbard Model](@ref).

You can initialize a determinant quantum Monte Carlo simulation of a given `model` simply through
```julia
dqmc = DQMC(model, beta=5.0)
```

Mandatory keywords are:

* `beta`: inverse temperature

Optional keywords are:

* `delta_tau::Float64 = 0.1`: imaginary time step size
* `safe_mult::Int = 10`: stabilize Green's function calculations every `safe_mult` steps (How many slice matrices can be multiplied until singular value information is lost due to numerical unaccuracy? See [StableDQMC.jl](https://github.com/crstnbr/StableDQMC.jl) for further details.)
* `checkerboard::Float64 = false`: use [Checkerboard decomposition](@ref) (faster)
* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `seed`: initialize DQMC with custom seed
* `all_checks::Bool = true`: turn off to suppress some numerical checks
* `measure_rate`: rate at which measurements are taken
* `print_rate`: rate at which prints happen for `verbose=true`


Afterwards, you can run the simulation by
```julia
run!(dqmc)
```

## Technical details

### Symmetric Suzuki-Trotter decomposition

We use the symmetric version of the Suzuki-Trotter decomposition, i.e.

```math
\begin{equation}
  e^{-\Delta\tau \sum_l T+V(l)} = \prod_l e^{-\Delta\tau T/2} e^{-\Delta\tau V(l)} e^{-\Delta\tau T/2} + \mathcal{O}(\Delta\tau^3)
\end{equation}
```

where $T$ is the hopping matrix and $V(l)$ is the interaction matrix with $l$ an imaginary time slice index indicating an auxiliary field dependence. One can verify this equality by expanding the exponentials. We define the factors in the product as slice matrices

```math
\begin{equation}
  B_l = e^{-\Delta\tau T/2} e^{-\Delta\tau V(l)} e^{-\Delta\tau T/2}
\end{equation}
```

The equal-time Greens function is given by

```math
\begin{equation}
  G(l=1) = \langle c_{i\sigma}c_{j\sigma}^\dagger \rangle
  = \left( 1 + B_M \cdots B_1 \right)^{-1}
\end{equation}
```

!!! warning

    For performance reasons, the greens function `mc.s.greens` and the slice matrices used at runtime are not as defined above. This has some implications for the Monte Carlo update, noted below.

    The greens function returned by `greens(dqmc)` follows the definition from above and should be used when measuring observables.


### Monte Carlo update


After some transformations, the partition function takes the form $Z = \sum_C w_C$ with $w_C = \det \left[1 + B_M \cdots B_1\right]$. To performs a Monte Carlo update we need to determine a weight $R = \frac{w_{C^\prime}}{w_C}$.

To do this efficiently we need to introduce an effective Greens function $G_{eff} = e^{\Delta\tau T/2} G e^{-\Delta\tau t/2}$. The propagators for it are given by

```math
\begin{equation}
  B_l = e^{-\Delta\tau T/2} e^{-\Delta\tau T/2} e^{-\Delta\tau V(l)}
\end{equation}
```

An update to the auxillary field causes a change in the interaction matrix $V(l) \to V(l)^\prime$. We can express this change on the slice matrix level by **defining**

```math
\begin{equation}
  V^\prime(l) \equiv V(l) [I + \Delta(i, l)]
\end{equation},
```

which gives $B^\prime_l = B_l [I + \Delta(i, l)]$. We can now write the Monte Carlo weight $R$ as

```math
\begin{equation}
  R = \frac{w_{C^\prime}}{w_C} = \frac{\det \left[1 + B_M \cdots B_{l} [I + \Delta(i, l)] B_{l-1} \cdots B_1\right]}{\det \left[1 + B_M \cdots B_1\right]}
\end{equation}
```

Using $\det A / \det B = \det AB^{-1}$ and $\det(I + AB) = \det(I + BA)$ we eventually get

```math
\begin{equation}
  R = \det\left[I + \Delta(i, l) B_{l-1} \cdots B_1 (I + B_M \cdots B_1)^{-1} B_M \cdots B_{l} \right]
\end{equation}
```

This can be written in terms of the effective equal time Greens function using $A(I + BA)^{-1} B = I - (I + AB)^{-1}$, resulting in

```math
\begin{equation}
  R = \det\left[I + \Delta(i, l) (I - G_{eff}(l)) \right]
\end{equation}
```

For further details on DQMC, see for example:
* [Introduction to quantum Monte Carlo simulations for fermionic systems, dos Santos](https://dx.doi.org/10.1590/S0103-97332003000100003)
* [Quantum Monte Carlo Methods](https://www.cambridge.org/de/academic/subjects/physics/condensed-matter-physics-nanoscience-and-mesoscopic-physics/quantum-monte-carlo-methods-algorithms-lattice-models?format=HB&isbn=9781107006423)


### Checkerboard decomposition

We provide a general algorithm to construct the "checkerboard" split up of a generic `AbstractLattice`. The only requirement is that the `AbstractLattice` implements a method `_neighbors(::Nothing, l::MyLattice, directed::Val{false})` or includes a field `bonds` and implements the trait `has_bonds_table(::MyLattice) = HasBondsTable()`. Either should give access to a bond matrix of size `(nbonds, 3)`, where each row contains a bond with a source site index, a target site index and a bond type (integer).

Of course, one can also manually construct a (more efficient) checkerboard split up by overloading the following function for the specific `AbstractLattice` subtype.

```@docs
MonteCarlo.build_checkerboard(l::MonteCarlo.AbstractLattice)
```

### Effective slice matrices and Green's function

Combining the symmetric Suzuki-Trotter and checkerboard decomposition we can write (assuming two checkerboard groups $a$ and $b$)

```math
\begin{align}
e^{-\Delta\tau \sum_l T+V(l)} &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} \\\\
&\times \left( \prod_j e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a} e^{-\Delta\tau T_b/2} e^{-\Delta\tau V} \right) e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2} + \mathcal{O}(\Delta\tau^2)
\end{align}
```

For performance resons we internally work with effective imaginary time slice matrices

$$B_l^{\text{eff}} = e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a} e^{-\Delta\tau T_b/2} e^{-\Delta\tau V}$$

instead of the original $B_l$s above.

!!! warning

    Note that one consequence is that the field `dqmc.s.greens` isn't the actual Green's function but an effective one defined by

    ```math
    \begin{align}
    G &= \left( 1 + B_M \cdots B_1 \right)^{-1} \\\\
    &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} \left( 1 + B^{\text{eff}}_M \cdots B^{\text{eff}}_1 \right)^{-1} e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2} \\\\
    &= e^{\Delta\tau T_a/2} e^{\Delta\tau T_b/2} G^{\text{eff}} e^{-\Delta\tau T_b/2} e^{-\Delta\tau T_a/2}
    \end{align}
    ```

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
