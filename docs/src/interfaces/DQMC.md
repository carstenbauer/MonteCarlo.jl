# Interface: Determinant Quantum Monte Carlo (DQMC)

Any model that wants to be simulated by means of DQMC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the determinant Monte Carlo flavor [Determinant Quantum Monte Carlo (DQMC)](@ref).

Example models: [Attractive Hubbard Model](@ref)

## Mandatory fields

 * `flv::Int`: number of distinct fermion flavors (e.g. spins, bands, etc.). The Green's function will have shape `(flv*N, flv*N)`, where `N` is the number of sites. Note that different fermion flavors are often related by symmetry and it's advisable to use this symmetry to work with smaller Green's function matrices. Have a look at the [Attractive Hubbard Model](@ref) as an example where `flv=1` although it has spinful fermions.

## Index of all methods

```@index
Pages = ["DQMC.md"]
```

## Mandatory methods

```@meta
CurrentModule = MonteCarlo
```

```@docs
conftype(::Type{DQMC}, m::Model)
```

```@docs
rand(::DQMC, ::Model)
```

```@docs
hopping_matrix(mc::DQMC, m::Model)
```

```@docs
interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.)
```

```@docs
propose_local(mc::DQMC, m::Model, i::Int, conf, E_boson::Float64)
```

```@docs
accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)
```

!!! warning

    The Monte Carlo update depends on definitions used during the derivation. In this case, the determinant ratio is given by

    \begin{equation}
        R = \det\left[I + \Delta(i, l) (I - G(l))\right]
    \end{equation}

    where $G(l)$ is the effective Greens function stored in `mc.s.greens` at the time of the update. $\Delta(i, l)$ is defined by

    \begin{equation}
        e^{-\Delta\tau V^\prime(l)} = e^{-\Delta\tau V(l)} * [I + \Delta(i, l)]
    \end{equation}

    with `i` the site index passed to `propose_local` and `update_local`, $l$ the current time slice and $V^\prime(l)$ the updated interaction matrix. This definition follows from the effective Greens function used and may differ from other derivations.

    The Monte Carlo update of the Greens function, which should be performed by `accept_local` may differ from literature for the same reason. Here we have

    \begin{equation}
        G(l) = G(l) - (I - G(l)) [I - \Delta(i, l) (I - G(l))]^{-1} \Delta(i, l) G(l)
    \end{equation}

    which may differ from literature in the placement of $I - G(l)$. Note that this equation can be optimized by using the sparsity of $\Delta(i, l)$


## Optional methods

```@docs
greenseltype(::Type{DQMC}, m::Model)
```

```@docs
energy_boson(mc::DQMC, m::Model, conf)
```

```@docs
prepare_observables(mc::DQMC, m::Model)
```

```@docs
measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf)
```

```@docs
finish_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable})
```
