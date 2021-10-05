# Crosscheck with 2020 Paper

This will be a rather extensive example and crosscheck with the 2020 paper ["Superconductivity, pseudogap, and phase separation in topological flat bands:'a quantum Monte Carlo study"](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.201112) ([arxiv](https://arxiv.org/pdf/1912.08848.pdf)) by Hofmann, Berg and Chowdhury. 

## Introduction

The paper investigates an attractive square lattice Hubbard model with complex hoppings up to the 5th order. It generates a flat and a dispersive band, where the flatness of the former can increased with the right ratio of hoppings. At partial filling of the flat band superconductivity is observed. The model reads

```math
\begin{aligned}
	H_{kin} &= \left[
        -t_1 \sum_{\langle i, j \rangle_1, \sigma} e^{i\phi_{ij}^\sigma} c_{i, \sigma}^\dagger c_{j, \sigma}
        -t_2 \sum_{\langle i, j \rangle_2, \sigma} s_{\langle i, j \rangle_2} c_{i, \sigma}^\dagger c_{j, \sigma}
        -t_5 \sum_{\langle i, j \rangle_5, \sigma} c_{i, \sigma}^\dagger c_{j, \sigma}
        + h.c.
    \right] - \mu \sum_i n_i \\
    H_{int} = - \frac{U}{2} \sum_i (n_i - 1)^2
\end{aligned}
```

where $$t_n$$ refers to n-th nearest neighbor hopping and $$\langle i, j \rangle_n$$ refers to the involved site pairs. We will discuss the prefactors more closely when implementing the lattice model. The interactive term is a variation of what we use in our standard attractive Hubbard model and can be dealt with using the same Hirsch transformation.



# Implementation



## The Lattice


The model is defined for a square lattice, however the paper suggests defining it via two site basis $$A = (0, 0)$$, $$B = (0, 1)$$ with lattice vector $$a_1 = (1, 1)$$ and $$a_2 = (1, -1)$$. We will follow this suggestion. The model uses first, second and fifth neighbor hoppings.

The nearest neighbors are directed, catching different values for $$\phi{ij}^\sigma$$ as a result. We need to create two groups, one with directions as indicated in figure 1a) in the paper, and one with the reverse. For second nearest neighbors the prefactor $$s_{\langle i, j \rangle_2}$$ depends on the combination of sublattice and direction. In $$a_1$$ direction the value is positive (negative) on the A (B) sublattice, and in $$a_2$$ it is negative (positive) on the A (B) sublattice. The fifth nearest neighbors always have the same weight and thus do not require special grouping.

We implement the lattice with [LatticePhysics.jl](https://github.com/janattig/LatticePhysics.jl). The package requires us to define a unitcell with all bonds we want to see in the full lattice. 

```julia
using LatticePhysics, LatPhysUnitcellLibrary

function LatPhysUnitcellLibrary.getUnitcellSquare(
            unitcell_type  :: Type{U},
            implementation :: Val{17}
        ) :: U where {LS,LB,S<:AbstractSite{LS,2},B<:AbstractBond{LB,2}, U<:AbstractUnitcell{S,B}}

    # return a new Unitcell
    return newUnitcell(
        # Type of the unitcell
        U,

        # Bravais lattice vectors
        [[1.0, +1.0], [1.0, -1.0]],
        
        # Sites
        S[
            newSite(S, [0.0, 0.0], getDefaultLabelN(LS, 1)),
            newSite(S, [0.0, 1.0], getDefaultLabelN(LS, 2))
        ],

        # Bonds
        B[
            # NN, directed
            # bonds from ref plot, π/4 weight for spin up
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (0, 1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (-1, 0)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (+1, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (0, 0)),

            # NN reversal
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (0, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (+1, 0)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (-1, +1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (0, 0)),
            
            # NNN
            # positive weight (we need forward and backward facing bonds here too)
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (+1, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (-1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, +1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, -1)),
            # negative weight
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, +1)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, -1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (+1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (-1, 0)),
            
            # Fifth nearest neighbors
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, 2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, 2)),  
            # backwards facing bonds
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, -2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, -2)), 
        ]
    )
end
```

With this implementation we can then generate a lattice of arbitrary size with

```julia
L = 8
uc = LatticePhysics.getUnitcellSquare(17)
lpl = getLatticePeriodic(uc, L)
```

where `L` is the linear system size. Note that due to the two basis sites the total number of sites is $$2L^2$$. To verify our lattice implementation it is useful to create a comparable plot. In Makie, for example, we may run

```julia
using GLMakie

# get small lattice without periodic bonds
uc = LatticePhysics.getUnitcellSquare(17)
lpl = getLatticeOpen(uc, 3)

# create figure and axis without background grid and stretching
fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect(), xgridvisible = false, ygridvisible = false)

# collect list of bonds grouped by label
ps = Point2f.(point.(sites(lpl)))
ls = [Point2f[] for _ in 1:5]
for b in bonds(lpl)
    push!(ls[b.label], ps[b.from], ps[b.to])
end

# Draw arrows for NN groups
ds = ls[1][2:2:end] .- ls[1][1:2:end]
arrows!(ax, ls[1][1:2:end] .+ 0.35 .* ds, 0.55 .* ds, color = :black)
ds = ls[2][2:2:end] .- ls[2][1:2:end]
arrows!(ax, ls[2][1:2:end] .+ 0.65 .* ds, 0.25 .* ds, color = :lightgray)

# NNN
linesegments!(ax, ls[3], color = :black, linewidth=1)
linesegments!(ax, ls[4], color = :black, linewidth=1, linestyle = :dash)

# 5th nearest neighbors
linesegments!(ax, ls[5] .+ Point2f(0, 0.05), color = :red)

# draw A and B sites
As = [Point2f(point(s)) for s in sites(lpl) if s.label == 1]
Bs = [Point2f(point(s)) for s in sites(lpl) if s.label == 2]
scatter!(ax, As, color = :black, markersize = 10)
scatter!(ax, Bs, color = :black, marker='■', markersize = 8)

# Label A and B sites
text!(ax, "A", position = Point2f(-0.2, 0), align = (:right, :center))
text!(ax, "B", position = Point2f(-0.2, 1), align = (:right, :center))

Makie.save("HBC_lattice.png", fig)
fig
```

![](assets/HBC/HBC_lattice.png)

In the plot we indicate first group of nearest neighbor with black arrows and the second, i.e. the reversals with light gray ones. Next nearest neighbors are indicated with full (group 3) or dashed lines (group 4) like in the paper. The fifth nearest neighbors (group 5) are drawn in red like in the reference.


## Hopping and Interaction Matrix


Now that we have the lattice we can generate a fitting hopping matrix. But before we do this, let us briefly discuss some optimizations. 

[LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) is a great tool when pushing for peak single threaded/single core linear algebra performance. The linear algebra needed for DQMC is reimplemented in MonteCarlo.jl using it for both `Float64` and `ComplexF64`. The latter uses `MonteCarlo.CMat64` and `MonteCarlo.CVec64` as concrete array types which are based on [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) under the hood. They should be used in this model. Furthermore we can make use of `MonteCarlo.BlockDiagonal` as we have no terms with differing spin indices. Thus we set

```julia
MonteCarlo.@with_kw_noshow struct HBCModel{LT<:AbstractLattice} <: HubbardModel
    # parameters with defaults based on paper
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0 / sqrt(2.0)
    t5::Float64 = (1 - sqrt(2)) / 4

    # lattice
    l::LT

    # two fermion flavors (up, down)
    flv::Int = 2
    
    # temp storage to avoid allocations in propose_local and accept_local
    IG::CMat64  = StructArray(Matrix{ComplexF64}(undef, length(l), 2))
    IGR::CMat64 = StructArray(Matrix{ComplexF64}(undef, length(l), 2))
    R::Diagonal{ComplexF64, CVec64} = Diagonal(StructArray(Vector{ComplexF64}(undef, 2)))
end

MonteCarlo.hoppingeltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.hopping_matrix_type(::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
MonteCarlo.greenseltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.greens_matrix_type( ::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
```

for our model. The definition of the hopping matrix then follow from the various weights in the Hamiltonian as

```julia
function MonteCarlo.hopping_matrix(mc::DQMC, m::HBCModel{<: LatPhysLattice})
    # number of sites
    N = length(m.l)

    # spin up and spin down blocks of T
    tup = diagm(0 => fill(-ComplexF64(m.mu), N))
    tdown = diagm(0 => fill(-ComplexF64(m.mu), N))

    # positive and negative prefactors for t1, t2
    t1p = m.t1 * cis(+pi/4) # ϕ_ij^↑ = + π/4
    t1m = m.t1 * cis(-pi/4) # ϕ_ij^↓ = - π/4
    t2p = + m.t2
    t2m = - m.t2
    
    for b in bonds(m.l.lattice)
        # NN paper direction
        if b.label == 1 
            tup[b.from, b.to]   = - t1p
            tdown[b.from, b.to] = - t1m
        
        # NN reverse direction
        elseif b.label == 2
            tup[b.from, b.to]   = - t1m
            tdown[b.from, b.to] = - t1p
            
        # NNN solid bonds
        elseif b.label == 3
            tup[b.from, b.to]   = - t2p
            tdown[b.from, b.to] = - t2p

        # NNN dashed bonds
        elseif b.label == 4
            tup[b.from, b.to]   = - t2m
            tdown[b.from, b.to] = - t2m

        # Fifth nearest neighbors
        else
            tup[b.from, b.to]   = - m.t5
            tdown[b.from, b.to] = - m.t5
        end
    end

    return BlockDiagonal(StructArray(tup), StructArray(tdown))
end
```

We note that the hermitian conjugates of a hoppings $$c_j^\dagger c_i$$ can also be understood as reversing the bond direction. Since we include both directions in our lattice definitions, second and fifth nearest neighbor hermitian conjugates are taken care of. First nearest neighbors also get a phase shift from complex conjugation - this is included by swapping `t1p` and `t1m` between group one and two.

The interaction matrix can almost be copied from the repulsive Hubbard model. The only difference is that the spin up and spin down blocks get the same sign. 

```julia
@inline @bm function MonteCarlo.interaction_matrix_exp!(mc::DQMC, m::HBCModel,
            result::Diagonal, conf::HubbardConf, slice::Int, power::Float64=1.)
    dtau = mc.parameters.delta_tau
    lambda = acosh(exp(0.5 * m.U * dtau))
    N = length(lattice(m))
    
    # spin up block
    @inbounds for i in 1:N
        result.diag[i] = exp(sign(power) * lambda * conf[i, slice])
    end

    # spin down block
    @inbounds for i in 1:N
        result.diag[N+i] = exp(sign(power) * lambda * conf[i, slice])
    end
    nothing
end
```

In this case we do not need to set the matrix type since the (abstract) `HubbardModel` already uses `Diagonal` interaction matrices.


## Local Updates


Our next task is to implement `propose_local!` and `accept_local!`. Since those only rely on specific indices, columns or rows for a large part of their calculation we have to dig into the optimized matrix types a bit. `propose_local` aims to calculate the determinant ratio $$R$$ and bosonic energy difference $$\Delta E_Boson = V(C_{new}) - V(c_{old})$$ where $C$ is the auxiliary field configuration. The determinant ratio is defined as

```math
R = \prod_\sigma \left[
        1 + \left( \exp(\Delta E_Boson) - 1 \right) 
        \left( 1 - G_{ii}^{\sigma, \sigma}(\tau, \tau) \right)
    \right]
```

where $i$ and $\tau$ are the lattice index and time slice index of the proposed change in the auxiliary field. This formula already assumes that the greens matrix $$G$$ is zero for all differing spin indices (i.e. spin up-down or down-up). Therefore it is just a product of two terms. With this `propose_local` is implemented as

```julia
@inline @bm function MonteCarlo.propose_local(
        mc::DQMC, model::HBModel, i::Int, slice::Int, conf::HubbardConf
    )
    N = length(model.l)
    G = mc.stack.greens
    Δτ = mc.parameters.delta_tau
    R = model.R

    α = acosh(exp(0.5Δτ * model.U))
    ΔE_boson = -2.0α * conf[i, slice]
    Δ = exp(ΔE_boson) - 1.0

    # Unrolled R = I + Δ * (I - G)
    # up-up term
    R.diag.re[1] = 1.0 + Δ * (1.0 - G.blocks[1].re[i, i])
    R.diag.im[1] = - Δ * G.blocks[1].im[i, i]
    # down-down term
    R.diag.re[2] = 1.0 + Δ * (1.0 - G.blocks[2].re[i, i])
    R.diag.im[2] = - Δ * G.blocks[2].im[i, i]

    # Calculate "determinant"
    detratio = ComplexF64(
        R.diag.re[1] * R.diag.re[2] - R.diag.im[1] * R.diag.im[2],
        R.diag.re[1] * R.diag.im[2] + R.diag.im[1] * R.diag.re[2]
    )
    
    return detratio, ΔE_boson, Δ
end
```

Note that the fields of our special matrix types are directly indexed here. A `BlockDiagonal` matrix contains all of its data in `B.blocks`. We define the first (upper left) block as spin up and the second (lower right) as spin down. `CMat64` and `CVec64` have their real and imaginary values split into two matrices `x.re` and `x.im` respectively.

For `accept_local`  we need to update the auxiliary field and the currently active greens function. To avoid recalculating $$\Delta$$ it is returned in `propose_local` and will be passed to `accept_local`. The updated greens function is given by

```math
G_{jk}^{\sigma \sigma^\prime} = 
    G_{jk}^{\sigma \sigma^\prime} -
    \left(I - G^{\sigma \sigma^\prime}(\tau, \tau) \right)_{ji}
    R_{\sigma, \sigma^\prime}^{-1} 
    \Delta_{ii}^{\sigma \sigma^\prime}(\tau)
    G_{ik}^{\sigma \sigma^\prime}(\tau, \tau)
```

where $$i$$ is again the site index of the proposed flip. Let's go through some observations/simplifications. First we note that for $$\sigma \ne \sigma^\prime$$ the greens function is and remains zero. The inversion of $$R$$ is an inversion of a diagonal matrix and thus simplifies to calculating the inverse of each element. Finally, $$\Delta$$ has the same value for spin up and spin down so it simplifies to a number.

Using these observations and applying optimizations relevant to our matrix types `accept_local` can be implemented as

```julia
@inline @bm function MonteCarlo.accept_local!(
        mc::DQMC, model::HBModel, i::Int, slice::Int, conf::HubbardConf, 
        detratio, ΔE_boson, Δ)

    @bm "accept_local (init)" begin
        N = length(model.l)
        G = mc.stack.greens
        IG = model.IG
        IGR = model.IGR
        R = model.R
    end
    
    # compute R⁻¹ Δ, using that R is Diagonal, Δ is Number
    # using Δ / (a + ib) = Δ / (a^2 + b^2) * (a - ib)
    @bm "accept_local (inversion)" begin
        f = Δ / (R.diag.re[1]^2 + R.diag.im[1]^2)
        R.diag.re[1] = +f * R.diag.re[1]
        R.diag.im[1] = -f * R.diag.im[1]
        f = Δ / (R.diag.re[2]^2 + R.diag.im[2]^2)
        R.diag.re[2] = +f * R.diag.re[2]
        R.diag.im[2] = -f * R.diag.im[2]
    end

    # Compute (I - G) R^-1 Δ
    # Note IG is reduced to non-zero entries. Full IG would be
    # (I-G)[:, i]        0
    #     0         (I-G)[:, i+N]
    # our IG is [(I-G)[:, i]  (I-G)[:, i+N]]
    @bm "accept_local (IG, R)" begin
        # Calculate IG = I - G (relevant entries only)
        @turbo for m in axes(IG, 1)
            IG.re[m, 1] = -G.blocks[1].re[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.re[m, 2] = -G.blocks[2].re[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.im[m, 1] = -G.blocks[1].im[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.im[m, 2] = -G.blocks[2].im[m, i]
        end
        IG.re[i, 1] += 1.0
        IG.re[i, 2] += 1.0
        
        # Calculate IGR = IG * R where R = R⁻¹ Δ from the 
        # previous calculation (relevant entries only)
        # spin up-up block 
        @turbo for m in axes(IG, 1)
            IGR.re[m, 1] = IG.re[m, 1] * R.diag.re[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.re[m, 1] -= IG.im[m, 1] * R.diag.im[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 1] = IG.re[m, 1] * R.diag.im[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 1] += IG.im[m, 1] * R.diag.re[1]
        end
        
        # spin down-down block
        @turbo for m in axes(IG, 1)
            IGR.re[m, 2] = IG.re[m, 2] * R.diag.re[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.re[m, 2] -= IG.im[m, 2] * R.diag.im[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 2] = IG.re[m, 2] * R.diag.im[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 2] += IG.im[m, 2] * R.diag.re[2]
        end
    end

    # Update G according to G = G - (I - G)[:, i:N:end] * R⁻¹ * Δ * G[i:N:end, :]
    # We already have IG = (I - G)[:, i:N:end] * R⁻¹ * Δ
    @bm "accept_local (finalize computation)" begin
        # get blocks to write less
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.stack.greens_temp.blocks[1]
        temp2 = mc.stack.greens_temp.blocks[2]

        # compute temp = IG[:, i:N:end] * G[i:N:end, :]
        # spin up-up block
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] = IGR.re[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] -= IGR.im[m, 1] * G1.im[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] = IGR.im[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] += IGR.re[m, 1] * G1.im[i, n]
        end
        
        # spin down-down block
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] = IGR.re[m, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] -= IGR.im[m, 2] * G2.im[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] = IGR.im[m, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] += IGR.re[m, 2] * G2.im[i, n]
        end

        # Calculate G = G - temp
        # spin up-up block
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.re[m, n] = G1.re[m, n] - temp1.re[m, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.im[m, n] = G1.im[m, n] - temp1.im[m, n]
        end
        
        # spin down-down block
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.re[m, n] = G2.re[m, n] - temp2.re[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.im[m, n] = G2.im[m, n] - temp2.im[m, n]
        end

        # Update configuration
        conf[i, slice] *= -1
    end

    nothing
end
```


## Utilities and other functionality


Now that we have the lattice, the hopping and interaction matrix as well as `propose_local` and `accept_local!` we're done with all the difficult stuff. There are a couple of things one might want to add. For example, adding `energy_boson()` would enable global updates and boson energy measurements. Adding `save_model` and `_load` should help with reducing file size and help future proof things, but isn't strictly necessary. And adding `intE_kernel` would allow the interactive and total energy to be measured. Beyond that one might add some constructors and convenience function like `parameters`. 

The full code including these convenience functions can be found [here]()