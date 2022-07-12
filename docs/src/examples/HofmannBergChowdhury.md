# Topological Flat Band Model

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
    H_{int} &= - \frac{U}{2} \sum_i (n_i - 1)^2
\end{aligned}
```

where $$t_n$$ refers to n-th nearest neighbor hopping and $$\langle i, j \rangle_n$$ refers to the involved site pairs. We will discuss the prefactors more closely when implementing the lattice model. The interactive term is a variation of what we use in our standard attractive Hubbard model and can be dealt with using the same Hirsch transformation.



# Implementation



## The Lattice


The model is defined for a square lattice, however the paper suggests defining it via two site basis $$A = (0, 0)$$, $$B = (0, 1)$$ with lattice vector $$a_1 = (1, 1)$$ and $$a_2 = (1, -1)$$. We will follow this suggestion. The model uses first, second and fifth neighbor hoppings.

The nearest neighbors are directed, catching different values for $$\phi_{ij}^\sigma$$ as a result. We need to create two groups, one with directions as indicated in figure 1a) in the paper, and one with the reverse. For second nearest neighbors the prefactor $$s_{\langle i, j \rangle_2}$$ depends on the combination of sublattice and direction. In $$a_1$$ direction the value is positive (negative) on the A (B) sublattice, and in $$a_2$$ it is negative (positive) on the A (B) sublattice. The fifth nearest neighbors always have the same weight and thus do not require special grouping.

```julia
using MonteCarlo: UnitCell, Bond, Lattice

function HBCLattice(Lx, Ly = Lx)
    uc = UnitCell(
        # name
        "HBC Square",

        # Bravais lattice vectors
        ([1.0, +1.0], [1.0, -1.0]),
        
        # Sites
        [[0.0, 0.0], [0.0, 1.0]],

        # Bonds
        [
            # NN, directed
            # bonds from ref plot, π/4 weight for spin up
            Bond(1, 2, ( 0,  1), 1),
            Bond(1, 2, (-1,  0), 1),
            Bond(2, 1, (+1, -1), 1),
            Bond(2, 1, ( 0,  0), 1),

            # NN reversal
            Bond(2, 1, ( 0, -1), 2),
            Bond(2, 1, (+1,  0), 2),
            Bond(1, 2, (-1, +1), 2),
            Bond(1, 2, ( 0,  0), 2),
            
            # NNN
            # positive weight (we need forward and backward facing bonds here too)
            Bond(1, 1, (+1,  0), 3),
            Bond(1, 1, (-1,  0), 3),
            Bond(2, 2, ( 0, +1), 3),
            Bond(2, 2, ( 0, -1), 3),
            # negative weight
            Bond(1, 1, ( 0, +1), 4),
            Bond(1, 1, ( 0, -1), 4),
            Bond(2, 2, (+1,  0), 4),
            Bond(2, 2, (-1,  0), 4),
            
            # Fifth nearest neighbors
            Bond(1, 1, (2, 0), 5),
            Bond(2, 2, (2, 0), 5),
            Bond(1, 1, (0, 2), 5),
            Bond(2, 2, (0, 2), 5),
            # backwards facing bonds
            Bond(1, 1, (-2,  0), 5),
            Bond(2, 2, (-2,  0), 5),
            Bond(1, 1, ( 0, -2), 5),
            Bond(2, 2, ( 0, -2), 5),
        ]
    )

    return Lattice(uc, (Lx, Ly))
end
```

With this implementation we can then generate a lattice of arbitrary size with

```julia
L = 8
l = HBCLattice(L)
```

where `L` is the linear system size. Note that due to the two basis sites the total number of sites is $$2L^2$$. To verify our lattice implementation it is useful to create a comparable plot. In Makie, for example, we may run

```julia
using GLMakie

# get small lattice without periodic bonds
l = HBCLattice(3)

# create figure and axis without background grid and stretching
fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect(), xgridvisible = false, ygridvisible = false)

# collect list of bonds grouped by label
ps = Point2f.(positions(l))
ls = [Point2f[] for _ in 1:5]
for b in bonds_open(l, true)
    push!(ls[b.label], ps[b.from], ps[b.to])
end

# Draw arrows for NN groups
ds = ls[1][2:2:end] .- ls[1][1:2:end]
a = arrows!(ax, ls[1][1:2:end] .+ 0.35 .* ds, 0.55 .* ds, color = :black, arrowsize = 16)
ds = ls[2][2:2:end] .- ls[2][1:2:end]
arrows!(ax, ls[2][1:2:end] .+ 0.65 .* ds, 0.25 .* ds, color = :lightgray, arrowsize = 16)

# NNN
linesegments!(ax, ls[3], color = :black, linewidth=1)
linesegments!(ax, ls[4], color = :black, linewidth=1, linestyle = :dash)

# 5th nearest neighbors
linesegments!(ax, ls[5] .+ Point2f(0, 0.05), color = :red)

# draw A and B sites
As = ps[1, :, :][:]
Bs = ps[2, :, :][:]
scatter!(ax, As, color = :black, markersize = 10)
scatter!(ax, Bs, color = :black, marker='■', markersize = 16)

# Label A and B sites
text!(ax, "A", position = Point2f(2-0.2, 0), align = (:right, :center))
text!(ax, "B", position = Point2f(2-0.2, 1), align = (:right, :center))

Makie.save("HBC_lattice.png", fig)
fig


```

![](assets/HBC/HBC_lattice.png)

In the plot we indicate the first group of nearest neighbors with black arrows and the second, i.e. the reversals with light gray ones. Next nearest neighbors are indicated with full (group 3) or dashed lines (group 4) like in the paper. The fifth nearest neighbors (group 5) are drawn in red like in the reference.


## Hopping and Interaction Matrix


Now that we have the lattice we can generate a fitting hopping matrix. But before we do this, let us briefly discuss some optimizations. 

[LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) is a great tool when pushing for peak single threaded/single core linear algebra performance. The linear algebra needed for DQMC is reimplemented in MonteCarlo.jl using it for both `Float64` and `ComplexF64`. The latter uses `MonteCarlo.CMat64` and `MonteCarlo.CVec64` as concrete array types which are based on [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) under the hood. They should be used in this model. Furthermore we can make use of `MonteCarlo.BlockDiagonal` as we have no terms with differing spin indices. Thus we set

```julia
MonteCarlo.@with_kw_noshow struct HBCModel <: Model
    # parameters with defaults based on paper
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0 / sqrt(2.0)
    t5::Float64 = (1 - sqrt(2)) / 4

    # lattice
    l::Lattice{2}
    @assert l.unitcell.name == "HBC Square"
end

MonteCarlo.hoppingeltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.hopping_matrix_type(::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
MonteCarlo.greenseltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.greens_matrix_type( ::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
```

for our model. The definition of the hopping matrix then follows from the various weights in the Hamiltonian as

```julia
function MonteCarlo.hopping_matrix(m::HBCModel)
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
    
    for b in bonds(m.l.lattice, Val(true))
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

We note that the hermitian conjugates of a hopping $$c_j^\dagger c_i$$ can also be understood as reversing the bond direction. Since we include both directions in our lattice definitions, second and fifth nearest neighbor hermitian conjugates are taken care of. First nearest neighbors get a phase shift from complex conjugation, which is included by swapping `t1p` and `t1m` between group one and two.

To finish off the mandatory model interface we need to provide two more methods. The first is `lattice(model)` which simply return the lattice of the model. The other is `nflavors(model)` which returns the number of "active" flavors of the hopping matrix. This model has two flavors total, spin up and spin down, and both of these have an active effect on the hopping matrix, i.e. the values for spin up and spin down are different. Thus this method should return 2. 

```julia
MonteCarlo.lattice(m::HBCModel) = m.l
MonteCarlo.nflavors(::HBCModel) = 2
```

There are a few more methods we can implement for convenience. The most important of these is `choose_field(model)`, which sets a default field for our model. The best choice here should be `DensityHirschField` or `DensityGHQField` as the model uses an attractive interaction. Beyond this we could implement `intE_kernel` to enable energy measurements, `parameters(model)`, `save_model`, `_load_model` and printing.

The full code including these convenience functions can be found [here](HBC_model.jl)


# Simulation Setup



To keep the runtime of this crosscheck reasonable we used the smallest linear system size the paper considers, `L = 8`. We also set `U = 1` and the fifth nearest neighbor hopping `t5 = 0`. This corresponds to a flatness ratio $$F = 0.2$$. To be comparable to the paper we will need to tune the chemical potential $$\mu$$ to hit half filling. This can be done through trial and error on a smaller lattice. The optimal $$\mu$$, after running the main simulation with a small set of different values, seems to be $$\mu \approx -2.206$$. Thus the basic setup for our simulation becomes

```julia
l = HBCLattice(8)
m = HBCModel(l, t5 = 0.0, mu = -2.206) # other defaults match F = 0.2 setup
mc = DQMC(
    m, beta = beta, thermalization = 1000, sweeps = 5000, 
    measure_rate = 5, print_rate = 100, recorder = Discarder()
)
```

where beta needs to run over a reasonable set of inverse temperatures. We will use `[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0]`. 

!!! note

    In our actual simulation we used a `BufferedConfigRecorder` to record configurations. That way the simulation can be replayed with different measurements. This is very useful when you are still unsure about what you want to measure or how exactly those measurements are supposed to work.

## Measurements

We will consider the following measurements for comparison:

1. Z-spin susceptibility, solid red line in figure 1d)
2. Superfluid stiffness, figure 2b)
3. Reciprocal s-wave pairing susceptibility, solid red line in figure 4a)
4. Reciprocal charge susceptibility, solid blue line in figure 4a)

#### Z-Spin Susceptibility

The z-spin susceptibility $$\int_0^\beta d\tau \langle m_z(r^\prime, \tau) m_z(r, 0) \rangle$$ can be measure with

```julia
mc[:SDSz] = spin_density_susceptibility(mc, m, :z)
```

The integral will be evaluated by MonteCarlo.jl and the result, accessible with `mean(mc[:SDCz])`, will return the average result by direction. I.e. `mean(mc[:SDCz])[i]` will contain the average z-spin susceptibility in `directions(mc)[i]`.

#### Reciprocal Charge Susceptibility

The paper defines the charge susceptibility as $$\int_0^\beta d\tau \langle N(\tau) N(0) \rangle$$ where $$N(\tau) = \sum_j (n_j(\tau) - \nu)$$ and $$\nu$$ is the filling. The charge density susceptibility that MonteCarlo.jl defines, on the other hand, is $$\langle n_j(\tau) n_i(0)\rangle$$. To connect these two we expand the papers' definition:

```math
\begin{aligned}
	\langle N(\tau) N(0) \rangle 
        &= \langle \sum_j (n_j(\tau) - \nu) \sum_i (n_i(0) - \nu) \rangle \\
        &= \sum_{ij} \langle n_j(\tau) n_i(0) - n_j(\tau) \nu - \nu n_i(0) + \nu \nu \rangle \\
        &= \sum_{ij} \langle n_j(\tau) n_i(0) \rangle - \langle \sum_i n_i \rangle^2
\end{aligned}
```

In the last step we associated $$\nu = \sum_i \langle n_i \rangle / N$$, i.e. the average occupation. We can use this representation to calculate the reciprocal charge susceptibility as $$1 / (O - \nu^2)$$ where $$O$$ is MonteCarlo.jl's charge density susceptibility and $\nu$ is the average occupation. We measure

```julia
mc[:occ] = occupation(mc, m)
mc[:CDS] = charge_density_susceptibility(mc, m)
```

#### Reciprocal S-Wave Pairing Susceptibility

The paper defines the (s-wave) pairing susceptibilities as $$O(\tau) = \sum_j c_{j, \uparrow} c_{j, \downarrow} + h.c.$$. More generally you would consider a site offset between the pairs of operators and use weighted sums to get pairing susceptibilities of various symmetries like d-wave, p-wave, etc. For s-wave this offset is $$\vec{0}$$. In MonteCarlo.jl these offsets are set via the lattice iterator. For example you may use `EachLocalQuadByDistance([2, 4, 5])` to consider the `directions(mc)[[2, 4, 5]]` as offsets. 

The `pairing_suceptibility` constructors from MonteCarlo.jl is written with these offsets in mind. By default it will include offsets for all nearest neighbors as well as offsets of $$\vec{0}$$. To reduce computational complexity we may reduce these to just $$\vec{0}$$ offsets by requesting just first offset:


```julia
mc[:PS] = pairing_susceptibility(mc, m, 1)
```

#### Superfluid Stiffness

The superfluid stiffness is given by $$0.25 [- K_x - \Lambda_{xx}(q = 0)]$$ in the paper. Both the diamagnetic contribution $$K_x$$ and the Fourier transformed current-current correlation $$\Lambda_{xx}(q)$$ are things we need to measure individually.

The diamagnetic contribution $$K_x$$ is the simpler one. For that we refer to equations 15a - 15j in the paper. The sum of all of these is the $$K_x$$ we seek. Since all terms are quadratic in creation and annihilation operators we do not need to worry about expanding them with Wicks theorem. Instead we can simply measure the Greens matrix during the simulation. If we compare the equations with the Hamiltonian we will also notice that they are (almost) the same as the hopping terms. Thus we can get weights from the hopping matrix and apply them afterwards. We measure

```julia
mc[:G] = greens_measurement(mc, model)
```

For the current-current correlations we need to measure $$\int_0^\beta d \tau \langle J_x^\alpha(r^\prime, \tau) J_x^\beta(r, 0) \rangle$$ where $$J_x(r, \tau)$$ is given in equation 14a - 14j. These terms are already implemented by MonteCarlo.jl, but we should briefly discuss them regardless. (We will leave the Fourier transform for later.) 

Much like $$K_x$$ these terms are closely related to the hopping terms of the Hamiltonian. The index $$\alpha$$ ($$\beta$$) refers to the direction of the hopping. Each term contains a hopping in +x direction weighted by $$i$$ and a hopping in the opposite direction weighted by $$-i$$ via the Hermitian conjugate. The fifth nearest neighbor terms also catch a factor 2. For the measurement we need to apply Wicks theorem to every combination of any two currents $$J^\alpha_x$$ and also take care of an implied spin index. The result can be found in the `cc_kernel` in MonteCarlo.jl.

Let's get back to what we need to measure in our simulation. MonteCarlo.jl's `current_current_susceptibility` measures $$\int_0^\beta d \tau \langle J_x^\alpha(r^\prime, \tau) J_x^\beta(r, 0) \rangle$$ where $$\alpha$$ and $$\beta$$ are directions that get passedto the function. Since we set `t5 = 0` we can ignore equations 14g - 14j leaving 6 equations with 3 directions on 2 sublattices. MonteCarlo.jl does not care about sublattices, so we're left with three +x directions `[2, 6, 9]` which need to be passed. You can check these with `directions(mc)[[2, 6, 9]]` against the paper. Our measurement is given by

```julia
mc[:CCS] = current_current_susceptibility(mc, model, directions = [2, 6, 9])
```

## Running the simulations

To run the simulation we simply use `run!(mc)`.

We should point out that these simulations are lot more complex than the other two examples. We are working with 128 sites as opposed to 16 and inverse temperatures as large as 40 instead of $$\le 12$$. We are also using complex matrices which bring $$2 - 4$$ times the complexity and we need to consider both a spin up and down sector in the greens matrix. 

It is therefore advised that you run this on a cluster, in parallel. To figure out how much time is needed you can check the sweep time for the smallest $$\beta$$ with measurements. The scaling should be roughly linear (w.r.t. $$\beta$$). Note that you can pass a `safe_before::TimeType` to make sure the simulation saves and exits in time. If your cluster restricts you to full nodes it might be useful to create files for each simulation beforehand and distribute filenames to different cores on the same node. (src/mpi.jl might be helpful.)



# Results



In this section we plot the results from our simulations on top of the results from the paper. There are 5 points per temperature coming from different chemical potentials $$\mu$$. The filling varies from 0.22 to 0.265 between them. Note also that not every simulation did the full number of sweeps. The shortest simulation ran for about 3000 sweeps, which includes 1000 thermalization sweeps.


## Z-Spin Susceptibility

To get the results from the paper we need to perform a $$q = 0$$ Fourier transform which is simply a sum. We calculate `real(sum(mean(mc[:SDCz])))` and plot against `1 / mc.parameters.beta`.

![](assets/HBC/SDCz.png)



## Reciprocal Charge Susceptibility

For the reciprocal charge susceptibility we plot

```julia
CDS = real(sum(mean(mc[:CDS])))
occ = real(sum(mean(mc[:occ])))
xs = 1 / mc.parameters.beta
ys = 1 / (CDS - mc.parameters.beta * occ^2 / length(lattice(mc)))
```

where the factor `mc.parameters.beta / length(lattice(mc))` comes from the imaginary time integral.

![](assets/HBC/CDS.png)



## Reciprocal Pairing Susceptibility

The pairing susceptibility comes with three directional indices after taking `mean(mc[:PS])`. The first is associated with the distance $$r - r^\prime$$ between the two pairing operators $$\Delta(r)$$ in $$\langle \Delta^\alpha(r) \Delta^\beta(r^\prime) \rangle$$. The second and third are displacements inside them. Since we only care about s-wave pairing the internal displacements are zero or index 1. Thus we plot `1 / real(sum(mean(mc[:PS])[:, 1, 1]))` against `1 / mc.parameters.beta` for the reciprocal pairing susceptibility. 

![](assets/HBC/PS.png)



## Superfluid Stiffness

The superfluid stiffness still requires a good amount of work. Let's start with the diamagnetic contribution $$K_x$$. As mentioned before the prefactors of $$K_x$$ match those of the hoppings in the Hamiltonian (except for 5th nearest neighbors which catch a factor of 4). The expectation values for $$\langle c_j^\dagger c_i \rangle$$ follow from the measured Greens function $$G_{ij} = c_i c_j^\dagger$$ as $$\delta_{ij} - G_{ji}$$ (permutation of operators).

To pick the correct sites we can poke at the backend of the lattice iterator interface. There are a few maps that get cached, one of which returns a list of (source, target) site index pairs given a directional index. It can be fetched (and potentially generated) via `dir2srctrg = mc[Dir2SrcTrg()]`. The relevant directional indices can be determined from `directions(mc)`. Using that we calculate the total diamagnetic contribution $$K_x$$ as

```julia
function dia_K_x(mc)
    # directional indices for K_x (see directions(mc)[idxs]
    # These correspond to 
    # K1 & K4, K1 & K4 h.c., K2 & K5, K3 & K6 h.c., K2 & K5 h.c., K3 & K5
    idxs = [2, 4, 6, 7, 8, 9]
    
    # T contains the prefactors used in K_x, G contains ⟨c_i c_j^†⟩
    T = Matrix(MonteCarlo.hopping_matrix(mc, mc.model))
    G = mean(mc[:G])

    # We use the dir2srctrg map to get all (src, trg) pairs relevant to the 
    # directions we specified with `idxs` above
    dir2srctrg = mc[MonteCarlo.Dir2SrcTrg()]
    N = length(lattice(mc))
    
    Kx = ComplexF64(0)
    for dir_idx in idxs
        for (src, trg) in dir2srctrg[dir_idx]
            # c_j^† c_i = δ_ij - G[i, j], but δ always 0 (no onsite)
            Kx -= T[trg, src] * G[src, trg]         # up-up
            Kx -= T[trg+N, src+N] * G[src+N, trg+N] # down-down
        end
    end

    # normalize
    Kx /= N
end
```

For the Fourier transformed current-current correlation $$\Lambda_{xx}(q)$$ the paper uses two summations. The first (eq. 16) runs over positions without offsets from the basis. The second (eq. 17) then resolves those offsets with a shift by $$\hat{e}_y$$ and also translates positions to the centers of the two sites involved with each hopping term. Combining both equations in a MonteCarlo.jl compatible style yields

```math
\Lambda_{xx}(q) = \sum_{\Delta r_{12}} \sum_{\Delta r_1, \Delta r_2} e^{- i q (\Delta r_{12} + 0.5 (\Delta r_1 - \Delta r_2))}
                  \int_0^\beta \sum_{r_0} \langle J_x^{\Delta r_1}(r_0 + \Delta r_{12}, \tau) J_x^{\Delta r_2}(r_0, 0) \rangle d\tau

```

Here $$\Delta r_{12}$$ is the distance between any two sites including basis offsets and $$\Delta r_1$$ and $$\Delta r_2$$ are the hopping distances. The integral over imaginary time and the sum over $$r_0$$ are already performed during measuring. This leaves the following to be calculated after measuring:

```julia
function para_ccc(mc, q)
    # direction indices for J1 & J4, J2 & J5, J3 & J6 (h.c. included in measurement)
    idxs = [2, 6, 9]

    CCS = mean(mc[:CCS])
    dirs = directions(lattice(mc))
    Λxx = ComplexF64(0)

    for (i, dir) in enumerate(dirs)
        for j in idxs, k in idxs
            Λxx += CCS[i, j, k] * cis(-dot(dir + 0.5(dirs[j] .- dirs[k]), q))
        end
    end
    
    Λxx
end
```

To compute the superfluid stiffness we now just calculate $$D_S = \frac{1}{4} [ -K_x - \Lambda_{xx}(q = 0)]$$ (eq. 4).

![](assets/HBC/DS.png)
