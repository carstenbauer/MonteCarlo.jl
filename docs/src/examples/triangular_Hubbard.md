# Triangular Attractive Hubbard 

This example implements the model from the paper [Attractive Hubbard model on a triangular lattice](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.3976). This also serves as a cross-check for the DQMC and a few measurements, such as pairing correlations and occupations. 

In the paper simulations were done with $500-1000$ sweeps and $\Delta\tau = 0.125$. The first figure is done with interaction strength $U = -4$, linear system size $L = 4$ at $\beta = 2, 5, 7$ for varying $\mu$. The following simulations should take about 2 minutes.

```julia
using MonteCarlo

betas = (2.0, 5.0, 7.0)
mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
lattice = TriangularLattice(4)
dqmcs = []

counter = 0
@time for beta in betas, mu in mus
    counter += 1
    print("\r[", lpad("$counter", 2), "/81]")
    m = HubbardModelAttractive(l = lattice, t = 1.0, U = 4.0, mu = mu)
    dqmc = DQMC(
        m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
        thermalization = 1000, sweeps = 1000, measure_rate = 1,
        recorder = Discarder
    )
    dqmc[:occ] = occupation(dqmc, m)
    dqmc[:PC] = pairing_correlation(dqmc, m)
    run!(dqmc, verbose = false)

    # for simplicity we just keep the whole simulation around
    push!(dqmcs, dqmc)
end
```

After running all the simulations we need to do a little bit of post-processing on the measured data. 

```julia
betas = (2.0, 5.0, 7.0)
mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
occs = []
Δoccs = []
pcs = []
Δpcs = []

for i in 0:2
    # Measurements are saved in a LogBinner from BinningAnalysis by default.
    # Taking the mean (std_error) of a LogBinner will return the Monte Carlo 
    # average (error). Occupation measurements happen per site, so we need 
    # another mean afterwards. 
    _occs = [2 * mean(mean(dqmcs[27*i + j][:occ])) for j in 1:27]
    doccs = [2 * mean(std_error(dqmcs[27*i + j][:occ])) for j in 1:27]
    push!(occs, _occs)
    push!(Δoccs, doccs)

    # pairing correlations are saved in a partially processed state - a 3D matrix
    # where each index corresponds to vectors between sites
    # y_{i, j, k} = ∑_x ⟨c_{x, ↑} c_{x+j, ↓} c_{x+i+k, ↓}^† c_{x+i, ↑}^†
    # The vectors corresponding to the indices i, j, k are returned by 
    # directions(lattice(dqmc)). To compute the pairing correlation of a certain
    # symmetry, we need to apply the weights corresponding vector indices j, k.
    # For s-wave symmetry these weights are always (1, 0, ..., 0) (only vector 0).
    # To match the paper the index i should just be summed over. This is 
    # equivalent to a q=0 Fourier transform.
    _pcs = [sum(mean(dqmcs[27*i + j][:PC])[:, 1, 1]) for j in 1:27]
    dpcs = [sum(std_error(dqmcs[27*i + j][:PC])[:, 1, 1]) for j in 1:27]
    push!(pcs, _pcs)
    push!(Δpcs, dpcs)
end
```

With the data in a processed form we can now plot it. To make comparison easier, we plot our data directly over figure 1 from our [reference](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.397).

```julia
using CairoMakie, FileIO, Colors


fig = Figure(resolution = (800, 800))
top = Axis(fig[1, 1])
bot = Axis(fig[2, 1])

# References
p = pkgdir(MonteCarlo)
top_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref1_1.png"))
bot_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref1_2.png"))
ip = image!(top, -2..2, 0..2, top_ref'[:, end:-1:1])
translate!(ip, 0, 0, -1)
ip = image!(bot, -2..2, 0..2.5, bot_ref'[:, end:-1:1], transparency=true)
translate!(ip, 0, 0, -1)

c = HSV(250, 0.6, 1)
for (i, (ys, dys)) in enumerate(zip(occs, Δoccs))
    band!(top, mus, ys .- dys, ys .+ dys, color = (:red, 0.3))
    lines!(top, mus, ys, color = (c, 0.5), linewidth=2)
    scatter!(top, mus, ys, color = c, marker = ('■', '□', 'o')[i])
end

axislegend(top, top.scene.plots[4:3:end], ["β = 2", "β = 5", "β = 7"], position = :rb)

for (i, (ys, dys)) in enumerate(zip(pcs, Δpcs))
    band!(bot, mus, ys .- dys, ys .+ dys, color = (:red, 0.3), transparency=true)
    lines!(bot, mus, ys, color = (c, 0.5), linewidth=2)
    scatter!(bot, mus, ys, color = c, marker = ('■', '□', 'o')[i])
end

xlims!(top, -2 , 2)
ylims!(top, 0 , 2)
xlims!(bot, -2 , 2)
ylims!(bot, 0 , 2.5)

CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig1_comparison.png"), fig)
```

![](assets/fig1_comparison.png)