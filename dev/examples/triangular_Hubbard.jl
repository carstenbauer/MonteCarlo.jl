using MonteCarlo

betas = (2.0, 5.0, 6.0, 7.0)
mus = vcat(-2.0:0.5:-0.5, -0.1:0.1:1.1, 1.25, 1.5, 2.0)
# mus = vcat(-2.0:0.25:-0.25, -0.1:0.1:1.1, 1.25, 1.5, 2.0)
# mus = vcat(-2.0:0.5:-0.5, -0.1:0.1:1.1, 1.25, 1.5, 2.0)
lattice = TriangularLattice(4)
# lattice = TriangularLattice(6)
# lattice = TriangularLattice(8)
dqmcs = []

counter = 0
N = length(mus) * length(betas)
@time for beta in betas, mu in mus
    counter += 1
    print("\r[", lpad("$counter", 2), "/$N]")
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

N = length(mus)
occs = []
Δoccs = []
pcs = []
Δpcs = []

for i in 0:length(betas)-1
    # Measurements are saved in a LogBinner from BinningAnalysis by default.
    # Taking the mean (std_error) of a LogBinner will return the Monte Carlo 
    # average (error). Occupation measurements happen per site, so we need 
    # another mean afterwards. 
    _occs = [2 * mean(mean(dqmcs[N*i + j][:occ])) for j in 1:N]
    doccs = [2 * mean(std_error(dqmcs[N*i + j][:occ])) for j in 1:N]
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
    _pcs = [sum(mean(dqmcs[N*i + j][:PC])[:, 1, 1]) for j in 1:N]
    dpcs = [sum(std_error(dqmcs[N*i + j][:PC])[:, 1, 1]) for j in 1:N]
    push!(pcs, _pcs)
    push!(Δpcs, dpcs)
end

using CairoMakie, FileIO, Colors


fig = Figure(resolution = (800, 800))
top = Axis(fig[1, 1])
bot = Axis(fig[2, 1])

# References
p = pkgdir(MonteCarlo)
top_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref1_1.png"))
# top_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref2_1.png"))
# top_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref3_1.png"))
bot_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref1_2.png"))
# bot_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref2_2.png"))
# bot_ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hub_ref3_2.png"))
ip = image!(top, -2..2, 0..2, top_ref'[:, end:-1:1])
translate!(ip, 0, 0, -1)
ip = image!(bot, -2..2, 0..2.5, bot_ref'[:, end:-1:1], transparency=true)
# ip = image!(bot, -2..2, 0..3.5, bot_ref'[:, end:-1:1], transparency=true)
# ip = image!(bot, -2..2, 0..6.3, bot_ref'[:, end:-1:1], transparency=true)
translate!(ip, 0, 0, -1)

c = HSV(250, 0.6, 1)
for (i, (ys, dys)) in enumerate(zip(occs, Δoccs))
    band!(top, mus, ys .- dys, ys .+ dys, color = (:red, 0.3))
    lines!(top, mus, ys, color = (c, 0.5), linewidth=2)
    scatter!(top, mus, ys, color = c, marker = ('■', '□', '△', 'o')[i])
end

axislegend(top, top.scene.plots[4:3:end], ["β = 2", "β = 5", "β = 6", "β = 7"], position = :rb)

for (i, (ys, dys)) in enumerate(zip(pcs, Δpcs))
    band!(bot, mus, ys .- dys, ys .+ dys, color = (:red, 0.3), transparency=true)
    lines!(bot, mus, ys, color = (c, 0.5), linewidth=2)
    scatter!(bot, mus, ys, color = c, marker = ('■', '□', '△', 'o')[i])
end

xlims!(top, -2 , 2)
ylims!(top, 0 , 2)
xlims!(bot, -2 , 2)
ylims!(bot, 0 , 2.5)
# ylims!(bot, 0 , 3.5)
# ylims!(bot, 0 , 6.3)

display(fig)
# CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig1_comparison.png"), fig)
# CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig2_comparison.png"), fig)
# CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig3_comparison.png"), fig)


################################################################################


using MonteCarlo, LinearAlgebra

Ls = (6, 8)
betas = [8.0]
dqmcs = []

function my_kernel(mc, m, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    4 * (I[j, i] - G[j, i]) * G[i, j]
end

counter = 0
@time for beta in betas, L in Ls
    counter += 1
    print("\r[", lpad("$counter", 2), "/$(length(betas) * length(Ls))]")
    lattice = TriangularLattice(L)

    m = HubbardModelAttractive(l = lattice, t = 1.0, U = 4.0, mu = 0)
    dqmc = DQMC(
        m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
        thermalization = 1000, sweeps = 1000, measure_rate = 1,
        recorder = Discarder
    )
    dqmc[:CDC] = MonteCarlo.Measurement(dqmc, m, Greens, EachSitePairByDistance, my_kernel)
    run!(dqmc, verbose = false)

    # for simplicity we just keep the whole simulation around
    push!(dqmcs, dqmc)
end

qs = vcat(
    range(Float64[0, 0],   Float64[pi, 0], length=10),
    range(Float64[pi, 0],  Float64[pi, pi], length=10),
    range(Float64[pi, pi], Float64[0, 0], length=10),
)
ys = map(dqmcs) do dqmc
    MonteCarlo.fourier_transform(
        qs, directions(MonteCarlo.lattice(dqmc)), mean(dqmc[:CDC])
    ) |> real
end

using CairoMakie, FileIO, Colors

fig = Figure(resolution = (800, 800))
ax = Axis(fig[1, 1])

# References
p = pkgdir(MonteCarlo)
ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hubbard_CDSF.png"))
ip = image!(ax, 0..3, 0..1.5, ref'[:, end:-1:1])
translate!(ip, 0, 0, -1)

c = HSV(250, 0.6, 1)
for (i, ys) in enumerate(ys)
    xs = vcat(range(0, 1, length=10), range(1, 2, length=10), range(2, 3, length=10))
    Makie.lines!(ax, xs, ys, color = (c, 0.5), linewidth=2)
    scatter!(ax, xs, ys, color = c, marker = ('■', 'o', 'x', '+', '*')[i])
end

ax.xticks[] = ([0,1,2,3], ["(0,0)", "(π, 0)", "(π,π)", "(0,0)"])
xlims!(ax, 0, 3)
ylims!(ax, 0 , 1.5)

fig
# CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig6_comparison.png"), fig)


################################################################################


using MonteCarlo, LinearAlgebra

Ls = (6,)
betas = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
dqmcs = []

counter = 0
# @time for beta in (8.0,), L in (6, )
@time for beta in betas, L in Ls
    counter += 1
    print("\r[", lpad("$counter", 2), "/$(length(betas))]")
    lattice = TriangularLattice(L)

    m = HubbardModelAttractive(l = lattice, t = 1.0, U = 4.0, mu = 0)
    dqmc = DQMC(
        m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
        thermalization = 1000, sweeps = 1000, measure_rate = 1,
        recorder = Discarder
    )
    dqmc[:SDS] = spin_density_susceptibility(dqmc, m, :z)
    run!(dqmc, verbose = false)

    # for simplicity we just keep the whole simulation around
    push!(dqmcs, dqmc)
end

xs = map(dqmc -> 1.0 / dqmc.parameters.beta, dqmcs)
ys = map(dqmc -> (dqmc[:SDS] |> mean |> sum), dqmcs)
dys = map(dqmc -> sqrt((dqmc[:SDS] |> std_error).^2 |> sum), dqmcs)

using CairoMakie, FileIO, Colors

fig = Figure(resolution = (800, 800))
ax = Axis(fig[1, 1])

# References
p = pkgdir(MonteCarlo)
ref = FileIO.load(joinpath(p, "docs/src/examples/assets/tri_Hubbard_SDS.png"))
ip = image!(ax, 0..1, 0..0.20, ref'[:, end:-1:1])
translate!(ip, 0, 0, -1)

c = HSV(250, 0.6, 1)
band!(ax, xs, ys .- dys, ys .+ dys, color = (:red, 0.3))
Makie.lines!(ax, xs, ys, color = (c, 0.5), linewidth=2)
scatter!(ax, xs, ys, color = c, marker = ('■', 'o', 'x', '+', '*')[1])

xlims!(ax, 0, 1)
ylims!(ax, 0 , 0.20)

fig
# CairoMakie.save(joinpath(p, "docs/src/examples/assets/fig7_comparison.png"), fig)
