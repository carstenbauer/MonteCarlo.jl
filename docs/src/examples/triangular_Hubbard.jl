################################################################################
### Occupation + Pairing Correlation
################################################################################


### Simulation
########################################

using MonteCarlo

# Jump between the different figures (Fig 1, 2, 3)
run_idx = 1

begin
    # Setup for different figures
    if run_idx == 1
        betas = (2.0, 5.0, 7.0)
        l = TriangularLattice(4)
        mus = vcat(-2.0, -1.5, -1.25:0.05:-1.0, -0.8:0.2:0.8, 0.9:0.05:1.25, 1.5, 2.0)
    elseif run_idx == 2
        betas = (2.0, 5.0, 7.0)
        l = TriangularLattice(6)
        mus = vcat(-2.0:0.25:-0.25, -0.1:0.1:1.1, 1.25, 1.5, 2.0)
    elseif run_idx == 3
        betas = (2.0, 5.0, 6.0, 7.0)
        l = TriangularLattice(8)
        mus = vcat(-2.0:0.5:-0.5, -0.1:0.1:1.1, 1.25, 1.5, 2.0)
    else
        error("Invalid run_idx = $run_idx")
    end

    dqmcs = []
    counter = 0
    N = length(mus) * length(betas)
    @time for beta in betas, mu in mus
        counter += 1
        print("\r[", lpad("$counter", 2), "/$N]")

        # Setup Simulation
        # Note that the pairing correlation formula used the paper is not the 
        # one MonteCarlo.jl uses by default, so we switch to the correct 
        # version via `kernel = MonteCarlo.pc_kernel`
        m = HubbardModel(l = l, t = 1.0, U = 4.0, mu = mu)
        dqmc = DQMC(
            m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
            thermalization = 1000, sweeps = 1000, measure_rate = 1,
            recorder = Discarder()
        )
        dqmc[:occ] = occupation(dqmc, m)
        dqmc[:PC] = pairing_correlation(dqmc, m, kernel = MonteCarlo.pc_kernel)

        # Thermalize & measure
        run!(dqmc, verbose = false)

        # for simplicity we just keep the whole simulation around
        push!(dqmcs, dqmc)
    end

    # Extract points for plots
    N = length(mus)
    occs = []
    Δoccs = []
    pcs = []
    Δpcs = []

    for i in 0:length(betas)-1
        # MonteCarlo.jl records occupations for each site in a LogBinner from 
        # BinningAnalysis.jl. The paper plots the average per site occupation.
        # This iterates through the simulations, picks the occupation 
        # measurement, calculates first the Monte Carlo average or standard 
        # error and second the site average. The factor then maps 0..1 
        # occupations to 0..2 occupations. (spin up + spin down)
        _occs = [2 * mean(mean(dqmcs[N*i + j][:occ])) for j in 1:N]
        doccs = [2 * mean(std_error(dqmcs[N*i + j][:occ])) for j in 1:N]
        push!(occs, _occs)
        push!(Δoccs, doccs)

        # For Pairing correlations the MonteCarlo representation is more 
        # complicated. For pairing correlations we consider 4 sites:
        #    (trg1, b1) ---- (src1, b1) ----- (src2, b2) ---- (trg2, b2)
        #               dir1            dir12            dir2
        # where b is a basis index, src and trg are Bravais site indices and
        # dir is a Bravauis direction between them. This is represented in 
        # MonteCarlo.jl by EachLocalQuadByDistance. The output of the 
        # measurement takes the form of a 5D tensor:
        #    PC[dir12, dir1, dir2, b1, b2]
        # To get the local s-wave pairing correlation used in the paper, we 
        # want dir1 = dir2 = 1. Since the triangular lattice has only one site 
        # per unit cell b1 = b2 = 1. And finally the paper considers the q = 0
        # pairing correlation, which is given by summing over dir12.
        _pcs = [sum(mean(dqmcs[N*i + j][:PC])[:, 1, 1, 1, 1]) for j in 1:N]
        dpcs = [sum(std_error(dqmcs[N*i + j][:PC])[:, 1, 1, 1, 1]) for j in 1:N]
        push!(pcs, _pcs)
        push!(Δpcs, dpcs)
    end
end


### Plotting
########################################

using CairoMakie, FileIO, Colors, LaTeXStrings

begin
    # Create layout
    fig = Figure(resolution = (800, 800))
    top = Axis(fig[1, 1], ylabel = L"\rho", ylabelsize = 30)
    bot = Axis(fig[2, 1], xlabel = L"\mu", ylabel = L"P_s", xlabelsize = 30, ylabelsize = 30)

    # Load reference images from paper
    p = joinpath(pkgdir(MonteCarlo), "docs/src/examples/assets/triangular")
    if run_idx == 1
        top_ref = FileIO.load(joinpath(p, "tri_Hub_ref1_1.png"))
        bot_ref = FileIO.load(joinpath(p, "tri_Hub_ref1_2.png"))
    elseif run_idx == 2
        top_ref = FileIO.load(joinpath(p, "tri_Hub_ref2_1.png"))
        bot_ref = FileIO.load(joinpath(p, "tri_Hub_ref2_2.png"))
    elseif run_idx == 3
        top_ref = FileIO.load(joinpath(p, "tri_Hub_ref3_1.png"))
        bot_ref = FileIO.load(joinpath(p, "tri_Hub_ref3_2.png"))
    end

    # Plot reference images in the background
    ip = image!(top, -2..2, 0..2, top_ref'[:, end:-1:1])
    translate!(ip, 0, 0, -1)
    if run_idx == 1
        ip = image!(bot, -2..2, 0..2.5, bot_ref'[:, end:-1:1], transparency=true)
    elseif run_idx == 2
        ip = image!(bot, -2..2, 0..3.5, bot_ref'[:, end:-1:1], transparency=true)
    elseif run_idx == 3
        ip = image!(bot, -2..2, 0..6.3, bot_ref'[:, end:-1:1], transparency=true)
    end
    translate!(ip, 0, 0, -1)

    # Plot our occupation data
    c = HSV(250, 0.6, 1)
    markers = length(betas) == 3 ? ('■', '□', 'o') : ('■', '□', '△', 'o')
    scatter_plots = []
    for (i, (ys, dys)) in enumerate(zip(occs, Δoccs))
        band!(top, mus, ys .- dys, ys .+ dys, color = (:red, 0.3))
        lines!(top, mus, ys, color = (c, 0.5), linewidth=2)
        s = scatter!(top, mus, ys, color = c, marker = markers[i])
        push!(scatter_plots, s)
    end

    # Add legend for scatter plots
    labels = map(beta -> "β = $beta", collect(betas))
    axislegend(top, scatter_plots, labels, position = :rb)

    # Plot our pairing correlations
    for (i, (ys, dys)) in enumerate(zip(pcs, Δpcs))
        band!(bot, mus, ys .- dys, ys .+ dys, color = (:red, 0.3), transparency=true)
        lines!(bot, mus, ys, color = (c, 0.5), linewidth=2)
        scatter!(bot, mus, ys, color = c, marker = markers[i])
    end

    # match limits to paper
    xlims!(top, -2 , 2)
    ylims!(top, 0 , 2)
    xlims!(bot, -2 , 2)
    run_idx == 1 && ylims!(bot, 0 , 2.5)
    run_idx == 2 && ylims!(bot, 0 , 3.5)
    run_idx == 3 && ylims!(bot, 0 , 6.3)

    display(fig)
end

# Save for docs
begin
    run_idx == 1 && CairoMakie.save(joinpath(p, "fig1_comparison.png"), fig)
    run_idx == 2 && CairoMakie.save(joinpath(p, "fig2_comparison.png"), fig)
    run_idx == 3 && CairoMakie.save(joinpath(p, "fig3_comparison.png"), fig)
end


################################################################################
### Charge Density
################################################################################


### Simulation
########################################

using MonteCarlo, LinearAlgebra

begin
    Ls = (6, 8)
    mus = (0.6, 0.6) # estimations for ρ = 1
    dqmcs = []

    counter = 0
    @time for (mu, L) in zip(mus, Ls)
        counter += 1
        print("\r[", lpad("$counter", 2), "/$(length(betas) * length(Ls))]")
        lattice = TriangularLattice(L)

        m = HubbardModelAttractive(l = lattice, t = 1.0, U = 4.0, mu = mu)
        dqmc = DQMC(
            m, beta = 8.0, delta_tau = 0.125, safe_mult = 8, 
            thermalization = 1000, sweeps = 1000, measure_rate = 1,
            recorder = Discarder()
        )
        # MonteCarlo.jl uses <n_i n_j> as the default pairing correlation, the
        # paper instead uses <n_i n_j> - <n_i><n_j>. The latter is given by
        # MonteCarlo.reduced_cdc_kernel
        dqmc[:CDC] = charge_density_correlation(dqmc, m, kernel = MonteCarlo.reduced_cdc_kernel)
        run!(dqmc, verbose = false)

        # for simplicity we just keep the whole simulation around
        push!(dqmcs, dqmc)
    end

    # Fourier transform
    ys = map(dqmcs) do dqmc
        # reciprocal path traced in paper
        L = size(lattice(dqmc))[1]
        N = div(L, 2)
        qs = vcat(
            range(Float64[0, 0],   Float64[pi, 0],  length = N+1),
            range(Float64[pi, 0],  Float64[pi, pi], length = N+1)[2:end],
            range(Float64[pi, pi], Float64[0, 0],   length = N+1)[2:end]
        )
        values = mean(dqmc[:CDC])
        dirs = directions(lattice(dqmc), dqmc[:CDC].lattice_iterator)
        map(qs) do q
            2 * real(sum(cis(dot(q, v)) * o for (v, o) in zip(dirs, values)))
        end
    end
end

### Plotting
########################################

using CairoMakie, FileIO, Colors, LaTeXStrings

begin
    fig = Figure(resolution = (800, 800))
    ax = Axis(fig[1, 1], ylabel = L"C(q_1, q_2)", ylabelsize = 30)

    # Load & Plot references
    p = joinpath(pkgdir(MonteCarlo), "docs/src/examples/assets/triangular")
    ref = FileIO.load(joinpath(p, "tri_Hubbard_CDSF.png"))
    ip = image!(ax, 0..3, 0..1.5, ref'[:, end:-1:1])
    translate!(ip, 0, 0, -1)

    # Plot our result
    c = HSV(250, 0.6, 1)
    for (i, ys) in enumerate(ys)
        xs = range(0, 3, length = length(ys))
        Makie.lines!(ax, xs, ys, color = (c, 0.5), linewidth=2)
        scatter!(ax, xs, ys, color = c, marker = ('■', 'o', 'x', '+', '*')[i])
    end

    # cleanup axis & limits
    ax.xticks[] = ([0,1,2,3], ["(0,0)", "(π, 0)", "(π,π)", "(0,0)"])
    xlims!(ax, 0, 3)
    ylims!(ax, 0 , 1.5)

    fig
end

# Save for docs
CairoMakie.save(joinpath(p, "fig6_comparison.png"), fig)


################################################################################
### Spin Susceptibility
################################################################################



### simulate
########################################

using MonteCarlo

begin
    Ls = (6,)
    betas = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    dqmcs = []

    counter = 0
    @time for beta in betas, L in Ls
        counter += 1
        print("\r[", lpad("$counter", 2), "/$(length(betas))]")
        lattice = TriangularLattice(L)

        m = HubbardModelAttractive(l = lattice, t = 1.0, U = 4.0, mu = 0.6)
        dqmc = DQMC(
            m, beta = beta, delta_tau = 0.125, safe_mult = 8, 
            thermalization = 1000, sweeps = 1000, measure_rate = 1,
            recorder = Discarder()
        )
        dqmc[:SDS] = spin_density_susceptibility(dqmc, m, :z)
        run!(dqmc, verbose = false)

        push!(dqmcs, dqmc)
    end

    xs = map(dqmc -> 1.0 / dqmc.parameters.beta, dqmcs)
    ys = map(dqmc -> (dqmc[:SDS] |> mean |> sum), dqmcs)
    dys = map(dqmc -> sqrt((dqmc[:SDS] |> std_error).^2 |> sum), dqmcs)
end


### Plotting
########################################

using CairoMakie, FileIO, Colors, LaTeXStrings

begin
    fig = Figure(resolution = (800, 800))
    ax = Axis(fig[1, 1], xlabel = "T(K)", ylabel = L"\chi(0, 0)", xlabelsize = 30, ylabelsize = 30)

    # References
    p = joinpath(pkgdir(MonteCarlo), "docs/src/examples/assets/triangular")
    ref = FileIO.load(joinpath(p, "tri_Hubbard_SDS.png"))
    ip = image!(ax, 0..1, 0..0.20, ref'[:, end:-1:1])
    translate!(ip, 0, 0, -1)

    # Our data
    c = HSV(250, 0.6, 1)
    band!(ax, xs, ys .- dys, ys .+ dys, color = (:red, 0.3))
    Makie.lines!(ax, xs, ys, color = (c, 0.5), linewidth=2)
    scatter!(ax, xs, ys, color = c, marker = ('■', 'o', 'x', '+', '*')[1])

    xlims!(ax, 0, 1)
    ylims!(ax, 0 , 0.20)

    fig
end

# Save for docs
CairoMakie.save(joinpath(p, "fig7_comparison.png"), fig)
