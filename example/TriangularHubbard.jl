using MonteCarlo, Statistics, LinearAlgebra
using PyPlot, LaTeXStrings

# This example reproduces data from
# "Attractive Hubbard model on a triangular lattice" - dos Santos, 1993
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.3976

# The paper investigated the attractive Hubbard model on a triangular lattice
# with
Ls = [4, 6, 8]
U = abs(-4.0)
betas = [2.0, 5.0, 7.0]   # also 6 once
sweeps = 1000       # 500 to 1000
Δτ = 0.125

# Note that the number of sweeps is relatively low, so numerical differences
# are to be expected. The Trotter error is of the order of Δτ^2 ≈ 0.016
# according to the paper.


################################################################################


μs = -2.0:0.25:2.0

for L in Ls
    # initialize Plots
    fig, ax = subplots(2, 1, figsize=(8, 8))
    fmts = ["ks-", "bv-", "go-", "rd-"]

    for (j, beta) in enumerate(betas)

        # Data storage for plotting
        occs = zeros(length(μs))
        Δoccs = zeros(length(μs))
        pcs = zeros(length(μs))
        Δpcs = zeros(length(μs))

        for (i, μ) in enumerate(μs)
            @info "[$L] [$beta] [$μ]"
            # Set up the model
            # TODO make method requiring l instead of L and dims
            model = HubbardModelAttractive(
                L = L, dims = 2,
                l = MonteCarlo.TriangularLattice(L),
                U = U, mu = μ
            )

            # Set up DQMC Simulation
            dqmc = DQMC(
                model,
                beta = beta, delta_tau = Δτ, safe_mult = 8,
                thermalization = sweeps, sweeps=sweeps,
                measure_rate = 1,
                # checkerboard = true,
                # Clear all measurements:
                measurements = Dict{Symbol, MonteCarlo.AbstractMeasurement}()
            )

            # Set up measurements
            push!(dqmc, :G => MonteCarlo.GreensMeasurement) # occupations
            push!(dqmc, :PC => MonteCarlo.PairingCorrelationMeasurement)

            # simulate!
            run!(dqmc, verbose=false)

            # save measurements for plotting
            measured = measurements(dqmc)
            # Inner mean is the MC mean
            # diag grabs entries c_iσ c_iσ^†
            # Outer mean averages over sites and flavours/spins
            # 1.0 - mean(...) changes c_iσ c_iσ^† to c_iσ^† c_iσ
            # 2 reframes them as per-site occuptions
            occs[i] = 2.0 * (1.0 - mean(diag(mean(measured[:G]))))
            Δoccs[i] = 2.0 * maximum(diag(std_error(measured[:G])))
            # sum(...) / N computes the uniform (q=0) Fourier transform
            pcs[i] = sum(mean(measured[:PC])) / MonteCarlo.nsites(model)
            Δpcs[i] = maximum(std_error(measured[:PC]))
        end

        # Plot data
        ax[1].errorbar(μs, occs, yerr=Δoccs, fmt=fmts[j], label=string(beta))
        ax[2].errorbar(μs, pcs, yerr=Δpcs, fmt=fmts[j])
    end

    # Finalize plot
    ax[1].set_ylim([0, 2])
    ax[1].set_xlim([-2, 2])
    ax[2].set_xlim([-2, 2])
    ax[2].set_xlabel(L"\mu")
    ax[1].set_ylabel("Occupation")
    ax[2].set_ylabel("Equal-time pairing correlation (s-wave)")
    ax[1].set_title("L = $L")
end

# TODO L = [6, 8], beta=8.0, U = abs(-4.0) CD structure factor
