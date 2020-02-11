using MonteCarlo, Statistics, LinearAlgebra

# This example reproduces data from
# "Attractive Hubbard model on a triangular lattice" - dos Santos, 1993
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.3976

# The paper investigated the attractive Hubbard model on a triangular lattice
# with
Ls = [4, 6, 8]
U = abs(-4.0)
betas = [2, 5, 7]   # also 6 once
sweeps = 1000       # 500 to 1000
Δτ = 0.125

# Note that the number of sweeps is relatively low, so numerical differences
# are to be expected. The Trotter error is of the order of Δτ^2 ≈ 0.016
# according to the paper.


################################################################################


μs = -2.0:0.25:2.0

for L in Ls
    # TODO
    # make figure
    for beta in betas

        # Data storage for plotting
        occs = zeros(length(μs))
        pcs = zeros(length(μs))

        for (i, μ) in enumerate(μs)
            # Set up the model
            # TODO make method requiring l instead of L and dims
            model = HubbardModelAttractive(
                L = L, dims = 2,
                l = TriangularLattice(L=L),
                U = U
            )

            # Set up DQMC Simulation
            dqmc = DQMC(
                model,
                beta = beta, delta_tau = Δτ, safe_mult = 8,
                thermalization = sweeps, sweeps=sweeps,
                measure_rate = 1,
                # Clear all measurements:
                measurements = Dict{Symbol, MonteCarlo.AbstractMeasurement}()
            )

            # Set up measurements
            push!(dqmc, :G => MonteCarlo.GreensMeasurement) # occupations
            push!(dqmc, :PC => MonteCarlo.PairingCorrelationMeasurement)

            # simulate!
            run!(dqmc)

            # save measurements for plotting
            m = measurements(dqmc)
            # Inner mean is the MC mean
            # diag grabs entries c_iσ c_iσ^†
            # Outer mean averages over sites and flavours/spins
            # 1.0 - mean(...) changes c_iσ c_iσ^† to c_iσ^† c_iσ
            # 2 reframes them as per-site occuptions
            occs[i] = 2.0 * (1.0 - mean(diag(mean(m[:G]))))
            # sum(...) / N computes the uniform (q=0) Fourier transform
            pcs[i] = sum(mean(m[:PC])) / nsites(model)
        end

        # TODO Plot data

    end
    # TODO: finalize plot
end

# TODO L = [6, 8], beta=8.0, U = abs(-4.0) CD structure factor

# TODO remove
@testset "DQMC: triangular Hubbard model vs dos Santos Paper" begin
    # > Attractive Hubbard model on a triangular lattice
    # dos Santos
    Random.seed!()
    sample_size = 5

    @time for (k, (mu, lit_oc, lit_pc,  beta, L)) in enumerate([
            (-2.0, 0.12, 1.0,  5.0, 4),
            (-1.2, 0.48, 1.50, 5.0, 4),
            ( 0.0, 0.88, 0.95, 5.0, 4),
            ( 1.2, 1.25, 1.55, 5.0, 4),
            ( 2.0, 2.00, 0.0,  5.0, 4)

            # (-2.0, 0.12, 1.0,  8.0, 4),
            # (-1.2, 0.48, 1.82, 8.0, 4),
            # ( 0.0, 0.88, 0.95, 8.0, 4),
            # ( 1.2, 1.25, 1.65, 8.0, 4),
            # ( 2.0, 2.00, 0.0,  8.0, 4),

            # (-2.0, 0.40, 1.0,  5.0, 6),
            # (-1.2, 0.40, 1.05, 5.0, 6),
            # (0.01, 0.80, 1.75, 5.0, 6),
            # ( 1.2, 1.40, 2.0,  5.0, 6),
            # ( 2.0, 2.00, 0.0,  5.0, 6)
        ])
        @info "[$(k)/5] μ = $mu (literature check)"
        m = HubbardModelAttractive(
            dims=2, L=L, l = MonteCarlo.TriangularLattice(L),
            t = 1.0, U = 4.0, mu = mu
        )
        OC_sample = []
        OC_errors = []
        PC_sample = []
        PC_errors = []
        for i in 1:sample_size
            mc = DQMC(
                m, beta=5.0, delta_tau=0.125, safe_mult=8,
                thermalization=2000, sweeps=2000, measure_rate=1,
                measurements = Dict{Symbol, MonteCarlo.AbstractMeasurement}()
            )
            push!(mc, :G => MonteCarlo.GreensMeasurement)
            push!(mc, :PC => MonteCarlo.PairingCorrelationMeasurement)
            run!(mc, verbose=false)
            measured = measurements(mc)

            # mean(measured[:G]) = MC mean
            # diag gets c_i c_i^† terms
            # 2 (1 - mean(c_i c_i^†)) = 2 mean(c_i^† c_i) where 2 follows from 2 spins
            occupation = 2 - 2(measured[:G].obs |> mean |> diag |> mean)
            push!(OC_sample, occupation)
            push!(OC_errors, 2(measured[:G].obs |> std_error |> diag |> mean))
            push!(PC_sample, measured[:PC].uniform_fourier |> mean)
            push!(PC_errors, measured[:PC].uniform_fourier |> std_error)
        end
        # min_error should compensate read-of errors & errors in the results
        # dos Santos used rather few sweeps, which seems to affect PC peaks strongly
        @test stat_equal(lit_oc, OC_sample, OC_errors, min_error=0.025)
        @test stat_equal(lit_pc, PC_sample, PC_errors, min_error=0.05)
    end
end
