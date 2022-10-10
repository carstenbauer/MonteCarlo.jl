function restructure_compare(Q, G, L)
    N = L^2
    K = div(size(Q, 1), N)

    for k1 in 0:N:N*K-1, k2 in 0:N:N*K-1
        for src_x in 1:L, src_y in 1:L
            src = src_x + L * (src_y - 1) + k1
            for trg_x in 1:L, trg_y in 1:L
                trg = trg_x + L * (trg_y - 1) + k2
                dx = ((trg_x - src_x + L) % L)
                dy = ((trg_y - src_y + L) % L)
                dr = 1 + dx + L * dy + k2

                if !(Q[src, dr] ≈ G[src, trg])
                    @info "G[$src, $trg] = $(G[src, trg]) ≠ $(Q[src, dr]) = Q[$src, $dr]"
                    return false
                end
            end
        end
    end

    return true
end

function _setup_measurements!(mc)
    m = mc.model
    mc[:PC] = pairing_susceptibility(mc, m)
    mc[:PC1] = pairing_susceptibility(mc, m, kernel = MonteCarlo.pc_kernel)
    mc[:PC2] = pairing_susceptibility(mc, m, kernel = MonteCarlo.pc_alt_kernel)
    mc[:CDC] = charge_density_susceptibility(mc, m)
    mc[:SDCx]  = spin_density_susceptibility(mc, m, :x)
    mc[:SDCy]  = spin_density_susceptibility(mc, m, :y)
    mc[:SDCz]  = spin_density_susceptibility(mc, m, :z)
    mc[:CCS] = current_current_susceptibility(mc, m)
    
    mc[:rPC] = pairing_susceptibility(mc, m, wrapper = MonteCarlo.Restructure)
    mc[:rPC1] = pairing_susceptibility(
        mc, m, wrapper = MonteCarlo.Restructure, kernel = MonteCarlo.pc_kernel
    )
    mc[:rPC2] = pairing_susceptibility(
        mc, m, wrapper = MonteCarlo.Restructure, kernel = MonteCarlo.pc_alt_kernel
    )
    mc[:rCCS] = current_current_susceptibility(mc, m, wrapper = MonteCarlo.Restructure)
    mc[:rCDC] = charge_density_susceptibility(mc, m, wrapper = MonteCarlo.Restructure)
    mc[:rSDCx] = spin_density_susceptibility(mc, m, :x, wrapper = MonteCarlo.Restructure)
    mc[:rSDCy] = spin_density_susceptibility(mc, m, :y, wrapper = MonteCarlo.Restructure)
    mc[:rSDCz] = spin_density_susceptibility(mc, m, :z, wrapper = MonteCarlo.Restructure)
    return
end

function complete_currents(mc, key)
    # Currents are computed with all bonds rather than a subset of bonds for
    # restructered greens functions. (This makes things more cache friendly)
    # We also moved iT[src, trg]out, since hoppings should distance based and 
    # therefore applicable after the simulation. 
    l = lattice(mc)
    Lx, Ly = l.Ls
    m = mc[key]
    x = mean(m)
    # bs = view(l.unitcell.bonds, m.lattice_iterator.wrapped.bond_idxs)
    bs = l.unitcell.bonds
    reversals = MonteCarlo.reverse_bond_map(l)
    # This should only keep one of reversals[i] = j & reversals[j] = i
    reversals = [f => r for (f, r) in enumerate(reversals) if r > f]

    output = Array{eltype(x)}(undef, size(x, 1), size(x, 2), length(reversals), length(reversals))

    for (f2, r2) in reversals
        c, d = bs[f2].uc_shift
        for (f1, r1) in reversals
            a, b = bs[f1].uc_shift

            for j in axes(x, 2), i in axes(x, 1)
                output[i, j, f1, f2] = x[i, j, f1, f2]
            end

            for j in axes(x, 2), i in axes(x, 1)
                output[i, j, f1, f2] -= x[mod1(i+c+Lx, Lx), mod1(j+d+Ly, Ly), f1, r2]
            end

            for j in axes(x, 2), i in axes(x, 1)
                output[i, j, f1, f2] -= x[mod1(i-a+Lx, Lx), mod1(j-b+Ly, Ly), r1, f2]
            end

            for j in axes(x, 2), i in axes(x, 1)
                output[i, j, f1, f2] += x[mod1(i-a+c+Lx, Lx), mod1(j-b+d+Ly, Ly), r1, r2]
            end

        end
    end

    # NOTE i^2 here currently
    return -output
end

@testset "Restructuring" begin
    m = RepulsiveHubbardModel(L = 2, dims = 2)
    mc = DQMC(m, beta=1.0)
    MonteCarlo.init!(mc)

    # The restructuring code figures out the number of flavors by comparing 
    # the input matrix size and the lattice size, so we can test both 1 and 2 
    # flavors here
    G1 = MonteCarlo.GreensMatrix(1, 3, rand(8, 8)) # 2 flavors
    G2 = MonteCarlo.StructArray(rand(ComplexF64, 4, 4)) # 1 flavor
    G3 = MonteCarlo.BlockDiagonal(rand(4, 4), rand(4,4)) # 1 flavor
    G4 = MonteCarlo.DiagonallyRepeatingMatrix(rand(4, 4)) # 1 flavor

    Q1, Q2, Q3, Q4 = deepcopy((G1, G2, G3, G4))
    MonteCarlo.restructure!(mc, Q1, temp = similar(Q1))
    MonteCarlo.restructure!(mc, Q2, temp = similar(Q2))
    MonteCarlo.restructure!(mc, (Q3,))
    MonteCarlo.restructure!(mc, Q4, temp = mc.stack.curr_U.blocks[1])
    
    @test Q1 isa GreensMatrix
    @test size(Q1) == (8, 8)
    @test restructure_compare(Q1, G1, 2)
    
    @test Q2 isa MonteCarlo.StructArray
    @test size(Q2) == (4, 4)
    @test restructure_compare(Q2.re, G2.re, 2)
    @test restructure_compare(Q2.im, G2.im, 2)
    
    @test Q3 isa MonteCarlo.BlockDiagonal
    @test length(Q3.blocks) == 2
    @test size(Q3.blocks[1]) == (4, 4)
    @test size(Q3.blocks[2]) == (4, 4)
    @test restructure_compare(Q3.blocks[1], G3.blocks[1], 2)
    @test restructure_compare(Q3.blocks[2], G3.blocks[2], 2)

    @test Q4 isa MonteCarlo.DiagonallyRepeatingMatrix
    @test size(Q4.val) == (4, 4)
    @test restructure_compare(Q4.val, G4.val, 2)

    @testset "DiagonallyRepeating Matrix, real, 2 basis sites, 1 flavor" begin
        m = HubbardModel(Honeycomb(3))
        # m = HubbardModel(L = 4, dims = 2)
        mc = DQMC(m, beta = 2.8, measure_rate = 1, silent = true)
        _setup_measurements!(mc)
        run!(mc, verbose = false)
        for key in (:PC, :PC1, :PC2, :CDC, :SDCx, :SDCy, :SDCz)
            matches = mean(mc[key]) ≈ mean(mc[Symbol('r', key)])
            matches || @info key
            @test matches
        end
        @test mean(mc[:CCS]) ≈ complete_currents(mc, :rCCS)
    end

    @testset "BlockDiagonal Matrix, real, 1 basis site, 2 flavors" begin
        m = RepulsiveHubbardModel(L = 4, dims = 2)
        mc = DQMC(m, beta = 2.3, measure_rate = 1, silent = true)
        _setup_measurements!(mc)
        run!(mc, verbose = false)
        for key in (:PC, :PC1, :PC2, :CDC, :SDCx, :SDCy, :SDCz)
            matches = mean(mc[key]) ≈ mean(mc[Symbol('r', key)])
            matches || @info key
            @test matches
        end
        @test mean(mc[:CCS]) ≈ complete_currents(mc, :rCCS)
    end

    # TODO 2 flavors normal Matrix - needs special model?
end


function calc_measured_greens(mc::DQMC, G::Matrix)
    eThalfminus = mc.stack.hopping_matrix_exp
    eThalfplus = mc.stack.hopping_matrix_exp_inv

    eThalfplus * G * eThalfminus
end

@testset "Measured Greens function" begin
    m = HubbardModel(8, 2, mu=0.5)
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.init!(mc)
    MonteCarlo.build_stack(mc, mc.stack)
    MonteCarlo.propagate(mc)

    # greens(mc) matches expected output
    @test greens(mc).val ≈ calc_measured_greens(mc, mc.stack.greens)

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice - k, -1)
    end
    # greens(mc) matches expected output
    @test greens(mc).val ≈ calc_measured_greens(mc, mc.stack.greens)
end

@testset "DQMC Measurement constructors" begin
    for m1 in (HubbardModel(4, 2), HubbardModel(4, 2, U = -1.0))
        mc = DQMC(m1, beta=1.0, safe_mult=1)

        fi = if MonteCarlo.unique_flavors(mc) == 2
            [(1, 1), (1, 2), (2, 1), (2, 2)]
        else
            [(1, 1),]
        end

        # Greens
        m = greens_measurement(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator === nothing
        @test m.flavor_iterator === nothing
        @test m.kernel == MonteCarlo.greens_kernel
        @test m.observable isa LogBinner{Matrix{Float64}}
        @test m.temp === nothing

        # Occupation
        m = occupation(mc, m1)
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == Greens()
        @test m.lattice_iterator === nothing
        @test m.flavor_iterator === nothing
        @test m.kernel == MonteCarlo.occupation_kernel
        @test m.observable isa LogBinner{Vector{Float64}}
        @test m.temp isa Vector{Float64}

        for time in (:equal, :unequal)
            # Charge densities
            if time == :equal
                m = charge_density_correlation(mc, m1)
                @test m.greens_iterator == Greens()
            else
                m = charge_density_susceptibility(mc, m1)
                @test m.greens_iterator == TimeIntegral(mc)
            end
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.lattice_iterator == EachSitePairByDistance()
            @test m.flavor_iterator == fi
            @test m.kernel == MonteCarlo.full_cdc_kernel
            @test m.observable isa LogBinner{Array{Float64, 3}}
            @test m.temp isa Array{Float64, 3}

            # Spin densities
            for dir in (:x, :y, :z)
                if time == :equal
                    m = spin_density_correlation(mc, m1, dir)
                    @test m.greens_iterator == Greens()
                else
                    m = spin_density_susceptibility(mc, m1, dir)
                    @test m.greens_iterator == TimeIntegral(mc)
                end
                @test m isa MonteCarlo.DQMCMeasurement
                @test m.lattice_iterator == EachSitePairByDistance()
                @test m.kernel == Core.eval(MonteCarlo, Symbol(:full_sdc_, dir, :_kernel))
                @test m.observable isa LogBinner{Array{Float64, 3}}
                @test m.temp isa Array{Float64, 3}
            end

            # pairings
            if time == :equal
                m = pairing_correlation(mc, m1)
                @test m.greens_iterator == Greens()
            else
                m = pairing_susceptibility(mc, m1)
                @test m.greens_iterator == TimeIntegral(mc)
            end
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.lattice_iterator == EachLocalQuadByDistance(1:5)
            @test m.flavor_iterator == 2
            @test m.kernel == MonteCarlo.pc_combined_kernel
            @test m.observable isa LogBinner{Array{Float64, 5}}
            @test m.temp isa Array{Float64, 5}
        end

        # Magnetizations
        for dir in (:x, :y, :z)
            m = magnetization(mc, m1, dir)
            @test m isa MonteCarlo.DQMCMeasurement
            @test m.greens_iterator == Greens()
            @test m.lattice_iterator == EachSite()
            @test m.flavor_iterator == 2
            @test m.kernel == Core.eval(MonteCarlo, Symbol(:m, dir, :_kernel))
            @test m.observable isa LogBinner{Vector{Float64}}
            @test m.temp isa Vector{Float64}
        end

        # Current Current susceptibility
        m = current_current_susceptibility(mc, m1, lattice_iterator = EachLocalQuadBySyncedDistance(2:5))
        @test m isa MonteCarlo.DQMCMeasurement
        @test m.greens_iterator == TimeIntegral(mc)
        @test m.lattice_iterator == EachLocalQuadBySyncedDistance(2:5)
        @test m.flavor_iterator == fi
        @test m.kernel == MonteCarlo.cc_kernel
        @test m.observable isa LogBinner{Array{Float64, 4}}
        @test m.temp isa Array{Float64, 4}
    end

    m = HubbardModel(4, 2)
    mc = DQMC(m, beta=1.0, safe_mult=1)
    add_default_measurements!(mc)

    @test !haskey(mc, :occ) # skipped with :G
    @test !haskey(mc, :Mx) # skipped with :G
    @test !haskey(mc, :My) # skipped with :G
    @test !haskey(mc, :Mz) # skipped with :G
    @test haskey(mc, :G) && (mc[:G].kernel == MonteCarlo.greens_kernel)
    @test haskey(mc, :K) && (mc[:K].kernel == MonteCarlo.kinetic_energy_kernel)
    @test haskey(mc, :V) && (mc[:V].kernel == MonteCarlo.interaction_energy_kernel)
    @test !haskey(mc, :E) # skipped with :K, :V

    @test haskey(mc, :CDC)  && (mc[:CDC].kernel == MonteCarlo.full_cdc_kernel)
    @test haskey(mc, :PC)   && (mc[:PC].kernel == MonteCarlo.pc_combined_kernel)
    @test haskey(mc, :SDCx) && (mc[:SDCx].kernel == MonteCarlo.full_sdc_x_kernel)
    @test haskey(mc, :SDCy) && (mc[:SDCy].kernel == MonteCarlo.full_sdc_y_kernel)
    @test haskey(mc, :SDCz) && (mc[:SDCz].kernel == MonteCarlo.full_sdc_z_kernel)

    @test haskey(mc, :CDS)  && (mc[:CDS].kernel == MonteCarlo.full_cdc_kernel)
    @test haskey(mc, :PS)   && (mc[:PS].kernel == MonteCarlo.pc_combined_kernel)
    @test haskey(mc, :SDSx) && (mc[:SDSx].kernel == MonteCarlo.full_sdc_x_kernel)
    @test haskey(mc, :SDSy) && (mc[:SDSy].kernel == MonteCarlo.full_sdc_y_kernel)
    @test haskey(mc, :SDSz) && (mc[:SDSz].kernel == MonteCarlo.full_sdc_z_kernel)
    @test haskey(mc, :CCS)  && (mc[:CCS].kernel == MonteCarlo.cc_kernel)
end