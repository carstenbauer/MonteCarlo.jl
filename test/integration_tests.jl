@testset "MC: IsingModel Simulation" begin
    Random.seed!(123)
    m = IsingModel(dims=2, L=8);
    mc = MC(m, beta=0.35);
    run!(mc, sweeps=1000, thermalization=10, verbose=false);

    # Check measurements
    measured = measurements(mc)
    @test   25.47  ≈ measured[:Magn].M |> mean         atol=0.01
    @test    0.82  ≈ measured[:Magn].M |> std_error    atol=0.01
    @test  887.    ≈ measured[:Magn].M2 |> mean        atol=1.0
    @test   46.    ≈ measured[:Magn].M2 |> std_error   atol=1.0
    @test    0.398 ≈ measured[:Magn].m |> mean         atol=0.001
    @test    0.013 ≈ measured[:Magn].m |> std_error    atol=0.001
    @test    1.300 ≈ measured[:Magn].chi |> mean       atol=0.001

    @test  -55.10  ≈ measured[:Energy].E |> mean       atol=0.01
    @test    0.88  ≈ measured[:Energy].E |> std_error  atol=0.01
    @test 3342.    ≈ measured[:Energy].E2 |> mean      atol=1.0
    @test  104.    ≈ measured[:Energy].E2 |> std_error atol=1.0
    @test   -0.861 ≈ measured[:Energy].e |> mean       atol=0.001
    @test    0.014 ≈ measured[:Energy].e |> std_error  atol=0.001
    @test    0.585 ≈ measured[:Energy].C |> mean       atol=0.001

    @test [
        -0.028 -0.02 0.022 0.018 0.002 -0.008 -0.026 -0.058; -0.016 -0.028 0.004 0.014 0.024 0.024 -0.026 -0.044; -0.048 -0.018 0.004 0.02 0.02 0.002 -0.016 -0.018; -0.028 -0.016 0.016 -0.02 -0.018 -0.024 -0.026 -0.03; -0.02 -0.016 -0.024 -0.044 -0.028 -0.006 -0.026 -0.05; -0.008 -0.006 0.004 -0.038 -0.038 -0.016 -0.03 -0.022; -0.028 -0.018 0.014 -0.01 -0.018 0.008 0.008 -0.05; -0.038 -0.028 0.016 0.026 -0.018 -0.022 -0.018 -0.054
    ] ≈ measured[:conf].obs |> mean                    atol=0.001
    @test last(measured[:conf].obs) == mc.conf
end


@testset "DQMC: HubbardModel Simulation" begin
    Random.seed!(123)
    m = HubbardModelAttractive(dims=2, L=4);
    mc = DQMC(m, beta=1.0);
    run!(mc, sweeps=1000, thermalization=10, verbose=false);

    # Check measurements
    measured = measurements(mc)
    @test [
        0.477228 -0.148092 0.000650671 -0.152739 -0.156428 0.0050096 0.0343558 0.00164354 0.000686254 0.0314209 -0.00315709 0.0363441 -0.149735 9.70741e-5 0.0335553 0.00358477; -0.153774 0.454577 -0.160642 0.000648542 0.0126497 -0.159687 0.00669076 0.0333576 0.037061 0.00569243 0.0358175 -0.00404491 0.00355269 -0.16096 0.0045801 0.0340806; 0.00137329 -0.149875 0.525756 -0.156449 0.0352624 0.000278655 -0.152092 -0.00291045 0.000989166 0.0322977 -0.00177604 0.0358907 0.0322688 -0.00133419 -0.151118 0.00029095; -0.145036 0.000208847 -0.148504 0.520469 0.00232344 0.0322997 -0.00403395 -0.141013 0.0340725 -0.00152847 0.0315167 -0.0043278 -0.0022816 0.032141 -0.00106526 -0.149073; -0.153225 0.0116195 0.0344808 0.00197823 0.459859 -0.154248 0.00721501 -0.148024 -0.157576 0.00551732 0.0349254 0.00233362 0.00219748 0.0371333 -0.00627411 0.03385; 0.00507538 -0.151558 -0.000391151 0.0329975 -0.155022 0.479573 -0.150905 0.00226697 0.00295499 -0.148607 0.00301816 0.0347823 0.0315428 0.00392446 0.0326532 -0.00249034; 0.0351943 0.00306724 -0.159841 -0.00715043 0.00575413 -0.155764 0.491007 -0.148957 0.0353336 0.00576441 -0.153046 -0.00141562 -0.000898893 0.0362212 0.00138285 0.0337357; 0.00118721 0.0348373 -0.00172386 -0.166127 -0.168602 0.000460718 -0.16182 0.508259 -0.000832228 0.036381 0.000702338 -0.165049 0.0363685 0.000530091 0.0375223 3.16583e-7; -0.000531546 0.031598 0.000765815 0.0349322 -0.150373 0.00357689 0.032698 -0.00108294 0.507991 -0.14935 0.00185427 -0.154108 -0.147832 -0.00104978 0.032203 -0.000816359; 0.0354625 0.00767388 0.0356703 0.000856574 0.00576076 -0.152291 0.0023841 0.0328979 -0.157559 0.475779 -0.156894 -0.00115001 -0.00177241 -0.158056 0.00557598 0.0338655; -0.00215687 0.0327828 -0.00226684 0.0351669 0.0352777 0.00517752 -0.149678 -0.000851672 0.00202061 -0.151299 0.480858 -0.157663 0.0346624 6.83759e-5 -0.155145 0.000856897; 0.0346939 -0.00253293 0.0333441 -0.00604537 -0.000972427 0.034324 -0.000290598 -0.147195 -0.160874 -0.00191725 -0.157674 0.52118 -0.00359494 0.0360121 0.00140374 -0.153626; -0.156798 0.00415648 0.0344931 -0.00219987 0.00366139 0.0365817 -0.00100057 0.0324253 -0.163216 -0.00268529 0.0365863 -0.00464487 0.526158 -0.162021 0.00374309 -0.151599; -0.000494114 -0.148824 -0.00554303 0.0321295 0.0336398 0.00246292 0.0314112 -0.00022372 -0.00163466 -0.149309 0.00169644 0.0332757 -0.149364 0.494472 -0.146979 -0.001565; 0.0343884 0.00516688 -0.153669 -0.00249342 -0.00596883 0.0360426 0.00222142 0.0311754 0.0350533 0.00385871 -0.154668 0.00290686 4.46472e-5 -0.155343 0.498268 -0.146554; 0.00117938 0.0336081 0.000884403 -0.159131 0.0366746 -0.00326657 0.0348769 0.00402202 -0.00051693 0.0347719 0.00395268 -0.160173 -0.154205 0.000746075 -0.154492 0.47354
    ] ≈ measured[:Greens].obs |> mean           atol=0.0001
    @test [
        0.0205191 0.00466986 0.00319713 0.00433101 0.0063448 0.00298264 0.00188447 0.00270596 0.00279388 0.00147149 0.00223948 0.00191749 0.00544694 0.0034801 0.00166324 0.00279062; 0.00581179 0.0173809 0.00615989 0.00298333 0.00264239 0.00544051 0.00288374 0.00180361 0.00208151 0.00286832 0.0019582 0.0017646 0.00267498 0.00597742 0.00256961 0.00180452; 0.00304324 0.00535371 0.0171345 0.00603167 0.00188977 0.00215002 0.00496448 0.00224604 0.0019083 0.00161783 0.00226727 0.00188134 0.00147511 0.00266665 0.00549195 0.00243711; 0.00409947 0.00231981 0.00510606 0.0172372 0.00209903 0.00155016 0.00232761 0.00370052 0.0018203 0.00180111 0.00127228 0.00238768 0.00236927 0.00137936 0.00228196 0.00505507; 0.0059045 0.0025362 0.00187802 0.00270614 0.0191752 0.00584846 0.00234163 0.00566209 0.00677845 0.00217692 0.001687 0.00235139 0.00203264 0.00186346 0.00162346 0.00230803; 0.00271169 0.00545808 0.00263302 0.0016385 0.00582896 0.0177921 0.00586849 0.00208916 0.00277508 0.00461105 0.00264704 0.00144147 0.00176448 0.00242271 0.0019095 0.00155428; 0.00196409 0.00206585 0.00588655 0.00288169 0.00244775 0.00538384 0.0147285 0.00616923 0.00185163 0.00234098 0.00470289 0.00209398 0.00170172 0.00177406 0.00201058 0.00195253; 0.00291793 0.00171317 0.00288986 0.00509084 0.00732493 0.00205185 0.00623073 0.0158797 0.00265401 0.00204915 0.00334973 0.00524786 0.00180304 0.00206516 0.00213687 0.00277024; 0.00228005 0.00158917 0.00184363 0.00168331 0.00600594 0.0023704 0.00159086 0.00215774 0.0179357 0.00458805 0.00265191 0.0057394 0.0048724 0.00230553 0.00160043 0.00256597; 0.00188113 0.00260856 0.00165067 0.00188637 0.00246927 0.00535851 0.00277197 0.00174793 0.00452975 0.0186795 0.00587128 0.00293908 0.00256491 0.00550431 0.00254789 0.00182041; 0.00199224 0.00198167 0.00239671 0.00148865 0.0019963 0.00225782 0.0048999 0.00253693 0.00227267 0.00495478 0.0167915 0.00544362 0.00193831 0.00261395 0.00605124 0.00213894; 0.00201428 0.00189837 0.00177325 0.00217569 0.00219486 0.00157229 0.00262323 0.00533245 0.00639645 0.00236338 0.00582814 0.0141972 0.00214637 0.00170952 0.00242774 0.00652343; 0.00551784 0.00314772 0.0018408 0.00244508 0.00239146 0.00211826 0.00165435 0.00167652 0.00526101 0.00247197 0.00198295 0.00232103 0.0166794 0.00479359 0.00235695 0.0051415; 0.00299558 0.00716651 0.00302339 0.00157788 0.00177353 0.00249744 0.00176033 0.00164171 0.00242951 0.00543594 0.00212568 0.00147208 0.00535451 0.0167751 0.00525319 0.00226052; 0.00194073 0.0021867 0.00667736 0.00247544 0.00198623 0.00230368 0.00222494 0.00156614 0.00178054 0.00234639 0.00585425 0.00259014 0.00238915 0.00565968 0.0169088 0.00539021; 0.00288388 0.00177993 0.00292207 0.00446213 0.00215593 0.00206059 0.00171043 0.00265896 0.00261172 0.00179292 0.00246213 0.00612522 0.00551238 0.00288897 0.00570215 0.0156827
    ] ≈ measured[:Greens].obs |> std_error      atol=0.0001

    @test  0.784 ≈ measured[:BosonEnergy].obs |> mean           atol=0.001
    @test  0.430 ≈ measured[:BosonEnergy].obs |> std_error      atol=0.001

    @test [
        0.16 -0.06 0.1 0.04 0.16 -0.1 -0.06 0.16 0.04 0.0; -0.04 0.1 0.14 0.16 0.22 0.32 0.06 0.12 0.06 -2.22045e-18; -0.16 0.14 -0.1 -0.02 0.02 0.1 -0.22 -0.16 -0.04 -0.08; 0.04 -0.06 -0.08 -0.06 -0.08 -0.16 -0.08 -0.1 -0.04 0.06; 0.1 0.12 0.14 0.18 0.14 0.08 -2.22045e-18 0.18 -0.02 0.04; 0.22 0.04 0.22 -0.16 0.04 -0.08 -0.04 0.06 0.12 9.99201e-18; -0.08 0.06 0.16 0.08 0.04 -0.06 -0.08 0.18 0.0 -0.04; -0.2 0.04 0.0 0.06 -0.06 0.04 -2.10942e-17 -0.06 0.18 -0.16; -0.1 0.06 0.1 -0.12 -0.08 0.08 -8.88178e-18 -0.12 -0.08 0.1; 0.16 -0.04 0.06 -0.02 -0.1 0.1 0.16 -0.06 0.16 0.04; 0.08 0.1 -0.12 0.06 -0.06 0.2 0.06 -0.06 0.04 0.04; 0.08 -0.2 -0.04 -0.02 -0.12 -0.08 -0.1 0.14 -0.14 -0.02; -0.06 -0.1 -0.02 0.06 -0.08 -0.08 -0.02 0.1 -0.14 -0.16; 0.1 -0.08 -0.08 0.12 0.04 -0.02 -0.22 0.08 -0.06 0.12; -0.16 0.04 0.16 0.12 0.08 0.1 0.22 -0.16 -0.12 -0.02; -2.22045e-18 0.06 0.02 0.04 0.1 0.06 0.22 0.16 -0.06 -0.02
    ] ≈ measured[:conf].obs |> mean                    atol=0.01
    @test last(measured[:conf].obs) == mc.conf
end



# TODO
# remove this / make this an example / make this faster
#=


"""
    stat_equal(
        expected_value, actual_values, standard_errors;
        min_error = 0.1^3, order=2, rtol = 0, debug=false
    )

Compare an `expected_value` (i.e. literature value, exact result, ...) to a set
of `actual_values` and `standard_errors` (i.e. calculated from DQMC or MC).

- `order = 2`: Sets the number of σ-intervals included. (This affects the
accuracy of the comaprison and the number of matches required)
- `min_error = 0.1^3`: Sets a lower bound for the standard error. (If one
standard error falls below `min_error`, `min_error` is used instead. This
happens before `order` is multiplied.)
- `rtol = 0`: The relative tolerance passed to `isapprox`.
- `debug = false`: If `debug = true` information on comparisons is always printed.
"""
function stat_equal(
        expected_value, actual_values::Vector, standard_errors::Vector;
        min_error = 0.001, order=2, debug=false, rtol=0.0
    )

    @assert order > 1
    N_matches = floor(length(actual_values) * (1 - 1 / order^2))
    if N_matches == 0
        error("No matches required. Try increasing the sample size or σ-Interval")
    elseif N_matches < 3
        @warn "Only $N_matches out of $(length(actual_values)) are required!"
    end

    is_approx_equal = [
        isapprox(expected_value, val, atol=order * max(min_error, err), rtol=rtol)
        for (val, err) in zip(actual_values, standard_errors)
    ]
    does_match = sum(is_approx_equal) >= N_matches

    if debug || !does_match
        printstyled("────────────────────────────────\n", color = :light_magenta)
        print("stat_equal returned ")
        printstyled("$(does_match)\n\n", color = does_match ? :green : :red)
        print("expected: $expected_value\n")
        print("values:   [")
        for i in eachindex(actual_values)
            if i < length(actual_values)
                printstyled("$(actual_values[i])", color = is_approx_equal[i] ? :green : :red)
                print(", ")
            else
                printstyled("$(actual_values[i])", color = is_approx_equal[i] ? :green : :red)
            end
        end
        print("]\n")
        print("$(order)-σ:      [")
        for i in eachindex(standard_errors)
            if i < length(standard_errors)
                printstyled("$(standard_errors[i])", color = is_approx_equal[i] ? :green : :red)
                print(", ")
            else
                printstyled("$(standard_errors[i])", color = is_approx_equal[i] ? :green : :red)
            end
        end
        print("]\n")
        print("checks:   [")
        for i in eachindex(standard_errors)
            if i < length(standard_errors)
                printstyled("$(is_approx_equal[i])", color = is_approx_equal[i] ? :green : :red)
                print(", ")
            else
                printstyled("$(is_approx_equal[i])", color = is_approx_equal[i] ? :green : :red)
            end
        end
        print("]\n")
        printstyled("────────────────────────────────\n", color = :light_magenta)
    end
    does_match
end



@testset "DQMC: triangular Hubbard model vs dos Santos Paper" begin
    # > Attractive Hubbard model on a triangular lattice
    # dos Santos
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.3976
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

=#
