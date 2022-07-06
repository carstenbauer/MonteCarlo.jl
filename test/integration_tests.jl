@time @testset "MC: IsingModel Simulation" begin
    m = IsingModel(dims=2, L=4);
    mc = MC(m, beta=1.0, sweeps=10_000, thermalization=10_000, measure_rate = 1);
    run!(mc, verbose=false);

    # Check measurements
    # means from 1000k + 100k sweeps
    # errors from above (because standard error changes with #sweeps)
    measured = measurements(mc)
    @test 15.99     ≈ measured[:Magn].M |> mean         atol=0.2
    @test 0.001726  ≈ measured[:Magn].M |> std_error    atol=0.001
    @test 255.65    ≈ measured[:Magn].M2 |> mean        atol=1.0
    @test 0.05181   ≈ measured[:Magn].M2 |> std_error   atol=0.02
    @test 0.9993    ≈ measured[:Magn].m |> mean         atol=0.001
    @test 0.0001079 ≈ measured[:Magn].m |> std_error    atol=0.0001
    @test 0.001589  ≈ measured[:Magn].chi |> mean       atol=0.001 # long run
    # @test 0.0009960 ≈ measured[:Magn].chi |> mean       atol=0.01 # short run

    @test -31.95    ≈ measured[:Energy].E |> mean       atol=0.1
    @test 0.006907  ≈ measured[:Energy].E |> std_error  atol=0.002
    @test 1021.4    ≈ measured[:Energy].E2 |> mean      atol=5.0
    @test 0.3868    ≈ measured[:Energy].E2 |> std_error atol=0.2
    @test -1.9971   ≈ measured[:Energy].e |> mean       atol=0.01
    @test 0.0004317 ≈ measured[:Energy].e |> std_error  atol=0.0002
    @test 0.02378   ≈ measured[:Energy].C |> mean       atol=0.015 # long run
    # @test 0.01594   ≈ measured[:Energy].C |> mean       atol=0.007 # short run
    @test isempty(mc.configs) == true
end