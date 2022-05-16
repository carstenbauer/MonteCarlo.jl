@time @testset "MC: IsingModel Simulation" begin
    m = IsingModel(dims=2, L=4);
    mc = MC(m, beta=1.0, sweeps=10_000, thermalization=1_000);
    run!(mc, verbose=false);

    # Check measurements
    measured = measurements(mc)
    @test 15.9   ≈ measured[:Magn].M |> mean         atol=0.2
    @test 0.0015 ≈ measured[:Magn].M |> std_error    atol=0.001
    @test 255.7  ≈ measured[:Magn].M2 |> mean        atol=1.0
    @test 0.05   ≈ measured[:Magn].M2 |> std_error   atol=0.01
    @test 0.999  ≈ measured[:Magn].m |> mean         atol=0.001
    @test 0.0001 ≈ measured[:Magn].m |> std_error    atol=0.0001
    @test 0.001  ≈ measured[:Magn].chi |> mean       atol=0.001

    @test  -31.9  ≈ measured[:Energy].E |> mean       atol=0.1
    @test    0.007  ≈ measured[:Energy].E |> std_error  atol=0.002
    @test 1020.    ≈ measured[:Energy].E2 |> mean      atol=5.0
    @test  0.36    ≈ measured[:Energy].E2 |> std_error atol=0.05
    @test   -1.99 ≈ measured[:Energy].e |> mean       atol=0.01
    @test    0.0004 ≈ measured[:Energy].e |> std_error  atol=0.0002
    @test    0.024  ≈ measured[:Energy].C |> mean       atol=0.005
    @test isempty(mc.configs) == true
end