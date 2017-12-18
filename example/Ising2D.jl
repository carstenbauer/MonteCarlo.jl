using MonteCarlo

m = MonteCarlo.IsingModel(dims=2, L=10)
mc = MonteCarlo.MC(m)

MonteCarlo.run!(mc, sweeps=100)
