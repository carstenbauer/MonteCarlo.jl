using MonteCarlo

model = IsingModel(dims=2, L=64)

mc = DiscreteMC(model)

run!(mc, sweeps=1000)
