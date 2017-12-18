using MonteCarlo

m = MonteCarlo.IsingModel(dims=2, L=10)
mc = MonteCarlo.MC(m)

MonteCarlo.run!(mc, sweeps=100)

# Wolff cluster example
m = MonteCarlo.IsingModel(dims=2, L=10)
mc = MonteCarlo.MC(m)
mc.p.global_moves = true # enable Wolff cluster
mc.p.global_rate = 1
MonteCarlo.run!(mc)
