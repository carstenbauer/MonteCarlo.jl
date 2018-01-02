using MonteCarlo

# single runs
m = IsingModel(dims=2, L=10)
mc = MC(m)
run!(mc, sweeps=1000, thermalization=1000)

# Wolff cluster example
m = IsingModel(dims=2, L=10)
mc = MC(m)
mc.p.global_moves = true # enable Wolff cluster
mc.p.global_rate = 1
run!(mc)


# Full analysis
using Distributions, PyPlot, DataFrames
# Tspace = linspace(1.2, 3.8, n_Ts)
Tdist = Normal(MonteCarlo.IsingTc, .64)
n_Ts = 2^6 # 2^8
Ts = rand(Tdist, n_Ts)
Ts = Ts[Ts.>=1.2]
Ts = Ts[Ts.<=3.8]
L = 2^3
therm = 2^10
sweeps = 2^10

df = DataFrame(L=Int[], T=Float64[], M=Float64[], χ=Float64[], E=Float64[], C_V=Float64[])

for (i, T) in enumerate(Ts)
	println("T = ", T)
	β = 1/T
	model = IsingModel(dims=2, L=L, β=β)
	mc = MC(model)
	# mc.p.global_moves = true # enable Wolff cluster
	# mc.p.global_rate = 1
	obs = run!(mc, sweeps=sweeps, thermalization=therm, verbose=false)
	push!(df, [L, T, mean(obs["m"]), mean(obs["χ"]), mean(obs["e"]), mean(obs["C"])])
end

sort!(df, cols = [:L, :T])

# plot results
fig, ax = subplots(2,2, figsize=(12,8))
ax[1][:plot](df[:T], df[:E], "o-", color="darkred", markeredgecolor="black")
ax[1][:set_ylabel]("Energy")

ax[2][:plot](df[:T], df[:C_V], "o-", color="darkred", markeredgecolor="black")
ax[2][:set_ylabel]("Specific heat")

ax[3][:plot](df[:T], df[:M], "o-", color="C0", markeredgecolor="black")
ax[3][:axvline](x=MonteCarlo.IsingTc, color="black", linewidth=2.0, label="\$ T_c \$")
ax[3][:legend](loc="best")
ax[3][:set_ylabel]("Magnetization")

ax[4][:plot](df[:T], df[:χ], "o-", color="C0", markeredgecolor="black")
ax[4][:axvline](x=MonteCarlo.IsingTc, color="black", linewidth=2.0, label="\$ T_c \$")
ax[4][:legend](loc="best")
ax[4][:set_ylabel]("Susceptibility χ")
tight_layout()