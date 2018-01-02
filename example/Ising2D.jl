# using MonteCarlo

# # single runs
# m = IsingModel(dims=2, L=10)
# mc = MC(m)
# run!(mc, sweeps=1000, thermalization=1000)

# # Wolff cluster example
# m = IsingModel(dims=2, L=10)
# mc = MC(m)
# mc.p.global_moves = true # enable Wolff cluster
# mc.p.global_rate = 1
# run!(mc)


using MonteCarlo, Distributions, PyPlot, DataFrames
# Tspace = linspace(1.2, 3.8, n_Ts)
Tdist = Normal(MonteCarlo.IsingTc, .64)
n_Ts = 2^8 # 2^8
Ts = sort!(rand(Tdist, n_Ts))
Ts = Ts[Ts.>=1.2]
Ts = Ts[Ts.<=3.8]
therm = 10^4
sweeps = 10^3

df = DataFrame(L=Int[], T=Float64[], M=Float64[], χ=Float64[], E=Float64[], C_V=Float64[])

for L in 2.^[3, 4, 5, 6]
	println("L = ", L)
	for (i, T) in enumerate(Ts)
		println("\t T = ", T)
		β = 1/T
		model = IsingModel(dims=2, L=L, β=β)
		mc = MC(model)
		# mc.p.global_moves = true # enable Wolff cluster
		# mc.p.global_rate = 1
		obs = run!(mc, sweeps=sweeps, thermalization=therm, verbose=false)
		push!(df, [L, T, mean(obs["m"]), mean(obs["χ"]), mean(obs["e"]), mean(obs["C"])])
	end
	flush(STDOUT)
end

sort!(df, cols = [:L, :T])

grps = groupby(df, :L)

# plot results individually
for g in grps
	L = g[:L][1]
	fig, ax = subplots(2,2, figsize=(12,8))
	ax[1][:plot](g[:T], g[:E], "o", color="darkred", markeredgecolor="black")
	ax[1][:set_ylabel]("Energy")

	ax[2][:plot](g[:T], g[:C_V], "o", color="darkred", markeredgecolor="black")
	ax[2][:set_ylabel]("Specific heat")

	ax[3][:plot](g[:T], g[:M], "o", color="C0", markeredgecolor="black")
	ax[3][:axvline](x=MonteCarlo.IsingTc, color="black", linewidth=2.0, label="\$ T_c \$")
	# ax[3][:legend](loc="best")
	ax[3][:set_ylabel]("Magnetization")

	ax[4][:plot](g[:T], g[:χ], "o", color="C0", markeredgecolor="black")
	ax[4][:axvline](x=MonteCarlo.IsingTc, color="black", linewidth=2.0, label="\$ T_c \$")
	# ax[4][:legend](loc="best")
	ax[4][:set_ylabel]("Susceptibility χ")
	tight_layout()
	savefig("ising2d_L_$(L).pdf")
end

# plot results together
fig, ax = subplots(2,2, figsize=(12,8))

for g in grps
	L = g[:L][1]
	ax[1][:plot](g[:T], g[:E], "o", markeredgecolor="black", label="L=$L")
	ax[2][:plot](g[:T], g[:C_V], "o", markeredgecolor="black", label="L=$L")
	ax[3][:plot](g[:T], g[:M], "o", markeredgecolor="black", label="L=$L")
	ax[4][:plot](g[:T], g[:χ], "o", markeredgecolor="black", label="L=$L")
end
ax[1][:legend](loc="best")
ax[1][:set_ylabel]("Energy")

ax[2][:legend](loc="best")
ax[2][:set_ylabel]("Specific heat")

ax[3][:legend](loc="best")
ax[3][:set_ylabel]("Magnetization")
ax[3][:axvline](x=MonteCarlo.IsingTc, linewidth=2.0, color="black", label="\$ T_c \$")

ax[4][:set_ylabel]("Susceptibility χ")
ax[4][:axvline](x=MonteCarlo.IsingTc, linewidth=2.0, color="black", label="\$ T_c \$")
ax[4][:legend](loc="best")
tight_layout()
savefig("ising2d.pdf")