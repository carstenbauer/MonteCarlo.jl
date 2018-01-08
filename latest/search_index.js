var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Documentation-1",
    "page": "Home",
    "title": "Documentation",
    "category": "section",
    "text": "This is a package for numerically simulating physical systems in Julia. The purpose of this package is to supply efficient Julia implementations of Monte Carlo flavors for the study of physical models of spins, bosons and/or fermions. Examples that ship with the package areIsing spin model simulated by classical Monte Carlo\nFermionic Hubbard model simulated by variants of determinant quantum Monte Carlo"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install the package execute the following command in the REPL:Pkg.clone(\"https://github.com/crstnbr/MonteCarlo.jl\")Afterwards, you can use MonteCarlo.jl like any other package installed with Pkg.add():using MonteCarloTo obtain the latest version of the package just do Pkg.update() or specifically Pkg.update(\"MonteCarlo\")."
},

{
    "location": "manual/gettingstarted.html#",
    "page": "Getting started",
    "title": "Getting started",
    "category": "page",
    "text": ""
},

{
    "location": "manual/gettingstarted.html#Manual-1",
    "page": "Getting started",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual/gettingstarted.html#Installation-/-Updating-1",
    "page": "Getting started",
    "title": "Installation / Updating",
    "category": "section",
    "text": "To install the package execute the following command in the REPL:Pkg.clone(\"https://github.com/crstnbr/MonteCarlo.jl\")To obtain the latest version of the package just do Pkg.update() or specifically Pkg.update(\"MonteCarlo\")."
},

{
    "location": "manual/gettingstarted.html#Example-1",
    "page": "Getting started",
    "title": "Example",
    "category": "section",
    "text": "This is a simple demontration of how to perform a classical Monte Carlo simulation of the 2D Ising model:# load packages\nusing MonteCarlo, MonteCarloObservable\n\n# load your model\nm = IsingModel(dims=2, L=8, β=0.35);\n\n# choose a Monte Carlo flavor and run the simulation\nmc = MC(m);\nrun!(mc, sweeps=1000, thermalization=1000, verbose=false);\n\n# analyze results\nobservables(mc) # what observables do exist for that simulation?\nm = mc.obs[\"m\"] # magnetization\nmean(m)\nstd(m) # one-sigma error\n\n# create standard plots\nhist(m)\nplot(m)(Image: ) (Image: )"
},

{
    "location": "manual/examples.html#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "manual/examples.html#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": ""
},

{
    "location": "manual/examples.html#D-Ising-model-1",
    "page": "Examples",
    "title": "2D Ising model",
    "category": "section",
    "text": "Results: (Image: )Code:using MonteCarlo, Distributions, PyPlot, DataFrames, JLD\n\nTdist = Normal(MonteCarlo.IsingTc, .64)\nn_Ts = 2^8\nTs = sort!(rand(Tdist, n_Ts))\nTs = Ts[Ts.>=1.2]\nTs = Ts[Ts.<=3.8]\ntherm = 10^4\nsweeps = 10^3\n\ndf = DataFrame(L=Int[], T=Float64[], M=Float64[], χ=Float64[], E=Float64[], C_V=Float64[])\n\nfor L in 2.^[3, 4, 5, 6]\n	println(\"L = \", L)\n	for (i, T) in enumerate(Ts)\n		println(\"\\t T = \", T)\n		β = 1/T\n		model = IsingModel(dims=2, L=L, β=β)\n		mc = MC(model)\n		obs = run!(mc, sweeps=sweeps, thermalization=therm, verbose=false)\n		push!(df, [L, T, mean(obs[\"m\"]), mean(obs[\"χ\"]), mean(obs[\"e\"]), mean(obs[\"C\"])])\n	end\n	flush(STDOUT)\nend\n\nsort!(df, cols = [:L, :T])\n@save \"ising2d.jld\" df\n\n# plot results together\ngrps = groupby(df, :L)\nfig, ax = subplots(2,2, figsize=(12,8))\nfor g in grps\n	L = g[:L][1]\n	ax[1][:plot](g[:T], g[:E], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[2][:plot](g[:T], g[:C_V], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[3][:plot](g[:T], g[:M], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[4][:plot](g[:T], g[:χ], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\nend\nax[1][:legend](loc=\"best\")\nax[1][:set_ylabel](\"Energy\")\nax[1][:set_xlabel](\"Temperature\")\n\nax[2][:set_ylabel](\"Specific heat\")\nax[2][:set_xlabel](\"Temperature\")\nax[2][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[2][:legend](loc=\"best\")\n\nax[3][:set_ylabel](\"Magnetization\")\nax[3][:set_xlabel](\"Temperature\")\nx = linspace(1.2, MonteCarlo.IsingTc, 100)\ny = (1-sinh.(2.0 ./ (x)).^(-4)).^(1/8)\nax[3][:plot](x,y, \"k--\", label=\"exact\")\nax[3][:plot](linspace(MonteCarlo.IsingTc, 3.8, 100), zeros(100), \"k--\")\nax[3][:legend](loc=\"best\")\n\nax[4][:set_ylabel](\"Susceptibility χ\")\nax[4][:set_xlabel](\"Temperature\")\nax[4][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[4][:legend](loc=\"best\")\ntight_layout()\nsavefig(\"ising2d.pdf\")"
},

{
    "location": "manual/flavors/mc.html#",
    "page": "MC",
    "title": "MC",
    "category": "page",
    "text": ""
},

{
    "location": "manual/flavors/mc.html#Monte-Carlo-1",
    "page": "MC",
    "title": "Monte Carlo",
    "category": "section",
    "text": "This is plain simple classical Monte Carlo (MC). It can for example be used to simulate the Ising model (see 2D Ising Model).You can initialize a Monte Carlo simulation of a given model simply throughmc = MC(model)Allowed keywords are:sweeps: number of measurement sweeps\nthermalization: number of thermalization (warmup) sweeps\nglobal_moves: wether global moves should be proposed\nglobal_rate: frequency for proposing global movesAfterwards, you can run the simulation byrun!(mc)Note that you can just do another run!(mc, sweeps=1000) to continue the simulation."
},

{
    "location": "manual/flavors/mc.html#Model-interface-1",
    "page": "MC",
    "title": "Model interface",
    "category": "section",
    "text": "Any model that wants to be simulatable by means of MC must implement the following interface."
},

{
    "location": "manual/flavors/mc.html#Mandatory-fields-1",
    "page": "MC",
    "title": "Mandatory fields",
    "category": "section",
    "text": "β::Float64: inverse temperature\nl::Lattice: any Lattice"
},

{
    "location": "manual/flavors/mc.html#Mandatory-methods-1",
    "page": "MC",
    "title": "Mandatory methods",
    "category": "section",
    "text": "conftype: type of a configuration\nenergy: energy of configuration\nrand: random configuration\npropose_local: propose local move\naccept_local: accept a local movePrecise signatures can be found here: Methods: MC."
},

{
    "location": "manual/flavors/mc.html#Optional-methods-1",
    "page": "MC",
    "title": "Optional methods",
    "category": "section",
    "text": "global_move: propose and accept or reject a local move\nprepare_observables: initialize observables\nmeasure_observables!: measure observables\nfinish_observables!: finish measurementsPrecise signatures can be found here: Methods: MC."
},

{
    "location": "manual/flavors/mc.html#Potential-extensions-1",
    "page": "MC",
    "title": "Potential extensions",
    "category": "section",
    "text": "Pull requests are very much welcome!Heat bath (instead of Metropolis) option"
},

{
    "location": "manual/flavors/dqmc.html#",
    "page": "DQMC",
    "title": "DQMC",
    "category": "page",
    "text": ""
},

{
    "location": "manual/models/ising.html#",
    "page": "Ising model",
    "title": "Ising model",
    "category": "page",
    "text": ""
},

{
    "location": "manual/models/ising.html#Ising-model-1",
    "page": "Ising model",
    "title": "Ising model",
    "category": "section",
    "text": "The famous Hamiltonian of the Ising model is given by \\begin{align} \\mathcal{H} = -\\sum_{\\langle i,j \\rangle} \\sigma_i \\sigma_j , \\end{align}where $ \\langle i, j \\rangle $ indicates that the sum has to be taken over nearest neighbors.You can create an Ising model as follows,model = IsingModel(; dims::Int=2, L::Int=8, β::Float64=1.0)The following parameters can be set via keyword arguments:dims: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)\nL: linear system size\nβ: inverse temperaturenote: Note\nSo far only dims=2 is supported. Feel free to extend the model and create a pull request!"
},

{
    "location": "manual/models/ising.html#Supported-Monte-Carlo-flavors-1",
    "page": "Ising model",
    "title": "Supported Monte Carlo flavors",
    "category": "section",
    "text": "Monte Carlo"
},

{
    "location": "manual/models/ising.html#Square-lattice-(2D)-1",
    "page": "Ising model",
    "title": "Square lattice (2D)",
    "category": "section",
    "text": ""
},

{
    "location": "manual/models/ising.html#Analytic-results-1",
    "page": "Ising model",
    "title": "Analytic results",
    "category": "section",
    "text": "The model can be solved exactly by transfer matrix method (Onsager solution). This gives the following results.Critical temperature: $ T_c = \\frac{2}{\\ln{1+\\sqrt{2}}} $Magnetization (per site): $ m = \\left(1-\\left[\\sinh 2\\beta \\right]^{-4}\\right)^{\\frac {1}{8}} $See also 2D Ising model."
},

{
    "location": "manual/models/ising.html#Potential-extensions-1",
    "page": "Ising model",
    "title": "Potential extensions",
    "category": "section",
    "text": "Pull requests are very much welcome!Arbitrary dimensions\nMagnetic field\nMaybe explicit J instead of implicit J=1\nNon-cubic lattices (just add lattice::Lattice keyword)"
},

{
    "location": "manual/models/hubbard.html#",
    "page": "Hubbard model",
    "title": "Hubbard model",
    "category": "page",
    "text": ""
},

{
    "location": "manual/models/hubbard.html#Hubbard-model-1",
    "page": "Hubbard model",
    "title": "Hubbard model",
    "category": "section",
    "text": "The Hamiltonian of the repulsive Hubbard model is given by \\begin{align} \\mathcal{H} = -t \\sum_{\\langle i,j \\rangle, \\sigma} \\left( c^\\dagger_{i\\sigma} c_{j\\sigma} + \\text{h.c.} \\right) + U \\sum_j \\left( n_{j\\uparrow} - \\frac{1}{2} \\right) \\left( n_{j\\downarrow} - \\frac{1}{2} \\right) - \\mu\\sum_j\\left( n_{j\\uparrow} + n_{j\\downarrow} \\right), \\end{align}where sigma denotes spin, t is the hopping amplitude, U the on-site repulsive interaction strength, mu the chemical potential and $ \\langle i, j \\rangle $ indicates that the sum has to be taken over nearest neighbors. Note that (1) is the Hubbard model in particle-hole symmetric form which has the nice property that mu = 0 corresponds to half-filling.You can create a Hubbard model instance as follows,model = HubbardModel(; dims::Int=2, L::Int=8, β::Float64=1.0)The following parameters can be set via keyword arguments:dims: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)\nL: linear system size\nβ: inverse temperature"
},

{
    "location": "manual/custom/models.html#",
    "page": "Models",
    "title": "Models",
    "category": "page",
    "text": ""
},

{
    "location": "manual/custom/models.html#Custom-models-1",
    "page": "Models",
    "title": "Custom models",
    "category": "section",
    "text": "Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, a key idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to extend the package and implement your own physical model. Sometimes examples say more than words, so feel encouraged to have a look at the implementations of the above mentioned models."
},

{
    "location": "manual/custom/models.html#Semantics-1",
    "page": "Models",
    "title": "Semantics",
    "category": "section",
    "text": "Loosely speeking, we define a Model to be a Hamiltonian on a lattice. Therefore, the lattice is part of a model. The motivation for this modeling is that the physics of a system does not only depend on the Hamiltonian but also (sometime drastically) on the underlying lattice. This is for example very obvious for spin systems which due to the lattice might become (geometrically) frustrated and show spin liquids physics. Also, from a technical point of view, lattice information is almost exclusively processed in energy calculations which both relate to the Hamiltonian (and therefore the model).note: Note\nWe will generally use the terminology Hamiltonian, energy and so on. However, this doesn't restrict you from defining your model as an Lagrangian with an action in any way as this just corresponds to a one-to-one mapping of interpretations."
},

{
    "location": "manual/custom/models.html#Mandatory-fields-and-methods-1",
    "page": "Models",
    "title": "Mandatory fields and methods",
    "category": "section",
    "text": "Any concrete model (type), let's call it MyModel in the following, must be a subtype of the abstract type MonteCarlo.Model. To work with a Monte Carlo flavor, it must internally have at least the following fields:β::Float64: inverse temperature\nl::Lattice: any LatticeFurthermore it must implement the following methods:conftype: type of a configuration\nenergy: energy of configuration\nrand: random configuration\npropose_local: propose local move\naccept_local: accept a local moveA full list of methods with precise signatures that should be implemented for MyModel can be found here: Methods: Models."
},

{
    "location": "manual/custom/models.html#Lattice-requirements-1",
    "page": "Models",
    "title": "Lattice requirements",
    "category": "section",
    "text": "The Hamiltonian of your model might impose some requirements on the Lattice object that you use as it must provide you with enough lattice information.It might be educating to look at the structure of the simple SquareLattice struct. mutable struct SquareLattice <: CubicLattice\n    L::Int\n    sites::Int\n    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx\n    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx\n    sql::Matrix{Int}\n    SquareLattice() = new()\nendIt only provides access to next nearest neighbors through the arrays neighs and neighs_cartesian. If your model's Hamiltonian requires higher order neighbor information, because of, let's say, a next next nearest neighbor hopping term, the SquareLattice doesn't suffice. You could either extend this Lattice or implement a NNSquareLattice for example."
},

{
    "location": "manual/custom/lattices.html#",
    "page": "Lattices",
    "title": "Lattices",
    "category": "page",
    "text": ""
},

{
    "location": "manual/custom/lattices.html#Custom-lattices-1",
    "page": "Lattices",
    "title": "Custom lattices",
    "category": "section",
    "text": "As described in Custom models a lattice is considered to be part of a model. Hence, most of the requirements for fields of a Lattice subtype come from potential models (see Lattice requirements). Below you'll find information on which fields are mandatory from a Monte Carlo flavor point of view."
},

{
    "location": "manual/custom/lattices.html#Mandatory-fields-and-methods-1",
    "page": "Lattices",
    "title": "Mandatory fields and methods",
    "category": "section",
    "text": "Any concrete lattice type, let's call it MyLattice in the following, must be a subtype of the abstract type MonteCarlo.Lattice. To work with a Monte Carlo flavor, it must internally have at least have the following field,sites: number of lattice sites.However, as already mentioned above depending on the physical model of interest it will typically also have (at least) something likeneighs: next nearest neighbors,as most Hamiltonian will need next nearest neighbor information.The only reason why such a field isn't generally mandatory is that the Monte Carlo routine doesn't care about the lattice much. Neighbor information is usually only used in the energy (difference) calculation of a particular configuration like done in energy or propose_local which both belong to a Model."
},

{
    "location": "manual/custom/flavors.html#",
    "page": "Flavors",
    "title": "Flavors",
    "category": "page",
    "text": ""
},

{
    "location": "methods/general.html#",
    "page": "General",
    "title": "General",
    "category": "page",
    "text": ""
},

{
    "location": "methods/general.html#Methods:-General-1",
    "page": "General",
    "title": "Methods: General",
    "category": "section",
    "text": "Below you find all general exports."
},

{
    "location": "methods/general.html#Index-1",
    "page": "General",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"general.md\"]"
},

{
    "location": "methods/general.html#MonteCarlo.init!-Tuple{MonteCarlo.MC}",
    "page": "General",
    "title": "MonteCarlo.init!",
    "category": "Method",
    "text": "init!(mc::MC[; seed::Real=-1])\n\nInitialize the classical Monte Carlo simulation mc. If seed !=- 1 the random generator will be initialized with srand(seed).\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.run!-Tuple{MonteCarlo.MC}",
    "page": "General",
    "title": "MonteCarlo.run!",
    "category": "Method",
    "text": "run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])\n\nRuns the given classical Monte Carlo simulation mc. Progress will be printed to STDOUT if verborse=true (default).\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.MC",
    "page": "General",
    "title": "MonteCarlo.MC",
    "category": "Type",
    "text": "Classical Monte Carlo simulation\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.MC-Union{Tuple{M}, Tuple{M}} where M<:MonteCarlo.Model",
    "page": "General",
    "title": "MonteCarlo.MC",
    "category": "Method",
    "text": "MC(m::M) where M<:Model\n\nCreate a classical Monte Carlo simulation for model m with default parameters.\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.IsingModel",
    "page": "General",
    "title": "MonteCarlo.IsingModel",
    "category": "Type",
    "text": "Famous Ising model on a cubic lattice.\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.IsingModel-Tuple{Int64,Int64,Float64}",
    "page": "General",
    "title": "MonteCarlo.IsingModel",
    "category": "Method",
    "text": "IsingModel(dims::Int, L::Int, β::Float64)\nIsingModel(; dims::Int=2, L::Int=8, β::Float64=1.0)\n\nCreate Ising model on dims-dimensional cubic lattice with linear system size L and inverse temperature β.\n\n\n\n"
},

{
    "location": "methods/general.html#Documentation-1",
    "page": "General",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"MC.jl\", \"IsingModel.jl\"]"
},

{
    "location": "methods/models.html#",
    "page": "Models",
    "title": "Models",
    "category": "page",
    "text": ""
},

{
    "location": "methods/models.html#Methods:-Models-1",
    "page": "Models",
    "title": "Methods: Models",
    "category": "section",
    "text": "Below you find all methods that any particular physical model (subtype of the abstract type MonteCarlo.Model) should implement. See also Custom models for more information."
},

{
    "location": "methods/models.html#Index-1",
    "page": "Models",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"models.md\"]"
},

{
    "location": "methods/models.html#Documentation-1",
    "page": "Models",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nOrder   = [:function]\nPages = [\"abstract_model.jl\", \"abstract_functions.jl\"]"
},

]}
