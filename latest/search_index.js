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
    "text": "This is a package for numerically simulating physical systems in Julia. The purpose of this package is to supply efficient Julia implementations of Monte Carlo flavors for the study of physical models of spins, bosons and/or fermions. Examples that ship with the package areIsing spin model simulated by Monte Carlo\nFermionic Hubbard model simulated by variants of determinant quantum Monte Carlo"
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
    "location": "manual/gettingstarted.html#Usage-1",
    "page": "Getting started",
    "title": "Usage",
    "category": "section",
    "text": "This is a simple demontration of how to perform a Monte Carlo simulation of the 2D Ising model:# load packages\nusing MonteCarlo, MonteCarloObservable\n\n# load your model\nm = IsingModel(dims=2, L=8);\n\n# choose a Monte Carlo flavor and run the simulation\nmc = MC(m, beta=0.35);\nrun!(mc, sweeps=1000, thermalization=1000, verbose=false);\n\n# analyze results\nobservables(mc) # what observables do exist for that simulation?\nm = mc.obs[\"m\"] # magnetization\nmean(m)\nstd(m) # one-sigma error\n\n# create standard plots\nhist(m)\nplot(m)(Image: )"
},

{
    "location": "manual/gettingstarted.html#Create-custom-models-1",
    "page": "Getting started",
    "title": "Create custom models",
    "category": "section",
    "text": "Probably the most important idea underlying the package design is extensibility. Users should be able to define custom physical models and utilize already implemented Monte Carlo flavors to study them. To that end, all Monte Carlo flavors have rather well defined interfaces, that is specifications of mandatory and optional fields and methods, that the user must implement for any model that he wants to simulate. The definition of the interface for the above used Monte Carlo can for example be found here: Interface: Monte Carlo (MC). Practically, it is probably a good idea to start from a copy of one of the preimplemented models.We hope that MonteCarlo.jl allows the user to put his focus on the physical model rather than having to tediously implement general Monte Carlo schemes, often over and over again."
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
    "text": "Results: (Image: )Code:using MonteCarlo, Distributions, PyPlot, DataFrames, JLD\n\nTdist = Normal(MonteCarlo.IsingTc, .64)\nn_Ts = 2^8\nTs = sort!(rand(Tdist, n_Ts))\nTs = Ts[Ts.>=1.2]\nTs = Ts[Ts.<=3.8]\ntherm = 10^4\nsweeps = 10^3\n\ndf = DataFrame(L=Int[], T=Float64[], M=Float64[], χ=Float64[], E=Float64[], C_V=Float64[])\n\nfor L in 2.^[3, 4, 5, 6]\n	println(\"L = \", L)\n	for (i, T) in enumerate(Ts)\n		println(\"\\t T = \", T)\n		beta = 1/T\n		model = IsingModel(dims=2, L=L)\n		mc = MC(model, beta=beta)\n		run!(mc, sweeps=sweeps, thermalization=therm, verbose=false)\n		push!(df, [L, T, mean(mc.obs[\"m\"]), mean(mc.obs[\"χ\"]), mean(mc.obs[\"e\"]), mean(mc.obs[\"C\"])])\n	end\n	flush(STDOUT)\nend\n\nsort!(df, cols = [:L, :T])\n@save \"ising2d.jld\" df\n\n# plot results together\ngrps = groupby(df, :L)\nfig, ax = subplots(2,2, figsize=(12,8))\nfor g in grps\n	L = g[:L][1]\n	ax[1][:plot](g[:T], g[:E], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[2][:plot](g[:T], g[:C_V], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[3][:plot](g[:T], g[:M], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[4][:plot](g[:T], g[:χ], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\nend\nax[1][:legend](loc=\"best\")\nax[1][:set_ylabel](\"Energy\")\nax[1][:set_xlabel](\"Temperature\")\n\nax[2][:set_ylabel](\"Specific heat\")\nax[2][:set_xlabel](\"Temperature\")\nax[2][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[2][:legend](loc=\"best\")\n\nax[3][:set_ylabel](\"Magnetization\")\nax[3][:set_xlabel](\"Temperature\")\nx = linspace(1.2, MonteCarlo.IsingTc, 100)\ny = (1-sinh.(2.0 ./ (x)).^(-4)).^(1/8)\nax[3][:plot](x,y, \"k--\", label=\"exact\")\nax[3][:plot](linspace(MonteCarlo.IsingTc, 3.8, 100), zeros(100), \"k--\")\nax[3][:legend](loc=\"best\")\n\nax[4][:set_ylabel](\"Susceptibility χ\")\nax[4][:set_xlabel](\"Temperature\")\nax[4][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[4][:legend](loc=\"best\")\ntight_layout()\nsavefig(\"ising2d.pdf\")"
},

{
    "location": "models/ising.html#",
    "page": "Ising model",
    "title": "Ising model",
    "category": "page",
    "text": ""
},

{
    "location": "models/ising.html#IsingModel-1",
    "page": "Ising model",
    "title": "IsingModel",
    "category": "section",
    "text": ""
},

{
    "location": "models/ising.html#Hamiltonian-1",
    "page": "Ising model",
    "title": "Hamiltonian",
    "category": "section",
    "text": "The famous Hamiltonian of the Ising model is given by\\begin{align} \\mathcal{H} = -\\sum_{\\langle i,j \\rangle} \\sigma_i \\sigma_j , \\end{align}where $ \\langle i, j \\rangle $ indicates that the sum has to be taken over nearest neighbors."
},

{
    "location": "models/ising.html#Creating-an-Ising-model-1",
    "page": "Ising model",
    "title": "Creating an Ising model",
    "category": "section",
    "text": "You can create an Ising model as follows,model = IsingModel(; dims::Int=2, L::Int=8)The following parameters can be set via keyword arguments:dims: dimensionality of the cubic lattice (i.e. 1 = chain, 2 = square lattice, etc.)\nL: linear system sizenote: Note\nSo far only dims=2 is supported. Feel free to extend the model and create a pull request!"
},

{
    "location": "models/ising.html#Supported-Monte-Carlo-flavors-1",
    "page": "Ising model",
    "title": "Supported Monte Carlo flavors",
    "category": "section",
    "text": "Monte Carlo (MC) (Have a look at the examples section below)"
},

{
    "location": "models/ising.html#Examples-1",
    "page": "Ising model",
    "title": "Examples",
    "category": "section",
    "text": "You can find example simulations of the 2D Ising model under Getting started and here: 2D Ising model."
},

{
    "location": "models/ising.html#MonteCarlo.IsingModel",
    "page": "Ising model",
    "title": "MonteCarlo.IsingModel",
    "category": "type",
    "text": "Famous Ising model on a cubic lattice.\n\nIsingModel(; dims::Int=2, L::Int=8)\n\nCreate Ising model on dims-dimensional cubic lattice with linear system size L.\n\n\n\n"
},

{
    "location": "models/ising.html#MonteCarlo.IsingModel-Tuple{Union{Dict{String,Any}, Dict{Symbol,Any}}}",
    "page": "Ising model",
    "title": "MonteCarlo.IsingModel",
    "category": "method",
    "text": "IsingModel(kwargs::Dict{String, Any})\n\nCreate Ising model with (keyword) parameters as specified in kwargs dict.\n\n\n\n"
},

{
    "location": "models/ising.html#Exports-1",
    "page": "Ising model",
    "title": "Exports",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"IsingModel.jl\"]"
},

{
    "location": "models/ising.html#Analytic-results-1",
    "page": "Ising model",
    "title": "Analytic results",
    "category": "section",
    "text": ""
},

{
    "location": "models/ising.html#Square-lattice-(2D)-1",
    "page": "Ising model",
    "title": "Square lattice (2D)",
    "category": "section",
    "text": "The model can be solved exactly by transfer matrix method (Onsager solution). This gives the following results.Critical temperature: $ T_c = \\frac{2}{\\ln{1+\\sqrt{2}}} $Magnetization (per site): $ m = \\left(1-\\left[\\sinh 2\\beta \\right]^{-4}\\right)^{\\frac {1}{8}} $"
},

{
    "location": "models/ising.html#Potential-extensions-1",
    "page": "Ising model",
    "title": "Potential extensions",
    "category": "section",
    "text": "Pull requests are very much welcome!Arbitrary dimensions\nMagnetic field\nMaybe explicit J instead of implicit J=1\nNon-cubic lattices (just add lattice::Lattice keyword)"
},

{
    "location": "flavors/mc.html#",
    "page": "MC",
    "title": "MC",
    "category": "page",
    "text": ""
},

{
    "location": "flavors/mc.html#Monte-Carlo-(MC)-1",
    "page": "MC",
    "title": "Monte Carlo (MC)",
    "category": "section",
    "text": "This is plain simple Monte Carlo (MC). It can for example be used to simulate the Ising model (see 2D Ising model).You can initialize a Monte Carlo simulation of a given model simply throughmc = MC(model)Allowed keywords are:beta: inverse temperature\nsweeps: number of measurement sweeps\nthermalization: number of thermalization (warmup) sweeps\nglobal_moves: wether global moves should be proposed\nglobal_rate: frequency for proposing global moves\nseed: initialize MC with custom seedAfterwards, you can run the simulation byrun!(mc)Note that you can just do another run!(mc, sweeps=1000) to continue the simulation."
},

{
    "location": "flavors/mc.html#Examples-1",
    "page": "MC",
    "title": "Examples",
    "category": "section",
    "text": "You can find example simulations of the 2D Ising model under Getting started and here: 2D Ising model."
},

{
    "location": "flavors/mc.html#MonteCarlo.run!-Tuple{MonteCarlo.MC}",
    "page": "MC",
    "title": "MonteCarlo.run!",
    "category": "method",
    "text": "run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])\n\nRuns the given Monte Carlo simulation mc. Progress will be printed to STDOUT if verbose=true (default).\n\n\n\n"
},

{
    "location": "flavors/mc.html#MonteCarlo.MC",
    "page": "MC",
    "title": "MonteCarlo.MC",
    "category": "type",
    "text": "Monte Carlo simulation\n\n\n\n"
},

{
    "location": "flavors/mc.html#MonteCarlo.MC-Union{Tuple{M,Union{Dict{String,Any}, Dict{Symbol,Any}}}, Tuple{M}} where M<:MonteCarlo.Model",
    "page": "MC",
    "title": "MonteCarlo.MC",
    "category": "method",
    "text": "MC(m::M; kwargs::Dict{String, Any})\n\nCreate a Monte Carlo simulation for model m with (keyword) parameters as specified in the dictionary kwargs.\n\n\n\n"
},

{
    "location": "flavors/mc.html#MonteCarlo.MC-Union{Tuple{M}, Tuple{M}} where M<:MonteCarlo.Model",
    "page": "MC",
    "title": "MonteCarlo.MC",
    "category": "method",
    "text": "MC(m::M; kwargs...) where M<:Model\n\nCreate a Monte Carlo simulation for model m with keyword parameters kwargs.\n\n\n\n"
},

{
    "location": "flavors/mc.html#Exports-1",
    "page": "MC",
    "title": "Exports",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"MC.jl\"]"
},

{
    "location": "flavors/mc.html#Potential-extensions-1",
    "page": "MC",
    "title": "Potential extensions",
    "category": "section",
    "text": "Pull requests are very much welcome!Heat bath (instead of Metropolis) option"
},

{
    "location": "lattices.html#",
    "page": "Lattices",
    "title": "Lattices",
    "category": "page",
    "text": ""
},

{
    "location": "lattices.html#Lattices-1",
    "page": "Lattices",
    "title": "Lattices",
    "category": "section",
    "text": "The package ships with a couple of standard latticesType Description\nMonteCarlo.Chain 1-dimensional chain\nMonteCarlo.SquareLattice 2-dimensional square lattice\nMonteCarlo.CubicLattice D-dimensional cubic latticeIt also provides routines to load the following common lattice formatsType Description\nMonteCarlo.ALPSLattice ALPS simple lattice graph (XML file)"
},

{
    "location": "lattices.html#Didn\'t-find-your-desired-lattice?-1",
    "page": "Lattices",
    "title": "Didn\'t find your desired lattice?",
    "category": "section",
    "text": "Just implement your own lattice for later use in a model of choice. See Custom lattices."
},

{
    "location": "customize.html#",
    "page": "Customize",
    "title": "Customize",
    "category": "page",
    "text": ""
},

{
    "location": "customize.html#Customize-1",
    "page": "Customize",
    "title": "Customize",
    "category": "section",
    "text": ""
},

{
    "location": "customize.html#Custom-models-1",
    "page": "Customize",
    "title": "Custom models",
    "category": "section",
    "text": "Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, the central idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to extend the package and implement your own physical model (or variations of existing models). You can find the interfaces in the corresponding section of the documentation, for example: Interface: Monte Carlo (MC).Sometimes examples tell the most, so feel encouraged to have a look at the implementations of the above mentioned models to get a feeling of how to implement your own model."
},

{
    "location": "customize.html#General-remarks-for-lattice-models-1",
    "page": "Customize",
    "title": "General remarks for lattice models",
    "category": "section",
    "text": ""
},

{
    "location": "customize.html#Semantics-1",
    "page": "Customize",
    "title": "Semantics",
    "category": "section",
    "text": "For lattice models, we define a Model to be a Hamiltonian on a lattice. Therefore, the lattice is part of the model (and not the Monte Carlo flavor). The motivation for this modeling is that the physics of a system does not only depend on the Hamiltonian but also (sometime drastically) on the underlying lattice. This is for example very obvious for spin systems which due to the lattice might become (geometrically) frustrated and show spin liquids physics. Also, from a technical point of view, lattice information is almost exclusively processed in energy calculations which both relate to the Hamiltonian (and therefore the model).note: Note\nWe will generally use the terminology Hamiltonian, energy and so on. However, this doesn\'t restrict you from defining your model as an Lagrangian with an action in any way as this just corresponds to a one-to-one mapping of interpretations."
},

{
    "location": "customize.html#Lattice-requirements-1",
    "page": "Customize",
    "title": "Lattice requirements",
    "category": "section",
    "text": "The Hamiltonian of your model might impose some requirements on the Lattice object that you use as it must provide you with enough lattice information.It might be educating to look at the structure of the simple SquareLattice struct.mutable struct SquareLattice <: AbstractCubicLattice\n   L::Int\n   sites::Int\n   neighs::Matrix{Int} # row = up, right, down, left; col = siteidx\n   neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx\n   sql::Matrix{Int}\n   SquareLattice() = new()\nendIt only provides access to next nearest neighbors through the arrays neighs and neighs_cartesian. If your model\'s Hamiltonian requires higher order neighbor information, because of, let\'s say, a next next nearest neighbor hopping term, the SquareLattice doesn\'t suffice. You could either extend this Lattice or implement a NNSquareLattice for example."
},

{
    "location": "customize.html#Custom-lattices-1",
    "page": "Customize",
    "title": "Custom lattices",
    "category": "section",
    "text": "As described in Custom models a lattice is considered to be part of a model. Hence, most of the requirements for fields of a Lattice subtype come from potential models (see Lattice requirements). Below you\'ll find information on which fields are mandatory from a Monte Carlo flavor point of view."
},

{
    "location": "customize.html#Mandatory-fields-1",
    "page": "Customize",
    "title": "Mandatory fields",
    "category": "section",
    "text": "Any concrete lattice type, let\'s call it MyLattice in the following, must be a subtype of the abstract type MonteCarlo.Lattice. To work with a Monte Carlo flavor, it must internally have at least have the following field,sites: number of lattice sites.However, as already mentioned above depending on the physical model of interest it will typically also have (at least) something likeneighs: next nearest neighbors,as most Hamiltonian will need next nearest neighbor information.The only reason why such a field isn\'t generally mandatory is that the Monte Carlo routine doesn\'t care about the lattice much. Neighbor information is usually only used in the energy (difference) calculation of a particular configuration like done in energy or propose_local which both belong to a Model."
},

{
    "location": "customize.html#Custom-Monte-Carlo-flavors-1",
    "page": "Customize",
    "title": "Custom Monte Carlo flavors",
    "category": "section",
    "text": "Coming soon..."
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
    "location": "methods/general.html#MonteCarlo.observables-Tuple{MonteCarlo.MonteCarloFlavor}",
    "page": "General",
    "title": "MonteCarlo.observables",
    "category": "method",
    "text": "observables(mc::MonteCarloFlavor)\n\nGet a list of all observables defined for a given Monte Carlo simulation.\n\nReturns a Dict{String, String} where values are the observables names and keys are short versions of those names. The keys can be used to collect correponding observable objects from the Monte Carlo simulation, e.g. like mc.obs[key].\n\nNote, there is no need to implement this function for a custom MonteCarloFlavor.\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.reset!-Tuple{MonteCarlo.MonteCarloFlavor}",
    "page": "General",
    "title": "MonteCarlo.reset!",
    "category": "method",
    "text": "reset!(mc::MonteCarloFlavor)\n\nResets the Monte Carlo simulation mc. Previously set parameters will be retained.\n\n\n\n"
},

{
    "location": "methods/general.html#Documentation-1",
    "page": "General",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"abstract.jl\"]"
},

{
    "location": "interfaces/MC.html#",
    "page": "MC",
    "title": "MC",
    "category": "page",
    "text": ""
},

{
    "location": "interfaces/MC.html#Interface:-Monte-Carlo-(MC)-1",
    "page": "MC",
    "title": "Interface: Monte Carlo (MC)",
    "category": "section",
    "text": "Any model that wants to be simulated by means of MC must implement the following interface. Below you find all semantic definitions and precise signatures of mandatory fields and mandatory and optional methods that any model should implement to work with the Monte Carlo flavor Monte Carlo (MC)."
},

{
    "location": "interfaces/MC.html#Mandatory-fields-1",
    "page": "MC",
    "title": "Mandatory fields",
    "category": "section",
    "text": "l::Lattice: any Lattice"
},

{
    "location": "interfaces/MC.html#Index-of-all-methods-1",
    "page": "MC",
    "title": "Index of all methods",
    "category": "section",
    "text": "Pages = [\"MC.md\"]"
},

{
    "location": "interfaces/MC.html#Base.Random.rand-Tuple{MonteCarlo.MC,MonteCarlo.Model}",
    "page": "MC",
    "title": "Base.Random.rand",
    "category": "method",
    "text": "rand(mc::MC, m::Model)\n\nDraw random configuration.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.accept_local!-Tuple{MonteCarlo.MC,MonteCarlo.Model,Int64,Any,Float64,Any,Float64}",
    "page": "MC",
    "title": "MonteCarlo.accept_local!",
    "category": "method",
    "text": "accept_local(mc::MC, m::Model, i::Int, conf, E::Float64, delta_i, delta_E::Float64)\n\nAccept a local move for site i of current configuration conf with energy E. Arguments delta_i and delta_E correspond to output of propose_local() for that local move.\n\nSee also propose_local.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.conftype-Tuple{MonteCarlo.Model}",
    "page": "MC",
    "title": "MonteCarlo.conftype",
    "category": "method",
    "text": "conftype(m::Model)\n\nReturns the type of a configuration.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.energy-Tuple{MonteCarlo.MC,MonteCarlo.Model,Any}",
    "page": "MC",
    "title": "MonteCarlo.energy",
    "category": "method",
    "text": "energy(mc::MC, m::Model, conf)\n\nCalculate energy of configuration conf for Model m.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.propose_local-Tuple{MonteCarlo.MC,MonteCarlo.Model,Int64,Any,Float64}",
    "page": "MC",
    "title": "MonteCarlo.propose_local",
    "category": "method",
    "text": "propose_local(mc::MC, m::Model, i::Int, conf, E::Float64) -> delta_E, delta_i\n\nPropose a local move for lattice site i of current configuration conf with energy E. Returns local move information delta_i (e.g. new[i] - conf[i], will be forwarded to accept_local!) and energy difference delta_E = E_new - E_old.\n\nSee also accept_local!.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#Mandatory-methods-1",
    "page": "MC",
    "title": "Mandatory methods",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nOrder   = [:function]\nPages = [\"MC_mandatory.jl\"]"
},

{
    "location": "interfaces/MC.html#MonteCarlo.finish_observables!-Tuple{MonteCarlo.MC,MonteCarlo.Model,Dict{String,MonteCarloObservable.Observable}}",
    "page": "MC",
    "title": "MonteCarlo.finish_observables!",
    "category": "method",
    "text": "measure_observables!(mc::MC, m::Model, obs::Dict{String,Observable}, conf, E::Float64)\n\nMeasure observables and update corresponding MonteCarloObservable.Observable objects in obs.\n\nSee also prepare_observables and measure_observables!.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.global_move-Tuple{MonteCarlo.MC,MonteCarlo.Model,Any,Float64}",
    "page": "MC",
    "title": "MonteCarlo.global_move",
    "category": "method",
    "text": "global_move(mc::MC, m::Model, conf, E::Float64) -> accepted::Bool\n\nPropose a global move for configuration conf with energy E. Returns wether the global move has been accepted or not.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.measure_observables!-Tuple{MonteCarlo.MC,MonteCarlo.Model,Dict{String,MonteCarloObservable.Observable},Any,Float64}",
    "page": "MC",
    "title": "MonteCarlo.measure_observables!",
    "category": "method",
    "text": "measure_observables!(mc::MC, m::Model, obs::Dict{String,Observable}, conf, E::Float64)\n\nMeasures observables and updates corresponding MonteCarloObservable.Observable objects in obs.\n\nSee also prepare_observables and finish_observables!.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#MonteCarlo.prepare_observables-Tuple{MonteCarlo.MC,MonteCarlo.Model}",
    "page": "MC",
    "title": "MonteCarlo.prepare_observables",
    "category": "method",
    "text": "prepare_observables(m::Model) -> Dict{String, Observable}\n\nInitializes observables and returns a Dict{String, Observable}. In the latter, keys are abbreviations for the observables names and values are the observables themselves.\n\nSee also measure_observables! and finish_observables!.\n\n\n\n"
},

{
    "location": "interfaces/MC.html#Optional-methods-1",
    "page": "MC",
    "title": "Optional methods",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nOrder   = [:function]\nPages = [\"MC_optional.jl\"]"
},

]}
