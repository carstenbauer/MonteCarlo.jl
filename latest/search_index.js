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
    "text": "TODO"
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
    "text": "This is a simple demontration of how to perform a classical Monte Carlo simulation of the 2D Ising model:using PyPlotusing MonteCarlo, MonteCarloObservable\nm = IsingModel(dims=2, L=8, β=0.35);\nmc = MC(m);\nrun!(mc, sweeps=1000, thermalization=1000, verbose=false);\n\nobservables(m) # what observables do exist for that model?\n\nm = mc.obs[\"m\"] # take observable\nname(m) # ==== \"Magnetization (per site)\"\ntypeof(m) # === MonteCarloObservable\n\nmean(m) # estimate for the mean\nstd(m) # one-sigma error of mean from automated binning analysis\n\nhist(m) # histogram of time series\nbegin savefig(\"hist.svg\"); nothing end # hide\nplot(m) # plot time series\nbegin savefig(\"ts.svg\"); nothing end # hide\nnothing # hide(Image: )(Image: )"
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
    "location": "methods/general.html#MonteCarlo.init!-Union{Tuple{MonteCarlo.MC{#s83,S} where #s83<:MonteCarlo.Model}, Tuple{S}} where S",
    "page": "General",
    "title": "MonteCarlo.init!",
    "category": "Method",
    "text": "init!(mc::MC[; seed::Real=-1])\n\nInitialize the classical Monte Carlo simulation mc. If seed !=- 1 the random generator will be initialized with srand(seed).\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.run!-Union{Tuple{MonteCarlo.MC{#s12,S} where #s12<:MonteCarlo.Model}, Tuple{S}} where S",
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
    "location": "methods/general.html#MonteCarlo.observables-Tuple{MonteCarlo.Model}",
    "page": "General",
    "title": "MonteCarlo.observables",
    "category": "Method",
    "text": "observables(m::Model)\n\nGet all observables defined for a given model.\n\nReturns a Dict{String, String} where values are the observables names and keys are short versions of those names. The keys can be used to collect correponding observable objects from a Monte Carlo simulation, e.g. like mc.obs[key].\n\n\n\n"
},

{
    "location": "methods/general.html#Documentation-1",
    "page": "General",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"mc.jl\", \"IsingModel.jl\", \"abstract_functions.jl\"]"
},

]}
