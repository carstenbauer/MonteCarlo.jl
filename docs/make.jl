using Documenter, MonteCarlo

makedocs(
    modules = [MonteCarlo],
    doctest = false,
    sitename = "MonteCarlo.jl",
    pages = [
        "Introduction" => "index.md",
        "Manual" => [
            "Getting started" => "manual/gettingstarted.md",
            "Showcase" => "manual/showcase.md",
        ],
        "Physical models" => [
            "Ising model" => "models/ising.md",
            "Attractive Hubbard model" => "models/hubbardattractive.md",
        ],
        "Monte Carlo flavors" => [
            "MC" => "flavors/mc.md",
            "DQMC" => "flavors/dqmc.md",
        ],
        "Lattices" => "lattices.md",
        "Customize" => "customize.md",
        "Interfaces" => [
            "MC" => "interfaces/MC.md",
            "DQMC" => "interfaces/DQMC.md",
        ],
        "General exports" => ["General" => "methods/general.md"],
    ],
    # assets = ["assets/custom.css", "assets/custom.js"]
)

deploydocs(
    repo = "github.com/crstnbr/MonteCarlo.jl.git",
    push_preview = true,
    # target = "site",
)
