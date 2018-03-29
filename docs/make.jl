using Documenter, MonteCarlo

makedocs(
    # options

)

makedocs(
    modules = [MonteCarlo],
    format = :html,
    sitename = "MonteCarlo.jl",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting started" => "manual/gettingstarted.md",
            "Examples" => "manual/examples.md"
        ],
        "Physical models" => [
            "Ising model" => "models/ising.md",
            "Attractive Hubbard model" => "models/hubbardattractive.md"
        ],
        "Monte Carlo flavors" => [
            "MC" => "flavors/mc.md",
            "DQMC" => "flavors/dqmc.md"
        ],
        "Lattices" => "lattices.md",
        "Customize" => "customize.md",
        "Methods" => [
            "General" => "methods/general.md"
        ],
        "Interfaces" => [
            "MC" => "interfaces/MC.md",
            "DQMC" => "interfaces/DQMC.md"
        ]
    ]
)

deploydocs(
    repo   = "github.com/crstnbr/MonteCarlo.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    julia  = "release",
    osname = "linux"
)
