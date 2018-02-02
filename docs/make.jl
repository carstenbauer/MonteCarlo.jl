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
            "Examples" => "manual/examples.md",
            "Monte Carlo flavors" => [
                "MC" => "manual/flavors/mc.md"
                # "DQMC" => "manual/flavors/dqmc.md"
            ],
            "Physical models" => [
                "Ising model" => "manual/models/ising.md",
                "Hubbard model" => "manual/models/hubbard.md"
            ],
            "Custom" => [
                "Models" => "manual/custom/models.md",
                "Lattices" => "manual/custom/lattices.md",
                "Flavors" => "manual/custom/flavors.md"
            ]
        ],
        "Methods" => [
            "General" => "methods/general.md",
            "Flavors" => [
                "MC" => "methods/flavors/MC.md"
            ]
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
