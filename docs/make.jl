using Documenter, MonteCarlo

makedocs(
    modules = [MonteCarlo],
    doctest = false,
    sitename = "MonteCarlo.jl",
    pages = [
        "Introduction" => "index.md",
        "DQMC" => [
            "Introduction" => "DQMC/Introduction.md",
            "Model" => "DQMC/model.md",
            "Lattices" => "DQMC/lattice.md",
            "Fields" => "DQMC/fields.md",
            "DQMC" => "DQMC/dqmc.md",
            "Configuration Recorder" => "DQMC/recorder.md",
            "Update Scheduler" => "DQMC/scheduler.md",
            "Measurements" => "DQMC/measurements.md",
        ],
        "Examples" => [
            "examples/triangular_Hubbard.md",
            "examples/ALF1.md",
            "examples/HofmannBergChowdhury.md"
        ],
        "Legacy" => [
            "Introduction" => "legacy/index.md",
            "Manual" => [
                "Getting started" => "legacy/manual/gettingstarted.md",
                "Showcase" => "legacy/manual/showcase.md",
            ],
            "Physical models" => [
                "Ising model" => "legacy/models/ising.md",
                "Attractive Hubbard model" => "legacy/models/hubbardattractive.md",
            ],
            "Monte Carlo flavors" => [
                "MC" => "legacy/flavors/mc.md",
                "DQMC" => "legacy/flavors/dqmc.md",
            ],
            "Lattices" => "legacy/lattices.md",
            "Customize" => "legacy/customize.md",
            "Interfaces" => [
                "MC" => "legacy/interfaces/MC.md",
                "DQMC" => "legacy/interfaces/DQMC.md",
            ],
            "General exports" => [
                "General" => "legacy/methods/general.md"
            ],
        ],
        "DQMC Dev Docs" => [
            "devdocs/measurements.md"
        ]
    ],
    # assets = ["assets/custom.css", "assets/custom.js"]
)

deploydocs(
    repo = "github.com/carstenbauer/MonteCarlo.jl.git",
    push_preview = true,
    # target = "site",
)
