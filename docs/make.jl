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
            "Customize" => [
                "Models" => "manual/custommodels.md",
                "Lattices" => "manual/customlattices.md"
            ]
        ],
        "Methods" => [
            "General" => "methods/general.md",
            "Models" => "methods/models.md"
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