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
        "Functions" => "functions.md"
        # "Subsection" => [
        #     ...
        # ]
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