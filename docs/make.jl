using Documenter, MonteCarlo

makedocs(modules = [MonteCarlo], doctest=false)

deploydocs(
    deps   = Deps.pip("mkdocs", "mkdocs-material" ,"python-markdown-math", 
        "pygments", "pymdown-extensions"),
    repo   = "github.com/crstnbr/MonteCarlo.jl.git",
    julia  = "release",
    osname = "linux",
)