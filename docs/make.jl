using Documenter, DocumenterMarkdown, MonteCarlo

makedocs(
    modules = [MonteCarlo],
    doctest = false,
    sitename = "MonteCarlo.jl",
    format = Markdown(),
)

deploydocs(
    deps = Deps.pip(
        "mkdocs",
        "mkdocs-material",
        "python-markdown-math",
        "pygments",
        "pymdown-extensions",
    ),
    repo = "github.com/crstnbr/MonteCarlo.jl.git",
    make = () -> run(`mkdocs build`),
    target = "site",
    push_preview = true,
)
