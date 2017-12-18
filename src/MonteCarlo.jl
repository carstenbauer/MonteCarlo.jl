module MonteCarlo

include("abstract_types.jl")

# include all lattices
for l in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"lattices"))))
    include(joinpath(@__DIR__,"lattices",l))
end

# include all models
for m in filter(x->isdir(joinpath(@__DIR__,"models",x)),
                sort!(readdir(joinpath(@__DIR__,"models"))))
    for jlf in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"models",m))))
        include(joinpath(@__DIR__,"models",m,jlf))
    end
end

# include all mc flavors
for flv in filter(x->isdir(joinpath(@__DIR__,"flavors",x)),
                sort!(readdir(joinpath(@__DIR__,"flavors"))))
    for jlf in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"flavors",flv))))
        include(joinpath(@__DIR__,"flavors",flv,jlf))
    end
end

end # module
