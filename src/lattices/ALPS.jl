"""
Generic ALPS lattice parsed from XML file.
"""
mutable struct ALPSLattice <: Lattice
    # TODO
    # L::Int
    # neighs::Matrix{Int} # row = neighbors; col = siteidx
    ALPSLattice() = new()
end

# constructors
function ALPSLattice(filename::String)
    l = ALPSLattice()
    # TODO
    build_neighbortable!(l)
    return l
end

function build_neighbortable!(l::ALPSLattice)
    # TODO
end
