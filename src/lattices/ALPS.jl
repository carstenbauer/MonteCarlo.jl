"""
Generic ALPS lattice parsed from XML file.
"""
mutable struct ALPSLattice <: Lattice
    sites::Int # n_sites
    dim::Int
    neighs::Matrix{Int} # row = neighbors; col = siteidx (assumption: const. coordination nr.)
    n_neighs::Int
    bond_vecs::Matrix{Float64}

    # for generic checkerboard support
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type

    ALPSLattice() = new()
end

# constructors
function ALPSLattice(xmlfile::String)
    l = ALPSLattice()
    parse_alpslattice_xml(xmlfile, l)
    build_neighbortable!(l)
    return l
end

function build_neighbortable!(l::ALPSLattice)
    l.neighs = zeros(Int, l.n_neighs, l.sites)

    # we store src->trg before trg->src
    try
        nc = fill(1, l.sites)
        for b in 1:l.n_bonds
            src = l.bonds[b,1]
            trg = l.bonds[b,2]
            l.neighs[nc[src], src] = trg
            nc[src]+=1
        end
        for b in 1:l.n_bonds
            src = l.bonds[b,1]
            trg = l.bonds[b,2]
            l.neighs[nc[trg], trg] = src
            nc[trg]+=1
        end
    catch e
        if isa(e, BoundsError)
            warn("Lattice seems to not have a constant coordination number. Fields `n_neighs` and `neighs` shouldn't be used.")
        else
            throw(e)
        end
    end
    nothing
end

function parse_alpslattice_xml(filename::String, l::ALPSLattice)
  xdoc = parse_file(filename)
  graph = LightXML.root(xdoc)
  l.sites = 1

  l.sites = parse(Int, attribute(graph, "vertices"; required=true))
  l.dim = parse(Int, attribute(graph, "dimension"; required=true))

  edges = get_elements_by_tagname(graph, "EDGE")
  l.n_bonds = length(edges)

  # bonds & bond vectors
  l.bonds = zeros(l.n_bonds, 3)
  l.bond_vecs = zeros(l.n_bonds, l.dim)
  v = Vector{Float64}(l.dim)
  for (i, edge) in enumerate(edges)
    src = 0
    trg = 0
    src = parse(Int, attribute(edge, "source"; required=true))
    trg = parse(Int, attribute(edge, "target"; required=true))
    typ = parse(Int, attribute(edge, "type"; required=true))
    id = parse(Int, attribute(edge, "id"; required=true))
    v = [parse(Float64, f) for f in split(attribute(edge, "vector"; required=true)," ")]

    if id != i error("Edges in lattice file must be sorted from 1 to N!") end

    l.bonds[i, 1] = src
    l.bonds[i, 2] = trg
    l.bonds[i, 3] = typ
    l.bond_vecs[i, :] = v
  end

  l.n_neighs = count(x->x==1, l.bonds[:, 1]) + count(x->x==1, l.bonds[:, 2]) # neighbors of site 1
end
