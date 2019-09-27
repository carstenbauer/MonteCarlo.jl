"""
Generic ALPS lattice parsed from XML file.
"""
struct ALPSLattice <: AbstractLattice
    sites::Int # n_sites
    dim::Int
    neighs::Matrix{Int} # row = neighbors; col = siteidx (assumption: const. coordination nr.)
    n_neighs::Int
    bond_vecs::Matrix{Float64}

    # for generic checkerboard support
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

# constructors
function ALPSLattice(xmlfile::String)
    sites, dim, n_neighs, bond_vecs, n_bonds, bonds = parse_alpslattice_xml(xmlfile, l)
    neighs = build_neighbortable(ALPSLattice, n_neighs, sites, n_bonds, bonds)
    return ALPSLattice(sites, dim, neighs, n_neighs, bond_vecs, n_bonds, bonds)
end

function build_neighbortable(::Type{ALPSLattice}, n_neighs, sites, n_bonds, bonds)
    neighs = zeros(Int, n_neighs, sites)

    # we store src->trg before trg->src
    try
        nc = fill(1, sites)
        for b in 1:n_bonds
            src = bonds[b,1]
            trg = bonds[b,2]
            neighs[nc[src], src] = trg
            nc[src]+=1
        end
        for b in 1:n_bonds
            src = bonds[b,1]
            trg = bonds[b,2]
            neighs[nc[trg], trg] = src
            nc[trg]+=1
        end
    catch e
        if isa(e, BoundsError)
            warn("Lattice seems to not have a constant coordination number. Fields `n_neighs` and `neighs` shouldn't be used.")
        else
            throw(e)
        end
    end
    return neighs
end

function parse_alpslattice_xml(filename::String)
    xdoc = parse_file(filename)
    graph = LightXML.root(xdoc)
    sites = 1

    sites = parse(Int, attribute(graph, "vertices"; required=true))
    dim = parse(Int, attribute(graph, "dimension"; required=true))

    edges = get_elements_by_tagname(graph, "EDGE")
    n_bonds = length(edges)

    # bonds & bond vectors
    bonds = zeros(n_bonds, 3)
    bond_vecs = zeros(n_bonds, dim)
    v = Vector{Float64}(dim)
    for (i, edge) in enumerate(edges)
        src = 0
        trg = 0
        src = parse(Int, attribute(edge, "source"; required=true))
        trg = parse(Int, attribute(edge, "target"; required=true))
        typ = parse(Int, attribute(edge, "type"; required=true))
        id = parse(Int, attribute(edge, "id"; required=true))
        v = [parse(Float64, f) for f in split(attribute(edge, "vector"; required=true)," ")]

        if id != i error("Edges in lattice file must be sorted from 1 to N!") end

        bonds[i, 1] = src
        bonds[i, 2] = trg
        bonds[i, 3] = typ
        bond_vecs[i, :] = v

    end

    n_neighs = count(x->x==1, bonds[:, 1]) + count(x->x==1, bonds[:, 2]) # neighbors of site 1
    return sites, dim, n_neighs, bond_vecs, n_bonds, bonds
end

# Implement AbstractLattice interface: mandatory
@inline Base.length(l::ALPSLattice) = l.sites

# Implement AbstractLattice interface: optional
@inline neighbors_lookup_table(l::ALPSLattice) = copy(l.neighs)

# HasNeighborsTable and HasBondsTable traits
has_neighbors_table(::ALPSLattice) = HasNeighborsTable()
has_bonds_table(::ALPSLattice) = HasBondsTable()
