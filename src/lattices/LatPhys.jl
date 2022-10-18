import .LatticePhysics.LatPhysBase
import .LatticePhysics

function MonteCarlo.Lattice(l::LatPhysBase.AbstractLattice)
    uc = LatPhysBase.unitcell(l)
    lv = Tuple(LatPhysBase.latticeVectors(uc))
    Ls = map(LatPhysBase.latticeVectors(l), lv) do lv, ucv
        scaling = lv ./ ucv
        if all(x -> x ≈ scaling[1], scaling)
            return scaling[1]
        else
            @warn "Irregular scaling of unitcell vectors $scaling"
            NaN
        end
    end
    sites = LatPhysBase.point.(uc.sites)
    @assert prod(Ls) * length(sites) == LatPhysBase.numSites(l) "Failed to determine correct linear system size"
    if all(b -> b.label isa Integer, uc.bonds)
        bonds = map(b -> Bond(b.from, b.to, b.wrap, b.label), uc.bonds)
    else
        lbls = unique(LatPhysBase.label.(uc.bonds))
        mapping = Dict(lbls, eachindex(lbls))
        bonds = map(b -> Bond(b.from, b.to, b.wrap, mapping[b.label]), uc.bonds)
    end
    return MonteCarlo.Lattice(UnitCell(lv, sites, bonds), Tuple(round.(Int, Ls)))
end

# This is a bit weird, does it work?
function LatPhysBase.Lattice(l::MonteCarlo.Lattice)
    uc = l.unitcell
    lv = collect(lattice_vectors(l))
    sites = map(p -> LatPhysBase.Site{Int, length(p)}(p, 1), uc.sites)
    bonds = map(b -> LatPhysBase.Bond(b.from, b.to, b.label, b.uc_shift), uc.bonds)
    return LatticePhysics.getLatticePeriodic(
        LatPhysBase.Unitcell(lv, sites, bonds), l.Ls
    )
end