abstract type AbstractDQMCStack end

abstract type AbstractUpdateScheduler end
init!(s::AbstractUpdateScheduler, mc, model) = s

abstract type AbstractField end
abstract type AbstractFieldCache end


################################################################################
### Checkerboard
################################################################################


"""
    build_checkerboard(l::Lattice)

Generates a collection of groups containing (src, trg) pairs where each group 
only touches each site (src or trg) once. Each group also represents only one 
bond direction, though multiple groups can represent the same direction. Bond 
reversals (i.e. trg -> src relative to src -> trg) are not considered here, as 
they are always present.
"""
function build_checkerboard(l::AbstractLattice)
    @assert all(l.Ls .> 2) "Error become very large for tiny system sizes"
    bs = view(unitcell(l).bonds, unitcell(l)._directed_indices)

    # Unit cell bonds can match up with the Bravais lattice size to create 
    # duplicate pairs. For example a bond with uc_shift = (2, 0) on a L = 4 
    # lattice can generate src = (2, 0) -> trg = (4, 0) and it's reverse 
    # (2, 0) -> (0, 0) = (4, 0).
    # Since we later use the pairs generated here to read out the hopping 
    # matrix we don't want these duplicates to exist. Thus we need to explicitly
    # filter them out here.

    base_groups = map(bs) do uc_bond
        map(eachindex(Bravais(l))) do src
            global_bond = MonteCarlo._shift_Bravais(l, src, uc_bond)
            MonteCarlo.from(global_bond) => MonteCarlo.to(global_bond)
        end |> unique
        # unique to avoid duplicates
    end

    # avoid duplicates across groups
    for i in eachindex(base_groups)
        for j in i+1:length(base_groups)
            filter!(pair -> !(pair in base_groups[i]), base_groups[j])
        end
    end

    # I'm guessing either every src becomes a trg or none does...
    # this seems to do the trick for even lattice sizes
    # I guess for uneven L we need a rest group
    groups = Vector{Pair{Int, Int}}[]
    for base_group in base_groups
        src, trg = base_group[1]
        if any(t -> t[1] == trg, base_group)
            used1 = Int[]
            used2 = Int[]
            used3 = Int[]
            group1 = Pair{Int, Int}[]
            group2 = Pair{Int, Int}[]
            group3 = Pair{Int, Int}[]
            for (src, trg) in base_group
                if !(src in used1) && !(trg in used1)
                    push!(group1, src => trg)
                    push!(used1, src, trg)
                elseif !(src in used2) && !(trg in used2)
                    push!(group2, src => trg)
                    push!(used2, src, trg)
                elseif !(src in used3) && !(trg in used3)
                    push!(group3, src => trg)
                    push!(used3, src, trg)
                else
                    error("Failed to distribute bonds into groups.")
                end
            end
            push!(groups, group1)
            isempty(group2) || push!(groups, group2)
            isempty(group3) || push!(groups, group3)
        else
            push!(groups, base_group)
        end
    end

    return groups
end
