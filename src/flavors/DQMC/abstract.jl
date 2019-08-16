abstract type AbstractDQMCStack end

abstract type Checkerboard end
struct CheckerboardTrue <: Checkerboard end
struct CheckerboardFalse <: Checkerboard end


"""
    build_checkerboard(l::Lattice) -> checkerboard, groups, n_groups

Generic checkerboard construction.

The methods returns
    * `checkerboard::Matrix{Int}`: size `(3, n_bonds)` where
                rows = `source site`, `target site`, `bondid` and cols
                    correspond to bonds. Sorted (along columns) in such a way
                    that `checkerboard[3, groups[i]]` are all the bond indices
                    corresponding to the i-th group.
    * `groups::Vector{UnitRange}`: ranges indicating which columns of `checkerboard`
                    belong to which checkerboard group.
    * `n_groups::Int`: number of checkerboard groups.
"""
function build_checkerboard(l::AbstractLattice)
    bonds = neighbors(l)
    n_bonds = length(bonds)

    groups = UnitRange[]
    edges_used = zeros(Int64, n_bonds)
    checkerboard = zeros(Int64, 3, n_bonds)
    group_start = 1
    group_end = 1

    while minimum(edges_used) == 0
        sites_used = zeros(Int64, length(l))

        for (id, (src, trg)) in enumerate(bonds)
            if edges_used[id] == 1 continue end
            if sites_used[src] == 1 continue end
            if sites_used[trg] == 1 continue end

            edges_used[id] = 1
            sites_used[src] = 1
            sites_used[trg] = 1

            checkerboard[:, group_end] = [src, trg, id]
            group_end += 1
        end
        push!(groups, group_start:group_end-1)
        group_start = group_end
    end
    n_groups = length(groups)

    return checkerboard, groups, n_groups
end
