abstract type AbstractDQMCStack end

abstract type AbstractUpdateScheduler end
init!(s::AbstractUpdateScheduler, mc, model) = s

abstract type AbstractField end
abstract type AbstractFieldCache end


################################################################################
### Checkerboard
################################################################################


# TODO Rework this into a matrix wrapper

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
    # _bonds = collect(bonds(l, Val(true)))
    # n_bonds = length(_bonds)

    # groups = UnitRange[]
    # edges_used = zeros(Int64, n_bonds)
    # checkerboard = zeros(Int64, 3, n_bonds)
    # group_start = 1
    # group_end = 1

    # while minimum(edges_used) == 0
    #     sites_used = zeros(Int64, length(l))

    #     for (id, b) in enumerate(_bonds)
    #         src = b.from; trg = b.to
    #         if edges_used[id] == 1 continue end
    #         if sites_used[src] == 1 continue end
    #         if sites_used[trg] == 1 continue end

    #         edges_used[id] = 1
    #         sites_used[src] = 1
    #         sites_used[trg] = 1

    #         checkerboard[:, group_end] = [src, trg, id]
    #         group_end += 1
    #     end
    #     push!(groups, group_start:group_end-1)
    #     group_start = group_end
    # end
    # n_groups = length(groups)

    # return checkerboard, groups, n_groups

    # TODO: doesn't work with uneven L
    bs = unitcell(l).bonds

    base_groups = map(bs) do uc_bond
        map(eachindex(Bravais(l))) do src
            global_bond = MonteCarlo._shift_Bravais(l, src, uc_bond)
            MonteCarlo.from(global_bond) => MonteCarlo.to(global_bond)
        end
    end

    # I'm guessing either every src becomes a trg or none does...
    # this seems to do the trick for even lattice sizes
    # I guess for uneven L we need a rest group
    groups = Vector{Pair{Int, Int}}[]
    for base_group in base_groups
        src, trg = base_group[1]
        if any(t -> t[1] == trg, base_group)
            used = Int[]
            group1 = Pair{Int, Int}[]
            group2 = Pair{Int, Int}[]
            for (src, trg) in base_group
                if !(src in used) && !(trg in used)
                    push!(group1, src => trg)
                    push!(used, src, trg)
                else
                    push!(group2, src => trg)
                end
            end
            push!(groups, group1, group2)
        else
            push!(groups, base_group)
        end
    end

    Ns = vcat(0, cumsum(length.(groups)))
    cb = Matrix{Int}(undef, 3, last(Ns))
    ranges = [Ns[i]+1 : Ns[i+1] for i in eachindex(groups)]
    for i in eachindex(groups)
        for j in eachindex(groups[i])
            cb[1, Ns[i]+j] = groups[i][j][1]
            cb[2, Ns[i]+j] = groups[i][j][2]
        end
    end

    return cb, ranges, length(groups)
end


function build_checkerboard2(l::AbstractLattice)
    # TODO: doesn't work with uneven L
    # TODO: need to be wary of bonds who are there own inverses
    bs = view(unitcell(l).bonds, unitcell(l)._directed_indices)

    base_groups = map(bs) do uc_bond
        map(eachindex(Bravais(l))) do src
            global_bond = MonteCarlo._shift_Bravais(l, src, uc_bond)
            MonteCarlo.from(global_bond) => MonteCarlo.to(global_bond)
        end
    end

    # I'm guessing either every src becomes a trg or none does...
    # this seems to do the trick for even lattice sizes
    # I guess for uneven L we need a rest group
    groups = Vector{Pair{Int, Int}}[]
    for base_group in base_groups
        src, trg = base_group[1]
        if any(t -> t[1] == trg, base_group)
            used = Int[]
            group1 = Pair{Int, Int}[]
            group2 = Pair{Int, Int}[]
            for (src, trg) in base_group
                if !(src in used) && !(trg in used)
                    push!(group1, src => trg)
                    push!(used, src, trg)
                else
                    push!(group2, src => trg)
                end
            end
            push!(groups, group1, group2)
        else
            push!(groups, base_group)
        end
    end

    Ns = vcat(0, cumsum(length.(groups)))
    cb = Matrix{Int}(undef, 3, last(Ns))
    ranges = [Ns[i]+1 : Ns[i+1] for i in eachindex(groups)]
    for i in eachindex(groups)
        for j in eachindex(groups[i])
            cb[1, Ns[i]+j] = groups[i][j][1]
            cb[2, Ns[i]+j] = groups[i][j][2]
        end
    end

    return cb, ranges, length(groups)
end
