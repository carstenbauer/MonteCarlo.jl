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
function build_checkerboard(l::Lattice)
  # check if lattice is compatible with generic cb algorithm
  mandatory_fields = [:n_bonds, :bonds]
  all(map(f->in(f, fieldnames(typeof(l))), mandatory_fields)) || error("Lattice $(typeof(l)) "*
            "doesn't have all the necessary fields for generic checkerboard decomposition.")

  groups = UnitRange[]
  edges_used = zeros(Int64, l.n_bonds)
  checkerboard = zeros(Int64, 3, l.n_bonds)
  group_start = 1
  group_end = 1

  while minimum(edges_used) == 0
    sites_used = zeros(Int64, l.sites)

    for id in 1:l.n_bonds
      src, trg, typ = l.bonds[id,1:3]

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
