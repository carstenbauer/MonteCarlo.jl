################################################################################
### Lattice Iterator
################################################################################


struct SuperfluidStiffness{IT, N, T} <: AbstractLatticeIteratorTemplate
    template::IT
    dq::Vector{Float64}
end
function SuperfluidStiffness(template::DeferredLatticeIteratorTemplate, dq)
    SuperfluidStiffness(template, dq)
end

struct _SuperfluidStiffness{LI <: DeferredLatticeIterator} <: LatticeIterationWrapper{LI}
    iter::LI
    weights::Matrix{ComplexF64}
end
function (x::SuperfluidStiffness)(cache::LatticeIteratorCache, lattice::AbstractLattice)
    # We need to calculate 0.25 (Λₓₓ(dq) - Λₓₓ(0)) where
    # Λₓₓ(q) = CCS[i, j, k] * dot(eₓ, dir[j]) * dot(eₓ, dir[k]) * 
    #           cis(- dot(dq, dir[i], + 0.5 (dir[j] - dir[k])))
    # we collect the 0.25, the cis from q=dq and the cis from q=0 (1) into weights
    # Idk why the dot(eₓ, dir[j/k]) is there but https://arxiv.org/pdf/1912.08848.pdf
    # has them so I'll have them too (in J_x as prefactor 1 or 2)
    # The 0.5 (dir[j] - dir[k]) comes from the same paper, eq 17-19
    all_dirs = directions(lattice)
    hop_idxs = x.template.directions
    dr = normalize(x.dq)
    fs = map(idx -> dot(dr, all_dirs[idx]), hop_idxs)
    weights = Matrix{ComplexF64}(undef, length(all_dirs), length(hop_idxs), length(hop_idxs))
    for (i, dr12) in enumerate(all_dirs)
        for (j, dj) in enumerate(hop_idxs), (k, dk) in enumerate(hop_idxs)
            weights[i,j,k] = 0.25 * fs[j] * fs[k] *
                (cis(-dot(x.dq, dr12 + 0.5 * (dirs[dj] - dirs[dk]))) - 1)
        end
    end

    _SuperfluidStiffness(x.template(cache, lattice), weights)
end

Base.iterate(s::_SuperfluidStiffness) = iterate(s.iter)
Base.iterate(s::_SuperfluidStiffness, state) = iterate(s.iter, state)
Base.length(s::_SuperfluidStiffness) = length(s.iter)
Base.eltype(s::_SuperfluidStiffness) = eltype(s.iter)



################################################################################
### Adjustments to measurement call stack
################################################################################



# binner contains Complex numbers
_binner_zero_element(mc, model, ::SuperfluidStiffness, eltype) = zero(ComplexF64)

# compute 0.25 (Λₓₓ(dq) - Λₓₓ(0)) = sum_{i,j,k} weight[i,j,k] * CCS[i,j,k]
function commit!(s::_SuperfluidStiffness, m)
    @assert size(s.weight) == size(m.temp)
    final = zero(ComplexF64)
    for i in eachindex(m.temp)
        final += s.weight[i] * m.temp[i]
    end
    push!(m.observable, final)
end



################################################################################
### Measurement constructor
################################################################################



"""
    superfluid_stiffness(dqmc, model[; Ls, kwargs...])

Returns a measurement that calculates the superfluid stiffness as 
`0.25 (Λₓₓ((dq, 0)) - Λₓₓ(0))`. 

Note that this might be slightly different from `0.25 (-K_x - Λₓₓ(0))` and that
some sources use `0.25 (Λₓₓ((dq, 0)) - Λₓₓ((0, dq)))` instead. (I.e. a 
longitudinal Λₓₓ minus a transversal Λₓₓ.)
"""
function superfluid_stiffness(
        dqmc::DQMC, model::Model; Ls = size(lattice(model)), kwargs...
    )
    # TODO
    # I'm still not sure how exactly shift_dir/long should be picked.

    # 0.25 [Λₓₓ(q = (Δq, 0) - Λₓₓ(0))]
    shift_dir = Float64[1, 0]
    
    # find all hopping directions (skipping on-site)
    dir2srctrg = mc[Dir2SrcTrg()]
    T = hopping_matrix(mc, model)
    valid_directions = Int64[]
    
    for i in 2:length(dir2srctrg)
        for (src, trg) in dir2srctrg[i]
            if T[trg, src] != 0
                push!(valid_directions, i)
                break
            end
        end
    end

    # reduce to directions that have a positive component in shift_dir
    dirs = directions(lattice(dqmc))
    filter!(idx -> dot(dirs[idx], shift_dir) > 0, valid_directions)

    # create lattice iterator
    long = 2pi / Ls[1] * shift_dir
    li = SuperfluidStiffness(EachLocalQuadByDistance(valid_directions), long)

    current_current_susceptibility(dqmc, model, lattice_iterator = li)
end