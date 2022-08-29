################################################################################
### Fourier transformed CCS
################################################################################

#=
# TODO:
This is effectively just a in-simulation Fourier transform (at a specific 
q-vector). It shouldn't be too hard to generalize this as a Fourier wrapper.

Prototyping:
struct Fourier
    iter               # may restrict recorder directions
    fourier_weights    # cis values
    prefactors         # constant prefactors (weights of symmmetries?)
end

Problem: 
fourier_weights may need a q-index, r1-index and r2-index? or multiple spatial 
indices for unit cell and bravais lattice positions? Or are just total distances
enough?
=# 
struct CCSFourier{IT <: DeferredLatticeIterator} <: DeferredLatticeIterator
    iter::IT
    dr::Vector{Float64}
    dq::Vector{Float64}
    weights::Array{ComplexF64, 5}
end

function CCSFourier(iter, dr, dq, l::Lattice)
    CCSFourier(iter, dr, dq, Array{ComplexF64}(undef, output_size(iter, l)))
end

function with_lattice(wrapper::CCSFourier, lattice::Lattice)
    # We need to calculate 0.25 (-Kₓ - Λₓₓ(dq)) where
    # Λₓₓ(q) = CCS[i, j, k] * dot(dr, dir[j]) * dot(dr, dir[k]) * 
    #           cis(- dot(dq, dir[i], + 0.5 (dir[j] - dir[k])))
    # we collect the 0.25, the cis from q=dq into weights

    all_dirs = directions(lattice)
    bravais_dirs = directions(Bravais(lattice))
    hop_idxs = wrapper.iter.directions
    fs = map(idx -> dot(wrapper.dr, all_dirs[idx]), hop_idxs)

    # hop_idxs are related to full lattice directions
    # dr12 needs to be a Bravais lattice direction because EachLocalQuadByDistance
    # uses them (with uc1, uc2)
    for (k, dk) in enumerate(hop_idxs), (j, dj) in enumerate(hop_idxs)
        for (i, dr12) in enumerate(bravais_dirs)
            for uc2 in axes(wrapper.weights, 2), uc1 in axes(wrapper.weights, 1)
                duc = lattice.unitcell.sites[uc2] .- lattice.unitcell.sites[uc1]
                dr = dr12 + duc + 0.5 * (all_dirs[dj] - all_dirs[dk])
                wrapper.weights[uc1, uc2, i, j, k] = 0.25 * fs[j] * fs[k] *
                    cis(-dot(wrapper.dq, dr))
            end
        end
    end

    return WithLattice(wrapper, lattice)
end

_iterate(s::CCSFourier, l::Lattice) = _iterate(s.iter, l)
_iterate(s::CCSFourier, l::Lattice, state) = _iterate(s.iter, l, state)
_length(s::CCSFourier) = _length(s.iter)
_eltype(s::CCSFourier) = _eltype(s.iter)
output_size(s::CCSFourier, l::Lattice) = output_size(s.iter, l)
_binner_zero_element(mc, li::CCSFourier, eltype) = zero(ComplexF64)

# measurement call stack
# compute 0.25 (Λₓₓ(dq) - Λₓₓ(0)) = sum_{i,j,k} weight[i,j,k] * CCS[i,j,k]
@bm function finish!(s::CCSFourier, m, mc)
    # TODO this assumes directions match order of (uc, uc, dir) sorting
    @assert size(s.weights) == size(m.temp) "$(size(s.weights)) != $(size(m.temp))"
    output = zero(ComplexF64)

    for i in eachindex(s.weights)
        output += s.weights[i] * m.temp[i]
    end

    push!(m.observable, output / length(lattice(mc)))
end


################################################################################
### Kₓ
################################################################################


# Is it bad to inherit from Function?
struct DirectedEnergy{T <: Number} <: Function 
    src_sites::Vector{Int}
    trg_sites::Vector{Int}
    weights::Vector{T}
end

function DirectedEnergy(mc::DQMC, dir_idxs::Vector{Int}, prefactor::Number = 3 - unique_flavors(mc))
    # prefactor handles identical flavors
    
    if !isdefined(mc.stack, :hopping_matrix)
        MonteCarlo.init_hopping_matrices(mc, mc.model)
    end
    T = Matrix(mc.stack.hopping_matrix)

    l = lattice(mc)
    srctrg2dir = l[:srctrg2dir]

    src_sites = Vector{Int}()
    trg_sites = Vector{Int}()
    weights = Vector{promote_type(typeof(prefactor), eltype(T))}()

    for i in axes(T, 1), j in axes(T, 2)
        if srctrg2dir[i, j] in dir_idxs
            push!(src_sites, i)
            push!(trg_sites, j)
            push!(weights, prefactor * T[i, j] / length(lattice(mc)))
        end
    end

    return DirectedEnergy(src_sites, trg_sites, weights)
end

function (cache::DirectedEnergy{T})(::DQMC, ::Model, ::Nothing, G, ::Val) where T
    output = zero(promote_type(T, eltype(G)))

    for (src, trg, w) in zip(cache.src_sites, cache.trg_sites, cache.weights)
        # We use c_j^† c_i = δ_ij - G[i, j], but δ always 0 because no onsite
        output -= w * G[src, trg]
    end

    return output
end


################################################################################
### Measurement constructor
################################################################################



struct MultiMeasurement <: AbstractMeasurement
    finalize::Function
    measurements::Vector{DQMCMeasurement}
end

"""
    Multimeasurement(f, measurements...)

Creates a MultiMeasurement holding on to any number of individual measurements. 
The function (or Functor) `f` will be used to finalize the measurement by 
calculating a mean-variance pair from all of the individual measurements.
"""
function MultiMeasurement(f::Function, measurements::AbstractMeasurement...)
    MultiMeasurement(f, collect(measurements))
end

function Base.show(io::IO, mime::MIME"text/plain", m::MultiMeasurement)
    print(io, "Multimeasurement\n")
    for x in m.measurements
        print(io, "  ")
        show(io, mime, x)
        print(io, '\n')
    end
    return
end

BinningAnalysis.mean(m::MultiMeasurement) = m.finalize(m.measurements...)[1]
BinningAnalysis.std_error(m::MultiMeasurement) = m.finalize(m.measurements...)[2]
Base.length(m::MultiMeasurement) = length(m.measurements[1])
Base.isempty(m::MultiMeasurement) = isempty(m.measurements[1])
Base.empty!(m::MultiMeasurement) = empty!.(m.measurements)


"""
    superfluid_stiffness(dqmc, model[; Ls, kwargs...])

Returns a measurement that calculates the superfluid stiffness as 
`0.25 (-Kₓ - Λₓₓ^T)`. 

In this formula, Kₓ is the kinetic energy of bonds that have a component in x 
direction, where x is (in general) an arbitrary direction. Λₓₓ(q, ω) is the 
(time and space) Fourier transformed current current correlation function in 
x direction. If q || x we call it longitudinal Λₓₓ^L == -Kₓ, and if q ⟂ x we 
call it transversal Λₓₓ^T. In the formula above we specifically care about q → 0.

To get a minimal q vector in the transversal direction we choose it to be a 
reciprocal lattice vector. From that we derive the x direction as a normalized, 
perpendicular vector from it.

Note: Whether this choice is good or not may depend on the lattice. You can 
manually set `transversal` as a keyword argument if you wish to overwrite the 
default.
"""
function superfluid_stiffness(
        dqmc::DQMC, model::Model; 
        transversal = reciprocal_vectors(lattice(dqmc))[1], kwargs...
    )
    @assert length(transversal) == 2
    shift_dir = normalize([0 1; -1 0] * transversal)
    
    # find all hopping directions (skipping on-site)
    valid_directions = hopping_directions(model)

    # reduce to directions that have a positive component in shift_dir
    # (the negative version is already in the kernel function)
    dirs = directions(lattice(dqmc))
    filter!(idx -> dot(dirs[idx], shift_dir) > 0, valid_directions)

    # create lattice iterator
    li = CCSFourier(
        EachLocalQuadByDistance(valid_directions), shift_dir, transversal, lattice(dqmc)
    )

    # NOTE: tuple measurements should work because we implemented them in generate_groups
    return MultiMeasurement(
        _superfluid_stiffness,
        Measurement(dqmc, model, Greens(), nothing, DirectedEnergy(dqmc, valid_directions)),
        current_current_susceptibility(dqmc, model, lattice_iterator = li, eltype = ComplexF64)
    )
end

function _superfluid_stiffness(m1::DQMCMeasurement, m2::DQMCMeasurement)
    Kx  = mean(m1);  ΔKx  = max(0.0, real(std_error(m1)))
    Λxx = mean(m2);  ΔΛxx = max(0.0, real(std_error(m2)))
    return 0.25 * (-Kx - Λxx), 0.25 * sqrt(ΔKx^2 + ΔΛxx^2)
end