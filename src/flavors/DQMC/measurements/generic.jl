# call stack:
# run/replay 
#   > calls apply(GreensIterator, group, dqmc, model, sweep)
#       > calls measure(LatticeIterator, measurement, dqmc, model, sweep, GreensResults...)
#           > calls apply!(LatticeIterator, measurement, dqmc, model, GreensResults...)
#               > calls kernel(measurement, dqmc, model, MaskResult..., GreensResults...)

# Does:
# creates groups, runs simulation
#   > resolves GreensIterator (calclates G, Gkk, Gkl or whatever)
#       > resolves sweep (skipping, frequency based measurements) and commits
#           > resolves Mask (to Greens indices) 
#               > calculate element from Wicks theorem

# TODO
# - deprecate old measurement types
# - _save/_load to autoconvert
# - re-add Greens -> Occupation maybe?


struct DQMCMeasurement{
        GreensIterator, LatticeIterator, F <: Function, OT, T
    } <: AbstractMeasurement
    kernel::F
    observable::OT
    output::T
end

function DQMCMeasurement{GI, LI}(kernel::FT, observable::OT, output::T) where {GI, LI, FT, OT, T}
    DQMCMeasurement{GI, LI, FT, OT, T}(kernel, observable, output)
end

function Measurement(
        dqmc, model, GreensIterator, LatticeIterator, kernel::F;
        capacity = _default_capacity(dqmc), eltype = geltype(dqmc),
        shape = _get_shape(dqmc, model, LatticeIterator)
    ) where F
    temp = shape === nothing ? zero(eltype) : zeros(eltype, shape)
    obs = LogBinner(temp, capacity=capacity)
    DQMCMeasurement{GreensIterator, LatticeIterator}(kernel, obs, temp)
end



################################################################################
### DQMCMeasurement utilities
################################################################################



function Base.show(io::IO, ::MIME"text/plain", m::DQMCMeasurement{GI, LI}) where {GI, LI}
    max = capacity(m.observable)
    current = length(m.observable)
    print(io, "[$current/$max] DQMCMeasurement{$GI, $LI}($(m.kernel))")
end

"""
    _default_capacity(mc::DQMC)

Returns 100_000 if the configuration measurement is not available or empty, and
the length of the configuration measurement if it is not.

This should be useful for `run` - `setup measurements` - `replay` workflow, 
returning the exact capacity needed.
"""
function _default_capacity(mc::DQMC)
    k = if isdefined(mc, :measurements)
        findfirst(v -> v isa ConfigurationMeasurement, mc.measurements)
    else
        nothing
    end
    if k === nothing
        return 100_000
    else
        N = length(mc.measurements[k].obs)
        return N == 0 ? 100_000 : N
    end
end

# _get_shape(model, ::Nothing) = (length(lattice(model)),)
_get_shape(model, mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(model, mask::DistanceMask) = length(mask)

_get_shape(mc, model, LI::Type) = _get_shape(model, LI(mc, model))
_get_shape(model, ::Nothing) = nothing
_get_shape(model, ::EachSite) = length(lattice(model))
_get_shape(model, ::EachSiteAndFlavor) = nflavors(model) * length(lattice(model))
_get_shape(model, ::EachSitePair) = (length(lattice(model)), length(lattice(model)))
_get_shape(model, iter::EachSitePairByDistance) = ndirections(iter)
_get_shape(model, iter::EachLocalQuadByDistance) = ndirections(iter)
_get_shape(model, iter::EachLocalQuadBySyncedDistance) = ndirections(iter)


# Some type piracy to make things easier
Base.Nothing(::DQMC, ::Model) = nothing

# To identify the requirement of equal-time Greens functions
struct Greens <: AbstractGreensIterator end
Greens(::DQMC, ::Model)= Greens()

struct GreensAt{k, l} <: AbstractUnequalTimeGreensIterator end
GreensAt(l::Integer) = GreensAt{l, l} 
GreensAt(k::Integer, l::Integer) = GreensAt{k, l} 
GreensAt{k, l}(::DQMC, ::Model) where {k, l} = GreensAt{k, l}() 

# maybe we want to some stuff to get custom indexing?
# How would this integrate with greens iterators? maybe:
# greens(mc::DQMC, model::Model) = greens(mc)


requires(::AbstractMeasurement) = (Nothing, Nothing)
requires(::DQMCMeasurement{GI, LI}) where {GI, LI} = (GI, LI)

@bm function generate_groups(mc, model, measurements)
    # maybe instead:
    requirements = requires.(measurements)
    GIs = tuple(unique(first.(requirements))...)
    LIs = tuple(unique(last.(requirements))...)
    lattice_iterators = map(T -> T(mc, model), LIs)

    if any(T -> T <: AbstractUnequalTimeGreensIterator, GIs)
        initialize_stack(mc, mc.ut_stack)
    end

    # greens_iterator => [
    #     (lattice_iterator, measurement), 
    #     (lattice_iterator, measurement), 
    #     ...
    # ]
    map(enumerate(GIs)) do (i, G)
        ms = filter(m -> G == requires(m)[1], measurements)
        group = map(ms) do m
            LI = requires(m)[2]
            j = findfirst(==(LI), LIs)
            @assert j !== nothing
            (lattice_iterators[j], m)
        end
        G(mc, model) => group
    end
end


lattice_iterator(::DQMCMeasurement{GI, LI}, mc, model) where {GI, LI} = LI(mc, model)


# TODO
# Saving the kernel function as a symbol is kinda risky because the function
# definition is not guaranteed to be the same
# though maybe that's a good thing - changing function definitions doesn't
# break stuff this way
function _save(file::JLDFile, m::DQMCMeasurement{GI, LI}, key::String) where {GI, LI}
    write(file, "$key/VERSION", 1)
    write(file, "$key/type", DQMCMeasurement)
    write(file, "$key/GI", GI)
    write(file, "$key/LI", LI)
    # maybe add module for eval?
    write(file, "$key/kernel", Symbol(m.kernel))
    write(file, "$key/obs", m.observable)
    write(file, "$key/output", m.output)
end

# TODO
# I think eval will fail if kernel is not defined in MonteCarlo
function _load(data, ::Type{T}) where {T <: DQMCMeasurement}
    kernel = try
        eval(data["kernel"])
    catch e
        @warn "Failed to load kernel in module MonteCarlo." exception=e
        missing_kernel
    end
    DQMCMeasurement{data["GI"], data["LI"]}(kernel, data["obs"], data["output"])
end

missing_kernel(args...) = error("kernel couldn't be loaded.")



################################################################################
### Greens function related
################################################################################



@bm function apply!(::Nothing, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (lattice_iterator, measurement) in combined
        # Clear output if necessary
        prepare!(lattice_iterator, model, measurement)
        # Write measurement to ouput
        measure!(lattice_iterator, measurement, mc, model, sweep)
        # Finalize output and commit
        finish!(lattice_iterator, model, measurement)
    end

    nothing
end

@bm function apply!(::Greens, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    G = greens!(mc)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
        measure!(lattice_iterator, measurement, mc, model, sweep, G)
        finish!(lattice_iterator, model, measurement)
    end
    nothing
end

@bm function apply!(::GreensAt{k, l}, combined::Vector{<: Tuple}, mc::DQMC, model, sweep) where {k, l}
    G = greens!(mc, k, l)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
        measure!(lattice_iterator, measurement, mc, model, sweep, G)
        finish!(lattice_iterator, model, measurement)
    end
    nothing
end

@bm function apply!(iter::CombinedGreensIterator, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
    end

    G00 = greens!(mc)
    for (G0l, Gl0, Gll) in iter
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, model, sweep, (G00, G0l, Gl0, Gll))
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, model, measurement, mc.p.delta_tau)
    end
    nothing
end



################################################################################
### measure!
################################################################################



@bm function measure!(lattice_iterator, measurement, mc::DQMC, model, sweep, packed_greens)
    # ignore sweep
    apply!(lattice_iterator, measurement, mc, model, packed_greens)
    nothing
end

# Lattice irrelevant
@bm function measure!(::Nothing, measurement, mc::DQMC, model, sweep, packed_greens)
    push!(measurement.observable, measurement.kernel(mc, model, packed_greens))
    nothing
end



################################################################################
### LatticeIterator related
################################################################################



# If LatticeIterator is Nothing, then things should be handled in measure!
prepare!(::Nothing, model, m) = nothing
prepare!(::AbstractLatticeIterator, model, m) = m.output .= zero(eltype(m.output))

finish!(::Nothing, model, m) = nothing # handled in measure!
finish!(::AbstractLatticeIterator, model, m) = push!(m.observable, m.output)
function finish!(::AbstractLatticeIterator, model, m, factor)
    m.output .*= factor
    push!(m.observable, m.output)
end
function finish!(::EachSitePairByDistance, model, m, factor=1.0)
    m.output .*= factor / length(lattice(model))
    push!(m.observable, m.output)
end
function finish!(::EachLocalQuadByDistance, model, m, factor=1.0)
    m.output .*= factor / length(lattice(model))
    push!(m.observable, m.output)
end
function finish!(::EachLocalQuadBySyncedDistance, model, m, factor=1.0)
    m.output .*= factor / length(lattice(model))
    push!(m.observable, m.output)
end



# Call kernel for each site (linear index)
@bm function apply!(iter::EachSiteAndFlavor, measurement, mc::DQMC, model, packed_greens)
    for i in iter
        measurement.output[i] += measurement.kernel(mc, model, i, packed_greens)
    end
    nothing
end

# Call kernel for each site (linear index)
@bm function apply!(iter::EachSite, measurement, mc::DQMC, model, packed_greens)
    for i in iter
        measurement.output[i] += measurement.kernel(mc, model, i, packed_greens)
    end
    nothing
end

# Call kernel for each pair (src, trg) (Nsties² total)
@bm function apply!(iter::EachSitePair, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        measurement.output[i, j] += measurement.kernel(mc, model, i, j, packed_greens)
    end
    nothing
end

# Call kernel for each pair (site, site) (i.e. on-site) 
@bm function apply!(iter::OnSite, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        measurement.output[i] += measurement.kernel(mc, model, i, j, packed_greens)
    end
    nothing
end

# Call kernel for each pair (src, trg) and sum those that point in the same direction
@bm function apply!(iter::EachSitePairByDistance, measurement, mc::DQMC, model, packed_greens)
    @inbounds for (dir, src, trg) in iter
        measurement.output[dir] += measurement.kernel(mc, model, src, trg, packed_greens)
    end
    nothing
end

# Call kernel for each pair (src1, trg1, src2, trg) and sum those that have the 
# same `dir12 = pos[src2] - pos[src1]`, `dir1 = pos[trg1] - pos[src1]` and 
# `dir2 = pos[trg2] - pos[src2]`
@bm function apply!(iter::EachLocalQuadByDistance, measurement, mc::DQMC, model, packed_greens)
    # lin is a linear index for (dir12, dir1, dir2)
    for (lin, src1, trg1, src2, trg2) in iter
        measurement.output[lin] += measurement.kernel(
            mc, model, src1, trg1, src2, trg2, packed_greens
        )
    end
    nothing
end

# Call kernel for each pair (src1, trg1, src2, trg) and sum those that have the 
# same `dir12 = pos[src2] - pos[src1]` and 
# `dir1 = pos[trg1] - pos[src1] = dir2 = pos[trg2] - pos[src2] = dir_ii`
@bm function apply!(iter::EachLocalQuadBySyncedDistance, measurement, mc::DQMC, model, packed_greens)
    # lin is a linear index for (dir12, dir_ii)
    for (lin, src1, trg1, src2, trg2) in iter
        measurement.output[lin] += measurement.kernel(
            mc, model, src1, trg1, src2, trg2, packed_greens
        )
    end
    nothing
end


include("measurements.jl")
include("deprecated.jl")