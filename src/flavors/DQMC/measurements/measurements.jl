# Prototyping


# Maybe reorganize in general:
# - sort measurement into Integrator types (in run!/replay!)
# - call integrate(::Integrator, measurement_collection)
# - during measuring:
#   - prepare! to clear stuff
#   - measure! to apply mask
#       - default measure! calls kernel! (which needs to be implemented)
#   - finish!/commit! to save the measurement/commit it to an observable/Binner
# - maybe add read-only wrapper for Greens (including indexing?)
#   - indexing kinda sucks for performance though
#     ... so probably do dispatch on Model instead in prepare!/measure!/finish!


abstract type Requirements end
struct NoRequirements <: Requirements end       # uses conf
struct EqualTime <: Requirements end            # uses greens
struct StandardIntegrator <: Requirements end   # uses CombinedGreensIterator
# default should probably pass G
requires(::AbstractMeasurement) = EqualTime()
# generate something like
# (requirement => keys, ...) or (requirement => measurements, ...)


# Move to lattices/masks.jl
struct EmptyMask end
mask(::AbstractMeasurement) = EmptyMask()


function integrate(::StandardIntegrator, dqmc::DQMC, model, measurements)
    # Clear out temps
    for m in measurements
        prepare!(m)
    end

    # Compute
    G00 = greens!(dqmc)
    for (k, (Gk0, Gkk)) in enumerate(CombinedGreensIterator(dqmc))
        for measurement in measurements
            measure!(mask(measurement), measurement, model, k-1, G00, Gk0, Gkk)
        end
    end

    # Commit to final storage
    for m in measurements
        finish!(m) # or commit!()
    end
end

function measure!(mask::MaybeType, measurement::TYPED, model::MaybeType, args...)
    # If there are multiple temp outputs they should/need to be handled here
    # because kernel! should be independant of masks 
    # (i.e. kernel! doesn't know how to index output)
    for (dir, src, trg) in getorder(mask)
        output[dir] += kernel!(measurement, model, src, trg, args...)
    end
end

function kernel!(measurement::TYPED, model::MaybeType, src, trg, k, G00, Gk0, Gkk)
    # dispatch on measurement and probably model <- handle flv index with this
    # COMPUTE element from
    # lattice indices (src, trg) (or more)
    # maybe time slice index
    # maybe G00 aka the equal time greens function
    # maybe Gk0, Gkk 
    
    # Do some shit like this to keep track of flv for HubbardModel
    N = 0
    delta(src, trg) - G00[src, trg] ...
    # or, you know... but delta might be faster... 
    # though it probably doesn't matter?
    I[src, trg] - G00[src, trg]
end

delta(i, j) = i == j


function integrate(::EqualTime, ...)
    # Clear out temps
    for m in measurements
        prepare!(m)
    end

    # Compute
    G00 = greens!(dqmc)
    for measurement in measurements
        measure!(mask(measurement), measurement, model, k-1, G00, Gk0, Gkk)
    end

    # Commit to final storage
    for m in measurements
        finish!(m) # or commit!()
    end  
end


# Read-only arrays
using ReadOnlyArrays
# Add an eror for this to drive home that this is forbidden
function Base.setindex!(A::ReadOnlyArray, idxs...)
    error("Indexing this array is explicitly forbidden as it may invalidate other measurements")
end



# Sidenote: No performance penalty between ReadOnlyArrays and Arrays here
function foo(A)
    sum = 0
    @inbounds for i in axes(A, 1), j in axes(A, 2)
        sum += A[i, j] * A[j, j] + A[i, i] * A[i, j] 
    end
    sum
end




################################################################################



# Source from ReadOnlyArrays
#=
struct ReadOnlyArray{T,N,P} <: AbstractArray{T,N}
    parent::P
    ReadOnlyArray(parent::AbstractArray{T,N}) where{T,N} =
        new{T, N, typeof(parent)}(parent)
end

Base.IteratorSize(::Type{<:ReadOnlyArray{T,N,P}}) where {T,N,P} =
    Base.IteratorSize(P)
Base.IteratorEltype(::Type{<:ReadOnlyArray{T,N,P}}) where {T,N,P} =
    Base.IteratorEltype(P)
Base.eltype(::Type{<:ReadOnlyArray{T,N,P}}) where {T,N,P} =
    eltype(P)
Base.size(roa::ReadOnlyArray, args...) = size(roa.parent, args...)
Base.@propagate_inbounds Base.getindex(roa::ReadOnlyArray, I...) =
    getindex(roa.parent, I...)
Base.firstindex(roa::ReadOnlyArray) = firstindex(roa.parent)
Base.lastindex(roa::ReadOnlyArray) = lastindex(roa.parent)
Base.IndexStyle(::Type{<:ReadOnlyArray{T,N,P}}) where {T,N,P} = IndexStyle(P)
Base.iterate(roa::ReadOnlyArray, args...) = iterate(roa.parent, args...)
Base.length(roa::ReadOnlyArray) = length(roa.parent)

Base.axes(roa::ReadOnlyArray) = axes(roa.parent)
Base.strides(roa::ReadOnlyArray) = strides(roa.parent)
Base.unsafe_convert(p::Type{Ptr{T}}, roa::ReadOnlyArray) where {T} =
    Base.unsafe_convert(p, roa.parent)
Base.stride(roa::ReadOnlyArray, i::Int) = stride(roa.parent, i)
Base.parent(roa::ReadOnlyArray) = roa.parent
=#