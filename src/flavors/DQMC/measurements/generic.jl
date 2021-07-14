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
# try replacing global function rather than creatin new local ones


struct DQMCMeasurement{GI, LI, F <: Function, OT, T} <: AbstractMeasurement
    greens_iterator::GI
    lattice_iterator::LI
    kernel::F
    observable::OT
    temp::T
end

missing_kernel(args...) = error("kernel couldn't be loaded.")

function DQMCMeasurement(
        m::DQMCMeasurement;
        greens_iterator = m.greens_iterator, lattice_iterator = m.lattice_iterator,
        kernel = m.kernel,
        observable = m.observable, temp = m.temp,
        capacity = nothing
    )
    if capacity === nothing
        DQMCMeasurement(greens_iterator, lattice_iterator, kernel, observable, temp)
    else
        binner = rebuild(observable, capacity)
        DQMCMeasurement(greens_iterator, lattice_iterator, kernel, binner, temp)
    end
end
rebuild(B::LogBinner, capacity) = LogBinner(B, capacity=capacity)
rebuild(B::T, capacity) where T = T(B, capacity=capacity)

function Measurement(
        dqmc, _model, greens_iterator, lattice_iterator, kernel;
        capacity = _default_capacity(dqmc), eltype = geltype(dqmc),
        temp = let
            shape = _get_temp_shape(dqmc, _model, lattice_iterator)
            shape === nothing ? nothing : Array{eltype}(undef, shape)
        end,
        obs = let
            shape = _get_final_shape(dqmc, _model, lattice_iterator)
            _zero = shape === nothing ? zero(eltype) : zeros(eltype, shape)
            LogBinner(_zero, capacity=capacity)
        end
    )
    DQMCMeasurement(greens_iterator, lattice_iterator, kernel, obs, temp)
end



################################################################################
### DQMCMeasurement utilities
################################################################################



function Base.show(io::IO, ::MIME"text/plain", m::DQMCMeasurement)
    max = applicable(capacity, m.observable) ? capacity(m.observable) : Inf
    current = length(m.observable)
    GI = m.greens_iterator
    LI = if m.lattice_iterator isa Type
        m.lattice_iterator
    else
        typeof(m.lattice_iterator)
    end
    print(io, "[$current/$max] DQMCMeasurement($GI, $LI)")
end


MonteCarloObservable.mean(m::DQMCMeasurement) = mean(m.observable)
MonteCarloObservable.var(m::DQMCMeasurement) = var(m.observable)
MonteCarloObservable.std_error(m::DQMCMeasurement) = std_error(m.observable)
MonteCarloObservable.tau(m::DQMCMeasurement) = tau(m.observable)
Base.length(m::DQMCMeasurement) = length(m.observable)
Base.isempty(m::DQMCMeasurement) = isempty(m.observable)
Base.empty!(m::DQMCMeasurement) = empty!(m.observable)


"""
    _default_capacity(mc::DQMC)

Returns a default capacity based in the number of sweeps and the measure rate.
"""
_default_capacity(mc::DQMC) = 2 * ceil(Int, mc.parameters.sweeps / mc.parameters.measure_rate)


# _get_temp_shape is the shape of the temporary array
# nothing is interpreted as "It's not needed"
_get_temp_shape(mc, model, li) = _get_shape(mc, model, li)
_get_temp_shape(mc, model, ::Type{<: Sum}) = 1
function _get_temp_shape(mc, model, ::LatticeIterationWrapper{LI}) where {LI}
    _get_shape(mc, model, LI)
end

# final_shape refers to the shape of what the observable saves
# here `nothing` means saving the eltype instead of an array 
_get_final_shape(mc, model, li) = _get_shape(mc, model, li)
_get_final_shape(mc, model, ::Type{<: Sum}) = nothing
_get_final_shape(mc, model, ::SuperfluidDensity) = nothing
function _get_final_shape(mc, model, s::ApplySymmetries{LI, N}) where {LI, N}
    if LI <: EachLocalQuadByDistance || LI <: EachLocalQuadBySyncedDistance
        shape = _get_shape(mc, model, LI)
        (first(shape), N)
    else
        throw(MethodError(_get_final_shape, (mc, model, s)))
    end
end

_get_shape(model, mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(model, mask::DistanceMask) = length(mask)

_get_shape(mc, model, LI::Type) = _get_shape(LI(mc, model))
_get_shape(mc, model, ::Type{Nothing}) = nothing
_get_shape(mc, model, ::Type{EachSite}) = length(lattice(model))
_get_shape(mc, model, ::Type{EachSiteAndFlavor}) = nflavors(model) * length(lattice(model))
_get_shape(mc, model, ::Type{EachSitePair}) = (length(lattice(model)), length(lattice(model)))
_get_shape(iter::DeferredLatticeIterator) = ndirections(iter)


# Some type piracy to make things easier
Base.Nothing(::DQMC, ::Model) = nothing

# To identify the requirement of equal-time Greens functions
struct Greens <: AbstractGreensIterator end
Greens(::DQMC, ::Model)= Greens()

struct GreensAt <: AbstractUnequalTimeGreensIterator
    k::Int
    l::Int
end
GreensAt(l::Integer) = GreensAt(l, l)
GreensAt(k::Integer, l::Integer) = GreensAt(k, l)
# GreensAt{k, l}(::DQMC, ::Model) where {k, l} = GreensAt(k, l)
Base.:(==)(a::GreensAt, b::GreensAt) = (a.k == b.k) && (a.l == b.l)

# maybe we want to some stuff to get custom indexing?
# How would this integrate with greens iterators? maybe:
# greens(mc::DQMC, model::Model) = greens(mc)


requires(::AbstractMeasurement) = (Nothing, Nothing)
requires(m::DQMCMeasurement) = (m.greens_iterator, m.lattice_iterator)
# TODO ApplySymmetries{LI}


@bm function generate_groups(mc, model, measurements)
    # maybe instead:
    requirements = requires.(measurements)
    GIs = unique(first.(requirements))
    LIs = unique(last.(requirements))
    lattice_iterators = map(T -> T(mc, model), LIs)

    if any(x -> (x isa Type ? x : typeof(x)) <: AbstractUnequalTimeGreensIterator, GIs)
        initialize_stack(mc, mc.ut_stack)
    end

    # greens_iterator => [
    #     (lattice_iterator, measurement), 
    #     (lattice_iterator, measurement), 
    #     ...
    # ]
    output = map(enumerate(GIs)) do (i, G)
        ms = filter(m -> G == requires(m)[1], measurements)
        group = map(ms) do m
            LI = requires(m)[2]
            j = findfirst(==(LI), LIs)
            @assert j !== nothing
            (lattice_iterators[j], m)
        end
        (G isa Type ? G(mc, model) : G) => group
    end

    if length(measurements) != mapreduce(x -> length(x[2]), +, output, init = 0)
        for (G, group) in output
            println(G)
            for (li, m) in group
                println("\t", typeof(li), typeof(m.kernel))
            end
        end
        N = length(measurements)
        M = mapreduce(x -> length(x[2]), +, output, init = 0)
        error("Oh no. We lost some measurements. $N-> $M")
    end

    return output
end


lattice_iterator(m::DQMCMeasurement, mc, model) = m.lattice_iterator(mc, model)


# TODO
# Saving the kernel function as a symbol is kinda risky because the function
# definition is not guaranteed to be the same
# though maybe that's a good thing - changing function definitions doesn't
# break stuff this way
function _save(file::JLDFile, m::DQMCMeasurement, key::String)
    write(file, "$key/VERSION", 1)
    write(file, "$key/type", DQMCMeasurement)
    write(file, "$key/GI", m.greens_iterator)
    write(file, "$key/LI", m.lattice_iterator)
    # maybe add module for eval?
    write(file, "$key/kernel", Symbol(m.kernel))
    write(file, "$key/obs", m.observable)
    write(file, "$key/temp", m.temp)
end

function _load(data, ::Type{T}) where {T <: DQMCMeasurement}
    temp = haskey(data, "temp") ? data["temp"] : data["output"]
    
    kernel = try
        eval(data["kernel"])
    catch e
        @warn "Failed to load kernel in module MonteCarlo." exception=e
        missing_kernel
    end
    DQMCMeasurement(data["GI"], data["LI"], kernel, data["obs"], temp)
end


################################################################################
### Greens function related
################################################################################



@bm function apply!(::Nothing, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (lattice_iterator, measurement) in combined
        # Clear temp if necessary
        prepare!(lattice_iterator, model, measurement)
        # Write measurement to ouput
        measure!(lattice_iterator, measurement, mc, model, sweep)
        # Finalize computation (temp) and commit
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

@bm function apply!(g::GreensAt, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    G = greens!(mc, g.k, g.l)
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
    for (G0l, Gl0, Gll) in init(mc, iter)
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, model, sweep, (G00, G0l, Gl0, Gll))
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, model, measurement, mc.parameters.delta_tau)
    end
    nothing
end

@bm function apply!(iter::AbstractGreensIterator, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, model, measurement)
    end

    for packed_greens in init(mc, iter)
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, model, sweep, packed_greens)
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, model, measurement, 1.0)
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
### apply Lattice Iterators 
################################################################################



# Call kernel for each site (linear index)
@bm function apply!(iter::DirectLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    for i in iter
        measurement.temp[i] += measurement.kernel(mc, model, i, packed_greens)
    end
    nothing
end

# Call kernel for each pair (src, trg) (Nsties² total)
@bm function apply!(iter::EachSitePair, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        measurement.temp[i, j] += measurement.kernel(mc, model, (i, j), packed_greens)
    end
    nothing
end

# Call kernel for each pair (site, site) (i.e. on-site) 
@bm function apply!(iter::OnSite, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        measurement.temp[i] += measurement.kernel(mc, model, (i, j), packed_greens)
    end
    nothing
end

@bm function apply!(iter::DeferredLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        measurement.temp[first(idxs)] += measurement.kernel(mc, model, idxs[2:end], packed_greens)
    end
    nothing
end


# Sums
@bm function apply!(iter::Sum{<: DirectLatticeIterator}, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        measurement.temp[1] += measurement.kernel(mc, model, idxs, packed_greens)
    end
    nothing
end
@bm function apply!(iter::Sum{<: DeferredLatticeIterator}, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        measurement.temp[1] += measurement.kernel(mc, model, idxs[2:end], packed_greens)
    end
    nothing
end

@inline apply!(s::LatticeIterationWrapper, m, mc, model, pg) = apply!(s.iter, m, mc, model, pg)



################################################################################
### LatticeIterator preparation and finalization
################################################################################



# If LatticeIterator is Nothing, then things should be handled in measure!
@inline prepare!(::Nothing, model, m) = nothing
@inline prepare!(::AbstractLatticeIterator, model, m) = m.temp .= zero(eltype(m.temp))
@inline prepare!(s::Sum, args...) = prepare!(s.iter, args...)

@inline finish!(::Nothing, args...) = nothing # handled in measure!
@inline function finish!(li, model, m, factor=1.0)
    finalize_temp!(li, model, m, factor)
    commit!(li, m)
end

@inline function finalize_temp!(::AbstractLatticeIterator, model, m, factor)
    m.temp .*= factor
end
@inline function finalize_temp!(::DeferredLatticeIterator, model, m, factor)
    m.temp .*= factor / length(lattice(model))
end
@inline function finalize_temp!(s::LatticeIterationWrapper, model, m , factor)
    finalize_temp!(s.iter, model, m, factor)
end

@inline commit!(::AbstractLatticeIterator, m) = push!(m.observable, m.temp)
@inline commit!(::Sum, m) = push!(m.observable, m.temp[1])


function commit!(s::_ApplySymmetries{<: EachLocalQuadByDistance}, m)
    final = zeros(eltype(m.temp), size(m.temp, 1), length(s.symmetries))
    # This calculates
    # ∑_{a a'} O(Δr, a, a') f_ζ(a) f_ζ(a')
    # where a, a' are typically nearest neighbor directions and
    # f_ζ(a) is the weight in direction a for a symmetry ζ
    for (i, sym) in enumerate(s.symmetries)
        for k in 1:length(sym), l in 1:length(sym)
            @. final[:, i] += m.temp[:, k, l] * sym[k] * sym[l]
        end
    end
    push!(m.observable, final)
end
function commit!(s::_ApplySymmetries{<: EachLocalQuadBySyncedDistance}, m)
    final = zeros(eltype(m.temp), size(m.temp, 1), length(s.symmetries))
    # Same as above but with a = a'
    for (i, sym) in enumerate(s.symmetries)
        for k in 1:length(sym)
            @. final[:, i] += m.temp[:, k] * sym[k] * sym[k]
        end
    end
    push!(m.observable, final)
end


function commit!(s::_SuperfluidDensity{<: EachLocalQuadBySyncedDistance}, m)
    final = zero(eltype(m.observable))
    for (shift, long, trans) in zip(s.dir_idxs, s.long_qs, s.trans_qs)
        for (i, dir) in enumerate(s.dirs)
            final += m.temp[i, shift] * (cis(-dot(dir, long)) - cis(-dot(dir, trans)))
        end
    end
    push!(m.observable, final)
end