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
        temp = _measurement_buffer(dqmc, _model, lattice_iterator, eltype),
        obs = LogBinner(
            _binner_zero_element(dqmc, _model, lattice_iterator, eltype), 
            capacity=capacity
        ),
        # let
        #     shape = _get_final_shape(dqmc, _model, lattice_iterator)
        #     _zero = shape === nothing ? zero(eltype) : zeros(eltype, shape)
        #     LogBinner(_zero, capacity=capacity)
        # end
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



################################################################################
### Buffers
################################################################################



# function _measurement_buffer(mc::DQMC, m::Model, li, eltype)
#     shape = _get_temp_shape(dqmc, _model, lattice_iterator)
#     shape === nothing ? nothing : Array{eltype}(undef, shape)
# end

# General (Float64)
_measurement_buffer(mc, model, li, eltype) = _simple_buffer(mc, li, eltype)
_measurement_buffer(mc, model, ::Sum, eltype) = zeros(eltype, 1)
function _measurement_buffer(mc, model, ::LatticeIterationWrapper{LI}, eltype) where {LI}
    _measurement_buffer(mc, model, LI, eltype)
end

# StructArrays / ComplexF64
# This unwraps wrapper to see if any of the wrapped types matches what we're 
# looking for. Currently looks for T1 in T2
function is_wrapped_type_of(T1, T2)
    if T1 == T2
        return true
    else
        for t2 in T2.types
            is_subtype_of(T1, t2) && return true
        end
    end
    false
end

function maybe_structarray(mc, A)
    is_wrapped_type_of(CMat64, gmattype(mc)) ? StructArray(A) : A
end

_measurement_buffer(mc, model, li, ::ComplexF64) = maybe_structarray(mc, _simple_buffer(mc, li, eltype))
_measurement_buffer(mc, model, ::Sum, ::ComplexF64) = maybe_structarray(mc, zeros(ComplexF64, 1))
function _measurement_buffer(mc, model, ::LatticeIterationWrapper{LI}, ::ComplexF64) where {LI}
    maybe_structarray(mc, _measurement_buffer(mc, model, LI, eltype))
end


# can be determined from type
_simple_buffer(mc, t::Type, T) = _simple_buffer(mc, t(), T)
_simple_buffer(mc, ::Type{Nothing}, T) = nothing
_simple_buffer(mc, ::Nothing, T) = nothing
_simple_buffer(mc, ::EachSite, T) = zeros(T, length(lattice(mc)))
_simple_buffer(mc, ::EachSiteAndFlavor, T) = zeros(T, nflavors(model(mc)) * length(lattice(mc)))
_simple_buffer(mc, ::EachSitePair, T) = zeros(T, length(lattice(mc)), length(lattice(mc)))


# determined from type data
_simple_buffer(mc, li::DeferredLatticeIteratorTemplate, T) = zeros(T, ndirections(mc, li)) # TODO

# zero element for binner
_binner_zero_element(mc, model, li, eltype) = _simple_buffer(mc, li, eltype)
_binner_zero_element(mc, model, ::Type{Nothing}, eltype) = zero(eltype)
_binner_zero_element(mc, model, ::Nothing, eltype) = zero(eltype)
_binner_zero_element(mc, model, ::Sum, eltype) = zero(eltype)
# _binner_zero_element(mc, model, ::SuperfluidDensity, eltype) = zero(eltype)
function _binner_zero_element(mc, model, li::ApplySymmetries{LI, N}, eltype) where {LI, N}
    if LI <: EachLocalQuadByDistance || LI <: EachLocalQuadBySyncedDistance
        return zeros(eltype, first(ndirection(LI(mc, model))), N) # TODO
    else
        throw(MethodError(_binner_zero_element, (mc, model, li, eltype)))
    end
end




# Once we change greens iterators to follow the "template -> iter" structure
# this will be compat only
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


@bm function generate_groups(mc, model, measurements)
    empty!(mc.lattice_iterator_cache)

    # get unique requirements
    requirements = requires.(measurements)
    GIs = unique(first.(requirements))

    # init requirements
    # lattice_iterators = map(T -> T(mc, model), LIs)
    if any(x -> (x isa Type ? x : typeof(x)) <: AbstractUnequalTimeGreensIterator, GIs)
        initialize_stack(mc, mc.ut_stack)
    end

    # Group measurements with the same greens iterator together
    # lattice_iterators
    # greens_iterator => [
    #     (lattice_iterator, measurement), 
    #     (lattice_iterator, measurement), 
    #     ...
    # ]
    output = map(enumerate(GIs)) do (i, G)
        ms = filter(m -> G == requires(m)[1], measurements)
        group = map(ms) do m
            LI = requires(m)[2]
            (LI === nothing ? nothing : LI(mc, model), m)
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
        error("Oh no. We lost some measurements. $N -> $M")
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
    write(file, "$key/tag", "DQMCMeasurement")
    write(file, "$key/GI", m.greens_iterator)
    write(file, "$key/LI", m.lattice_iterator)
    # maybe add module for eval?
    write(file, "$key/kernel", Symbol(m.kernel))
    write(file, "$key/obs", m.observable)
    write(file, "$key/temp", m.temp)
end

function _load(data, ::Val{:DQMCMeasurement})
    temp = haskey(data, "temp") ? data["temp"] : data["output"]
    
    kernel = try
        eval(data["kernel"])
    catch e
        @warn "Failed to load kernel in module MonteCarlo." exception=e
        missing_kernel
    end
    DQMCMeasurement(data["GI"], data["LI"], kernel, data["obs"], temp)
end

to_tag(::Type{<: DQMCMeasurement}) = Val(:DQMCMeasurement)



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
    apply!(measurement.temp, lattice_iterator, measurement, mc, model, packed_greens)
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


# @bm function apply!(
#         temp::StructArray{ComplexF64}, iter::AbstractLatticeIterator, measurement, mc::DQMC, model, 
#         G::GreensMatrix{ComplexF64, CMat64}
#     )
#     # TODO temp needs to be CVec64
#     # TODO apply!() needs to be passed a temp array
#     # TODO BlockDiagonal too ree
#     # TODO factor for im^2?

#     # wait wtf this doesn't even work :<
#     # cause we have some form of G[i, j] * G[k, l] in the kernel which needs to be
#     # split but how are we gonna do this
#     apply!(temp.re, iter, measurement, mc, model, GreensMatrix(G.k, G.l, G.val.re))
#     apply!(temp.re, iter, measurement, mc, model, GreensMatrix(G.k, G.l, G.val.im))
#     apply!(temp.re, iter, measurement, mc, model, GreensMatrix(G.k, G.l, G.val.re))
#     apply!(temp.re, iter, measurement, mc, model, GreensMatrix(G.k, G.l, G.val.re))
#     nothing
# end


# Call kernel for each site (linear index)
@bm function apply!(temp::Array, iter::DirectLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    for i in iter
        temp[i] += measurement.kernel(mc, model, i, packed_greens)
    end
    nothing
end

# Call kernel for each pair (src, trg) (Nsties² total)
@bm function apply!(temp::Array, iter::EachSitePair, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        temp[i, j] += measurement.kernel(mc, model, (i, j), packed_greens)
    end
    nothing
end

# Call kernel for each pair (site, site) (i.e. on-site) 
@bm function apply!(temp::Array, iter::_OnSite, measurement, mc::DQMC, model, packed_greens)
    for (i, j) in iter
        temp[i] += measurement.kernel(mc, model, (i, j), packed_greens)
    end
    nothing
end

@bm function apply!(temp::Array, iter::DeferredLatticeIterator, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        temp[first(idxs)] += measurement.kernel(mc, model, idxs[2:end], packed_greens)
    end
    nothing
end


# Sums
@bm function apply!(temp::Array, iter::_Sum{<: DirectLatticeIterator}, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        temp[1] += measurement.kernel(mc, model, idxs, packed_greens)
    end
    nothing
end
@bm function apply!(temp::Array, iter::_Sum{<: DeferredLatticeIterator}, measurement, mc::DQMC, model, packed_greens)
    @inbounds for idxs in iter
        temp[1] += measurement.kernel(mc, model, idxs[2:end], packed_greens)
    end
    nothing
end

@inline function apply!(temp::Array, s::LatticeIterationWrapper, m, mc, model, pg)
    apply!(temp, s.iter, m, mc, model, pg)
end



################################################################################
### LatticeIterator preparation and finalization
################################################################################



# If LatticeIterator is Nothing, then things should be handled in measure!
@inline prepare!(::Nothing, model, m) = nothing
@inline prepare!(::AbstractLatticeIterator, model, m) = m.temp .= zero(eltype(m.temp))
@inline prepare!(s::_Sum, args...) = prepare!(s.iter, args...)

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
@inline commit!(::_Sum, m) = push!(m.observable, m.temp[1])


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