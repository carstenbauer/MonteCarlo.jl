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

include("greens_iterators.jl")


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
        temp = _measurement_buffer(dqmc, lattice_iterator, eltype),
        obs = LogBinner(
            _binner_zero_element(dqmc, lattice_iterator, eltype), 
            capacity=capacity
        )
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


BinningAnalysis.mean(m::DQMCMeasurement) = mean(m.observable)
BinningAnalysis.var(m::DQMCMeasurement) = var(m.observable)
BinningAnalysis.std_error(m::DQMCMeasurement) = std_error(m.observable)
BinningAnalysis.tau(m::DQMCMeasurement) = tau(m.observable)
Base.length(m::DQMCMeasurement) = length(m.observable)
Base.isempty(m::DQMCMeasurement) = isempty(m.observable)
Base.empty!(m::DQMCMeasurement) = empty!(m.observable)


"""
    _default_capacity(mc::DQMC)

Returns a default capacity based in the number of sweeps and the measure rate.
"""
_default_capacity(mc::DQMC) = 2 * ceil(Int, mc.parameters.sweeps / mc.parameters.measure_rate)

# TODO
# Saving the kernel function as a symbol is kinda risky because the function
# definition is not guaranteed to be the same
# though maybe that's a good thing - changing function definitions doesn't
# break stuff this way
function _save(file::FileLike, key::String, m::DQMCMeasurement)
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", "DQMCMeasurement")
    _save(file, "$key/GI", m.greens_iterator)
    _save(file, "$key/LI", m.lattice_iterator)
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
    gi = _load(data["GI"], Val(:GreensIterator))
    li = _load(data["LI"], Val(:LatticeIterator))
    DQMCMeasurement(gi, li, kernel, data["obs"], temp)
end

to_tag(::Type{<: DQMCMeasurement}) = Val(:DQMCMeasurement)


################################################################################
### Buffers
################################################################################


# I think this all now
_measurement_buffer(mc, li, eltype) = zeros(eltype, output_size(li, lattice(mc)))
_measurement_buffer(mc, ::Nothing, eltype) = nothing

function _binner_zero_element(mc, li, eltype)
    shape = output_size(li, lattice(mc))
    return shape == (1,) ? zero(eltype) : zeros(eltype, shape)
end
_binner_zero_element(mc, ::Nothing, eltype) = zero(eltype)


################################################################################
### Initialization
################################################################################


requires(::AbstractMeasurement) = (Nothing, Nothing)
requires(m::DQMCMeasurement) = (m.greens_iterator, m.lattice_iterator)


@bm function generate_groups(mc, model, measurements)
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
        group = map(m -> (requires(m)[2], m), ms)
        (G isa Type ? G(mc, model) : G) => group
    end

    if length(measurements) != mapreduce(x -> length(x[2]), +, output, init = 0)
        for (G, group) in output
            println(G)
            for (li, m) in group
                println("\t", typeof(li), " ", typeof(m.kernel))
            end
        end
        N = length(measurements)
        M = mapreduce(x -> length(x[2]), +, output, init = 0)
        error("Oh no! We lost some measurements: $N -> $M")
    end

    return output
end


function lattice_iterator(m::DQMCMeasurement, mc)
    return with_lattice(m.lattice_iterator, lattice(mc))
end



################################################################################
### Greens function related
################################################################################



@bm function apply!(::Nothing, combined::Vector{<: Tuple}, mc::DQMC)
    for (lattice_iterator, measurement) in combined
        # Clear temp if necessary
        prepare!(lattice_iterator, measurement, mc)
        # Write measurement to ouput
        measure!(lattice_iterator, measurement, mc)
        # Finalize computation (temp) and commit
        finish!(lattice_iterator, measurement, mc)
    end

    nothing
end

@bm function apply!(::Greens, combined::Vector{<: Tuple}, mc::DQMC)
    G = greens!(mc)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
        measure!(lattice_iterator, measurement, mc, G)
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(g::GreensAt, combined::Vector{<: Tuple}, mc::DQMC)
    G = greens!(mc, g.k, g.l)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
        measure!(lattice_iterator, measurement, mc, G)
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(iter::TimeIntegral, combined::Vector{<: Tuple}, mc::DQMC)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
    end

    G00 = greens!(mc)
    M = nslices(mc)
    for (i, (G0l, Gl0, Gll)) in enumerate(init(mc, iter))
        weight = ifelse(i in (1, M+1), 0.5, 1.0) * mc.parameters.delta_tau
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, (G00, G0l, Gl0, Gll), weight)
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(iter::AbstractGreensIterator, combined::Vector{<: Tuple}, mc::DQMC)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
    end

    G00 = greens!(mc)
    for (G0l, Gl0, Gll) in init(mc, iter)
        for (lattice_iterator, measurement) in combined
            measure!(lattice_iterator, measurement, mc, (G00, G0l, Gl0, Gll))
        end
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end



################################################################################
### measure!
################################################################################



@bm function measure!(lattice_iterator, measurement, mc::DQMC, packed_greens, weight = 1.0)
    # ignore sweep
    apply!(measurement.temp, lattice_iterator, measurement, mc, packed_greens, weight)
    nothing
end

# Lattice irrelevant
@bm function measure!(::Nothing, measurement, mc::DQMC, packed_greens)
    flv = Val(nflavors(mc))
    push!(measurement.observable, measurement.kernel(mc, mc.model, packed_greens, flv))
    nothing
end



################################################################################
### apply Lattice Iterators 
################################################################################


@bm function apply!(
        temp::Array, iter::DirectLatticeIterator, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    flv = Val(nflavors(mc))
    for idx in with_lattice(iter, lattice(mc))
        val = getindex(temp, CartesianIndex(idx))
        val += weight * measurement.kernel(mc, mc.model, idx, packed_greens, flv)
        setindex!(temp, val, CartesianIndex(idx))
    end
    nothing
end


@bm function apply!(
        temp::Array, iter::DeferredLatticeIterator, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    flv = Val(nflavors(mc))
    @inbounds for idxs in with_lattice(iter, lattice(mc))
        temp[first(idxs)] += weight * measurement.kernel(mc, mc.model, idxs[2:end], packed_greens, flv)
    end
    nothing
end


################################################################################
### LatticeIterator preparation and finalization
################################################################################



# If LatticeIterator is Nothing, then things should be handled in measure!
@inline prepare!(::Nothing, m, mc) = nothing
@inline prepare!(::AbstractLatticeIterator, m, mc) = m.temp .= zero(eltype(m.temp))


@inline finish!(::Nothing, args...) = nothing # handled in measure!
@inline function finish!(li, m, mc)
    finalize_temp!(li, m, mc)
    commit!(li, m)
end

@inline function finalize_temp!(::AbstractLatticeIterator, m, mc)
    nothing
end
@inline function finalize_temp!(::DeferredLatticeIterator, m, mc)
    m.temp ./= length(lattice(mc))
end

@inline commit!(::AbstractLatticeIterator, m) = push!(m.observable, m.temp)
@inline commit!(::Sum, m) = push!(m.observable, m.temp[1])


# function commit!(s::ApplySymmetries{<: EachLocalQuadByDistance}, m)
#     final = zeros(eltype(m.temp), size(m.temp, 1), length(s.symmetries))
#     # This calculates
#     # ∑_{a a'} O(Δr, a, a') f_ζ(a) f_ζ(a')
#     # where a, a' are typically nearest neighbor directions and
#     # f_ζ(a) is the weight in direction a for a symmetry ζ
#     for (i, sym) in enumerate(s.symmetries)
#         for k in 1:length(sym), l in 1:length(sym)
#             @. final[:, i] += m.temp[:, k, l] * sym[k] * sym[l]
#         end
#     end
#     push!(m.observable, final)
# end
# function commit!(s::ApplySymmetries{<: EachLocalQuadBySyncedDistance}, m)
#     final = zeros(eltype(m.temp), size(m.temp, 1), length(s.symmetries))
#     # Same as above but with a = a'
#     for (i, sym) in enumerate(s.symmetries)
#         for k in 1:length(sym)
#             @. final[:, i] += m.temp[:, k] * sym[k] * sym[k]
#         end
#     end
#     push!(m.observable, final)
# end