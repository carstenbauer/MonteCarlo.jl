#=
# Process for DQMCMeasurements:

1. Measurements are created with a
- greens_iterator, specifying which greens functions are needed
- lattice_iterator, specifying which lattice indices are needed (and how the summation runs)
- kernel, which is Function calculating the Wicks expanded Observable for a 
  given set of Matrices and indices
- temp Array for intermediate storage
- binner for long term storage and error analysis

2. generate_groups groups ḿeasurements by greens iterator and initializes the 
   unequal time stack if necessary (from run!(mc))

3. At runtime groups are processed in the following order (callstack):
  - `process_greens_iterator!(greens_iterator, group, mc)`
    - calls `prepare!(lattice_iter, measurement, mc)` clears `temp` for each measurement
    - generate Greens matrices `Gs`
    - calls `measure!(lattice_iter, measurement, mc, Gs)` for each measurement
      - could apply sweep based stuff (skipping, simulation time dependent stuff)
      - calls `apply!(temp, lattice_iter, measurement, mc, greens, weight)`
        - runs through lattice iterator (does summation)
        - calls `kernel(mc, model, lattice_idxs, greens, flv)`
          - computes Wicks expanded Observable for given indices
    - calls `finish!(lattice_iterator, measurement, mc)`
      - calls `finalize_temp!(lattice_iter, measurement, mc)`
        - applies lattice normalization
      - calls `commit!(lattice_iter, measurement)`
        - stores the result
=#


struct Restructure{T}
    wrapped::T
end


struct DQMCMeasurement{F <: Function, GI, LI, FI, OT, T} <: AbstractMeasurement
    greens_iterator::GI
    lattice_iterator::LI
    flavor_iterator::FI
    kernel::F
    observable::OT
    temp::T
end

missing_kernel(args...) = error("kernel couldn't be loaded.")

function DQMCMeasurement(
        m::DQMCMeasurement;
        greens_iterator = m.greens_iterator, 
        lattice_iterator = m.lattice_iterator,
        flavor_iterator = m.flavor_iterator,
        kernel = m.kernel,
        observable = m.observable, temp = m.temp,
        capacity = nothing
    )
    if capacity === nothing
        DQMCMeasurement(greens_iterator, lattice_iterator, flavor_iterator, kernel, observable, temp)
    else
        binner = rebuild(observable, capacity)
        DQMCMeasurement(greens_iterator, lattice_iterator, flavor_iterator, kernel, binner, temp)
    end
end
rebuild(B::LogBinner, capacity) = LogBinner(B, capacity=capacity)
rebuild(B::T, capacity) where T = T(B, capacity=capacity)

"""
    DQMCMeasurement(
        mc, model, greens_iterator, lattice_iterator, flavor_iterator, kernel; 
        kwargs...
    )

Constructs a `DQMCMeasurement` from the given arguments.

## Optional Keyword Arguments

- `temp` sets up the temporary storage for the measurement. Defaults to 
`_measurement_buffer(mc, lattice_iterator, eltype)`.
- `obs` sets the observable for the measurement. By default this is a 
`LogBinner` from BinningAnalysis.jl. The zero element follows from 
`_binner_zero_element(mc, lattice_iterator, eltype)`.
- `capacity` sets the capacity of the default observable. Defaults to 
`_default_capacity(mc)` which is at least double the number of measurements the 
simulation is set up to take.
- `eltype` sets the element type for the temporary storage and observable. By 
default this follows from element type of the Greens function `geltype(mc)`.
"""
function DQMCMeasurement(
        dqmc, _model, greens_iterator, lattice_iterator, 
        flavor_iterator, kernel;
        capacity = _default_capacity(dqmc), eltype = geltype(dqmc),
        temp = _measurement_buffer(dqmc, lattice_iterator, eltype),
        obs = LogBinner(
            _binner_zero_element(dqmc, lattice_iterator, eltype), 
            capacity=capacity
        )
    )
    DQMCMeasurement(greens_iterator, lattice_iterator, flavor_iterator, kernel, obs, temp)
end

@deprecate Measurement DQMCMeasurement false


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
    write(file, "$key/VERSION", 2)
    write(file, "$key/tag", "DQMCMeasurement")
    _save(file, "$key/GI", m.greens_iterator)
    _save(file, "$key/LI", m.lattice_iterator)
    write(file, "$key/FI", m.flavor_iterator)
    # maybe add module for eval?
    write(file, "$key/kernel", Symbol(m.kernel))
    _save(file, "$key/obs", m.observable)
    write(file, "$key/temp", m.temp)
end

struct ErrorIterator end
function Base.iterate(::ErrorIterator, state = nothing)
    error(
        "Attempting to iterate an `ErrorIterator()`. This most likely " *
        "happened due to loading a simulation with measurements lacking " *
        " flavor iterators."
    )
end

function _load(data, ::Val{:DQMCMeasurement})
    temp = haskey(data, "temp") ? data["temp"] : data["output"]
    kernel_name = data["kernel"]
    kernel = try
        eval(kernel_name)
    catch e
        @warn "Failed to load kernel in module MonteCarlo." exception=e
        missing_kernel
    end
    gi = _load(data["GI"], Val(:GreensIterator))
    li = _load(data["LI"], Val(:LatticeIterator))
    fi = if haskey(data, "FI")
        data["FI"]
    else
        # TODO generate defaults based on kernel name
        # don't think I can without mc being available because I need 
        # to call unique_flavors(mc)
        @error "Need to make an iterator that errors on iteration here" maxlog = 1
        ErrorIterator()
    end
    obs = _load(data["obs"])
    DQMCMeasurement(gi, li, fi, kernel, obs, temp)
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

function _flatten_measurements(measurements)
    output = Vector{DQMCMeasurement}()
    for m in measurements
        if m isa DQMCMeasurement
            push!(output, m)

        elseif m isa MultiMeasurement
            for _m in m.measurements
                if _m isa DQMCMeasurement
                    push!(output, _m)
                else
                    @warn "$_m not recognized as DQMCMeasurement. Ignoring."
                end
            end
            
        else
            @warn "Failed to convert $m into DQMCMeasurements"
        end
    end

    return output
end

@bm function generate_groups(mc, model, measurements)
    measurements = _flatten_measurements(measurements)

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


function maybe_repeating(mc, G)
    if unique_flavors(mc) == 1
        GreensMatrix(G.k, G.l, DiagonallyRepeatingMatrix(G.val))
    else
        G
    end
end

@bm function apply!(::Nothing, combined::Vector{<: Tuple}, mc::DQMC)
    for (lattice_iterator, measurement) in combined
        # Clear temp if necessary
        prepare!(lattice_iterator, measurement, mc)
        # Write measurement to ouput
        measure!(lattice_iterator, measurement, mc, nothing)
        # Finalize computation (temp) and commit
        finish!(lattice_iterator, measurement, mc)
    end

    nothing
end

@bm function apply!(::Greens, combined::Vector{<: Tuple}, mc::DQMC)
    G = maybe_repeating(mc, greens!(mc))
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
    end

    # measure normal Greens order first, then restructure! and measure rest
    for (lattice_iterator, measurement) in combined
        measurement.lattice_iterator isa Restructure && continue
        measure!(lattice_iterator, measurement, mc, G)
    end

    restructure!(mc, G)

    for (lattice_iterator, measurement) in combined
        measurement.lattice_iterator isa Restructure || continue
        measure!(lattice_iterator, measurement, mc, G)
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(g::GreensAt, combined::Vector{<: Tuple}, mc::DQMC)
    G = maybe_repeating(mc, greens!(mc, g.k, g.l))
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
    end

    # measure normal Greens order first, then restructure! and measure rest
    for (lattice_iterator, measurement) in combined
        measurement.lattice_iterator isa Restructure && continue
        measure!(lattice_iterator, measurement, mc, G)
    end

    restructure!(mc, G)

    for (lattice_iterator, measurement) in combined
        measurement.lattice_iterator isa Restructure || continue
        measure!(lattice_iterator, measurement, mc, G)
    end

    for (lattice_iterator, measurement) in combined
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(iter::TimeIntegral, combined::Vector{<: Tuple}, mc::DQMC)
    for (lattice_iterator, measurement) in combined
        prepare!(lattice_iterator, measurement, mc)
    end

    # To avoid doing the very expensive calculation of the G0l, Gl0, Gll 
    # multiple times we need to restructure in the greens iterator loop.
    # That also forces us to recalculate G00 every iteration, though that's 
    # relatively cheap.
    M = nslices(mc)
    for (i, (g0l, gl0, gll)) in enumerate(init(mc, iter))
        weight = ifelse(i in (1, M+1), 0.5, 1.0) * mc.parameters.delta_tau
        G00 = maybe_repeating(mc, greens!(mc))
        G0l = maybe_repeating(mc, g0l)
        Gl0 = maybe_repeating(mc, gl0)
        Gll = maybe_repeating(mc, gll)
        packed_greens = (G00, G0l, Gl0, Gll)

        for (lattice_iterator, measurement) in combined
            measurement.lattice_iterator isa Restructure && continue
            measure!(lattice_iterator, measurement, mc, packed_greens, weight)
        end

        restructure!(mc, packed_greens)

        for (lattice_iterator, measurement) in combined
            measurement.lattice_iterator isa Restructure || continue
            measure!(lattice_iterator, measurement, mc, packed_greens, weight)
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
    push!(measurement.observable, measurement.kernel(mc, mc.model, nothing, packed_greens, nothing))
    nothing
end



################################################################################
### apply Lattice Iterators 
################################################################################


function apply!(
        temp::Array, iter::DirectLatticeIterator, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::DirectLatticeIterator, ::$(typeof(measurement.kernel)))" begin
        @inbounds @fastmath for σ in measurement.flavor_iterator
            for idx in with_lattice(iter, lattice(mc))
                val = getindex(temp, CartesianIndex(idx))
                val += weight * measurement.kernel(mc, mc.model, idx, packed_greens, σ)
                setindex!(temp, val, CartesianIndex(idx))
            end
        end 
    end
    nothing
end


function apply!(
        temp::Array, iter::DeferredLatticeIterator, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::DeferredLatticeIterator, ::$(typeof(measurement.kernel)))" begin
        @inbounds @fastmath for σ in measurement.flavor_iterator
            for idxs in with_lattice(iter, lattice(mc))
                temp[first(idxs)] += weight * measurement.kernel(
                    mc, mc.model, idxs[2:end], packed_greens, σ
                )
            end
        end
    end
    nothing
end

function apply!(
        temp::Array, iter::EachSitePairByDistance, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::EachSitePairByDistance, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        Bsrctrg2dir = l[:Bravais_srctrg2dir]::Matrix{Int}
        B = length(unitcell(l))
        N = length(Bravais(l))

        @inbounds @fastmath for σ in measurement.flavor_iterator
            for b2 in 1:B, b1 in 1:B
                uc1 = N * (b1-1)
                uc2 = N * (b2-1)
                for trg in 1:N
                    @simd for src in 1:N
                        dir = Bsrctrg2dir[src, trg]
                        temp[dir, b1, b2] += weight * measurement.kernel(
                            mc, mc.model, (src + uc1, trg + uc2), packed_greens, σ
                        )
                    end
                end
            end
        end
    end

    return 
end


function apply!(
        temp::Array, iter::EachLocalQuadByDistance, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::EachLocalQuadByDistance, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        srcdir2trg = l[:srcdir2trg]::Matrix{Int}
        Bsrctrg2dir = l[:Bravais_srctrg2dir]::Matrix{Int}
        B = length(unitcell(l))
        subN = _length(l, iter.directions)::Int
        Ndir = length(l[:Bravais_dir2srctrg])::Int

        @inbounds @fastmath for σ in measurement.flavor_iterator
            for src1 in eachindex(l)
                uc1, Bsrc1 = fldmod1(src1, Ndir)
                dirs1 = _dir_idxs_uc(l, iter.directions, uc1)::Vector{Pair{Int, Int}}

                for src2 in eachindex(l)
                    uc2, Bsrc2 = fldmod1(src2, Ndir)
                    dir12 = Bsrctrg2dir[Bsrc1, Bsrc2]
                    dirs2 = _dir_idxs_uc(l, iter.directions, uc2)::Vector{Pair{Int, Int}}
                    
                    for (sub_idx1, dir1) in dirs1
                        trg1 = srcdir2trg[src1, dir1]
                        trg1 == 0 && continue
                        
                        for (sub_idx2, dir2) in dirs2
                            trg2 = srcdir2trg[src2, dir2]
                            trg2 == 0 && continue
                            
                            combined_dir = _sub2ind(
                                (Ndir, subN, subN, B, B), 
                                (dir12, sub_idx1, sub_idx2, uc1, uc2)
                            )

                            temp[combined_dir] += weight * measurement.kernel(
                                mc, mc.model, (src1, trg1, src2, trg2), packed_greens, σ
                            )
                        end
                    end
                end
            end

        end
    end

    return 
end


# These work with cache[1 + x1 - x2 + Lx] or cache[x + dx]
modcachex(l::Lattice) = collect(mod1.(1:(1+3l.Ls[1]), l.Ls[1]))
modcachey(l::Lattice) = collect(mod1.(1:(1+3l.Ls[2]), l.Ls[2]))


function apply!(
        temp::Array, iter::EachBondPairByBravaisDistance, 
        measurement, mc::DQMC, packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::EachBondPairByBravaisDistance, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        Lx, Ly = l.Ls
        bs = view(l.unitcell.bonds, iter.bond_idxs)
        modx = get!(l, :modcachex, modcachex)::Vector{Int}
        mody = get!(l, :modcachey, modcachey)::Vector{Int}

        @inbounds @fastmath for σ in measurement.flavor_iterator
            for s1y in 1:Ly, s1x in 1:Lx
                for s2y in 1:Ly, s2x in 1:Lx
                    # output "directions"
                    # 1 because I want onsite at index 1
                    dx = modx[1 + s2x - s1x + Lx] 
                    dy = mody[1 + s2y - s1y + Ly]

                    for (i, b1) in enumerate(bs)
                        s1 = _sub2ind(l, (s1x, s1y, from(b1)))
                        x, y = b1.uc_shift
                        t1 = _sub2ind(l, (modx[s1x+x+Lx], mody[s1y+y+Ly], to(b1)))

                        for (j, b2) in enumerate(bs)
                            s2 = _sub2ind(l, (s2x, s2y, from(b2)))
                            x, y = b2.uc_shift
                            t2 = _sub2ind(l, (modx[s2x+x+Lx], mody[s2y+y+Ly], to(b2)))

                            # println((dx, dy, i, j, s1, t1, s2, t2))
                            temp[dx, dy, i, j] += weight * measurement.kernel(
                                mc, mc.model, (s1, t1, s2, t2), packed_greens, σ
                            )
                        end
                    end
                end
            end
        end
    end
    return 
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