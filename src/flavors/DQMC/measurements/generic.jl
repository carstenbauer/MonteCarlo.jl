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


# TODO
# try replacing global function rather than creatin new local ones


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
        measure!(lattice_iterator, measurement, mc, G)
        finish!(lattice_iterator, measurement, mc)
    end
    nothing
end

@bm function apply!(g::GreensAt, combined::Vector{<: Tuple}, mc::DQMC)
    G = maybe_repeating(mc, greens!(mc, g.k, g.l))
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

    
    M = nslices(mc)
    for (i, (g0l, gl0, gll)) in enumerate(init(mc, iter))
        weight = ifelse(i in (1, M+1), 0.5, 1.0) * mc.parameters.delta_tau
        G00 = maybe_repeating(mc, greens!(mc))
        G0l = maybe_repeating(mc, g0l)
        Gl0 = maybe_repeating(mc, gl0)
        Gll = maybe_repeating(mc, gll)

        for (lattice_iterator, measurement) in combined
            measurement.kernel == temp_kernel && continue
            measure!(lattice_iterator, measurement, mc, (G00, G0l, Gl0, Gll), weight)
        end
        for (lattice_iterator, measurement) in combined
            measurement.kernel == temp_kernel || continue
            measure!(lattice_iterator, measurement, mc, (G00, G0l, Gl0, Gll), weight)
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

# #=

function apply!(
        temp::Array, iter::EachBondPairByBravaisDistance, 
        measurement::DQMCMeasurement{<: typeof(cc_kernel)}, 
        mc::DQMC, 
        packed_greens, weight = 1.0
    )
    @timeit_debug "apply!(::EachBondPairByBravaisDistance, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        Lx, Ly = l.Ls
        bs = view(l.unitcell.bonds, iter.bond_idxs)
        modx = get!(l, :modcachex, modcachex)::Vector{Int}
        mody = get!(l, :modcachey, modcachey)::Vector{Int}
        
        # G00 = packed_greens[1].val.val
        # Gll = packed_greens[4].val.val
        # println((
        #     Gll[1, 2] * G00[1, 2],
        #     Gll[2, 1] * G00[1, 2],
        #     Gll[1, 2] * G00[2, 1],
        #     Gll[2, 1] * G00[2, 1],
        #     Gll[1, 2] * G00[1, 2] - Gll[2, 1] * G00[1, 2] - Gll[1, 2] * G00[2, 1] + Gll[2, 1] * G00[2, 1]
        # ))

        #@inbounds 
        @fastmath for σ in measurement.flavor_iterator
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
    # error()
    return 
end

# =#


function restructure!(mc, Gs::NTuple{4, GreensMatrix}; kwargs...)
    restructure!.((mc,), Gs; kwargs...)
    return
end
function restructure!(mc, G::GreensMatrix; kwargs...)
    restructure!(mc, G.val; kwargs...)
    return
end
function restructure!(mc, G::BlockDiagonal; kwargs...)
    restructure!.((mc,), G.blocks; kwargs...)
    return
end
function restructure!(mc, G::StructArray; kwargs...)
    restructure!(mc, G.re; kwargs...)
    restructure!(mc, G.im; kwargs...)
    return
end
function restructure!(mc, G::DiagonallyRepeatingMatrix; kwargs...)
    restructure!(mc, G.val; kwargs...)
    return
end
function restructure!(mc, G::Matrix; target = G, temp = mc.stack.curr_U)
    N = length(lattice(mc))
    K = div(size(G, 1), N)
    dir2srctrg = lattice(mc)[:Bravais_dir2srctrg]::Vector{Vector{Int}}

    if target === G
        for k1 in 0:N:N*K-1, k2 in 0:N:N*K-1 # flv and basis
            for dir in eachindex(dir2srctrg)
                for (src, trg) in enumerate(dir2srctrg[dir])
                    temp[src+k1, dir+k2] = G[src+k1, trg+k2]
                end
            end
        end

        copyto!(target, temp)
    else
        for k1 in 0:N:N*K-1, k2 in 0:N:N*K-1 # flv and basis
            for dir in eachindex(dir2srctrg)
                for (src, trg) in enumerate(dir2srctrg[dir])
                    target[src+k1, dir+k2] = G[src+k1, trg+k2]
                end
            end
        end
    end

    return
end

# #=

function temp_kernel() end

function apply!(
        temp::Array, iter::EachBondPairByBravaisDistance, 
        measurement::DQMCMeasurement{<: typeof(temp_kernel)}, 
        mc::DQMC, 
        packed_greens, weight = 1.0
    )

    # NOTES
    # this version assumes:
    # - source and time indepedent hopping matrix (i.e. one constant value per bond)
    # - all bonds included (so skipping of reverse)
    # - real matrix with 1 effective flavor
    # - 2 real flavors

    # TODO
    # - I swapped G00 and Gll for comparison but I think it should be swapped back

    @timeit_debug "apply!(::EachBondPairByBravaisDistance, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        Lx, Ly = l.Ls
        N = Lx * Ly #length(l)
        bs = view(l.unitcell.bonds, iter.bond_idxs)

        restructure!(mc, packed_greens)        
        # restructure!(mc, mc.stack.hopping_matrix, target = mc.stack.curr_U)

        #           x, y
        # trg1 ---- src1 ---- src2 ---- trg2
        #    dx1, dy1   dx, dy   dx2, dy2
        #       b1                  b2
        # uc12      uc11      uc21      uc22

        G00, G0l, Gl0, Gll = packed_greens

        T0 = zero(eltype(temp))
        T1 = one(eltype(temp))

        @inbounds @fastmath for σ in measurement.flavor_iterator
            # Assuming 2D
            for (bi, b1) in enumerate(bs)
                uc11 = from(b1)
                uc12 = to(b1)
                dx1, dy1 = b1.uc_shift

                # mod0
                dx1 = ifelse(dx1 < 0, Lx + dx1, dx1)
                dy1 = ifelse(dy1 < 0, Ly + dy1, dy1)

                # one based; direction + target unitcell
                d1 = 1 + dx1 + Lx * dy1 + N * (uc12 - 1) 

                for (bj, b2) in enumerate(bs)
                    uc21 = from(b2)
                    uc22 = to(b2)
                    dx2, dy2 = b2.uc_shift

                    # mod0
                    dx2 = ifelse(dx2 < 0, Lx + dx2, dx2)
                    dy2 = ifelse(dy2 < 0, Ly + dy2, dy2)

                    d2 = 1 + dx2 + Lx * dy2 + N * (uc22 - 1)

                    for dy in 0:Ly-1, dx in 0:Lx-1
                        # val = I1 * I2
                        val = T0

                        # d = 1 + dx + Lx * dy # one based

                        # I[i + (dx, dy), i + b1]
                        # I3 = ifelse(
                        #     (uc21 == uc12) && (dx == dx1) && (dy == dy1) && (G0l.l == G0l.k),
                        #     T1, T0
                        # )

                        # Inlined mod1, see comment below
                        # # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                        # px1 = dx1 - dx; px1 = ifelse(px1 < 0, px1 + Lx, px1)
                        # py1 = dy1 - dy; py1 = ifelse(py1 < 0, py1 + Ly, py1)
                        # p1 = 1 + px1 + Lx * py1 + N * (uc12 - 1)
                        # # println((dx, dy, dx1, dy1, px1, py1, p1))

                        # # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                        # px2 = dx + dx2; px2 = ifelse(px2 ≥ Lx, px2 - Lx, px2)
                        # py2 = dy + dy2; py2 = ifelse(py2 ≥ Ly, py2 - Ly, py2)
                        # p2 = 1 + px2 + Lx * py2 + N * (uc22 - 1)
                        # # println((dx, dy, dx2, dy2, px2, py2, p2))

                        # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                        px1 = dx1 - dx; px1 = ifelse(px1 < 0, px1 + Lx, px1)
                        py1 = dy1 - dy; py1 = ifelse(py1 < 0, py1 + Ly, py1)
                        p1 = 1 + px1 + Lx * py1 + N * (uc12 - 1)
                        
  
                        # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                        px2 = dx + dx2; px2 = ifelse(px2 ≥ Lx, px2 - Lx, px2)
                        py2 = dy + dy2; py2 = ifelse(py2 ≥ Ly, py2 - Ly, py2)
                        p2 = 1 + px2 + Lx * py2 + N * (uc22 - 1)
                        # if bi in (1, 2) && bj == 2 && dx == 1 && dy == 0
                        #     println((dx, dy, dx1, dy1, px1, py1, p1))
                        #     println((dx, dy, dx2, dy2, px2, py2, p2))
                        #     println("---")
                        # end

                        I3 = ifelse(
                            (uc11 == uc22) && (p2 == 1) && (G0l.l == G0l.k),
                            T1, T0
                        )

                        # TODO permutation loop/iterations
                        # No, we can do permutations by including all directions
                        # after the simulation
                        # hopping matrix as well
                        # TODO flavor weights
                        @simd for y in 1:Ly
                            # This is a repalcement for mod1(y+dy, Ly). Another 
                            # alternative is explicit loop splitting, which is
                            # used for the x loop. For the y loop it doesn't 
                            # seem to matter if we use ifelse or loop splitting,
                            # for x loop splitting is faster.
                            y2 = ifelse(y > Ly - dy, y - Ly, y)
                            for x in 1:Lx-dx
                                i = x + Lx * (y-1) + N * (uc11 - 1)
                                i2 = x + dx + Lx * (y2 + dy - 1) + N * (uc21 - 1)

                                # println((dx+1, dy+1, bi, bj, i, i2, d1, d2, p1, p2))
                                # if bi in (1, 2) && bj == 2 && dx == 1 && dy == 0
                                #     println(
                                #         "[1] ($x, $y) -> ($i, $p2) ($i2, $p1) ", 
                                #         2 * (I3 - G0l.val.val[i, p2]) * Gl0.val.val[i2, p1]
                                #     )
                                # end

                                val += begin
                                    # I1 and I2 are always 0

                                    # initial version:
                                    # 4 * (I1 - Gll.val.val[i, d1]) * (I2 - G00.val.val[i2, d2]) +
                                    # 2 * (I3 - G0l.val.val[i2, p1]) * Gl0.val.val[i, p2]

                                    # matched first to old ccs:
                                    4 * Gll.val.val[i2, d2] * G00.val.val[i, d1] +
                                    2 * (I3 - G0l.val.val[i, p2]) * Gl0.val.val[i2, p1]
                                end
                            end

                            # in this loop dx > 0
                            for x in Lx-dx+1:Lx
                                i  = x + Lx * (y-1) + N * (uc11 - 1)
                                # needs a -Lx which is here  ⤵
                                i2 = x + dx + Lx * (y2 + dy - 2) + N * (uc21 - 1)
                                # println((dx+1, dy+1, bi, bj, i, i2, d1, d2, p1, p2))

                                # if bi in (1, 2) && bj == 2 && dx == 1 && dy == 0
                                #     println(
                                #         "[2] ($x, $y) -> ($i, $p2) ($i2, $p1) ", 
                                #         2 * (I3 - G0l.val.val[i, p2]) * Gl0.val.val[i2, p1]
                                #     )
                                # end

                                val += begin
                                    # 4 * (I1 - Gll.val.val[i, d1]) * (I2 - G00.val.val[i2, d2]) +
                                    
                                    4 * Gll.val.val[i2, d2] * G00.val.val[i, d1] +
                                    2 * (I3 - G0l.val.val[i, p2]) * Gl0.val.val[i2, p1]
                                    
                                    # 2 * Gl0.val.val[i, p1] * (I3 - G0l.val.val[i2, p2])
                                    # 2 * (I3 - G0l.val.val[i2, p2]) * Gl0.val.val[i, p1]
                                    # 2 * (I3 - G0l.val.val[i, p1]) * Gl0.val.val[i2, p2]
                                end
                            end
                        end # source loops

                        temp[dx+1, dy+1, bi, bj] += weight * val

                    end # main distance loop

                end # selected distance
            end # selected distance

        end # flavor
    end # benchmark
    # error()
    return 
end

# =#


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