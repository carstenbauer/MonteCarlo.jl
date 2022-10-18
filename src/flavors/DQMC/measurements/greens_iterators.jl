################################################################################
### Generic
################################################################################


abstract type AbstractGreensIterator end
abstract type AbstractUnequalTimeGreensIterator <: AbstractGreensIterator end

init!(::AbstractUnequalTimeGreensIterator) = nothing
"""
    verify(iterator::AbstractUnequalTimeGreensIterator[, maxerror=1e-6])

Returns true if the given `iterator` is accurate up to the given `maxerror`.

See also: (@ref)[`accuracy`]
"""
verify(it::AbstractUnequalTimeGreensIterator, maxerror=1e-6) = maximum(accuracy(it)) < maxerror

function _save(file::FileLike, key::String, m::T) where {T <: Union{AbstractGreensIterator, Nothing}}
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", "GreensIterator")
    write(file, "$key/name", nameof(T))
    write(file, "$key/fields", getfield.((m,), fieldnames(T)))
    return
end

function _load(data, ::Val{:GreensIterator})
    # ifelse maybe long but should be better for compile time than adding a 
    # bunch more _load methods and better for runtime than an eval
    tag = haskey(data, "name") ? data["name"] : data["tag"]
    fields = data["fields"]
    for T in _all_greens_iterator_types
        if tag == nameof(T)
            return T(fields...)
        end
    end

    # Fallback
    return eval(:($tag($(fields)...)))
end


################################################################################
### nothing (No Iteration)
################################################################################

# Once we change greens iterators to follow the "template -> iter" structure
# this will be compat only
# Some type piracy to make things easier
Base.Nothing(::DQMC, ::Model) = nothing


################################################################################
### Greens (equal time)
################################################################################


# To identify the requirement of equal-time Greens functions
struct Greens <: AbstractGreensIterator end
Greens(::DQMC, ::Model)= Greens()


################################################################################
### GreensAt (specific unequal time)
################################################################################


struct GreensAt <: AbstractUnequalTimeGreensIterator
    k::Int
    l::Int
end
GreensAt(l::Integer) = GreensAt(l, l)
GreensAt(k::Integer, l::Integer) = GreensAt(k, l)
# GreensAt{k, l}(::DQMC, ::Model) where {k, l} = GreensAt(k, l)
Base.:(==)(a::GreensAt, b::GreensAt) = (a.k == b.k) && (a.l == b.l)


################################################################################
### TimeIntegral (wrapper for CombinedGreensIterator, returns G0l, Gl0, Gll)
################################################################################


struct TimeIntegral <: AbstractUnequalTimeGreensIterator
    recalculate::Int
    TimeIntegral(recalculate::Int = -1) =  new(recalculate)
end
TimeIntegral(::DQMC, recalculate::Int = -1) = TimeIntegral(recalculate)  
TimeIntegral(::DQMC, ::Model, recalculate::Int = -1) = TimeIntegral(recalculate)  
# There is no point differentiating based on recalculate
Base.:(==)(a::TimeIntegral, b::TimeIntegral) = true

function init(mc, ti::TimeIntegral)
    if ti.recalculate == -1
        _TimeIntegral(init(mc, CombinedGreensIterator(
            mc, start = 0, stop = mc.parameters.slices
        )))
    else
        _TimeIntegral(init(mc, CombinedGreensIterator(
            mc, start = 0, stop = mc.parameters.slices, recalculate = ti.recalculate
        )))
    end
end


################################################################################
### CombinedGreensIterator (G0l, Gl0, Gll)
################################################################################


"""
    CombinedGreensIterator(mc::DQMC)

Returns an iterator which efficiently calculates and returns triples 
(G(0, l), G(l, 0), G(l, l)) for each iteration.

By default `l` runs from 1 (Δτ) to M (β), including both. This can be adjusted
by the keyword arguments `start = 1` and `stop = mc.parameters.slices`.

## Warning

Calling `greens(mc)`, `greens(mc, slice)`, `greens(mc, slice1, slice2)` or 
similar functions will break the iteration sequence. If you need the result of 
one of these, call it before starting the iteration. The result will remain 
valid.

This happens because the iterator uses a bunch of temporary matrices as storage
between iterations. Unlike the functions mentioned above, the iterator does not 
usually calculate greens function from scratch. Instead results from the last
iteration are used to calculate the next. Note that even with matrix 
stabilization this only works for some time and therefore a full recalculation
happens every so often. This can be changed by adjusting the keyword argument 
`recalculate = 2mc.paremeters.safe_mult`. 

You can also set `recalculate = nothing` and pass a `max_delta = 1e-7`. In this
case the iterator will pick `recalculate` such that the difference between the 
iterator and the respective results from `greens(...)` is at most `max_delta`. 
Note that this is an experimental feature - the dependence of the error on the 
state of the simulation has not been thoroughly investigated.

For reference:

outputs: (do not overwrite)
`stack.tmp1`, `stack.tmp2`, `uts.greens` 
`uts.complex_greens_temp1`, `uts.complex_greens_temp2`, `uts.complex_greens_temp3`

storage: (do not overwrite)
`stack.Ul`, `stack.Dl`, `stack.Tl`
`stack.Ur`, `stack.Dr`, `stack.Tr`
`uts.U`, `uts.D`, `uts.T`

temporary: (overwritten here but can be overwritten outside)
`stack.curr_U`, `uts.tmp`
"""
struct CombinedGreensIterator <: AbstractUnequalTimeGreensIterator
    recalculate::Int
    start::Int
    stop::Int
end

function CombinedGreensIterator(
        mc::DQMC, model::Model = mc.model; 
        recalculate = 2mc.parameters.safe_mult, max_delta = 1e-7,
        start = 0, stop = mc.parameters.slices
    )
    if recalculate === nothing
        iter = CombinedGreensIterator(typemax(Int64), start, stop)
        recalc = estimate_recalculate(iter, max_delta)
        CombinedGreensIterator(recalc, start, stop)
    else
        CombinedGreensIterator(recalculate, start, stop)
    end
end

Base.length(it::CombinedGreensIterator) = it.stop - it.start + 1 # both included
function Base.:(==)(a::CombinedGreensIterator, b::CombinedGreensIterator)
    (a.recalculate == b.recalculate) && (a.start == b.start) && (a.stop == b.stop)
end
function init(mc::DQMC, it::CombinedGreensIterator)
    _CombinedGreensIterator(mc, it, isdefined(mc.stack, :complex_greens_temp))
end


################################################################################
### CombinedGreensIterator (implementation)
################################################################################


struct _CombinedGreensIterator{T <: DQMC}
    mc::T
    spec::CombinedGreensIterator
    # TODO rework measurements to work well with StructArrays and remove this
    copy_to_array::Bool
end

Base.length(it::_CombinedGreensIterator) = length(it.spec)

# Fast specialized version
@bm function Base.iterate(it::_CombinedGreensIterator)
    s = it.mc.stack
    uts = it.mc.ut_stack

    # Need full built stack for curr_U to be save
    build_stack(it.mc, uts)
    # force invalidate stack because we modify uts.greens
    uts.last_slices = (-1, -1)

    if it.spec.start in (0, 1)
        # Get G00
        if current_slice(it.mc) == 1
            # The measure system triggers here to be a bit more performant
            copyto!(s.Tl, s.greens)
        else
            # calculate G00
            calculate_greens_full1!(it.mc, it.mc.ut_stack, 0, 0)
            copyto!(s.Tl, uts.greens)
        end
        copyto!(s.tmp1, s.Tl)

        # Prepare next iteration
        # G01 = (G00-I)B_1^-1; then G0l+1 = G0l B_{l+1}^-1
        vsub!(s.Tr, s.Tl, I)

        # prepare UDT stacks
        udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, it.mc.stack.pivot, it.mc.stack.tempv)
        copyto!(uts.U, s.Ul)
        copyto!(uts.D, s.Dl)
        copyto!(uts.T, s.Tl)

        # G01 = (G00-I)B_1^-1; then G0l+1 = G0l B_{l+1}^-1
        udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, it.mc.stack.pivot, it.mc.stack.tempv)

        if it.spec.start == 0
            # return for G00 iteration
            Gll = _greens!(it.mc, uts.greens, s.tmp1, s.tmp2)
            Gl0 = copyto!(s.tmp1, uts.greens)
            G0l = copyto!(s.tmp2, uts.greens)
            if it.copy_to_array
                return ((
                    GreensMatrix(0, 0, copyto!(uts.complex_greens_temp1, G0l)), 
                    GreensMatrix(0, 0, copyto!(uts.complex_greens_temp2, Gl0)), 
                    GreensMatrix(0, 0, copyto!(uts.complex_greens_temp3, Gll))
                ), 1)
            else
                return ((
                    GreensMatrix(0, 0, G0l), 
                    GreensMatrix(0, 0, Gl0), 
                    GreensMatrix(0, 0, Gll)
                ), 1)
            end
        elseif it.spec.start == 1
            # start with l = 1 iteration
            return iterate(it, 1)
        end
    else
        # See recalculate step below
        calculate_greens_full1!(it.mc, it.mc.ut_stack, it.spec.start, 0)
        copyto!(s.curr_U, uts.greens)
        calculate_greens_full2!(it.mc, it.mc.ut_stack, 0, it.spec.start)
        copyto!(uts.tmp, uts.greens)
        calculate_greens_full1!(it.mc, it.mc.ut_stack, it.spec.start, it.spec.start)

        copyto!(uts.T, uts.greens)
        Gll = _greens!(it.mc, uts.greens, uts.T, s.tmp2) 
        udt_AVX_pivot!(uts.U, uts.D, uts.T, s.pivot, s.tempv)

        Gl0 = _greens!(it.mc, s.tmp1, s.curr_U, s.tmp2)
        copyto!(s.Tl, s.curr_U)
        udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, s.pivot, s.tempv)
        
        G0l = _greens!(it.mc, s.tmp2, uts.tmp, s.curr_U)
        copyto!(s.Tr, uts.tmp)
        udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, s.pivot, s.tempv)

        if it.copy_to_array
            l = it.spec.start
            return ((
                GreensMatrix(0, l, copyto!(uts.complex_greens_temp1, G0l)), 
                GreensMatrix(l, 0, copyto!(uts.complex_greens_temp2, Gl0)), 
                GreensMatrix(l, l, copyto!(uts.complex_greens_temp3, Gll))
            ), l+1)
        else
            return ((
                GreensMatrix(0, it.spec.start, G0l), 
                GreensMatrix(it.spec.start, 0, Gl0), 
                GreensMatrix(it.spec.start, it.spec.start, Gll)
            ), it.spec.start+1)
        end
    end
end

@bm function Base.iterate(it::_CombinedGreensIterator, l)
    # l is 1-based
    s = it.mc.stack
    uts = it.mc.ut_stack
    # force invalidate stack because we modify uts.greens
    uts.last_slices = (-1, -1)
    # if start is 1 we need to stabilize like if it were 0
    shift = (it.spec.start != 1) * it.spec.start 

    if l > it.spec.stop
        return nothing
    elseif (l - shift) % it.spec.recalculate == 0 
        # Recalculation will overwrite 
        # Ul, Dl, Tl, Ur, Dr, Tr, U, D, T, tmp1, tmp2, greens
        # curr_U is used only if the stack is rebuilt
        # this leaves uts.tmp (and curr_U)

        calculate_greens_full1!(it.mc, it.mc.ut_stack, l, 0)
        copyto!(s.curr_U, uts.greens)
        calculate_greens_full2!(it.mc, it.mc.ut_stack, 0, l)
        copyto!(uts.tmp, uts.greens)
        calculate_greens_full1!(it.mc, it.mc.ut_stack, l, l)

        # input and output can be the same here
        copyto!(uts.T, uts.greens)
        Gll = _greens!(it.mc, uts.greens, uts.T, s.tmp2) 
        udt_AVX_pivot!(uts.U, uts.D, uts.T, s.pivot, s.tempv)

        Gl0 = _greens!(it.mc, s.tmp1, s.curr_U, s.tmp2)
        copyto!(s.Tl, s.curr_U)
        udt_AVX_pivot!(s.Ul, s.Dl, s.Tl, s.pivot, s.tempv)
        
        G0l = _greens!(it.mc, s.tmp2, uts.tmp, s.curr_U)
        copyto!(s.Tr, uts.tmp)
        udt_AVX_pivot!(s.Ur, s.Dr, s.Tr, s.pivot, s.tempv)

        if it.copy_to_array
            return ((
                GreensMatrix(0, l, copyto!(uts.complex_greens_temp1, G0l)), 
                GreensMatrix(l, 0, copyto!(uts.complex_greens_temp2, Gl0)), 
                GreensMatrix(l, l, copyto!(uts.complex_greens_temp3, Gll))
            ), l+1)
        else
            return ((
                GreensMatrix(0, l, G0l), 
                GreensMatrix(l, 0, Gl0), 
                GreensMatrix(l, l, Gll)
            ), l+1)
        end

    elseif ((l - shift) % it.spec.recalculate) % it.mc.parameters.safe_mult == 0
        # Stabilization        
        # Reminder: These overwrite s.tmp1 and s.tmp2
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, s.Ul) 
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, s.Tr)
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, uts.U)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, uts.T)

        # Gl0
        vmul!(s.tmp1, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp2, s.tmp1, s.Tl)
        udt_AVX_pivot!(s.Ul, s.Dl, s.tmp1, s.pivot, s.tempv)
        vmul!(s.curr_U, s.tmp1, s.Tl)
        copyto!(s.Tl, s.curr_U)
        Gl0 = _greens!(it.mc, s.tmp1, s.tmp2, s.curr_U)
        # tmp1 must not change anymore
        
        # G0l
        vmul!(s.curr_U, Diagonal(s.Dr), s.Tr)
        vmul!(uts.greens, s.Ur, s.curr_U)
        copyto!(s.Tr, s.curr_U)
        udt_AVX_pivot!(s.tmp2, s.Dr, s.Tr, s.pivot, s.tempv)
        vmul!(s.curr_U, s.Ur, s.tmp2)
        copyto!(s.Ur, s.curr_U)
        G0l = _greens!(it.mc, s.tmp2, uts.greens, s.curr_U)
        # tmp1, tmp2 must not change anymore
        
        # Gll
        vmul!(uts.tmp, uts.U, Diagonal(uts.D))
        vmul!(uts.greens, uts.tmp, uts.T)
        udt_AVX_pivot!(s.curr_U, uts.D, uts.tmp, it.mc.stack.pivot, it.mc.stack.tempv)
        vmul!(uts.U, uts.tmp, uts.T)
        vmul!(uts.T, Diagonal(uts.D), uts.U)
        udt_AVX_pivot!(uts.tmp, uts.D, uts.T, it.mc.stack.pivot, it.mc.stack.tempv)
        vmul!(uts.U, s.curr_U, uts.tmp)
        Gll = _greens!(it.mc, uts.greens, uts.greens, s.curr_U)

        if it.copy_to_array
            return ((
                GreensMatrix(0, l, copyto!(uts.complex_greens_temp1, G0l)), 
                GreensMatrix(l, 0, copyto!(uts.complex_greens_temp2, Gl0)), 
                GreensMatrix(l, l, copyto!(uts.complex_greens_temp3, Gll))
            ), l+1)
        else
            return ((
                GreensMatrix(0, l, G0l), 
                GreensMatrix(l, 0, Gl0), 
                GreensMatrix(l, l, Gll)
            ), l+1)
        end

    else
        # Quick advance
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, s.Ul) 
        # multiply_slice_matrix_inv_left!(it.mc, it.mc.model, nslices(it.mc)-l+1, s.Ur)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, s.Tr)
        multiply_slice_matrix_left!(it.mc, it.mc.model, l, uts.U)
        multiply_slice_matrix_inv_right!(it.mc, it.mc.model, l, uts.T)

        # Gl0
        vmul!(s.curr_U, s.Ul, Diagonal(s.Dl))
        vmul!(s.tmp2, s.curr_U, s.Tl)
        Gl0 = _greens!(it.mc, s.tmp1, s.tmp2, s.curr_U)

        # G0l
        vmul!(s.curr_U, s.Ur, Diagonal(s.Dr))
        vmul!(uts.greens, s.curr_U, s.Tr)
        G0l = _greens!(it.mc, s.tmp2, uts.greens, s.curr_U)

        # Gll
        vmul!(s.curr_U, uts.U, Diagonal(uts.D))
        vmul!(uts.greens, s.curr_U, uts.T)
        Gll = _greens!(it.mc, uts.greens, uts.greens, s.curr_U)

        if it.copy_to_array
            return ((
                GreensMatrix(0, l, copyto!(uts.complex_greens_temp1, G0l)), 
                GreensMatrix(l, 0, copyto!(uts.complex_greens_temp2, Gl0)), 
                GreensMatrix(l, l, copyto!(uts.complex_greens_temp3, Gll))
            ), l+1)
        else
            return ((
                GreensMatrix(0, l, G0l), 
                GreensMatrix(l, 0, Gl0), 
                GreensMatrix(l, l, Gll)
            ), l+1)
        end
    end
end


################################################################################
### TimeIntegral (implementation)
################################################################################


struct _TimeIntegral{T}
    iter::_CombinedGreensIterator{T}
end
Base.iterate(iter::_TimeIntegral) = iterate(iter.iter)
Base.iterate(iter::_TimeIntegral, i) = iterate(iter.iter, i)
Base.length(iter::_TimeIntegral) = length(iter.iter)



################################################################################
### Utilities
################################################################################


function accuracy(mc::DQMC, it::CombinedGreensIterator)
    iter = _CombinedGreensIterator(mc, it)
    Gk0s = [deepcopy(greens(mc, k, 0).val) for k in 1:nslices(mc)]
    G0ks = [deepcopy(greens(mc, 0, k).val) for k in 1:nslices(mc)]
    Gkks = [deepcopy(greens(mc, k, k).val) for k in 1:nslices(mc)]
    map(enumerate(iter)) do (i, Gs)
        (
            maximum(abs.(G0ks[i] .- Gs[1].val)),
            maximum(abs.(Gk0s[i] .- Gs[2].val)),
            maximum(abs.(Gkks[i] .- Gs[3].val))
        )
    end
end

function estimate_recalculate(iter::CombinedGreensIterator, max_delta=1e-7)
    deltas = accuracy(iter)
    idx = findfirst(ds -> any(ds .> max_delta), deltas)
    if idx === nothing
        return typemax(Int)
    else
        return idx
    end
end



################################################################################
### GreensIterator (generic and untested)
################################################################################




# Maybe split into multiple types?
struct GreensIterator{slice1, slice2, T <: DQMC} <: AbstractUnequalTimeGreensIterator
    mc::T
    recalculate::Int64
end

"""
    GreensIterator(mc::DQMC[, ks=Colon(), ls=0, recalculate=4mc.parameters.safe_mult])

Creates an Iterator which calculates all `[G[k <- l] for l in ls for k in ks]`
efficiently. Currently only allows `ks = :, ls::Integer`. 

For stability it is necessary to fully recompute `G[k <- l]` every so often. one
can adjust the frequency of recalculations via `recalculate`. To estimate the 
resulting accuracy, one may use `accuracy(GreensIterator(...))`.

This iterator requires the `UnequalTimeStack`. Iteration overwrites the 
`DQMCStack` variables `curr_U`, `Ul`, `Dl`, `Tl`, `Ur`, `Dr`, `Tr`, `tmp1` and 
`tmp2`, with `curr_U` acting as the output. For the iteration to remain valid, 
the UnequalTimeStack must not be overwritten. As such:
- `greens!(mc)` can be called and remains valid
- `greens!(mc, slice)` can be called and remains valid
- `greens!(mc, k, l)` will break iteration but remains valid (call before iterating)
"""
function GreensIterator(
        mc::T, slice1 = Colon(), slice2 = 0, recalculate = 4mc.parameters.safe_mult
    ) where {T <: DQMC}
    GreensIterator{slice1, slice2, T}(mc, recalculate)
end
init!(it::GreensIterator) = initialize_stack(it.mc, it.mc.ut_stack)
Base.length(it::GreensIterator{:, i}) where {i} = it.mc.parameters.slices + 1 - i

# Slower, versatile version:
function Base.iterate(it::GreensIterator{:, i}) where {i}
    s = it.mc.ut_stack
    calculate_greens_full1!(it.mc, s, i, i)
    copyto!(s.T, s.greens)
    udt_AVX_pivot!(s.U, s.D, s.T, it.mc.stack.pivot, it.mc.stack.tempv)
    G = _greens!(it.mc, it.mc.stack.curr_U, s.greens, it.mc.stack.Ur)
    return (GreensMatrix(i, i, G), (i+1, i))
end
function Base.iterate(it::GreensIterator{:}, state)
    s = it.mc.ut_stack
    k, l = state
    if k > it.mc.parameters.slices
        return nothing
    elseif k % it.recalculate == 0
        # Recalculate
        calculate_greens_full1!(it.mc, s, k, l) # writes s.greens
        G = _greens!(it.mc, it.mc.stack.curr_U, s.greens, it.mc.stack.Tl)
        copyto!(s.T, s.greens)
        udt_AVX_pivot!(s.U, s.D, s.T, it.mc.stack.pivot, it.mc.stack.tempv)
        return (GreensMatrix(k, l, G), (k+1, l))
    elseif k % it.mc.parameters.safe_mult == 0
        # Stabilization
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.stack.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.stack.tmp1, it.mc.stack.curr_U, s.T)
        udt_AVX_pivot!(s.U, s.D, it.mc.stack.curr_U, it.mc.stack.pivot, it.mc.stack.tempv)
        vmul!(it.mc.stack.tmp2, it.mc.stack.curr_U, s.T)
        copyto!(s.T, it.mc.stack.tmp2)
        G = _greens!(it.mc, it.mc.stack.curr_U, it.mc.stack.tmp1, it.mc.stack.tmp2)
        return (GreensMatrix(k, l, G), (k+1, l))
    else
        # Quick advance
        multiply_slice_matrix_left!(it.mc, it.mc.model, k, s.U)
        vmul!(it.mc.stack.curr_U, s.U, Diagonal(s.D))
        vmul!(it.mc.stack.tmp1, it.mc.stack.curr_U, s.T)
        G = _greens!(it.mc, it.mc.stack.curr_U, it.mc.stack.tmp1, it.mc.stack.Ur)
        s.last_slices = (-1, -1) # for safety
        return (GreensMatrix(k, l, G), (k+1, l))
    end
end
"""
    accuracy(iterator::AbstractUnequalTimeGreensIterator)

Compares values from the given iterator to more verbose computations, returning 
the maximum differences for each. This can be used to check numerical stability.
"""
function accuracy(iter::GreensIterator{:, l}) where {l}
    mc = iter.mc
    Gk0s = [deepcopy(greens(mc, k, l).val) for k in l:nslices(mc)]
    [maximum(abs.(Gk0s[i] .- G.val)) for (i, G) in enumerate(iter)]
end

const _all_greens_iterator_types = [
    Nothing, Greens, GreensAt, TimeIntegral, GreensIterator, CombinedGreensIterator
]