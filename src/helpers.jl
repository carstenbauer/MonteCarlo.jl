"""
    Enum ExitCode

An ExitCode is returned by `run!(mc)` once it finishes. Currently there are four
possible codes:

- `GENERIC_FAILURE` is the fallback for a failed or cancelled run. Usually there
is no save file in this case.
- `SUCCESS` indicates that the simulation finished normally. In this case the 
simulation has not created/overwritten a save file.
- `CANCELLED_TIME_LIMIT` indicates that the simulation ran normally but didn't 
finish due to reaching a set time limit. In this case the simulation has created
or overwritten a save file.
- `CANCELLED_LOW_ACCEPTANCE` means the simulation cancelled due to a lack of 
accepted updates. A save file is created in this case.
"""
@enum ExitCode begin
    GENERIC_FAILURE = -1
    SUCCESS = 0
    CANCELLED_TIME_LIMIT = 1
    CANCELLED_LOW_ACCEPTANCE = 2
end


"""
    sparsity(A)

Calculates the sparsity of the given array.
The sparsity is defined as number of zero-valued elements divided by
total number of elements.
"""
function sparsity(A::AbstractArray{T}) where T<:Number
    (length(A)-countnz(A))/length(A)
end

"""
    reldiff(A, B)

Relative difference of absolute values of `A` and `B` defined as

``
\\operatorname{reldiff} = 2 \\dfrac{\\operatorname{abs}(A - B)}{\\operatorname{abs}(A+B)}.
``
"""
function reldiff(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
    return 2*abs.(A-B)./abs.(A+B)
end


"""
    effreldiff(A, B, threshold=1e-14)

Same as `reldiff(A,B)` but with all elements set to zero where corresponding element of
`absdiff(A,B)` is smaller than `threshold`. This is useful in avoiding artificially large
relative errors.
"""
function effreldiff(
        A::AbstractArray{T}, B::AbstractArray{S}, threshold::Float64=1e-14
    ) where {T <: Number, S <: Number}
    r = reldiff(A,B)
    r[findall(x -> abs.(x)<threshold, absdiff(A,B))] .= 0.
    return r
end


"""
    absdiff(A, B)

Difference of absolute values of `A` and `B`.
"""
function absdiff(A::AbstractArray{T}, B::AbstractArray{S}) where {T<:Number, S<:Number}
    return abs.(A-B)
end


"""
    compare(A, B)

Compares two matrices `A` and `B`, prints out the maximal absolute and relative differences
and returns a boolean indicating wether `isapprox(A,B)`.
"""
function compare(A::AbstractArray{T}, B::AbstractArray{S}) where {T<:Number, S<:Number}
    @printf("max absdiff: %.1e\n", maximum(absdiff(A,B)))
    @printf("mean absdiff: %.1e\n", mean(absdiff(A,B)))
    @printf("max reldiff: %.1e\n", maximum(reldiff(A,B)))
    @printf("mean reldiff: %.1e\n", mean(reldiff(A,B)))

    r = effreldiff(A,B)
    @printf("effective max reldiff: %.1e\n", maximum(r))
    @printf("effective mean reldiff: %.1e\n", mean(r))

    return isapprox(A,B)
end


# NOTE
# Currenlty julia/sparearrays does not implement this function (type signature)
# once it does this can be removed/depracted in favor of mul!
# see also: test/slice_matrices.jl
occursin(
    "SparseArrays",
    string(which(mul!, (Matrix, Matrix, SparseMatrixCSC)).file)
) && @warn(
    "A Method `mul!(::Matrix, ::Matrix, ::SparseMatrixCSC)` now exists in " *
    "`SparseArrays`. The method defined in `helpers.jl` is likely to be  " *
    "unnecessary now."
)

function SparseArrays.mul!(C::StridedMatrix, X::StridedMatrix, A::SparseMatrixCSC)
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(C, zero(eltype(C)))
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mX
            C[multivec_row, col] += X[multivec_row, ki] * kv
        end
    end
    C
end


# Taken from Base
if !isdefined(Base, :splitpath)
    splitpath(p::AbstractString) = splitpath(String(p))

    if Sys.isunix()
        const path_dir_splitter = r"^(.*?)(/+)([^/]*)$"
    elseif Sys.iswindows()
        const path_dir_splitter = r"^(.*?)([/\\]+)([^/\\]*)$"
    else
        error("path primitives for this OS need to be defined")
    end

    _splitdir_nodrive(path::String) = _splitdir_nodrive("", path)
    function _splitdir_nodrive(a::String, b::String)
        m = match(path_dir_splitter,b)
        m === nothing && return (a,b)
        a = string(a, isempty(m.captures[1]) ? m.captures[2][1] : m.captures[1])
        a, String(m.captures[3])
    end

    function splitpath(p::String)
        drive, p = splitdrive(p)
        out = String[]
        isempty(p) && (pushfirst!(out,p))  # "" means the current directory.
        while !isempty(p)
            dir, base = _splitdir_nodrive(p)
            dir == p && (pushfirst!(out, dir); break)  # Reached root node.
            if !isempty(base)  # Skip trailing '/' in basename
                pushfirst!(out, base)
            end
            p = dir
        end
        if !isempty(drive)  # Tack the drive back on to the first element.
            out[1] = drive*out[1]  # Note that length(out) is always >= 1.
        end
        return out
    end
end


"""
    @bm function ... end
    @bm foo(args...) = ...
    @bm "name" begin ... end

Wraps the body of a function with `@timeit_debug <function name> begin ... end`.
This macro can also be used on a code block to generate
`@timeit_debug <name> begin ... end`.

The `@timeit_debug` macro can be disabled per module. Using `@bm` will make
sure that the module is always `MonteCarlo`. One can enable and disable
benchmarking with `enable_benchmarks()` and `disable_benchmarks()`. If they are
disabled they should come with zero overhead. See TimerOutputs.jl for more
details.

Benchmarks/Timings can be retrieved using `print_timer()` and reset with
`reset_timer!()`.
"""
macro bm(args...)
    if length(args) == 1 && args[1].head in (Symbol("="), :function)
        expr = args[1]
        code = TimerOutputs.timer_expr(
            MonteCarlo, true,
            _to_typed_name(expr.args[1]),
            :(begin $(expr.args[2]) end) # inner code block
        )
        Expr(
            expr.head,     # function or =
            esc(expr.args[1]),  # function name w/ args
            code
        )
    else
        # Not a function, just do the same as timeit_debug
        # This is copied from TimerOutputs.jl
        # With __module__ replaced by MonteCarlo because we want to have all
        # timings in the MonteCarlo namespace (otherwise they are not tied to
        # MonteCarlo.timeit_debug_enabled())
        TimerOutputs.timer_expr(MonteCarlo, true, args...)
    end
end

function _to_typed_name(e::Expr)
    if e.head == :where
        _to_typed_name(e.args[1])
    elseif e.head == :call
        string(e.args[1]) * "(" * join(_to_type.(e.args[2:end]), ", ") * ")"
    else
        dump(e)
        "ERROR"
    end
end

_to_type(s::Symbol) = "::Any"
function _to_type(e::Expr)
    if e.head == Symbol("::")
        return "::" * string(e.args[end])
    elseif e.head == Symbol("...")
        return _to_type(e.args[1]) * "..."
    elseif e.head == :parameters # keyword args
        return _to_type(e.args[1])
    elseif e.head == :kw # default value
        return _to_type(e.args[1])
    else
        dump(e)
        return "???"
    end
end

timeit_debug_enabled() = false

"""
    enable_benchmarks()

Enables benchmarking for `MonteCarlo`.

This affects every function with the `MonteCarlo.@bm` macro as well as any
`TimerOutputs.@timeit_debug` blocks. Benchmarks are recorded to the default
TimerOutput `TimerOutputs.DEFAULT_TIMER`. Results can be printed via
`TimerOutputs.print_timer()`.

[`disable_benchmarks`](@ref)
"""
enable_benchmarks() = TimerOutputs.enable_debug_timings(MonteCarlo)

"""
    disable_benchmarks()

Disables benchmarking for `MonteCarlo`.

This affects every function with the `MonteCarlo.@bm` macro as well as any
`TimerOutputs.@timeit_debug` blocks.

[`enable_benchmarks`](@ref)
"""
disable_benchmarks() = TimerOutputs.disable_debug_timings(MonteCarlo)


# map matrix indices to linear indices

# This can be called with one less element in Ns than in idxs (tight left)
"""
    _sub2ind(size, indices)

Calculates a flat array index from a set of (1-based) `indices` with the help 
of the `size` of the array.

Note that only `size[1:N-1]` are need for `N = length(indices)`. Any size values
beycond that are ignored. Note as well that the methods for tuples are more 
optimized (up to 4-tuples).

See also: [`_sub2ind0`](@ref)
"""
function _sub2ind(Ns, idxs)
    idx = idxs[end] - 1
    @inbounds for d in length(idxs)-1:-1:1
        idx = idx * Ns[d] + (idxs[d] - 1)
    end
    return idx + 1
end

"""
    _sub2ind0(size, indices)

Calculates a flat 0-based array index from a set of 0-based `indices` with the 
help of the `size` of the array.

Note that only `size[1:N-1]` are need for `N = length(indices)`. Any size values
beycond that are ignored. Note as well that the methods for tuples are more 
optimized (up to 4-tuples).

See also: [`_sub2ind`](@ref)
"""
function _sub2ind0(Ns, idxs)
    idx = idxs[end]
    @inbounds for d in length(idxs)-1:-1:1
        idx = idx * Ns[d] + idxs[d]
    end
    return idx
end

@generated function _sub2ind0(Ns::T, idxs::T) where T
    if Ns <: NTuple{1}
        :(@inbounds return idxs[1])
    elseif Ns <: NTuple{2}
        :(@inbounds return muladd(Ns[1], idxs[2], idxs[1]))
    elseif Ns <: NTuple{3}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2], idxs[1])
            @inbounds return muladd(Ns[2], idxs[3], idx)
        end
    elseif Ns <: NTuple{4}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2], idxs[1])
            @inbounds idx = muladd(Ns[2], idxs[3], idx)
            @inbounds return muladd(Ns[3], idxs[4], idx)
        end
    else
        quote
            idx = idxs[end]
            @inbounds for d in length(idxs)-1:-1:1
                idx = idx * Ns[d] + idxs[d]
            end
            return idx
        end
    end
end

@generated function _sub2ind(Ns::T, idxs::T) where T
    if Ns <: NTuple{1}
        :(@inbounds return idxs[1])
    elseif Ns <: NTuple{2}
        :(@inbounds return muladd(Ns[1], idxs[2]-1, idxs[1]))
    elseif Ns <: NTuple{3}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2]-1, idxs[1])
            @inbounds return muladd(Ns[2], idxs[3]-1, idx)
        end
    elseif Ns <: NTuple{4}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2]-1, idxs[1])
            @inbounds idx = muladd(Ns[2], idxs[3]-1, idx)
            @inbounds return muladd(Ns[3], idxs[4]-1, idx)
        end
    else
        quote
            #@boundscheck length(idxs)-2 < length(Ns)
            @inbounds idx = idxs[end]-1
            @inbounds for d in length(idxs)-1:-1:1
                idx = muladd(idx, Ns[d], idxs[d]-1)
            end
            return idx
        end
    end
end
