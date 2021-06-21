using Base: Callable
# TODO
# - need to re-add spin index
# because that's a variable shift

import LinearAlgebra

# We wanna implement some special stuff which might be bad for Base.Expr
struct MyExpr
    e::Expr

    function MyExpr(e::Expr)
        if e.head == :call
            new(Expr(e.args...))
        else
            new(e)
        end
    end
end
MyExpr(args...) = MyExpr(flatten_once(Expr(args...)))
Base.getproperty(e::MyExpr, field::Symbol) = getproperty(getfield(e, :e), field)
Base.:(==)(a::MyExpr, b::MyExpr) = getfield(a, :e) == getfield(b, :e)
expr2myexpr(x::Expr) = MyExpr(x)
expr2myexpr(x) = x


# I.e. a c or c^†
struct Operator{T1 <: Union{Integer, MyExpr, Symbol}, T2 <: Union{Integer, MyExpr, Symbol}}
    name::Symbol
    daggered::Bool
    index::T1
    time::T2
    function Operator(n, d, i, t)
        _i = expr2myexpr(i); _t = expr2myexpr(t)
        new{typeof(_i), typeof(_t)}(n, d, _i, _t)
    end
end

# constructors
create(idx, time=0; name = :c) = Operator(name, true, idx, time)
annihilate(idx, time=0; name = :c) = Operator(name, false, idx, time)
c(idx, time=0; name = :c) = annihilate(idx, time, name=name)
cd(idx, time=0; name = :c) = create(idx, time, name=name)

Literal(x, args...) = MyExpr(:Literal, x, args...)
ExpectationValue(x::MyExpr) = MyExpr(:ExpectationValue, x)

struct ArrayElement
    name::Symbol
    daggered::Bool
    indices::Vector{Union{Integer, MyExpr, Symbol}}
    ArrayElement(name::Symbol, d::Bool, idxs::Vector) = new(name, d, expr2myexpr.(idxs))
end
ArrayElement(name::Symbol, idxs...) = ArrayElement(name, false, collect(idxs))

# base extensions
function Base.:(==)(a::Operator, b::Operator)
    a.name == b.name && a.daggered == b.daggered && a.index == b.index && a.time == b.time
end
LinearAlgebra.adjoint(o::Operator) = Operator(o.name, !o.daggered, o.index, o.time)
const OpOrExpr = Union{ArrayElement, Operator, MyExpr}
Base.:(-)(a::OpOrExpr) = MyExpr(:*, Literal(-1), a)
Base.:(*)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:*, a, b)
Base.:(+)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:+, a, b)
Base.:(-)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:+, a, -b)
Base.:(*)(a::OpOrExpr, b::OpOrExpr, c, ds...) = *(MyExpr(:*, a, b), c, ds...)
Base.:(+)(a::OpOrExpr, b::OpOrExpr, c, ds...) = +(MyExpr(:+, a, b), c, ds...)


# just printing
function Base.show(io::IO, o::Operator)
    print(io, o.name)
    o.daggered && print(io, "^†")
    print(io, "_{", string(o.index), "}(", o.time, ")")
end

 print_expr(e::MyExpr) = print_expr(stdout, e)
function print_expr(io::IO, e::MyExpr)
    _print_expr(io, e)
    println(io)
end
function _print_expr(io, e::MyExpr)
    e.head == :+ && print(io, "(")
    e.head == :ExpectationValue && print(io, "⟨")
    for (i, child) in enumerate(e.args)
        i > 1 && print(io, " ", e.head, " ")
        _print_expr(io, child)
    end
    e.head == :ExpectationValue && print(io, "⟩")
    e.head == :+ && print(io, ")")
    nothing
end
function _print_expr(io, e::ArrayElement)
    print(io, e.name, "_{")
    for (i, idx) in enumerate(e.indices)
        i > 1 && print(io, ", ")
        _print_expr(io, idx)
    end
    print(io, "}")
end
_print_expr(io, e) = print(io, e)


# expand a * (b + c) -> a*b + a*c
# flatten nested operations
function expand(e::MyExpr)
    for i in eachindex(e.args)
        e.args[i] = expand(e.args[i])
    end
    new_expr = _expand(e)
    while new_expr != e
        e = new_expr
        new_expr = _expand(e)
    end
    return e
end
expand(e) = e

function _expand(e::MyExpr)
    # expand *(+(a, b), x) -> +(*(a, x), *(b, x))
    if e.head == :*
        i = 1
        while i <= length(e.args)
            child = e.args[i]
            if child isa MyExpr && child.head == :+
                products = map(child.args) do x
                    MyExpr(:*, e.args[1:i-1]..., x, e.args[i+1:end]...)
                end
                new_expr = MyExpr(:+, products...)
                # deal with other possible +(c, d) terms
                for j in eachindex(products)
                    new_expr.args[j] = _expand(new_expr.args[j])
                end
                # possibly flatten additions
                return _expand(new_expr)
            elseif child isa MyExpr && child.head == :*
                deleteat!(e.args, i)
                for x in child.args
                    insert!(e.args, i, x)
                    i += 1
                end
            else
                i += 1
            end
        end
    elseif e.head == :+
        i = 1
        e = deepcopy(e)
        while i < length(e.args)
            child = e.args[i]
            if child isa MyExpr && child.head == :+
                deleteat!(e.args, i)
                append!(e.args, child.args)
            else
                i += 1
            end
        end
    elseif e.head == :ExpectationValue
        if e.args[1].head == :+
            terms = [MyExpr(:ExpectationValue, x) for x in e.args[1].args]
            return MyExpr(:+, terms...)
        end
    end
    return e
end
_expand(e) = e

# flatten one level of nested operators
function flatten_once(e::Expr)
    new_args = Any[]
    for child in e.args
        if (child isa Expr || child isa MyExpr) && child.head == e.head
            append!(new_args, child.args)
        else
            push!(new_args, child)
        end
    end
    Expr(e.head, new_args...)
end


# Wicks theorem
# ⟨abcd⟩ ->  ⟨ab⟩⟨cd⟩ - ⟨ac⟩⟨bd⟩ + ⟨ad⟩⟨bc⟩ etc
function wicks(e)
    if e.head == :+
        products = wicks.(e.args)
        MyExpr(:+, products...)
    elseif e.head == :ExpectationValue
        wicks(e.args[1])
    else
        _wicks(e)
    end
end

function _wicks(e)
    @assert e.head == :*
    @assert all(x -> x isa Operator, e.args)
    skip = filter(i -> !(e.args[i] isa Operator), eachindex(e.args))
    groups = generate_pairings(e.args, copy(skip))
    products = map(enumerate(groups)) do (i, g)
        evs = map(p -> MyExpr(:ExpectationValue, MyExpr(:*, p...)), g)
        if isodd(i)
            # index 1 3 5 ... have even numbers pair-permutations
            MyExpr(:*, e.args[skip]..., evs...)
        else
            MyExpr(:*, Literal(-1), e.args[skip]..., evs...)
            # vcat(Literal(-1), e.args[skip], g)
        end
    end
    MyExpr(:+, products...)
end

function generate_pairings(v::Vector{T}, skip=Int[]) where T
    @assert length(v) % 2 == 0
    combos = Vector{Any}[]
    first_idx = findfirst(i -> !(i in skip), eachindex(v))
    push!(skip, first_idx)

    for i in first_idx+1:length(v)
        if !(i in skip)
            if length(skip)+1 < length(v)
                rest = generate_pairings(v, [skip; i])
                append!(combos, [vcat(Pair(v[first_idx], v[i]), r) for r in rest])
            else
                push!(combos, [Pair(v[first_idx], v[i])])
            end
        end
    end

    return combos
end



function to_greens(e::MyExpr)
    if e.head == :+
        new_args = Any[]
        for i in eachindex(e.args)
            converted = _to_greens(e.args[i])
            converted == 0.0 || push!(new_args, converted)
        end
        return MyExpr(:+, new_args...)
    else
        return _to_greens(e)
    end
end

function _to_greens(e::MyExpr)
    @assert e.head == :*

    outputs = []
    for (i, x) in enumerate(e.args)
        if x isa MyExpr && x.head == :ExpectationValue
            y = x.args[1]
            y.head == :* || error(y)
            length(y.args) == 2 || error(y)
            a, b = y.args
            if a.daggered && !b.daggered
                # c^† c - like
                delta = if a.time == b.time
                    ifelse(a.index == b.index, Literal(1), ArrayElement(:I, a.index, b.index))
                elseif a.time isa Number && b.time isa Number
                    0
                else
                    ArrayElement(:I, a.index, b.index) * ArrayElement(:I, a.time, b.time)
                end
                x = delta - ArrayElement(Symbol(:G, b.time, a.time), b.index, a.index)
                push!(outputs, x)
            elseif !a.daggered && b.daggered
                # c c^†
                # GreensElement(a.indices[1], b.indices[2])
                push!(outputs, ArrayElement(Symbol(:G, a.time, b.time), a.index, b.index))
            else
                return 0.0
            end
        else
            push!(outputs, x)
        end
    end

    return *(outputs...)
end