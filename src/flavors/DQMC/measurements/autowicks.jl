import LinearAlgebra

# We wanna implement some special stuff which might be bad for Base.Expr
struct MyExpr
    e::Expr
end
MyExpr(args...) = MyExpr(flatten_once(Expr(args...)))
Base.getproperty(e::MyExpr, field::Symbol) = getproperty(getfield(e, :e), field)
Base.:(==)(a::MyExpr, b::MyExpr) = getfield(a, :e) == getfield(b, :e)


# I.e. a c or c^†
struct Operator
    daggered::Bool
    indices::Vector{Any}
end

# constructors
Operator(d, idxs...) = Operator(d, collect(idxs))
create(indices...) = Operator(true, collect(indices))
annihilate(indices...) = Operator(false, collect(indices))
c(indices...) = annihilate(indices...)
cd(indices...) = create(indices...)

Literal(x, args...) = MyExpr(:Literal, x, args...)
ExpectationValue(x::MyExpr) = MyExpr(:ExpectationValue, x)

struct ArrayElement
    daggered::Bool
    indices::Vector{Any}
end
ArrayElement(idxs...) = ArrayElement(collect(idxs))

# base extensions
Base.:(==)(a::Operator, b::Operator) = a.daggered == b.daggered && a.indices == b.indices
LinearAlgebra.adjoint(o::Operator) = Operator(!o.daggered, o.indices)
const OpOrExpr = Union{Operator, MyExpr}
Base.:(-)(a::OpOrExpr) = MyExpr(:*, Literal(-1), a)
Base.:(*)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:*, a, b)
Base.:(+)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:+, a, b)
Base.:(-)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:+, a, Literal(-1), b)
Base.:(*)(a::OpOrExpr, b::OpOrExpr, c, ds...) = *(MyExpr(:*, a, b), c, ds...)
Base.:(+)(a::OpOrExpr, b::OpOrExpr, c, ds...) = +(MyExpr(:+, a, b), c, ds...)


# just printing
function Base.show(io::IO, o::Operator)
    print(io, "c")
    o.daggered && print(io, "^†")
    print(io, "_{")
    join(io, string.(o.indices), ", ")
    print(io, "}")
end

function print_expr(e::MyExpr)

    e.head == :+ && print("(")
    e.head == :ExpectationValue && print("⟨")
    for (i, child) in enumerate(e.args)
        i > 1 && print(" ", e.head, " ")
        print_expr(child)
    end
    e.head == :ExpectationValue && print("⟩")
    e.head == :+ && print(")")
    nothing
end
print_expr(e) = print(e)


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

# TODO
# need to figure out spins
# need to figure out times
# need to add I - G

struct GreensElement
    idx::Tuple{Any, Any}
    times::Tuple{Any, Any}
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
    @assert all(x -> x isa Pair, e.args)

    for (i, (a, b)) in enumerate(e.args)
        if a.daggered && !b.daggered
            # c^† c - like
        elseif !a.daggered && b.daggered
            # c c^†
            GreensElement(a.indices[1], b.indices[2])
        else
            return 0.0
        end
    end
end