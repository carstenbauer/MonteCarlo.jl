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
create(indices...) = Operator(true, indices)
annihilate(indices...) = Operator(false, indices)
c(indices...) = annihilate(indices...)
cd(indices...) = create(indices...)

# base extensions
Base.:(==)(a::Operator, b::Operator) = a.daggered == b.daggered && a.indices == b.indices
LinearAlgebra.adjoint(o::Operator) = Operator(!o.daggered, o.indices)
const OpOrExpr = Union{Operator, MyExpr}
Base.:(*)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:*, a, b)
Base.:(+)(a::OpOrExpr, b::OpOrExpr) = MyExpr(:+, a, b)
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
Base.string(idx::VarIndex) = "ID$(idx.ID)"

function print_expr(e::MyExpr)
    e.head != :* && print("(")
    for (i, child) in enumerate(e.args)
        i != 1 && print(" ", e.head, " ")
        print_expr(child)
    end
    e.head != :* && print(")")
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
# ⟨abcd⟩ ->  ⟨ab⟩⟨cd⟩ + ⟨ac⟩⟨bd⟩ + ⟨ad⟩⟨bc⟩
# TODO: fix this sign ^
function wicks(e)
    if e.head == :+
        products = _wicks.(e.args)
        MyExpr(:+, products...)
    else
        _wicks(e)
    end
end

function _wicks(e)
    @assert e.head == :*
    @assert all(x -> x isa Operator, e.args)
    combos = generate_pairings(e.args)
    MyExpr(:+, map(c -> MyExpr(:*, c...), combos)...)
end

function generate_pairings(v::Vector{T}, skip=Int[]) where T
    @assert length(v) % 2 == 0
    combos = Vector{Pair{T}}[]
    first_idx = findfirst(i -> !(i in skip), eachindex(v))
    push!(skip, first_idx)

    for i in first_idx+1:length(v)
        if !(i in skip)
            if length(skip)+1 < length(v)
                rest = generate_pairings(v, [skip; i])
                append!(combos, [vcat(Pair(v[first_idx], v[i]), r) for r in rest])
            else
                push!(combos, Pair(v[first_idx], v[i]))
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