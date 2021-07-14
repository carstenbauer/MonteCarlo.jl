using Base: Symbol
import LinearAlgebra

# We wanna implement some special stuff which might be bad for Base.Expr
struct MyExpr
    e::Expr

    function MyExpr(e::Expr, should_cleanup::Bool = true)
        if should_cleanup
            new(cleanup(e))
        else
            new(flatten_once(e))
        end
    end
end
MyExpr(args...; cleanup = true) = MyExpr(Expr(args...), cleanup)
Base.getproperty(e::MyExpr, field::Symbol) = getproperty(getfield(e, :e), field)
Base.:(==)(a::MyExpr, b::MyExpr) = getfield(a, :e) == getfield(b, :e)
expr2myexpr(x::Expr) = MyExpr(x)
expr2myexpr(x) = x
function Base.show(io::IO, me::MyExpr)
    print(io, "MyExpr(")
    e = getfield(me, :e)
    show(io, e.head)
    for arg in e.args
        print(io, ", ")
        show(io, arg)
    end
    print(io, ")")
end

function cleanup(e::Expr)
    if e.head == :call
        head = e.args[1]
        args = e.args[2:end]
    else
        head = e.head
        args = e.args
    end

    new_args = Any[]

    for child in args
        if child isa Expr
            child = cleanup(child)
        end
        if (child isa Expr || child isa MyExpr) && child.head == e.head
            append!(new_args, child.args)
        elseif child isa LineNumberNode; nothing;
        else
            push!(new_args, child)
        end
    end
    Expr(head, new_args...)
end



# I.e. a c or c^†
struct Operator{T1, T2}
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
# cd(idx, time=0; name = :c) = create(idx, time, name=name)
function Base.:(==)(a::Operator, b::Operator)
    a.name == b.name && a.daggered == b.daggered && a.index == b.index && a.time == b.time
end

ExpectationValue(x::MyExpr) = MyExpr(:ExpectationValue, x)

struct ArrayElement
    name::Symbol
    indices::Vector{Any}
    ArrayElement(name::Symbol, idxs::Vector) = new(name, expr2myexpr.(idxs))
end
ArrayElement(name::Symbol, idxs...) = ArrayElement(name, collect(idxs))
function Base.:(==)(a::ArrayElement, b::ArrayElement)
    a.name == b.name && a.indices == b.indices
end

struct Sym
    name::Symbol
end
Base.:(==)(a::Sym, b::Sym) = a.name == b.name

# base extensions
LinearAlgebra.adjoint(o::Operator) = Operator(o.name, !o.daggered, o.index, o.time)
const OpOrExpr = Union{ArrayElement, Operator, MyExpr, Sym}
Base.:(-)(a::OpOrExpr) = MyExpr(:*, -1, a)
for T1 in (OpOrExpr, Number), T2 in (OpOrExpr, Number)
    T1 == Number && T2 == Number && continue
    Base.:(*)(a::T1, b::T2) = MyExpr(:*, a, b)
    Base.:(+)(a::T1, b::T2) = MyExpr(:+, a, b)
    Base.:(-)(a::T1, b::T2) = MyExpr(:+, a, -b)
    Base.:(*)(a::T1, b::T2, c, ds...) = *(MyExpr(:*, a, b), c, ds...)
    Base.:(+)(a::T1, b::T2, c, ds...) = +(MyExpr(:+, a, b), c, ds...)
end


########################################
### Printing
########################################


function Base.show(io::IO, o::Operator)
    print(io, o.name)
    o.daggered && print(io, "^†")
    print(io, "_{")
    _print_expr(io, o.index)
    print(io, "}(", o.time, ")")
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
_print_expr(io, e::Sym) = print(io, e.name)
_print_expr(io, e) = print(io, e)


########################################
### Expand brackets & flatten tree
########################################


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
                    MyExpr(:*, e.args[1:i-1]..., x, e.args[i+1:end]..., cleanup=false)
                end
                new_expr = MyExpr(:+, products..., cleanup=false)
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
            terms = [MyExpr(:ExpectationValue, x, cleanup=false) for x in e.args[1].args]
            return MyExpr(:+, terms..., cleanup=false)
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


########################################
### Apply Wicks theorem
########################################


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
    skip = filter(i -> !(e.args[i] isa Operator), eachindex(e.args))
    groups = generate_pairings(e.args, copy(skip))
    products = map(enumerate(groups)) do (i, g)
        evs = map(p -> MyExpr(:ExpectationValue, MyExpr(:*, p...)), g)
        if isodd(i)
            # index 1 3 5 ... have even numbers pair-permutations
            MyExpr(:*, e.args[skip]..., evs...)
        else
            MyExpr(:*, -1, e.args[skip]..., evs...)
        end
    end
    MyExpr(:+, products...)
end

function generate_pairings(v::Vector{T}, skip=Int[]) where T
    length(v) - length(skip) == 0 && return Vector{Any}[]
    @assert (length(v) - length(skip)) % 2 == 0 "$v"
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


########################################
### convert expectation values to Greens functions
########################################


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
                    ifelse(a.index == b.index, 1, ArrayElement(:I, a.index, b.index))
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


########################################
### Replacements
########################################


struct PLACEHOLDER{T} end
PLACEHOLDER() = PLACEHOLDER{Any}()
struct REPLACEMENT
    x::Int64
end

replace(e, replacements::Pair...) = _replace1(e, replacements)
function _replace1(e, replacements::Tuple)
    e = _replace2(e, replacements)
    for (matchable, replacement) in replacements
        is_match, captures = match(e, matchable)
        if is_match
            return apply_replace(replacement, captures)
        end
    end
    return e
end
function _replace2(e::MyExpr, replacements)
    MyExpr(
        _replace1(e.head, replacements),
        [_replace1(child, replacements) for child in e.args]...
    )
end
function _replace2(ae::ArrayElement, replacements)
    ArrayElement(
        _replace1(ae.name, replacements),
        [_replace1(idx, replacements) for idx in ae.indices]...
    )
end
_replace2(e, replacements) = e


match(e, ::PLACEHOLDER{Any}) = true, Any[e]
match(e::T, ::PLACEHOLDER{T}) where {T} = true, Any[e]
match(a, b) = a == b, Any[]
function match(e::ArrayElement, m::ArrayElement)
    passthrough = Any[]
    if e.name != m.name || length(e.indices) != length(m.indices) 
        return false, passthrough
    end
    for (a, b) in zip(e.indices, m.indices)
        is_match, x = match(a, b)
        is_match || return false, passthrough
        passthrough = vcat(passthrough, x)
    end
    return true, passthrough
end
function match(e::MyExpr, m::MyExpr)
    passthrough = Any[]
    if e.head != m.head || length(e.args) != length(m.args) 
        return false, passthrough
    end
    for (a, b) in zip(e.args, m.args)
        is_match, x = match(a, b)
        is_match || return false, passthrough
        passthrough = vcat(passthrough, x)
    end
    return true, passthrough
end


apply_replace(r::REPLACEMENT, captures::Vector) = captures[r.x]
apply_replace(x, captures::Vector) = x
function apply_replace(r::MyExpr, captures::Vector)
    MyExpr(
        apply_replace(r.head, captures),
        (apply_replace(x, captures) for x in r.args)...
    )
end
function apply_replace(r::ArrayElement, captures::Vector)
    ArrayElement(
        apply_replace(r.name, captures),
        (apply_replace(x, captures) for x in r.indices)...
    )
end


function cleanup(e::MyExpr)
    replace(
        e,
        MyExpr(:+, PLACEHOLDER(), 0) => REPLACEMENT(1),
        MyExpr(:+, 0, PLACEHOLDER()) => REPLACEMENT(1),
        MyExpr(:*, PLACEHOLDER(), 0) => 0,
        MyExpr(:*, 0, PLACEHOLDER()) => 0,
        MyExpr(:*, PLACEHOLDER(), 1) => REPLACEMENT(1),
        MyExpr(:*, 1, PLACEHOLDER()) => REPLACEMENT(1),
    )
end


########################################
### To code
########################################


function to_expr(e::MyExpr)
    if e.head in (:+, :*)
        return Expr(:call, e.head, to_expr.(e.args)...)
    elseif e.head == :ExpectationValue
        error("Expectation values must be transformed.")
    else
        error("What is a $(e.head)?")
    end
end
to_expr(ae::ArrayElement) = Expr(:ref, ae.name, to_expr.(ae.indices)...)
to_expr(s::Symbol) = s
to_expr(x::Number) = x



function generate_kernel_function(
        e::Expr; mc_type = Any, model_type = Any,
        index_unpacking = default_index_unpacking(e),
        greens_unpacking = default_greens_unpacking(e),
    )

    func_head = Expr(
        :tuple,
        Expr(Symbol("::"), :mc, Symbol(mc_type)),
        Expr(Symbol("::"), :model, Symbol(model_type)),
        :packed_indices, :packed_greens, 
    )
    func_body = Expr(
        :block,
        :(N = length(lattice(model))),
        index_unpacking, greens_unpacking,
        e
    )
    Expr(Symbol("->"), func_head, func_body)
end

function default_index_unpacking(e)
    indices = filter(idx -> idx != :N, collect_indices(e))
    if length(indices) == 2
        @assert indices == Set([:i, :j])
        return :(i, j = packed_indices)
    elseif length(indices) == 4
        @assert indices == Set([:i, :j, :k, :l])
        return :(i, j, k, l = packed_indices)
    else
        error("We can only deal with 2 or 4 indices atm. $(length(indices))")
    end
end

function default_greens_unpacking(e)
    Gs = filter(idx -> idx != :N, collect_greens(e))
    if length(Gs) == 1
        @assert Gs == Set([:G00])
        return :(G00 = packed_greens)
    elseif length(Gs) == 4
        @assert Gs == Set([:G00, :G0l, :Gl0, :Gll])
        return :(G00, G0l, Gl0, Gll = packed_greens)
    else
        error("We can only deal with 2 or 4 indices atm. $(length(Gs))")
    end
end


function collect_indices(e::Expr, indices=Set{Symbol}(), is_index=false)
    if e.head in (:ref, :call)
        for arg in e.args[2:end]
            collect_indices(arg, indices, true)
        end
    elseif e.head == :block
        for arg in e.args
            collect_indices(arg, indices, is_index)
        end
    else
        error("Failed to parse $(e.head)")
    end
    return indices
end
function collect_indices(s::Symbol, indices, is_index)
    is_index && push!(indices, s)
end
collect_indices(_, _, _) = nothing

function collect_greens(e::Expr, names=Set{Symbol}())
    if e.head == :ref
        @assert length(e.args) > 1
        push!(names, e.args[1])
    elseif e.head == :call
        for arg in e.args[2:end]
            collect_greens(arg, names)
        end
    elseif e.head == :block
        for arg in e.args
            collect_greens(arg, names)
        end
    else
        error("Failed to parse $(e.head)")
    end
    return filter(name -> startswith(string(name), 'G'), names)
end
collect_greens(_, _) = nothing

macro expand2formula(code)
    code |> expand |> wicks |> to_greens |> to_expr
end

macro expand2kernel(code)
    code |> expand |> wicks |> to_greens |> to_expr |> generate_kernel_function
end