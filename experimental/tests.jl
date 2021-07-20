
################################################################################
### Tests
################################################################################

using Test

@testset "MyExpr" begin
    @test getfield(MyExpr(:(x + y)), :e) == Expr(:+, :x, :y)

    a = MyExpr(:(x * y))
    b = MyExpr(Expr(:*, :x, :y))
    @test a == b
    @test MyExpr(:*, :x, :y) == b

    e = quote
        a = 1 + c
        b = 2a
    end
    simplified = Expr(:block, Expr(:(=), :a, Expr(:+, 1, :c)), Expr(:(=), :b, Expr(:*, 2, :a)))
    @test getfield(MyExpr(e), :e) == simplified
end

@testset "Operators" begin
    op1 = create(:i, :l, name = :d)
    @test op1.name == :d
    @test op1.daggered == true
    @test op1.index == :i
    @test op1.time == :l

    op2 = annihilate(:j, :k, name = :a)
    @test op2.name == :a
    @test op2.daggered == false
    @test op2.index == :j
    @test op2.time == :k

    @test c(:i, :l, name=:d)' == op1
    @test c(:j, :k, name=:a) == op2

    op = c(:(i + N))
    @test op.index isa MyExpr
    @test getfield(op.index, :e) == Expr(:+, :i, :N)
end

@testset "ArrayElement" begin
    x = ArrayElement(:A, :i, :j)
    @test x.name == :A
    @test x.indices == [:i, :j]

    x = ArrayElement(:vector, :(2i + 1))
    @test x.indices[1] isa MyExpr
    @test getfield(x.indices[1], :e) == Expr(:+, Expr(:*, 2, :i), 1)
end

@testset "Basic math" begin
    a = c(:i)
    b = ArrayElement(:A, :j)
    @test a + b == MyExpr(:+, a, b)
    @test a * b == MyExpr(:*, a, b)
    @test -a == MyExpr(:*, -1, a)

    d = MyExpr(:(x + y))
    @test a + b + d == MyExpr(:+, a, b, d)
    @test a * b * d == MyExpr(:*, a, b, d)

    @test (a + b) * d == MyExpr(:*, MyExpr(:+, a, b), d)
    @test a + b * d == MyExpr(:+, a, MyExpr(:*, b, d))
end

@testset "Expand" begin
    a = c(:i)
    b = ArrayElement(:A, :j)
    e = (a + b) * 2
    @test e == MyExpr(:*, MyExpr(:+, a, b), 2)
    e2 = expand(e)
    @test e2 == MyExpr(:+, MyExpr(:*, a, 2), MyExpr(:*, b, 2))
end

@testset "Wicks" begin
    op1 = c(:(i))'
    op2 = c(:(i))
    op3 = c(:(i + N))'
    op4 = c(:(i + N))
    e = wicks(op1 * op2 * op3 * op4)
    result = MyExpr(
        :+, 
        MyExpr(:*, ExpectationValue(op1 * op2), ExpectationValue(op3 * op4)),
        MyExpr(:*, -1, ExpectationValue(op1 * op3), ExpectationValue(op2 * op4)),
        MyExpr(:*, ExpectationValue(op1 * op4), ExpectationValue(op2 * op3))
    )
    @test e == result
end

@testset "to Greens elements" begin
    e = wicks(c(:(i))' * c(:(i)) * c(:(j))' * c(:(j)))
    g = to_greens(e)
    result = 
        (1 - ArrayElement(:G00, :i, :i)) * (1 - ArrayElement(:G00, :j, :j)) +
        (ArrayElement(:I, :i, :j) - ArrayElement(:G00, :j, :i)) * ArrayElement(:G00, :i, :j)
    @test g == result
end