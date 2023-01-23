function Chain(Lx)
    uc = UnitCell(
        "Chain",
        (Float64[1],),
        [Float64[0]],
        [Bond(1, 1, ( 1,)), Bond(1, 1, (-1,))]
    )

    Lattice(uc, (Lx,))
end

function SquareLattice(Lx, Ly = Lx)
    uc = UnitCell(
        "Square",
        (Float64[1, 0], Float64[0, 1]),
        [Float64[0, 0]],
        [
            Bond(1, 1, ( 1,  0)),
            Bond(1, 1, ( 0,  1)),
            Bond(1, 1, (-1,  0)),
            Bond(1, 1, ( 0, -1))
        ]
    )

    Lattice(uc, (Lx, Ly))
end

function CubicLattice(Lx, Ly = Lx, Lz = Lx)
    uc = UnitCell(
        "Cubic",
        (Float64[1, 0, 0], Float64[0, 1, 0], Float64[0, 0, 1]),
        [Float64[0, 0, 0]],
        [
            Bond(1, 1, ( 1,  0,  0)),
            Bond(1, 1, ( 0,  1,  0)),
            Bond(1, 1, ( 0,  0,  1)),
            Bond(1, 1, (-1,  0,  0)),
            Bond(1, 1, ( 0, -1,  0)),
            Bond(1, 1, ( 0,  0, -1))
        ]
    )

    Lattice(uc, (Lx, Ly, Lz))
end

function Honeycomb(Lx, Ly = Lx)
    uc = UnitCell(
        "Honeycomb",
        (Float64[sqrt(3.0)/2, -0.5], Float64[sqrt(3.0)/2, +0.5]),
        [Float64[0.0, 0.0], Float64[1/sqrt(3.0), 0.0]],
        [
            Bond(1, 2, (0, 0)), Bond(1, 2, (-1, 0)), Bond(1, 2, (0, -1)),
            Bond(2, 1, (0, 0)), Bond(2, 1, ( 1, 0)), Bond(2, 1, (0,  1)),
        ]
    )

    Lattice(uc, (Lx, Ly))
end

function TriangularLattice(Lx, Ly = Lx)
    uc = UnitCell(
        "Triangular",
        (Float64[sqrt(3.0)/2, -0.5], Float64[sqrt(3.0)/2, +0.5]),
        [Float64[0.0, 0.0]],
        [
            Bond(1, 1, ( 1,  0)), Bond(1, 1, ( 0,  1)), Bond(1, 1, (-1,  1)),
            Bond(1, 1, (-1,  0)), Bond(1, 1, ( 0, -1)), Bond(1, 1, ( 1, -1)),
        ]
    )

    Lattice(uc, (Lx, Ly))
end