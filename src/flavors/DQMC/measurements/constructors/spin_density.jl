function spin_density(
        dqmc, model, dir::Symbol, greens_iterator; 
        lattice_iterator = EachSitePairByDistance(), wrapper = nothing, 
        flavor_iterator = FlavorIterator(dqmc, 0),
        kernel = if dir == :x; full_sdc_x_kernel
        elseif dir == :y; full_sdc_y_kernel
        elseif dir == :z; full_sdc_z_kernel
        else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
        end,
        kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    return DQMCMeasurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end

"""
    spin_density_correlation(mc, model, dir; kwargs...)

Generates an equal-time spin density correlation measurement for the given x, y,
or z direction `dir`. Note that the result needs to be added to the simulation 
via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = full_sdc_x_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `full_sdc_x_kernel` and 
`reduced_sdc_x_kernel` as well as the `sdc_y` and `sdc_z` versions.
- `lattice_iterator = EachSitePairByDistance()` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `flavor_iterator = FlavorIterator(mc, 0)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
spin_density_correlation(args...; kwargs...) = spin_density(args..., Greens(); kwargs...)

"""
    spin_density_susceptibility(mc, model, dir; kwargs...)

Generates an time-integrated spin density susceptibility measurement for the 
given x, y, or z direction `dir`. Note that the result needs to be added to the 
simulation via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = full_sdc_x_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `full_sdc_x_kernel` and 
`reduced_sdc_x_kernel` as well as the `sdc_y` and `sdc_z` versions.
- `lattice_iterator = EachSitePairByDistance()` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `flavor_iterator = FlavorIterator(mc, 0)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
spin_density_susceptibility(mc, args...; kwargs...) = spin_density(mc, args..., TimeIntegral(mc); kwargs...)


################################################################################
### Full kernels
################################################################################


"""
    full_sdc_x_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_x(src, τ) m_x(trg, 0)⟩ with the same definitions for m_x as 
`mx_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_x_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return full_sdc_x_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    # could be rewritten as 
    # s1 = N * (f1 - 1); s2 = N * (2 - f1); s3 = N * (f2 - 1); s3 = N * (2 - f2)
    # Gll.val[i+s1, i+s2] * G00.val[j+s3, j+s4]
    Gll.val[i+N, i] * G00.val[j+N, j] + 
    Gll.val[i+N, i] * G00.val[j, j+N] + 
    Gll.val[i, i+N] * G00.val[j+N, j] + 
    Gll.val[i, i+N] * G00.val[j, j+N] + 

    # similarly:
    # (id - G0l.val[j+s3, i+s1]) * Gl0.val[i+s2, j+s4]
    (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] + 
    (id - G0l.val[j, i])     * Gl0.val[i+N, j+N] +
    (id - G0l.val[j+N, i+N]) * Gl0.val[i, j] + 
    (0  - G0l.val[j+N, i])   * Gl0.val[i, j+N]
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, model, ij::NTuple{2}, 
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flvs
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

    # similarly:
    (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[2][i, j] +
    (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[1][i, j]
end


"""
    full_sdc_y_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_y(src, τ) m_y(trg, 0)⟩ with the same definitions for m_y as 
`my_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_y_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return full_sdc_y_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    # same as x, but with some prefactor
    - Gll.val[i+N, i] * G00.val[j+N, j] 
    + Gll.val[i+N, i] * G00.val[j, j+N] + 
      Gll.val[i, i+N] * G00.val[j+N, j] +
    - Gll.val[i, i+N] * G00.val[j, j+N] + 

    - (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] + 
      (id - G0l.val[j, i])   * Gl0.val[i+N, j+N] +
      (id - G0l.val[j+N, i+N]) * Gl0.val[i, j]   +
    - (0  - G0l.val[j+N, i]) * Gl0.val[i, j+N]
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    return (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[2][i, j] +
           (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[1][i, j]
end


"""
    full_sdc_z_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_z(src, τ) m_z(trg, 0)⟩ with the same definitions for m_z as 
`mz_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_z_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return full_sdc_z_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    # this is an easy one for flavor iterator
      (1 - Gll.val[i, i])     * (1 - G00.val[j, j]) +
    - (1 - Gll.val[i, i])     * (1 - G00.val[j+N, j+N]) +
    - (1 - Gll.val[i+N, i+N]) * (1 - G00.val[j, j]) +
      (1 - Gll.val[i+N, i+N]) * (1 - G00.val[j+N, j+N]) +

      (id - G0l.val[j, i])     * Gl0.val[i, j] +
    - (0  - G0l.val[j+N, i])   * Gl0.val[i, j+N] +
    - (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] +
      (id - G0l.val[j+N, i+N]) * Gl0.val[i+N, j+N]
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

      (1 - Gll.val.blocks[1][i, i]) * (1 - G00.val.blocks[1][j, j]) +
    - (1 - Gll.val.blocks[1][i, i]) * (1 - G00.val.blocks[2][j, j]) +
    - (1 - Gll.val.blocks[2][i, i]) * (1 - G00.val.blocks[1][j, j]) +
      (1 - Gll.val.blocks[2][i, i]) * (1 - G00.val.blocks[2][j, j]) +

      (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[1][i, j] +
      (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[2][i, j]
end


################################################################################
### reduced kernels
################################################################################


"""
    reduced_sdc_x_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_x(src, τ) m_x(trg, 0)⟩ - ⟨m_x(src, τ)⟩⟨m_x(trg, 0)⟩ with the same 
definitions for m_x as `mx_kernel`.
"""
@inline Base.@propagate_inbounds function reduced_sdc_x_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return reduced_sdc_x_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function reduced_sdc_x_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] + 
    (id - G0l.val[j, i])     * Gl0.val[i+N, j+N] +
    (id - G0l.val[j+N, i+N]) * Gl0.val[i, j] + 
    (0  - G0l.val[j+N, i])   * Gl0.val[i, j+N]
end

# The other two methods don't have ⟨mₓ⟩⟨mₓ⟩ terms as mixed-spin sectors are zero 
# in them
@inline Base.@propagate_inbounds function reduced_sdc_x_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4, flvs
    )
    return full_sdc_x_kernel(mc, model, ij, packed_greens, flvs)
end


"""
    reduced_sdc_y_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_y(src, τ) m_y(trg, 0)⟩ - ⟨m_y(src, τ)⟩⟨m_y(trg, 0)⟩ with the same 
definitions for m_y as `my_kernel`.
"""
@inline Base.@propagate_inbounds function reduced_sdc_y_kernel(
        mc, model, ij::NTuple{2}, G::GreensMatrix, flv
    )
    return reduced_sdc_y_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function reduced_sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    - (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] + 
      (id - G0l.val[j, i])   * Gl0.val[i+N, j+N] +
      (id - G0l.val[j+N, i+N]) * Gl0.val[i, j]   +
    - (0  - G0l.val[j+N, i]) * Gl0.val[i, j+N]
end

# Like sdc_x
@inline Base.@propagate_inbounds function reduced_sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4, flvs
    )
    return full_sdc_y_kernel(mc, model, ij, packed_greens, flvs)
end

"""
    reduced_sdc_z_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_z(src, τ) m_z(trg, 0)⟩ - ⟨m_z(src, τ)⟩⟨m_z(trg, 0)⟩ with the same 
definitions for m_z as `mz_kernel`.
"""
@inline Base.@propagate_inbounds function reduced_sdc_z_kernel(
        mc, model, ij::NTuple{2}, G::GreensMatrix, flv
    )
    return reduced_sdc_z_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function reduced_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

      (id - G0l.val[j, i])     * Gl0.val[i, j] +
    - (0  - G0l.val[j+N, i])   * Gl0.val[i, j+N] +
    - (0  - G0l.val[j, i+N])   * Gl0.val[i+N, j] +
      (id - G0l.val[j+N, i+N]) * Gl0.val[i+N, j+N]
end

@inline Base.@propagate_inbounds function reduced_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function reduced_sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

    (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[1][i, j] +
    (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[2][i, j]
end


################################################################################
### Directional kernels
################################################################################


"""
    full_sdc_x_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_x(src, τ) m_x(trg, 0)⟩ with the same definitions for m_x as 
`mx_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, ::Model, sources, dirs, ucs, G::GreensMatrix, flv
    )
    return full_sdc_x_kernel(mc, model, sources, dirs, ucs, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: Matrix}, flv
    )
    N = length(lattice(mc))
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    # on-site but varying spin
    Gll.val[i+N, 1+uc1]   * G00.val[j+N, 1+uc2] + 
    Gll.val[i+N, 1+uc1]   * G00.val[j,   1+uc2+N] + 
    Gll.val[i,   1+uc1+N] * G00.val[j+N, 1+uc2] + 
    Gll.val[i,   1+uc1+N] * G00.val[j,   1+uc2+N] + 

    # shifted by distance + varying spin
    (0  - G0l.val[j,   Δji+N]) * Gl0.val[i+N, Δij] + 
    (id - G0l.val[j,   Δji])   * Gl0.val[i+N, Δij+N] +
    (id - G0l.val[j+N, Δji+N]) * Gl0.val[i,   Δij] + 
    (0  - G0l.val[j+N, Δji])   * Gl0.val[i,   Δij+N]
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    return 2 * (id - G0l.val.val[j, Δji]) * Gl0.val.val[i, Δij]
end

@inline Base.@propagate_inbounds function full_sdc_x_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: BlockDiagonal}, flv
    )

    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    # similarly:
    (id - G0l.val.blocks[1][j, Δji]) * Gl0.val.blocks[2][i, Δij] +
    (id - G0l.val.blocks[2][j, Δji]) * Gl0.val.blocks[1][i, Δij]
end


"""
    full_sdc_y_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_y(src, τ) m_y(trg, 0)⟩ with the same definitions for m_y as 
`my_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, ::Model, sources, dirs, ucs, G::GreensMatrix, flv
    )
    return full_sdc_y_kernel(mc, model, sources, dirs, ucs, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: Matrix}, flv
    )
    N = length(lattice(mc))
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    # same as x, but with some prefactor
    - Gll.val[i+N, 1+uc1]   * G00.val[j+N, 1+uc2] 
    + Gll.val[i+N, 1+uc1]   * G00.val[j,   1+uc2+N] + 
      Gll.val[i,   1+uc1+N] * G00.val[j+N, 1+uc2] +
    - Gll.val[i,   1+uc1+N] * G00.val[j,   1+uc2+N] + 

    - (0  - G0l.val[j,   Δji+N]) * Gl0.val[i+N, Δij] + 
      (id - G0l.val[j,   Δji])   * Gl0.val[i+N, Δij+N] +
      (id - G0l.val[j+N, Δji+N]) * Gl0.val[i,   Δij]   +
    - (0  - G0l.val[j+N, Δji])   * Gl0.val[i,   Δij+N]
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    return 2 * (id - G0l.val.val[j, Δji]) * Gl0.val.val[i, Δij]
end

@inline Base.@propagate_inbounds function full_sdc_y_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    return (id - G0l.val.blocks[1][j, Δji]) * Gl0.val.blocks[2][i, Δij] +
           (id - G0l.val.blocks[2][j, Δji]) * Gl0.val.blocks[1][i, Δij]
end


"""
    full_sdc_z_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨m_z(src, τ) m_z(trg, 0)⟩ with the same definitions for m_z as 
`mz_kernel`.
"""
@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, ::Model, sources, dirs, ucs, G::GreensMatrix, flv
    )
    return full_sdc_z_kernel(mc, model, sources, dirs, ucs, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: Matrix}, flv
    )
    N = length(lattice(mc))
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

    # this is an easy one for flavor iterator
      (1 - Gll.val[i,   1+uc1])   * (1 - G00.val[j,   1+uc2]) +
    - (1 - Gll.val[i,   1+uc1])   * (1 - G00.val[j+N, 1+uc2+N]) +
    - (1 - Gll.val[i+N, 1+uc1+N]) * (1 - G00.val[j,   1+uc2]) +
      (1 - Gll.val[i+N, 1+uc1+N]) * (1 - G00.val[j+N, 1+uc2+N]) +

      (id - G0l.val[j,   Δji])   * Gl0.val[i,   Δij] +
    - (0  - G0l.val[j+N, Δji])   * Gl0.val[i,   Δij+N] +
    - (0  - G0l.val[j,   Δji+N]) * Gl0.val[i+N, Δij] +
      (id - G0l.val[j+N, Δji+N]) * Gl0.val[i+N, Δij+N]
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))
    return 2 * (id - G0l.val.val[j, Δji]) * Gl0.val.val[i, Δij]
end

@inline Base.@propagate_inbounds function full_sdc_z_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))

      (1 - Gll.val.blocks[1][i, 1+uc1]) * (1 - G00.val.blocks[1][j, 1+uc2]) +
    - (1 - Gll.val.blocks[1][i, 1+uc1]) * (1 - G00.val.blocks[2][j, 1+uc2]) +
    - (1 - Gll.val.blocks[2][i, 1+uc1]) * (1 - G00.val.blocks[1][j, 1+uc2]) +
      (1 - Gll.val.blocks[2][i, 1+uc1]) * (1 - G00.val.blocks[2][j, 1+uc2]) +

      (id - G0l.val.blocks[1][j, Δji]) * Gl0.val.blocks[1][i, Δij] +
      (id - G0l.val.blocks[2][j, Δji]) * Gl0.val.blocks[2][i, Δij]
end