function spin_density(
        dqmc, model, dir::Symbol, greens_iterator; 
        lattice_iterator = EachSitePairByDistance(), wrapper = nothing, 
        flavor_iterator = FlavorIterator(dqmc, 0),
        kernel = if dir == :x; sdc_x_kernel
        elseif dir == :y; sdc_y_kernel
        elseif dir == :z; sdc_z_kernel
        else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
        end,
        kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    return Measurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end
spin_density_correlation(args...; kwargs...) = spin_density(args..., Greens(); kwargs...)
spin_density_susceptibility(mc, args...; kwargs...) = spin_density(mc, args..., TimeIntegral(mc); kwargs...)


################################################################################
### Methods
################################################################################


@inline Base.@propagate_inbounds function sdc_x_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return sdc_x_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function sdc_x_kernel(
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

@inline Base.@propagate_inbounds function sdc_x_kernel(
        mc, model, ij::NTuple{2}, 
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flvs
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function sdc_x_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

    # similarly:
    (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[2][i, j] +
    (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[1][i, j]
end



@inline Base.@propagate_inbounds function sdc_y_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return sdc_y_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function sdc_y_kernel(
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

@inline Base.@propagate_inbounds function sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]

    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function sdc_y_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

    return (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[2][i, j] +
           (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[1][i, j]
end



@inline Base.@propagate_inbounds function sdc_z_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv)
    return sdc_z_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function sdc_z_kernel(
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

@inline Base.@propagate_inbounds function sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    return 2 * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function sdc_z_kernel(
        mc, model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    id = I[i, j] * I[G0l.k, G0l.l]

      (1 - Gll.val.blocks[1][i, i]) * (1 - G00.val.blocks[1][j, j]) +
    - (1 - Gll.val.blocks[1][i, i]) * (1 - G00.val.blocks[2][j, j]) +
    - (1 - Gll.val.blocks[2][i, i]) * (1 - G00.val.blocks[1][j, j]) +
      (1 - Gll.val.blocks[2][i, i]) * (1 - G00.val.blocks[2][j, j]) +

      (id - G0l.val.blocks[1][j, i]) * Gl0.val.blocks[1][i, j] +
      (id - G0l.val.blocks[2][j, i]) * Gl0.val.blocks[2][i, j]
end