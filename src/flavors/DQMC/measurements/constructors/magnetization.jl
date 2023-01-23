"""
    magnetization(mc, model, dir; kwargs...)

Generates a measurements of magnetization in `:x`, `:y` or `:z` direction 
depending on `dir`,

## Optional Keyword Arguments

- `kernel = mx_kernel/my_kernel/my_kernel` sets the function representing the 
Wicks expanded expectation value of the measurement. In this case the kernel 
depends on the choice of `dir`. See the the individual kernels for more information.
- `lattice_iterator = EachSite()` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `flavor_iterator = FlavorIterator(mc, 0)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
function magnetization(
        mc::DQMC, model::Model, dir::Symbol; 
        greens_iterator = Greens(),
        lattice_iterator = EachSite(), wrapper = nothing, 
        flavor_iterator = FlavorIterator(mc, 0),
        kernel = if dir == :x; mx_kernel
        elseif dir == :y; my_kernel
        elseif dir == :z; mz_kernel
        else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
        end,
        kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    return DQMCMeasurement(mc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end


################################################################################
### kernel methods
################################################################################


"""
    mx_kernel(mc, model, i::Integer, G::GreensMatrix, flavor_indices)

Returns the per-site x-magnetization `⟨cᵢ↑^† cᵢ↓ + cᵢ↓^† cᵢ↑⟩`. 

Note that flavor/spin indices are handlded in the kernel, i.e. flavors should 
not be iterated over.
"""
@inline Base.@propagate_inbounds function mx_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return -G.val[i+N, i] - G.val[i, i+N]
end
@inline Base.@propagate_inbounds mx_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
@inline Base.@propagate_inbounds mx_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv) = 0.0


"""
    my_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site y-magnetization `-⟨cᵢ↑^† cᵢ↓ - cᵢ↓^† cᵢ↑⟩` without the 
imaginary prefactor.

Note that flavor/spin indices are handlded in the kernel, i.e. flavors should 
not be iterated over.
"""
@inline Base.@propagate_inbounds function my_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return G.val[i+N, i] - G.val[i, i+N]
end
@inline Base.@propagate_inbounds my_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
@inline Base.@propagate_inbounds my_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv) = 0.0


"""
    mz_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site z-magnetization `⟨nᵢ↑ - nᵢ↓⟩`.

Note that flavor/spin indices are handlded in the kernel, i.e. flavors should 
not be iterated over.
"""
@inline Base.@propagate_inbounds function mz_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return G.val[i+N, i+N] - G.val[i, i]
end
@inline Base.@propagate_inbounds mz_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
@inline Base.@propagate_inbounds function mz_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv)
    return G.val.blocks[2][i, i] - G.val.blocks[1][i, i]
end