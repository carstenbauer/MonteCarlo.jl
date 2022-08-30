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
    return Measurement(mc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end


################################################################################
### kernel methods
################################################################################


"""
    mx_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site x-magnetization `⟨cᵢ↑^† cᵢ↓ + cᵢ↓^† cᵢ↑⟩`.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function mx_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return -G.val[i+N, i] - G.val[i, i+N]
end
mx_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
mx_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv) = 0.0


"""
    my_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site y-magnetization `-⟨cᵢ↑^† cᵢ↓ - cᵢ↓^† cᵢ↑⟩` without the 
imaginary prefactor.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function my_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return G.val[i+N, i] - G.val[i, i+N]
end
my_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
my_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv) = 0.0


"""
    mz_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site z-magnetization `⟨nᵢ↑ - nᵢ↓⟩`.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function mz_kernel(mc, model, i, G::_GM{<: Matrix}, flv)
    N = length(lattice(model))
    return G.val[i+N, i+N] - G.val[i, i]
end
mz_kernel(mc, model, i, G::_GM{<: DiagonallyRepeatingMatrix}, flv) = 0.0
function mz_kernel(mc, model, i, G::_GM{<: BlockDiagonal}, flv)
    return G.val.blocks[2][i, i] - G.val.blocks[1][i, i]
end