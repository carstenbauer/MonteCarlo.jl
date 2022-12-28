"""
    build_checkerboard(l::Lattice)

Generates a collection of groups containing (src, trg) pairs where each group 
only touches each site (src or trg) once. Each group also represents only one 
bond direction, though multiple groups can represent the same direction. Bond 
reversals (i.e. trg -> src relative to src -> trg) are not considered here, as 
they are always present.

Returns groups in a nested structure `groups[idx1][idx2][idx3]`, where `idx1`
picks the bond direction, `idx2` a subset with non-overlapping sites and `idx3`
the individual site pairs involved in a bond.
"""
function build_checkerboard(l::AbstractLattice)
    # TODO
    # What's keeping this from generating staggered bonds
    # - - -
    #  - - -
    # - - - 
    # etc?
    # The setup of doing directions first?

    # Unit cell bonds can match up with the Bravais lattice size to create 
    # duplicate pairs. For example a bond with uc_shift = (2, 0) on a L = 4 
    # lattice can generate src = (2, 0) -> trg = (4, 0) and it's reverse 
    # (2, 0) -> (0, 0) = (4, 0).
    # Since we later use the pairs generated here to read out the hopping 
    # matrix we don't want these duplicates to exist. Thus we need to explicitly
    # filter them out here.

    # Generate groups based on bond direction. E.g. an x-directed and a y-directed 
    # group for a square lattice.
    bs = view(unitcell(l).bonds, unitcell(l)._directed_indices)

    base_groups = map(bs) do uc_bond
        group = map(eachindex(Bravais(l))) do src
            global_bond = MonteCarlo._shift_Bravais(l, src, uc_bond)
            MonteCarlo.from(global_bond) => MonteCarlo.to(global_bond)
        end

        # remove dublicates
        i = 1
        while i < length(group)
            j = i+1
            src, trg = group[i]
            while j <= length(group)
                if (Pair(src, trg) == group[j]) || (Pair(trg, src) == group[j])
                    deleteat!(group, j)
                else
                    j += 1
                end
            end
            i += 1
        end

        return group
    end

    # avoid duplicates across groups
    for i in eachindex(base_groups)
        for j in i+1:length(base_groups)
            filter!(base_groups[j]) do (src, trg)
               !((src, trg) in base_groups[i]) && !((trg, src) in base_groups[i])
            end
        end
    end

    # split groups like [(i, j), (j, k), (k, l), ...]
    # into two groups [(i, j), (k, l), ...], [(j, k), (l, ...), ...]
    # for uneven lattices we need a third group. 
    # For example 3 sites (i, j) (j, k), (k, i) need to split into 3 groups
    groups = [Vector{Pair{Int, Int}}[] for _ in base_groups]
    for (idx, base_group) in enumerate(base_groups)
        src, trg = base_group[1]
        if any(t -> t[1] == trg, base_group)
            used1 = Int[]
            used2 = Int[]
            used3 = Int[]
            group1 = Pair{Int, Int}[]
            group2 = Pair{Int, Int}[]
            group3 = Pair{Int, Int}[]
            for (src, trg) in base_group
                if !(src in used1) && !(trg in used1)
                    push!(group1, src => trg)
                    push!(used1, src, trg)
                elseif !(src in used2) && !(trg in used2)
                    push!(group2, src => trg)
                    push!(used2, src, trg)
                elseif !(src in used3) && !(trg in used3)
                    push!(group3, src => trg)
                    push!(used3, src, trg)
                else
                    error("Failed to distribute bonds into groups.")
                end
            end
            push!(groups[idx], group1)
            isempty(group2) || push!(groups[idx], group2)
            isempty(group3) || push!(groups[idx], group3)
        else
            push!(groups[idx], base_group)
        end
    end

    # Debug/Crosscheck for different group configuration sin 8x8 Square lattice
    # default is no stagger X X Y Y

    # X Y X Y, no stagger (irrelevant)
    # groups = [[groups[1][1], groups[2][1]], [groups[1][2], groups[2][2]]]

    # X X Y Y all staggered
    # groups = [[
    #         vcat((groups[1][mod1(i, 2)][4(i-1)+1 : 4i] for i in 1:8)...),
    #         vcat((groups[1][mod1(i+1, 2)][4(i-1)+1 : 4i] for i in 1:8)...),
    #     ], #groups[2]
    #     [
    #         vcat((groups[2][mod1(i, 2)][i] for i in 1:32)...),
    #         vcat((groups[2][mod1(i+1, 2)][i] for i in 1:32)...),
    #     ]
    # ]
    
    # Stagger only Y Y
    # groups = [
    #     groups[1],
    #     [
    #         vcat((groups[2][mod1(i, 2)][i] for i in 1:32)...),
    #         vcat((groups[2][mod1(i+1, 2)][i] for i in 1:32)...),
    #     ],
    # ]
    
    # Stagger only X X
    # groups = [
    #     [
    #         vcat((groups[1][mod1(i, 2)][4(i-1)+1 : 4i] for i in 1:8)...),
    #         vcat((groups[1][mod1(i+1, 2)][4(i-1)+1 : 4i] for i in 1:8)...),
    #     ],
    #     groups[2]
    # ]

    return groups
end



"""
    SparseMatrix(vals, is, js)

Reduced Representation of M[i, j] with M[j, i] implied in multiplications
"""
struct SparseCBMatrix{T} <: AbstractMatrix{T}
    # switching this to a single value seems oddly irrelevant... maybe test again later
    diag::Diagonal{Float64, Vector{Float64}}
    vals::Vector{T}
    is::Vector{Int}
    js::Vector{Int}
end

# printing
function Base.display(x::SparseCBMatrix{T}) where T
    println(stdout, "SparseCBMatrix{$T}:")
    println(stdout, "    diag = $(x.diag.diag)")
    for (i, j, v) in zip(x.is, x.js, x.vals)
        println(stdout, "    [$i, $j], [$j, $i] -> $v")
    end
    return
end

Base.show(io::IO, x::SparseCBMatrix) = show(io, MIME"text/plain"(), x)
function Base.show(io::IO, ::MIME"text/plain", x::SparseCBMatrix{T}) where T
    print(io, "SparseCBMatrix{$T} with $(length(x.vals)) elements (mime)")
end

function Base.Matrix(x::SparseCBMatrix{T}) where T
    N = length(x.diag.diag)
    output = Matrix{T}(undef, N, N)
    copyto!(output, x.diag)
    for idx in eachindex(x.is)
        output[x.is[idx], x.js[idx]] = x.vals[idx]
        output[x.js[idx], x.is[idx]] = x.vals[idx]
    end
    return output
end

Base.size(x::SparseCBMatrix) = size(x.diag)



struct CheckerboardDecomposed{T} <: AbstractMatrix{T}
    diag::Diagonal{T, Vector{T}}
    parts::Vector{SparseCBMatrix{T}}
end

# printing
function Base.display(x::CheckerboardDecomposed{T}) where T
    println(stdout, "CheckerboardDecomposed{$T} with 1+$(length(x.parts)) parts representing:")
    display(Matrix(x))
    return
end

function Base.Matrix(x::CheckerboardDecomposed{T}) where T
    N = length(x.diag.diag)
    output = Matrix{T}(undef, N, N)
    id = Matrix{T}(I, N, N)
    tmp = similar(output)
    vmul!(output, id, x, tmp)
    return output
end

Base.size(x::CheckerboardDecomposed) = (length(x.diag), length(x.diag))


function CheckerboardDecomposed(M::Matrix, lattice, factor; symmetric_trotter = true)
    N = length(lattice)
    flv = div(size(M, 1), N)
    groups = build_checkerboard(lattice)

    # This version assumes src -> trg to imply trg -> src to exist in T
    # exp(T) then becomes cosh(abs(T[src, trg])) on the diagonal
    # and sinh(abs(T[src, trg])) * (cos(f) + isin(f)) on the src/trg spaces
    # to avoid some multiplication we pull out the cosh (diag just becomes copy)
    # Actually this only works if we hit every index, i.e. not for odd lattice sizes

    D = Diagonal(exp.(factor * diag(M)))
    parts = SparseCBMatrix{eltype(M)}[]

    for bond_groups in groups
        temp_parts = SparseCBMatrix{eltype(M)}[]
        for group in bond_groups
            # through the magic of math and sparsity, exp(temp) = I + temp
            # So we introduce a type for it
            diag = ones(length(D.diag))
            vals = Vector{eltype(M)}(undef, flv*flv*length(group))
            is = Vector{Int}(undef, flv*flv*length(group))
            js = Vector{Int}(undef, flv*flv*length(group))
            idx = 1

            for (src, trg) in group
                for f1 in 0:N:(flv-1)*N, f2 in 0:N:(flv-1)*N
                    # transform should now just do -0.5 * dtau * temp
                    v = factor * M[src+f1, trg+f2]

                    diag[src+f1] = diag[trg+f2] = cosh(v)
                    vals[idx] = sinh(v)
                    is[idx] = src + f1
                    js[idx] = trg + f2
                    idx += 1

                    # pre = abs(v)
                    # _re = real(v/pre)
                    # _im = imag(v/pre)
                    # c = cosh(v) # or abs
                    # x[src + f1] *= c
                    # x[trg + f2] *= c
                    # vals[idx] = tanh(v)
                    # is[idx] = src + f1
                    # js[idx] = trg + f2
                    # idx += 1
                end
            end

            # hopefully this helps with cache coherence?
            _sort_by_ij!(vals, is, js)

            push!(temp_parts, SparseCBMatrix(Diagonal(diag), vals, is, js))
        end
        append!(parts, temp_parts)
    end

    # D.diag .*= x

    # TODO: 
    # this only works with constant diagonal part
    # otherwise we need to be careful about multiplication order of D
    # i.e. for inverted matrices the order needs to swap
    # I have also not checked if the current order is correct...
    if factor > 0.0 # inverted
        reverse!(parts)
    end

    return CheckerboardDecomposed(D, parts)
end

function _sort_by_ij!(vals, is, js)
    perm = sortperm(js)
    permute!(vals, perm)
    permute!(is, perm)
    permute!(js, perm)

    perm = sortperm(is)
    permute!(vals, perm)
    permute!(is, perm)
    permute!(js, perm)
    return
end

# Note
# getindex and Matrix() are not well defined... Depending on the transform we 
# used we may need to add or multiple the internal matrices

function vmul!(output::Matrix{T}, S::SparseCBMatrix{T}, M::Matrix{T}) where T
    # O(N^2 + N^2)
    # I * M
    # copyto!(output, M)
    vmul!(output, S.diag, M)

    # O(N^2)
    # P * M
    @inbounds @fastmath for k in axes(M, 2)
        @simd for n in eachindex(S.is)
            i = S.is[n]; j = S.js[n]
            output[i, k] = muladd(S.vals[n], M[j, k], output[i, k])
            output[j, k] = muladd(conj(S.vals[n]), M[i, k], output[j, k])
        end
    end
    return output
end


function vmul!(output::Matrix{T}, M::Matrix{T}, S::SparseCBMatrix{T}) where T
    # copyto!(output, M)
    vmul!(output, M, S.diag)

    @inbounds for n in eachindex(S.is)
        j = S.is[n]; k = S.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], S.vals[n], output[i, k])
            output[i, j] = muladd(M[i, k], conj(S.vals[n]), output[i, j])
        end
    end

    return M
end

function vmulc!(output::Matrix{T}, M::Matrix{T}, S::SparseCBMatrix{T}) where T
    # copyto!(output, M)
    vmul!(output, M, S.diag)

    @inbounds for n in eachindex(S.is)
        j = S.is[n]; k = S.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], conj(S.vals[n]), output[i, k])
            output[i, j] = muladd(M[i, k], S.vals[n], output[i, j])
        end
    end

    return M
end

function vmul!(output::Matrix{T}, M::Matrix{T}, S::Transpose{T, SparseCBMatrix{T}}) where T
    # copyto!(output, M)
    vmul!(output, M, S.parent.diag)

    @inbounds for n in eachindex(S.parent.is)
        k = S.parent.is[n]; j = S.parent.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], S.parent.vals[n], output[i, k])
            output[i, j] = muladd(M[i, k], conj(S.parent.vals[n]), output[i, j])
        end
    end

    return M
end




function vmul!(trg::Matrix{T}, src::Matrix{T}, cb::CheckerboardDecomposed{T}, tmp::Matrix{T}) where T
    #M P1 ⋯ PN D

    # N parts mults, diag mult inline
    if iseven(length(cb.parts))
        tmp_trg = trg
        tmp_src = tmp
    else # odd - same tmp_trg as above after vmul!
        tmp_trg = tmp
        tmp_src = trg
    end

    vmul!(tmp_src, src, cb.parts[1])
    
    @inbounds for i in 2:length(cb.parts)
        vmul!(tmp_trg, tmp_src, cb.parts[i])
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end

    rvmul!(tmp_src, cb.diag)
    @assert tmp_src === trg

    return trg
end


# What's better than A*B? That'S right, it's (B^T A^T)^T because 
# LoopVectorization is just that good

vmul!(trg::AbstractArray, A::AbstractArray, B::AbstractArray, tmp::AbstractArray) = vmul!(trg, A, B)
function vmul!(trg::Matrix{T}, cb::CheckerboardDecomposed{T}, src::Matrix{T}, tmp::Matrix{T}) where T
    # # PN ⋯ P1 D M = PN ⋯ P1 M D = ((D M)' P1' ⋯ PN')'
    # P1 ⋯ PN D M = P1 ⋯ PN M D = ((D M)' PN' ⋯ P1')'
    if iseven(length(cb.parts))
        vmul!(trg, cb.diag, src)
        tmp_src = trg
        tmp_trg = tmp
    else
        vmul!(tmp, cb.diag, src)
        tmp_src = tmp
        tmp_trg = trg
    end

    # transpose
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    x = tmp_trg
    tmp_trg = tmp_src
    tmp_src = x

    for i in length(cb.parts):-1:1
        vmul!(tmp_trg, tmp_src, transpose(cb.parts[i]))
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end
    
    # tranpose
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    @assert tmp_trg === trg "$(length(cb.parts))"

    return trg
end

function vmul!(trg::Matrix{T}, cb::Adjoint{T, <: CheckerboardDecomposed}, src::Matrix{T}, tmp::Matrix{T}) where {T <: Real}
    # (P1 ⋯ PN D)' M = D' PN' ⋯ P1' M = D' (M' P1 ⋯ PN)'

    if iseven(length(cb.parent.parts))
        tmp_trg = trg
        tmp_src = tmp
    else
        tmp_trg = tmp
        tmp_src = trg
    end

    # transpose
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = src[j, i]
    end

    x = tmp_trg
    tmp_trg = tmp_src
    tmp_src = x

    for P in cb.parent.parts
        vmulc!(tmp_trg, tmp_src, P)
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end

    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    vmul!(tmp_src, adjoint(cb.parent.diag), tmp_trg)

    @assert tmp_src === trg
    
    return trg
end
