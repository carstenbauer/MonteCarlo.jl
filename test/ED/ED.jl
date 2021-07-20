using LinearAlgebra, SparseArrays
using MonteCarlo
using MonteCarlo: @bm, @avx


struct State
    x::UInt16
end

const VOID = State(typemax(UInt16))
const UP = 1
const DOWN = 2

State(i::Integer) = State(UInt16(i))
function Base.getindex(s::State, i)
    mask = UInt16(1) << (i-1)
    s.x & mask == mask
end
function State(s::State, val::Bool, idx)
    mask = UInt16(1) << (idx-1)
    if val
        return State(s.x | mask)
    else
        return State(s.x & (~mask))
    end
end
function Base.show(io::IO, s::State)
    print(io, "State(", bitstring(s.x), ")")
end
function count_bits(s::State, upto=16)
    mask = VOID.x >> (16-upto)
    c = s.x & mask
    count_bits(c)
end
function count_bits(c::Integer)
    count = 0
    while c != 0
        count += 1
        c = c & (c-1)
    end
    count
end

function create(state::State, site, substate)
    lin = 2(site-1) + substate
    (state[lin] || (state == VOID)) && return 0.0, VOID

    n = count_bits(state, lin-1)
    mask = UInt16(1) << (lin-1)
    return 1.0 - 2.0 * (n%2), State(state.x | mask)
end
function annihilate(state, site, substate)
    lin = 2(site-1) + substate
    (!state[lin] || (state == VOID)) && return 0.0, VOID

    n = count_bits(state::State, lin-1)
    mask = UInt16(1) << (lin-1)
    return 1.0 - 2.0 * (n%2), State(state.x & (~mask))
end

Base.:(+)(s::State, x) = s.x + x
Base.iterate(s::State) = s == VOID ? nothing : (s, 1)
Base.iterate(s::State, _) = nothing
Base.eltype(::State) = State



function HamiltonMatrix(model::T) where {T <: HubbardModel}
    lattice = model.l
    t = model.t
    U = T <: HubbardModelAttractive ? -abs(model.U) : abs(model.U)
    mu = T <: HubbardModelAttractive ? model.mu : 0.0

    H = zeros(Float64, 4^length(lattice), 4^length(lattice))

    # -t ∑_ijσ c_iσ^† c_jσ
    # +U ∑_i (n_i↑ - 1/2)(n_i↓ - 1/2)
    # -μ ∑_i n_i
    for i in 1:4^length(lattice)
        lstate = State(i-1)
        for j in 1:4^length(lattice)
            rstate = State(j-1)

            E = 0.0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for substate in [1, 2]
                for (source, target) in neighbors(lattice, Val(true))
                    target == -1 && continue
                    # be careful not to change rstate 
                    # (or restore it with state_from_integer!)
                    _sign1, state = annihilate(rstate, source, substate)    
                    _sign2, state = create(state, target, substate)
                    if _sign1 * _sign2 != 0.0 && lstate == state
                        E -= _sign1 * _sign2 * t
                    end
                end
            end

            # # U, μ terms
            for p in 1:length(lattice) #1:4
                up_occ = rstate[2(p-1)+1]
                down_occ = rstate[2(p-1)+2]
                if lstate == rstate
                    E += U * (up_occ - 0.5) * (down_occ - 0.5)
                    E -= mu * (up_occ + down_occ)
                end
            end
            H[i, j] = E
        end
    end

    # So we don't need to recalculate eigenvalues
    eigen!(H)
end


################################################################################
### equal-time Observables / Expectation values
################################################################################


# Useful two-particle operators
"""
    hopping_operator(state, site1, substate1, site2, substate2)

This is `c^\\dagger_{site2, substate2} c_{site1, substate1} |state>` i.e. 
hopping from 1 to 2.
"""
@inline function _hopping_operator(state, site1, substate1, site2, substate2)
    sign1, _state = annihilate(state, site1, substate1)
    sign2, _state = create(_state, site2, substate2)
    return (sign1*sign2, _state)
end
@inline function _number_operator(state, site, substate)
    _hopping_operator(state, site, substate, site, substate)
end
@inline function _number_operator(state, site)
    p1, state1 = _number_operator(state, site, UP)
    p2, state2 = _number_operator(state, site, DOWN)
    return (p1, p2), (state1, state2)
end




function Greens(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = create(s, site1, substate1)
        # _sign1 == 0. && return 0.0, VOID
        _sign2, state = annihilate(state, site2, substate2)
        # _sign2 == 0. && return 0.0, VOID
        return _sign1 * _sign2, state
    end
end

# According to p188 QMCM this is equivalent
# can be used to check if Greens/ED is correct
function Greens_permuted(site1, site2, substate1, substate2)
    s -> begin
        delta = ((site1 == site2) && (substate1 == substate2)) ? 1.0 : 0.0
        _sign1, state = annihilate(s, site2, substate2)
        # if _sign1 == 0.0
        #     return delta, state
        # end
        if _sign1 == 0.0
            return delta == 0.0 ? (0.0, VOID) : (delta, s)
        end
        _sign2, state = create(state, site1, substate1)

        p = _sign1 * _sign2
        if p == 0.0 && delta == 0.0
            return 0.0, VOID
        else
            return delta - p, state
        end
    end
end

# Charge Density Correlation
function MonteCarlo.charge_density_correlation(site1::Integer, site2::Integer)
    state -> begin
        # states = State[]
        # prefactors = Float64[]
        # for substate2 in [UP, DOWN]
        #     sign1, _state = annihilate(state, site2, substate2)
        #     sign2, _state = create(_state, site2, substate2)
        #     sign1*sign2 == 0 && continue
        #     for substate1 in [UP, DOWN]
        #         sign3, __state = annihilate(_state, site1, substate1)
        #         sign4, __state = create(__state, site1, substate1)
        #         p = sign1 * sign2 * sign3 * sign4
        #         if p != 0.0
        #             push!(prefactors, p)
        #             push!(states, __state)
        #         end
        #     end
        # end
        # prefactors, states

        (_p1, _p2), (_state1, _state2) = _number_operator(state, site2)
        (p1, p2), (state1, state2) = _number_operator(_state1, site1)
        (p3, p4), (state3, state4) = _number_operator(_state2, site1)
        return (_p1*p1, _p1*p2, _p2*p3, _p2*p4), (state1, state2, state3, state4)
    end
end



# Magnetization
function m_x(site)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # _sign1, _state = annihilate(state, site, DOWN)
        # _sign2, _state = create(_state, site, UP)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, p)
        # end
        # _sign1, _state = annihilate(state, site, UP)
        # _sign2, _state = create(_state, site, DOWN)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, p)
        # end
        # return prefactors, states
        sign1, state1 = _hopping_operator(state, site, DOWN, site, UP)
        sign2, state2 = _hopping_operator(state, site, UP, site, DOWN)
        return (sign1, sign2), (state1, state2)
    end
end
function m_y(site)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # _sign1, _state = annihilate(state, site, DOWN)
        # _sign2, _state = create(_state, site, UP)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, p)
        # end
        # _sign1, _state = annihilate(state, site, UP)
        # _sign2, _state = create(_state, site, DOWN)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, -1.0 * p)
        # end
        # return -1im * prefactors, states
        sign1, state1 = _hopping_operator(state, site, DOWN, site, UP)
        sign2, state2 = _hopping_operator(state, site, UP, site, DOWN)
        return -1im .* (sign1, -sign2), (state1, state2)
    end
end
function m_z(site)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # _sign1, _state = annihilate(state, site, UP)
        # _sign2, _state = create(_state, site, UP)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, p)
        # end
        # _sign1, _state = annihilate(state, site, DOWN)
        # _sign2, _state = create(_state, site, DOWN)
        # p = _sign1 * _sign2
        # if p != 0
        #     push!(states, _state)
        #     push!(prefactors, -1.0 * p)
        # end
        # return prefactors, states
        sign1, state1 = _number_operator(state, site, UP)
        sign2, state2 = _number_operator(state, site, DOWN)
        return (sign1, -sign2), (state1, state2)
    end
end


# Spin Density Correlations (s_{x, i} * s_{x, j} etc)
function spin_density_correlation_x(site1, site2)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # for substates2 in [(UP, DOWN), (DOWN, UP)]
        #     sign1, _state = annihilate(state, site2, substates2[1])
        #     sign2, _state = create(_state, site2, substates2[2])
        #     sign1 * sign2 == 0.0 && continue
        #     for substates1 in [(UP, DOWN), (DOWN, UP)]
        #         sign3, __state = annihilate(_state, site1, substates1[1])
        #         sign4, __state = create(__state, site1, substates1[2])
        #         p = sign1 * sign2 * sign3 * sign4
        #         if p != 0.0
        #             push!(prefactors, p)
        #             push!(states, __state)
        #         end
        #     end
        # end
        # prefactors, states
        _p1, _state1 = _hopping_operator(state, site2, UP, site2, DOWN)
        p1, state1 = _hopping_operator(_state1, site1, UP, site1, DOWN)
        p2, state2 = _hopping_operator(_state1, site1, DOWN, site1, UP)
        
        _p2, _state2 = _hopping_operator(state, site2, DOWN, site2, UP)
        p3, state3 = _hopping_operator(_state2, site1, UP, site1, DOWN)
        p4, state4 = _hopping_operator(_state2, site1, DOWN, site1, UP)

        return (_p1*p1, _p1*p2, _p2*p3, _p2*p4), (state1, state2, state3, state4)
    end
end
function spin_density_correlation_y(site1, site2)
    state -> begin
        # states = typeof(state)[]
        # prefactors = ComplexF64[]
        # for substates2 in [(UP, DOWN), (DOWN, UP)]
        #     sign1, _state = annihilate(state, site2, substates2[1])
        #     sign2, _state = create(_state, site2, substates2[2])
        #     sign1*sign2 == 0.0 && continue
        #     for substates1 in [(UP, DOWN), (DOWN, UP)]
        #         # prefactor from the - in s_y
        #         c = substates1 == substates2 ? +1.0 : -1.0
        #         sign3, __state = annihilate(_state, site1, substates1[1])
        #         sign4, __state = create(__state, site1, substates1[2])
        #         p = sign1 * sign2 * sign3 * sign4
        #         if p != 0.0
        #             push!(prefactors, -1.0 * c * p)
        #             push!(states, __state)
        #         end
        #     end
        # end
        # prefactors, states
        _p1, _state1 = _hopping_operator(state, site2, UP, site2, DOWN)
        p1, state1 = _hopping_operator(_state1, site1, UP, site1, DOWN)
        p2, state2 = _hopping_operator(_state1, site1, DOWN, site1, UP)
        
        _p2, _state2 = _hopping_operator(state, site2, DOWN, site2, UP)
        p3, state3 = _hopping_operator(_state2, site1, UP, site1, DOWN)
        p4, state4 = _hopping_operator(_state2, site1, DOWN, site1, UP)

        return (-_p1*p1, _p1*p2, _p2*p3, -_p2*p4), (state1, state2, state3, state4)
    end
end
function spin_density_correlation_z(site1, site2)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # for substates2 in [(UP, UP), (DOWN, DOWN)]
        #     sign1, _state = annihilate(state, site2, substates2[1])
        #     sign2, _state = create(_state, site2, substates2[2])
        #     sign1 * sign2 == 0.0 && continue
        #     for substates1 in [(UP, UP), (DOWN, DOWN)]
        #         # prefactor from the - in s_z
        #         c = substates1 == substates2 ? +1.0 : -1.0
        #         sign3, __state = annihilate(_state, site1, substates1[1])
        #         sign4, __state = create(__state, site1, substates1[2])
        #         p = sign1 * sign2 * sign3 * sign4
        #         if p != 0.0
        #             push!(prefactors, c * p)
        #             push!(states, _state)
        #         end
        #     end
        # end
        # prefactors, states
        _p1, _state1 = _hopping_operator(state, site2, UP, site2, UP)
        p1, state1 = _hopping_operator(_state1, site1, UP, site1, UP) # n_i↑ n_j↑
        p2, state2 = _hopping_operator(_state1, site1, DOWN, site1, DOWN) # n_i↓ n_j↑
        
        _p2, _state2 = _hopping_operator(state, site2, DOWN, site2, DOWN)
        p3, state3 = _hopping_operator(_state2, site1, UP, site1, UP) # n_i↑ n_j↓
        p4, state4 = _hopping_operator(_state2, site1, DOWN, site1, DOWN) # n_i↓ n_j↓

        return (_p1*p1, -_p1*p2, -_p2*p3, _p2*p4), (state1, state2, state3, state4)
    end
end


# local s-wave
# NOTE - this may no longer match the order of other observables
# Δ_i Δ_j^† |ψ⟩
MonteCarlo.pairing_correlation(i::Integer, j::Integer) = pairing_correlation(i, i, j, j)
# general case
# Δ_i = c_{i, ↑} c_{j, ↓}
# Δ_j^† -> Δ_k^† = (c_{k, ↑} c_{l, ↓})^† = c_{l, ↓}^† c_{k, ↑}^†
#  Δ_i Δ_k^† |ψ⟩ = c_{i, ↑} c_{j, ↓} c_{l, ↓}^† c_{k, ↑}^† |ψ⟩
function MonteCarlo.pairing_correlation(i::Integer, j::Integer, k::Integer, l::Integer)
    state -> begin
        sign1, _state = create(state, k, UP)
        sign2, _state = create(_state, l, DOWN)
        sign3, _state = annihilate(_state, j, DOWN)
        sign4, _state = annihilate(_state, i, UP)
        p = sign1 * sign2 * sign3 * sign4
        return (p, _state)
    end
end


# = i \sum\sigma (T[trg, src] c^\dagger(trg,\sigma, \tau) c(src, \sigma, \tau) - T[src, trg] c^\dagger(src, \sigma, \tau) c(trg, \sigma \tau))
function current_density(src, trg, hopping_matrix::AbstractArray)
    state -> begin
        # states = typeof(state)[]
        # prefactors = Float64[]
        # for substate in (UP, DOWN)
        #     # T[trg, src] c^\dagger(trg,\sigma, \tau) c(src, \sigma, \tau)
        #     sign1, _state = annihilate(state, src, substate)
        #     sign2, _state = create(_state, trg, substate)
        #     if sign1*sign2 != 0.0
        #         push!(prefactors, sign1 * sign2 * hopping_matrix[trg, src])
        #         push!(states, _state)
        #     end

        #     # - T[src, trg] c^\dagger(src, \sigma, \tau) c(trg, \sigma \tau)
        #     sign1, _state = annihilate(state, trg, substate)
        #     sign2, _state = create(_state, src, substate)
        #     if sign1*sign2 != 0.0
        #         push!(prefactors, -sign1 * sign2 * hopping_matrix[src, trg])
        #         push!(states, _state)
        #     end
        # end
        # prefactors, states
        h1 = hopping_matrix[trg, src]
        h2 = -hopping_matrix[src, trg]

        p1, state1 = _hopping_operator(state, src, UP, trg, UP)
        p2, state2 = _hopping_operator(state, trg, UP, src, UP)
        p3, state3 = _hopping_operator(state, src, DOWN, trg, DOWN)
        p4, state4 = _hopping_operator(state, trg, DOWN, src, DOWN)

        return (h1*p1, h2*p2, h1*p3, h2*p4), (state1, state2, state3, state4)
    end
end
    


function scalarproduct(lstate::State, values::Vector, states::Vector)
    x = zero(eltype(values))
    for k in eachindex(states)
        x += (states[k] == lstate) * values[k]
        # if states[k] == lstate
        #     x += values[k]
        # end
    end
    x
end
function scalarproduct(lstate::State, values::NTuple{N, T}, states::Tuple) where {N, T}
    x = zero(T)
    for k in eachindex(states)
        x += (states[k] == lstate) * values[k]
        # if states[k] == lstate
        #     x += values[k]
        # end
    end
    x
end
scalarproduct(lstate::State, value, state::State) = (lstate == state) * value


@bm function expectation_value(
        observable::Function,
        H::Eigen;
        T=1.0, beta = 1.0 / T,
        N_sites = 4,
        N_substates = 2
    )
    vals, vecs = H
    Z = 0.0
    O = 0.0
    T = eltype(observable(State(0))[1])
    right_coefficients = zeros(T, size(vecs, 1))

    for i in eachindex(vals)
        # exp(βEᵢ)
        weight = exp(-beta * vals[i])
        Z += weight
        right_coefficients .= zero(eltype(right_coefficients))
        # |phi⟩ = |Ô|ψᵢ⟩
        for j in 1:size(vecs, 1)
            values, states = observable(State(j-1))
            for (l, state) in enumerate(states)
                state == VOID && continue
                right_coefficients[state+1] += values[l] * vecs[j, i]
            end
        end
        # ⟨ψᵢ|phi⟩
        O += weight * dot(vecs[:, i], right_coefficients)
    end

    O / Z
end



function calculate_Greens_matrix(H::Eigen, lattice; beta=1.0, N_substates=2)
    G = Matrix{Float64}(
        undef,
        length(lattice)*N_substates,
        length(lattice)*N_substates
    )
    for substate1 in 1:N_substates, substate2 in 1:N_substates
        for site1 in 1:length(lattice), site2 in 1:length(lattice)
            G[
                length(lattice) * (substate1-1) + site1,
                length(lattice) * (substate2-1) + site2
            ] = expectation_value(
                Greens(site1, site2, substate1, substate2),
                H,
                beta = beta,
                N_sites = length(lattice),
                N_substates=N_substates
            )
        end
    end
    G
end


function energy(H::Eigen; beta = 1.0)
    vals, vecs = H
    Z = 0.0
    O = 0.0

    for i in eachindex(vals)
        # E_i exp(βEᵢ)
        weight = exp(-beta * vals[i])
        Z += weight
        O += vals[i] * weight
    end

    O / Z
end


################################################################################
### unequal time
################################################################################


# ⟨a1(τ1) a2(τ2)⟩ w/ a1(τ1) = e^{τ1H} a1(τ1) e^{-τ2H}
@bm function expectation_value(
        obsτ1::Function, obsτ2::Function, H::Eigen, τ1, τ2;
        T=1.0, beta = 1.0 / T, N_sites = 4, N_substates = 2
    )
    @assert beta ≥ τ1 ≥ τ2 ≥ 0 "Time order must be respected!"
    @bm "init" begin
        O = 0.0
        # vals sorted small to large
        vals, vecs = H
        rstate = State(0)
        vals1, _ = obsτ1(rstate)
        vals2, _ = obsτ2(rstate)
        T = typeof(first(vals1) * first(vals2))
        obsτ1_mat = zeros(T, size(vecs))
        obsτ2_mat = zeros(T, size(vecs))
    end

    @bm "obs mat" begin
        @inbounds for i in eachindex(vals)
            lstate = State(i-1)
            for j in eachindex(vals)
                rstate = State(j-1)
                values, states = obsτ2(rstate)
                obsτ2_mat[i, j] = scalarproduct(lstate, values, states)
                
                rstate = State(j-1)
                values, states = obsτ1(rstate)
                obsτ1_mat[i, j] = scalarproduct(lstate, values, states)
            end
        end
    end

    @bm "prepare O" begin
        # A bit faster with allocations ¯\_(ツ)_/¯
        obsτ2_mat = vecs' * obsτ2_mat * vecs
        obsτ1_mat = vecs' * obsτ1_mat * vecs
        # Should be?
        # obsτ2_mat = vecs * obsτ2_mat * vecs'
        # obsτ1_mat = vecs * obsτ1_mat * vecs'

        Z = mapreduce(E -> exp(-beta * E), +, vals)
    end

    @bm "compute O" begin
        # This seems to run much faster if "prepare O" allocates ¯\_(ツ)_/¯
        # v = obsτ2_mat * exp.(-τ2 * vals)
        # v .*= exp.(-(τ1 - τ2) * vals)
        # w = obsτ1_mat * v
        # O = dot(exp.(-(beta - τ1) * vals), w)

        # Correct for τ1 = τ2
        O = 0.0
        for n in eachindex(vals), m in eachindex(vals)
            O += exp(-(beta-τ1)*vals[n]) * obsτ1_mat[n, m] * 
                 exp(-(τ1-τ2)*vals[m]) * obsτ2_mat[m, n] * exp(-τ2*vals[n])
        end
    end

    # <e^{-(β - τ2) H} c_i exp(-(τ2 - τ1) H) c^\dagger_j exp(-τ1 H)>

    O / Z
end


function calculate_Greens_matrix(H::Eigen, tau1, tau2, lattice; beta=1.0, N_substates=2)
    G = Matrix{Float64}(
        undef,
        length(lattice)*N_substates,
        length(lattice)*N_substates
    )
    # Respect time order:
    swap = tau1 < tau2
    for substate1 in 1:N_substates, substate2 in 1:N_substates
        for site1 in 1:length(lattice), site2 in 1:length(lattice)
            ctau1 = s -> annihilate(s, site2, substate2)
            ctau2 = s -> create(s, site1, substate1)

            G[
                length(lattice) * (substate1-1) + site1,
                length(lattice) * (substate2-1) + site2
            ] = expectation_value(
                swap ? ctau2 : ctau1,
                swap ? ctau1 : ctau2,
                H, swap ? tau2 : tau1, swap ? tau1 : tau2,
                # s -> annihilate!(s, site2, substate2),
                # s -> create!(s, site1, substate1),
                # H, tau1, tau2,
                beta = beta,
                N_sites = length(lattice),
                N_substates = N_substates
            ) * (1 - 2swap) # minus sign from permuting c^†(τ1) c(τ2)
        end
    end
    G
end



@bm function expectation_value_integrated(
        obsτ1::Function, obsτ2::Function, H::Eigen; step = 0.1,
        T=1.0, beta = 1.0 / T, N_sites = 4, N_substates = 2
    )
    @bm "init" begin
        O = 0.0
        vals, vecs = H
        @assert eltype(vecs) <: Real
        rstate = State(0)
        vals1, _ = obsτ1(rstate)
        vals2, _ = obsτ2(rstate)
        T1 = eltype(vals1)
        T2 = eltype(vals2)
        T = (T1 <: Complex || T2 <: Complex) ? ComplexF64 : Float64
        obsτ1_mat = zeros(T, size(vecs))
        obsτ2_mat = zeros(T, size(vecs))
        X = zeros(T, size(vecs))
    end

    @bm "obs mat" begin
        @inbounds for i in eachindex(vals)
            lstate = State(i-1)
            for j in eachindex(vals)
                rstate = State(j-1)
                values, states = obsτ2(rstate)
                obsτ2_mat[i, j] = scalarproduct(lstate, values, states)

                values, states = obsτ1(rstate)
                obsτ1_mat[i, j] = scalarproduct(lstate, values, states)
            end
        end
    end

    @bm "prepare O" begin
        mul!(X, adjoint(vecs), obsτ1_mat)
        mul!(obsτ1_mat, X, vecs)
        
        mul!(X, obsτ2_mat, vecs)
        mul!(obsτ2_mat, transpose(X), vecs)

        X .= obsτ1_mat .* obsτ2_mat

        # obsτ2_mat = transpose(vecs) * obsτ2_mat * vecs
        # obsτ1_mat = transpose(vecs) * obsτ1_mat * vecs
        # X = obsτ1_mat .* transpose(obsτ2_mat)
        Z = mapreduce(E -> exp(-beta * E), +, vals)
    end

    @bm "compute O" begin
        for τ in beta:-step:0.5step #0:step:beta-0.5step
            o = 0.0
            for n in eachindex(vals), m in eachindex(vals)
                o += exp(-(beta-τ)*vals[n] - τ*vals[m]) * X[n, m]
            end
            O += step * o / Z
        end
    end
    
    O
end


number_operator(site::Integer) = state -> _number_operator(state, site)


################################################################################
### utility
################################################################################


function state2string(state)
    sub, L = size(state)
    spin = ["↑", "↓"]
    chunks = String[]
    for p in 1:L
        str = ""
        for s in 1:2
            str *= state[s, p] ? spin[s] : "⋅"
        end
        push!(chunks, str)
    end
    join(chunks, " ")
end
