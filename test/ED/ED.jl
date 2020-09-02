using LinearAlgebra, SparseArrays
using MonteCarlo
using MonteCarlo: @bm

# State ∈ [0, up, down, updown] = [00, 10, 01, 11]
# 2x2 lattice -> 4 (exponent) -> 4^4 states = 256
const UP = 1
const DOWN = 2


# TODO: Maybe eventually allow more chunks?
function state_from_integer(int, sites=4, substates_per_site=2)
    @assert int <= (2substates_per_site)^sites-1
    out = BitArray(undef, (substates_per_site, sites))
    out.chunks[1] = int
    out
end
function state_from_integer!(input, int)
    @inbounds Base.setindex!(input.chunks, int, 1)
    input
end
integer_from_state(input) = input.chunks[1]


# NOTE: These output `0.0, state` if the state should be destroyed.
create(state, site, substate) = create!(copy(state), site, substate)
function create!(state, site, substate)
    # create(|1⟩) -> no state
    state[substate, site] && return 0.0, state

    # fermionic sign
    lindex = 2(site-1) + substate
    t = 0
    @inbounds for i in 1:lindex-1
        t += state[i]
    end
    _sign = 1.0 - 2.0 * (t%2)

    # create(|0⟩) -> |1⟩
    state[substate, site] = true
    return _sign, state
end

annihilate(state, site, substate) = annihilate!(copy(state), site, substate)
function annihilate!(state, site, substate)
    # annihilate(|0⟩) -> no state
    !state[substate, site] && return 0.0, state

    # fermionic sign
    lindex = 2(site-1) + substate
    t = 0
    @inbounds for i in 1:lindex-1
        t += state[i]
    end
    _sign = 1.0 - 2.0 * (t%2)

    # annihilate(|1⟩) -> |0⟩
    state[substate, site] = false
    return _sign, state
end



function HamiltonMatrix(model::HubbardModelAttractive)
    lattice = model.l
    t = model.t
    U = -abs(model.U)
    mu = model.mu

    H = zeros(Float64, 4^lattice.sites, 4^lattice.sites)

    lstate = state_from_integer(0, lattice.sites)
    rstate = state_from_integer(0, lattice.sites)

    # -t ∑_ijσ c_iσ^† c_jσ
    # +U ∑_i (n_i↑ - 1/2)(n_i↓ - 1/2)
    # -μ ∑_i n_i
    for i in 1:4^lattice.sites
        state_from_integer!(lstate, i-1)
        for j in 1:4^lattice.sites
            state_from_integer!(rstate, j-1)

            E = 0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for substate in [1, 2]
                for source in 1:lattice.sites
                    for target in lattice.neighs[:, source]
                        # be careful not to change rstate 
                        # (or restore it with state_from_integer!)
                        _sign1, state = annihilate(rstate, source, substate)
                        _sign2, state = create!(state, target, substate)
                        if _sign1 * _sign2 != 0.0 && lstate == state
                            E -= _sign1 * _sign2 * t
                        end
                    end
                end
            end

            # # U, μ terms
            for p in 1:4
                up_occ = rstate[1, p]
                down_occ = rstate[2, p]
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


function Greens(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = create!(s, site1, substate1)
        _sign1 == 0. && return Float64[], typeof(s)[]
        _sign2, state = annihilate!(state, site2, substate2)
        _sign2 == 0. && return Float64[], typeof(s)[]
        return [_sign1 * _sign2], [state]
    end
end

# According to p188 QMCM this is equivalent
# can be used to check if Greens/ED is correct
function Greens_permuted(site1, site2, substate1, substate2)
    s -> begin
        delta = ((site1 == site2) && (substate1 == substate2)) ? 1.0 : 0.0
        _sign1, state = annihilate!(s, site2, substate2)
        if _sign1 == 0.0
            return delta == 0.0 ? (Float64[], typeof(s)[]) : ([delta], [state])
        end
        _sign2, state = create!(state, site1, substate1)

        p = _sign1 * _sign2
        if p == 0.0 && delta == 0.0
            return Float64[], typeof(s)[]
        else
            return [delta - p], [state]
        end
    end
end

# Charge Density Correlation
function charge_density_correlation(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substate1 in [UP, DOWN]
            for substate2 in [UP, DOWN]
                sign1, _state = annihilate(state, site2, substate2)
                sign2, _state = create!(_state, site2, substate2)
                sign3, _state = annihilate!(_state, site1, substate1)
                sign4, _state = create!(_state, site1, substate1)
                p = sign1 * sign2 * sign3 * sign4
                if p != 0.0
                    push!(prefactors, p)
                    push!(states, _state)
                end
            end
        end
        prefactors, states
    end
end



# Magnetization
function m_x(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create!(_state, site, UP)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, p)
        end
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create!(_state, site, DOWN)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, p)
        end
        return prefactors, states
    end
end
function m_y(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create!(_state, site, UP)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, p)
        end
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create!(_state, site, DOWN)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, -1.0 * p)
        end
        return -1im * prefactors, states
    end
end
function m_z(site)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        _sign1, _state = annihilate(state, site, UP)
        _sign2, _state = create!(_state, site, UP)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, p)
        end
        _sign1, _state = annihilate(state, site, DOWN)
        _sign2, _state = create!(_state, site, DOWN)
        p = _sign1 * _sign2
        if p != 0
            push!(states, _state)
            push!(prefactors, -1.0 * p)
        end
        return prefactors, states
    end
end


# Spin Density Correlations (s_{x, i} * s_{x, j} etc)
function spin_density_correlation_x(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substates1 in [(UP, DOWN), (DOWN, UP)]
            for substates2 in [(UP, DOWN), (DOWN, UP)]
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create!(_state, site2, substates2[2])
                sign3, _state = annihilate!(_state, site1, substates1[1])
                sign4, _state = create!(_state, site1, substates1[2])
                p = sign1 * sign2 * sign3 * sign4
                if p != 0.0
                    push!(prefactors, p)
                    push!(states, _state)
                end
            end
        end
        prefactors, states
    end
end
function spin_density_correlation_y(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = ComplexF64[]
        for substates1 in [(UP, DOWN), (DOWN, UP)]
            for substates2 in [(UP, DOWN), (DOWN, UP)]
                # prefactor from the - in s_y
                c = substates1 == substates2 ? +1.0 : -1.0
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create!(_state, site2, substates2[2])
                sign3, _state = annihilate!(_state, site1, substates1[1])
                sign4, _state = create!(_state, site1, substates1[2])
                p = sign1 * sign2 * sign3 * sign4
                if p != 0.0
                    push!(prefactors, -1.0 * c * p)
                    push!(states, _state)
                end
            end
        end
        prefactors, states
    end
end
function spin_density_correlation_z(site1, site2)
    state -> begin
        states = typeof(state)[]
        prefactors = Float64[]
        for substates1 in [(UP, UP), (DOWN, DOWN)]
            for substates2 in [(UP, UP), (DOWN, DOWN)]
                # prefactor from the - in s_z
                c = substates1 == substates2 ? +1.0 : -1.0
                sign1, _state = annihilate(state, site2, substates2[1])
                sign2, _state = create!(_state, site2, substates2[2])
                sign3, _state = annihilate!(_state, site1, substates1[1])
                sign4, _state = create!(_state, site1, substates1[2])
                p = sign1 * sign2 * sign3 * sign4
                if _state != 0.0
                    push!(prefactors, c * p)
                    push!(states, _state)
                end
            end
        end
        prefactors, states
    end
end


# local s-wave
pairing_correlation(site1, site2) = pairing_correlation(site1, site1, site2, site2)
# general case
function pairing_correlation(site1, site2, site3, site4)
    state -> begin
        sign1, _state = create!(state, site1, UP)
        sign2, _state = create!(_state, site2, DOWN)
        sign3, _state = annihilate!(_state, site3, DOWN)
        sign4, _state = annihilate!(_state, site4, UP)
        p = sign1 * sign2 * sign3 * sign4
        if p == 0
            return Float64[], typeof(state)[]
        else
            return [p], [_state]
        end
    end
end


function scalarproduct(lstate::BitArray, values::Vector, states::Vector)
    x = zero(eltype(values))
    for k in eachindex(states)
        if states[k] == lstate
            x += values[k]
        end
    end
    x
end
scalarproduct(lstate::BitArray, value, state::BitArray) = (lstate == state) * value


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
    state = state_from_integer(0, N_sites, N_substates)

    right_coefficients = zeros(ComplexF64, size(vecs, 1))

    for i in eachindex(vals)
        # exp(βEᵢ)
        temp = exp(-beta * vals[i])
        Z += temp
        right_coefficients .= zero(eltype(right_coefficients))

        # ⟨ψᵢ|Ô|ψᵢ⟩
        for j in 1:size(vecs, 1)
            state_from_integer!(state, j-1)
            values, states = observable(state)
            for l in eachindex(values)
                # Assuming no (s, v) pair if state destroyed
                k = states[l].chunks[1]+1
                right_coefficients[k] += values[l] * vecs[j, i]
            end
        end
        O += temp * dot(vecs[:, i], right_coefficients)
    end
    O / Z
end



function calculate_Greens_matrix(H::Eigen, lattice; beta=1.0, N_substates=2)
    G = Matrix{Float64}(
        undef,
        lattice.sites*N_substates,
        lattice.sites*N_substates
    )
    for substate1 in 1:N_substates, substate2 in 1:N_substates
        for site1 in 1:lattice.sites, site2 in 1:lattice.sites
            G[
                lattice.sites * (substate1-1) + site1,
                lattice.sites * (substate2-1) + site2
            ] = expectation_value(
                Greens(site1, site2, substate1, substate2),
                H,
                beta = beta,
                N_sites = lattice.sites,
                N_substates=N_substates
            )
        end
    end
    G
end


################################################################################
### unequal time
################################################################################



@bm function expectation_value(
        obsτ2::Function, obsτ1::Function, H::Eigen, τ2, τ1;
        T=1.0, beta = 1.0 / T, N_sites = 4, N_substates = 2
    )
    @bm "init" begin
        O = 0.0
        # vals sorted small to large
        vals, vecs = H
        lstate = state_from_integer(0, N_sites)
        rstate = state_from_integer(0, N_sites)
        vals1, _ = obsτ1(rstate)
        vals2, _ = obsτ2(rstate)
        T = typeof(first(vals1) * first(vals2))
        obsτ1_mat = zeros(T, size(vecs))
        obsτ2_mat = zeros(T, size(vecs))
    end

    @bm "obs mat" begin
        @inbounds for i in eachindex(vals)
            state_from_integer!(lstate, i-1)
            for j in eachindex(vals)
                state_from_integer!(rstate, j-1)
                values, states = obsτ2(rstate)
                obsτ2_mat[i, j] = scalarproduct(lstate, values, states)

                state_from_integer!(rstate, j-1)
                values, states = obsτ1(rstate)
                obsτ1_mat[i, j] = scalarproduct(lstate, values, states)
            end
        end
    end

    @bm "prepare O" begin
        # A bit faster with allocations ¯\_(ツ)_/¯
        obsτ2_mat = vecs' * obsτ2_mat * vecs
        obsτ1_mat = vecs' * obsτ1_mat * vecs

        Z = mapreduce(E -> exp(-beta * E), +, vals)
    end

    @bm "compute O" begin
        # This seems to run much faster if "prepare O" allocates ¯\_(ツ)_/¯
        v = obsτ1_mat * exp.(-τ1 * vals)
        v .*= exp.(-(τ2 - τ1) * vals)
        w = obsτ2_mat * v
        O = dot(exp.(-(beta - τ2) * vals), w)
    end

    # <e^{-(β - τ2) H} c_i exp(-(τ2 - τ1) H) c^\dagger_j exp(-τ1 H)>

    O / Z
end


function calculate_Greens_matrix(H::Eigen, tau2, tau1, lattice; beta=1.0, N_substates=2)
    G = Matrix{Float64}(
        undef,
        lattice.sites*N_substates,
        lattice.sites*N_substates
    )
    for substate1 in 1:N_substates, substate2 in 1:N_substates
        for site1 in 1:lattice.sites, site2 in 1:lattice.sites
            G[
                lattice.sites * (substate1-1) + site1,
                lattice.sites * (substate2-1) + site2
            ] = expectation_value(
                s -> annihilate!(s, site2, substate2),
                s -> create!(s, site1, substate1),
                H, tau2, tau1,
                beta = beta,
                N_sites = lattice.sites,
                N_substates = N_substates
            )
        end
    end
    G
end



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
