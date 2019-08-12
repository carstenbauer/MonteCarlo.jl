using LinearAlgebra, SparseArrays

# State ∈ [0, up, down, updown] = [00, 10, 01, 11]
# 2x2 lattice -> 4 (exponent) -> 4^4 states = 256

function state_from_integer(int, sites=4, substates_per_site=2)
    int > (2substates_per_site)^sites-1 && return
    out = BitArray(undef, (substates_per_site, sites))
    out.chunks[1] = int
    out
end

function create(state, site, substate)
    # No state case
    state == 0 && return 0, 0.0
    # create(|1⟩) -> no state
    state[substate, site] && return 0, 0.0

    # fermionic sign
    lindex = 2(site-1) + substate
    _sign = iseven(sum(state[1:lindex-1])) ? +1.0 : -1.0

    # create(|0⟩) -> |1⟩
    s = copy(state)
    s[substate, site] = true
    return _sign, s
end

function annihilate(state, site, substate)
    # No state
    state == 0 && return 0, 0.0
    # annihilate(|0⟩) -> no state
    !state[substate, site] && return 0, 0.0

    # fermionic sign
    lindex = 2(site-1) + substate
    _sign = iseven(sum(state[1:lindex-1])) ? +1.0 : -1.0

    # annihilate(|1⟩) -> |0⟩
    s = copy(state)
    s[substate, site] = false
    return _sign, s
end



function HamiltonMatrix(model::HubbardModelAttractive)
    lattice = model.l
    t = model.t
    U = model.U
    mu = model.mu

    H = zeros(Float64, 4^lattice.sites, 4^lattice.sites)

    # -t ∑_ijσ c_iσ^† c_jσ
    # +U ∑_i (n_i↑ - 1/2)(n_i↓ - 1/2)
    # -μ ∑_i n_i
    for i in 1:4^lattice.sites
        lstate = state_from_integer(i-1, lattice.sites)
        for j in 1:4^lattice.sites
            rstate = state_from_integer(j-1, lattice.sites)

            E = 0
            # hopping (hermitian conjugate implied/included by lattice generation)
            for substate in [1, 2]
                for source in 1:lattice.sites
                    for target in lattice.neighs[:, source]
                        _sign1, state = annihilate(rstate, source, substate)
                        _sign2, state = create(state, target, substate)
                        if state != 0 && lstate == state
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
    H
end


# Greens function


function Greens(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = create(s, site1, substate1)
        state == 0 && return typeof(s)[], Float64[]
        _sign2, state = annihilate(state, site2, substate2)
        state == 0 && return typeof(s)[], Float64[]
        return [state], [_sign1 * _sign2]
    end
end

# According to p188 QMCM this is equivalent
# can be used to check if Greens/ED is correct
function Greens_permuted(site1, site2, substate1, substate2)
    s -> begin
        _sign1, state = annihilate(s, site2, substate2)
        _sign2, state = create(state, site1, substate1)
        delta = ((site1 == site2) && (substate1 == substate2)) ? 1.0 : 0.0

        if state == 0 && delta == 0.0
            # off-diagonal
            return typeof(s)[], Float64[]
        elseif state == 0
            # only delta function triggers (|0⟩ state)
            return [s], [delta]
        else
            # both trigger (|1⟩ state)
            return [state], [delta - _sign1 * _sign2]
        end
    end
end


function expectation_value(
        observable::Function,
        H;
        T=1.0, beta = 1.0 / T,
        N_sites = 4,
        N_substates = 2
    )

    vals, vecs = eigen(H)
    Z = 0.0
    O = 0.0
    for i in eachindex(vals)
        # exp(βEᵢ)
        temp = exp(-beta * vals[i])
        Z += temp

        # ⟨ψᵢ|Ô|ψᵢ⟩
        right_coefficients = zeros(eltype(vecs), size(vecs, 1))
        for j in 1:size(vecs, 1)
            state = state_from_integer(j-1, N_sites, N_substates)
            states, values = observable(state)
            for (s, v) in zip(states, values)
                # Assuming no (s, v) pair if state destroyed
                k = s.chunks[1]+1
                right_coefficients[k] += v * vecs[j, i]
            end
        end
        O += temp * dot(vecs[:, i], right_coefficients)
    end
    O / Z
end


function calculate_Greens_matrix(H, lattice; beta=1.0, N_substates=2)
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



# utility

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
