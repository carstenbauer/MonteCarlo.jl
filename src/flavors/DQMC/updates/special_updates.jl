# mutable struct ChemicalPotentialTuningMeasurement{T} <: AbstractMeasurement
#     N::T
#     count::Int
# end

# function ChemicalPotentialTuningMeasurement(mc)
#     T = greens_eltype(model(mc), field(mc))
#     ChemicalPotentialTuningMeasurement(zero(T), zero(T), 0)
# end

# function reset!(m::ChemicalPotentialTuningMeasurement{T}) where T
#     m.N = m.N2 = zero(T)
#     m.count = 0
#     nothing
# end
# requires(::ChemicalPotentialTuningMeasurement) = (Greens(), nothing)
# @bm function measure!(::Nothing, m::ChemicalPotentialTuningMeasurement, mc::DQMC, model, sweep, G)
#     N = diagmean(G)
#     m.N += N
#     m.N2 += N^2
#     m.count += 1
#     nothing
# end
# For occupations we'll throw away the complex part anyway...
diagmean(G::GreensMatrix) = diagmean(G.val)
diagmean(G::CMat64) = diagmean(G.re)
diagmean(G::Matrix) = mapreduce(i -> 1 - G[i], +, diagind(G)) / size(G, 1)
diagmean(G::BlockDiagonal) = mapreduce(diagmean, +, G.blocks) / length(G.blocks)




mutable struct ChemicalPotentialTuning <: AbstractUpdate
    target_occupation::Float64

    # Put these here because mus and Ns should be synced and ever changing
    mus::Vector{Float64}
    Ns::Vector{Float64}

    idx::Int
end
function ChemicalPotentialTuning(target_occ)
    ChemicalPotentialTuning(target_occ, Float64[], Float64[], 1)
end

function init!(mc, u::ChemicalPotentialTuning)
    push!(u.mus, mc.model.mu)
    sizehint!(u.Ns, mc.parameters.thermalization)
    sizehint!(u.mus, mc.parameters.thermalization + 1)
    nothing
end

should_be_unique(::ChemicalPotentialTuning) = true
generate_key(u::ChemicalPotentialTuning) = hash((u.target_occupation, u.mus, u.Ns, u.idx))
is_full_sweep(::ChemicalPotentialTuning) = false
name(::ChemicalPotentialTuning) = "ChemicalPotentialTuning"

window_mean(vals, range) = mapreduce(i -> vals[i], +, range) / length(range)
window_var(vals, range, mean) = mapreduce(i -> (vals[i] - mean)^2, +, range) / length(range)

function update(u::ChemicalPotentialTuning, mc, model, field)
    # Some notes on this
    # - because our adjustment goes with 1 / κ large κ result in small changes
    # - κ itself tends to be very small
    # - keeping κ large helps reduce extreme adjustements of µ
    # - the proposed κ min and max are entirely larger than κ
    # My thoughts:
    # κ is a derivative based on and at the window average of N and µ. The base
    # update we do `µ = mean(µ) + (target_N - mean(N)) / κ` pushes µ directly to
    # a value that should result in the target_N. But since we're using window 
    # averages this adjustments will take time and we overcompensate.
    # I think it makes sense to scale κ with the window size (in some form) to
    # slow down the adjustments and allow the window averages to catch up. Doing
    # this linearly seems like too much though.

    if mc.last_sweep < mc.parameters.thermalization

        push!(u.Ns, real(diagmean(greens!(mc))))

        window = div(u.idx, 2, RoundUp) : u.idx
        win_mean_mu = window_mean(u.mus, window)
        # win_var_mu = window_var(u.mus, window, win_mean_mu)
        win_mean_N = window_mean(u.Ns, window)
        win_var_N = window_var(u.Ns, window, win_mean_N)

        # κ_min = 1 / sqrt(u.idx) # length(lattice(mc)) / U
        # κ_min = sqrt(u.idx + 1) / 10 # length(lattice(mc)) / U
        # κ_max = sqrt(win_var_N / win_var_mu)
        κ = mc.parameters.beta * win_var_N
        # bounded_κ = clamp(κ, κ_min, κ_max)
        # bounded_κ = max(κ * sqrt(u.idx), 1 / u.idx)
        bounded_κ = max(κ, 1 / u.idx)

        new_mu = win_mean_mu + (u.target_occupation - win_mean_N) / bounded_κ
        push!(u.mus, new_mu)
        model.mu = new_mu
        u.idx += 1

        init_hopping_matrices(mc, model)
        reverse_build_stack(mc, mc.stack)
        propagate(mc)

    elseif u.idx > 1

        win_mean_mu = window_mean(u.mus, div(u.idx, 2, RoundUp) : u.idx)
        push!(u.mus, win_mean_mu)
        model.mu = win_mean_mu
        u.idx = -u.idx

        init_hopping_matrices(mc, model)
        reverse_build_stack(mc, mc.stack)
        propagate(mc)
    end
    return 0
end