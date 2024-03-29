# occupation = real diagonal of the Greens function
diagmean(G::GreensMatrix) = diagmean(G.val)
diagmean(G::CMat64) = diagmean(G.re)
diagmean(G::Matrix) = mapreduce(i -> 1 - G[i], +, diagind(G)) / size(G, 1)
diagmean(G::BlockDiagonal) = mapreduce(diagmean, +, G.blocks) / length(G.blocks)



"""
    ChemicalPotentialTuning(target_occupation[; max_dmu = 1.0])

Creates an update that tunes the chemical potential µ of a model to match a 
given `target_occupation` <n>.

For this the statistical derivative `d<n>/dµ` is used to adjust the chemical 
potential. These adjustments run every time the update takes place, so long as 
the simulation is in the thermalization stage. The stength of the adjustment
gets weaker each sweep and the absolute adjustment to the chemical potential is 
bounded by `max_dmu`.

This update assumes `model.mu` to refer to the chemical potential, and for it 
to be a mutable field. Furthermore it assumes `diagmean` to exist for the 
active GreensMatrix type.
"""
mutable struct ChemicalPotentialTuning <: AbstractUtilityUpdate
    target_occupation::Float64

    # Put these here because mus and Ns should be synced and ever changing
    mus::Vector{Float64}
    Ns::Vector{Float64}

    max_dmu::Float64

    idx::Int
end
function ChemicalPotentialTuning(target_occ; max_dmu = 1.0)
    ChemicalPotentialTuning(target_occ, Float64[], Float64[], max_dmu, 1)
end

function init!(mc, u::ChemicalPotentialTuning)
    push!(u.mus, mc.model.mu)
    sizehint!(u.Ns, mc.parameters.thermalization)
    sizehint!(u.mus, mc.parameters.thermalization + 1)
    nothing
end

function can_replace(u1::ChemicalPotentialTuning, u2::ChemicalPotentialTuning)
    (u1.target_occupation == u2.target_occupation) && (u1.mus == u2.mus) && 
    (u1.Ns == u2.Ns) && (u1.idx == u2.idx)
end
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

        new_mu = win_mean_mu + min(u.max_dmu, (u.target_occupation - win_mean_N) / bounded_κ)
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