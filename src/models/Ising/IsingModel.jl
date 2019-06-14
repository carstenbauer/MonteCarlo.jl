const IsingSpin = Int8 # can't use something more efficient here because of bug in MonteCarloObservable (see #10 on gitsrv)
const IsingDistribution = IsingSpin[-1,1]
const IsingConf = Array{IsingSpin}

const IsingTc = 1/(1/2*log(1+sqrt(2)))

"""
Famous Ising model on a cubic lattice.

    IsingModel(; dims, L)

Create Ising model on `dims`-dimensional cubic lattice
with linear system size `L`.
"""
@with_kw_noshow mutable struct IsingModel{C<:AbstractCubicLattice} <: Model # noshow because we override it below
    L::Int
    dims::Int
    l::C = choose_lattice(IsingModel, dims, L)
end

function choose_lattice(::Type{IsingModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    IsingModel(kwargs::Dict{String, Any})

Create Ising model with (keyword) parameters as specified in `kwargs` dict.
"""
IsingModel(kwargs::Union{Dict{String, Any}, Dict{Symbol, Any}}) =
            IsingModel(; convert(Dict{Symbol,Any}, kwargs)...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::IsingModel) = "$(model.dims)D-Ising model"
Base.show(io::IO, model::IsingModel) = print(io, "$(model.dims)D-Ising model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::IsingModel) = print(io, model)


# methods for using it with Monte Carlo flavor MC (Monte Carlo)
"""
    energy(mc::MC, m::IsingModel, conf::IsingConf)

Calculate energy of Ising configuration `conf` for Ising model `m`.
"""
function energy(mc::MC, m::IsingModel, conf::IsingConf)
  neigh = m.l.neighs
    E = 0.0
    for n in 1:m.dims
        @inbounds @simd for i in 1:m.l.sites
            E -= conf[i]*conf[neigh[n,i]]
        end
    end
    return E
end

"""
    energy(mc::MC, m::IsingModel{SquareLattice}, conf::IsingConf)

Calculate energy of Ising configuration `conf` for 2D Ising model `m`.
This method is a faster variant of the general method for the square lattice case.
(It is roughly twice as fast in this case.)
"""
function energy(mc::MC, m::IsingModel{SquareLattice}, conf::IsingConf)
  neigh = m.l.neighs
    E = 0.0
    @inbounds @simd for i in 1:m.l.sites
        E -= conf[i]*conf[neigh[1,i]] + conf[i]*conf[neigh[2,i]]
    end
    return E
end

import Base.rand
"""
    rand(mc::MC, m::IsingModel)

Draw random Ising configuration.
"""
rand(mc::MC, m::IsingModel) = rand(IsingDistribution, fill(m.L, m.dims)...)

"""
    conftype(::Type{MC}, m::IsingModel)

Returns the type of an Ising model configuration.
"""
conftype(::Type{MC}, m::IsingModel) = Array{IsingSpin, m.dims}

"""
    propose_local(mc::MC, m::IsingModel, i::Int, conf::IsingConf, E::Float64) -> delta_E, delta_i

Propose a local spin flip at site `i` of current configuration `conf`
with energy `E`. Returns the local move `delta_i = new[i] - conf[i]` and energy difference `delta_E = E_new - E_old`.
"""
@inline function propose_local(mc::MC, m::IsingModel, i::Int, conf::IsingConf, E::Float64)
    delta_E = 2. * conf[i] * sum(conf[m.l.neighs[:,i]])
    return delta_E, conf[i]==1 ? -2 : 2
end

"""
    accept_local(mc::MC, m::IsingModel, i::Int, conf::IsingConf, E::Float64, delta_i, delta_E::Float64)

Accept a local spin flip at site `i` of current configuration `conf`
with energy `E`. Arguments `delta_i` and `delta_E` correspond to output of `propose_local()`
for that spin flip.
"""
@inline function accept_local!(mc::MC, m::IsingModel, i::Int, conf::IsingConf, E::Float64, delta_i, delta_E::Float64)
    conf[i] *= -1
    nothing
end

"""
    global_move(mc::MC, m::IsingModel, conf::IsingConf, E::Float64) -> accepted::Bool

Constructs a Wolff cluster spinflip for configuration `conf` with energy `E`.
Returns wether a cluster spinflip has been performed (any spins have been flipped).
"""
function global_move(mc::MC, m::IsingModel, conf::IsingConf, E::Float64)
  N = m.l.sites
  neighs = m.l.neighs
  beta = mc.p.beta

    cluster = Array{Int, 1}()
    tocheck = Array{Int, 1}()

    s = rand(1:N)
    push!(tocheck, s)
    push!(cluster, s)

    while !isempty(tocheck)
        cur = pop!(tocheck)
        @inbounds for n in neighs[:,cur]

            @inbounds if conf[cur] == conf[n] && !(n in cluster) && rand() < (1 - exp(- 2.0 * beta))
                push!(tocheck, n)
                push!(cluster, n)
            end

        end
    end

    for spin in cluster
        conf[spin] *= -1
    end

    return length(cluster)>1
    #return length(cluster) # technically a misuse. allows us to see the cluster size * 100 as acc_rate_global.
end

"""
    prepare_observables(mc::MC, m::IsingModel)

Initializes observables for the Ising model and returns a `Dict{String, Observable}`.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
@inline function prepare_observables(mc::MC, m::IsingModel)
    obs = Dict{String,Observable}()
    obs["confs"] = Observable(conftype(MC, m), "Configurations")

    obs["E"] = Observable(Float64, "Total energy")
    obs["E2"] = Observable(Float64, "Total energy squared")
    obs["e"] = Observable(Float64, "Energy (per site)")

    obs["M"] = Observable(Float64, "Total magnetization")
    obs["M2"] = Observable(Float64, "Total magnetization squared")
    obs["m"] = Observable(Float64, "Magnetization (per site)")

    obs["χ"] = Observable(Float64, "Susceptibility")

    obs["C"] = Observable(Float64, "Specific Heat")

    return obs
end

"""
    measure_observables!(mc::MC, m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)

Measures observables and updates corresponding `Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
@inline function measure_observables!(mc::MC, m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)
  N = m.l.sites

    add!(obs["confs"], conf)

    # energie
    E2 = E^2
    add!(obs["E"], E)
    add!(obs["E2"], E2)
    add!(obs["e"], E/N)

    # magnetization
    M::Float64 = abs(sum(conf))
    M2 = M^2
    add!(obs["M"], M)
    add!(obs["M2"], M2)
    add!(obs["m"], M/N)

    nothing
end

"""
    measure_observables!(mc::MC, m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)

Calculates magnetic susceptibility and specific heat and updates corresponding `Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
@inline function finish_observables!(mc::MC, m::IsingModel, obs::Dict{String,Observable})
  N = m.l.sites
  beta = mc.p.beta

    # specific heat
  E = mean(obs["E"])
  E2 = mean(obs["E2"])
    add!(obs["C"], beta*beta*(E2/N - E*E/N))

    # susceptibility
  M = mean(obs["M"])
  M2 = mean(obs["M2"])
    add!(obs["χ"], beta*(M2/N - M*M/N))

    nothing
end
