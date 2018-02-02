mutable struct GaussianFunction <: Model
    mu::Array{Float64, 1}
    sigma::Array{Float64, 1}
    min_x::Array{Float64, 1}
    max_x::Array{Float64, 1}
end

"""
    GaussianFunction(; mu::Array{Float64, 1}=[0.], sigma::Array{Float64, 1}=[1.])

Prepare Gaussian with mean values mu and standard deviations sigma
"""
GaussianFunction(; mu::Array{Float64, 1}=[0.], sigma::Array{Float64, 1}=[1.], min_x::Array{Float64, 1}=[-10.], max_x::Array{Float64, 1} = [10.]) = GaussianFunction(mu, sigma, min_x, max_x)
"""
    GaussianFunction(kwargs::Dict{String, Any})

Create Gaussian with (keyword) parameters as specified in `kwargs` dict.
"""
GaussianFunction(kwargs::Dict{String, Any}) = GaussianFunction(; convert(Dict{Symbol,Any}, kwargs)...)


# methods to use it with Monte Carlo flavor Integrator
"""
    energy(mc::Integrator, m::GaussianFunction, value::IsingConf)

Calculate energy of Ising configuration `conf` for Ising model `m`.
"""
function energy(mc::Integrator, m::GaussianFunction, value::Array{Float64, 1})
    return prod(exp.(-(value - m.mu).^2 ./ m.sigma))
end

# cosmetics
import Base.summary
import Base.show
Base.summary(gf::GaussianFunction) = "GaussianFunction"
Base.show(io::IO, gf::GaussianFunction) = print(io, "GaussianFunction (Mean: $(round.(gf.mu, 3)), Std: $(round.(gf.sigma, 3)))")
Base.show(io::IO, m::MIME"text/plain", gf::GaussianFunction) = print(io, gf)
