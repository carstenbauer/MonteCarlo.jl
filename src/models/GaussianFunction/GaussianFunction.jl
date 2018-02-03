mutable struct GaussianFunction <: Model
    mu::Vector{Float64}
    sigma::Vector{Float64}
end

"""
    GaussianFunction(; mu::Vector{Float64}=[0.], sigma::Vector{Float64}=[1.])

Prepare (multidimensional) Gaussian with mean vector `mu` and
standard deviation vector `sigma`.
"""
GaussianFunction(; mu::Vector{Float64}=[0.], sigma::Vector{Float64}=[1.]) = GaussianFunction(mu, sigma)

"""
    GaussianFunction(kwargs::Dict{String, Any})

Create Gaussian with (keyword) parameters as specified in `kwargs` dict.
"""
GaussianFunction(kwargs::Dict{String, Any}) = GaussianFunction(; convert(Dict{Symbol,Any}, kwargs)...)

# methods to use it with Monte Carlo flavor Integrator
"""
    energy(mc::Integrator, m::GaussianFunction, x::Vector{Float64})

Calculate energy (i.e. function value) of the Gaussian function `m` at point `x`.
"""
function energy(mc::Integrator, m::GaussianFunction, x::Vector{Float64})
    return prod(exp.(-(x - m.mu).^2 ./ m.sigma))
end

# cosmetics
import Base.summary
import Base.show
Base.summary(gf::GaussianFunction) = "GaussianFunction"
Base.show(io::IO, gf::GaussianFunction) = print(io, "GaussianFunction (Mean: $(round.(gf.mu, 3)), Std: $(round.(gf.sigma, 3)))")
Base.show(io::IO, m::MIME"text/plain", gf::GaussianFunction) = print(io, gf)
