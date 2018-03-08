using Distributions

type Parameters
  lattice_file::String
  beta::Float64
  delta_tau::Float64
  slices::Int
  safe_mult::Int

  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure

  hsfield::Array{Int8, 2} # 1: spatial, dim 2: imag time
  boson_action::Float64

  mu::Float64
  lambda::Float64
  u::Float64
  flv::Int # flavors: GF matrix has size flv*l.sites x flv*l.sites. 1 for attr Hubbard

  Parameters() = new()
end