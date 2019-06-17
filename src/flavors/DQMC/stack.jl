# type
mutable struct DQMCStack{GreensEltype<:Number, HoppingEltype<:Number} <: AbstractDQMCStack
  eye_flv::Matrix{Float64}
  eye_full::Matrix{Float64}
  ones_vec::Vector{Float64}

  u_stack::Array{GreensEltype, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{GreensEltype, 3}

  Ul::Matrix{GreensEltype}
  Ur::Matrix{GreensEltype}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{GreensEltype}
  Tr::Matrix{GreensEltype}

  greens::Matrix{GreensEltype}
  greens_temp::Matrix{GreensEltype}
  log_det::Float64 # contains logdet of greens_{mc.p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  U::Matrix{GreensEltype}
  D::Vector{Float64}
  T::Matrix{GreensEltype}
  u::Matrix{GreensEltype}
  d::Vector{Float64}
  t::Matrix{GreensEltype}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:mc.p.slices+1, where 0 and mc.p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  # # -------- Global update backup
  # gb_u_stack::Array{GreensEltype, 3}
  # gb_d_stack::Matrix{Float64}
  # gb_t_stack::Array{GreensEltype, 3}

  # gb_greens::Matrix{GreensEltype}
  # gb_log_det::Float64

  # gb_conf::Array{Float64, 3}
  # # --------


  # preallocated, reused arrays
  curr_U::Matrix{GreensEltype}
  eV::Matrix{GreensEltype}

  # hopping matrices
  hopping_matrix_exp::Matrix{HoppingEltype} # mu included
  hopping_matrix_exp_inv::Matrix{HoppingEltype} # mu included

  # checkerboard hopping matrices
  checkerboard::Matrix{Int} # src, trg, bondid
  groups::Vector{UnitRange}
  n_groups::Int
  chkr_hop_half::Vector{SparseMatrixCSC{HoppingEltype, Int64}}
  chkr_hop_half_inv::Vector{SparseMatrixCSC{HoppingEltype, Int64}}
  chkr_hop_half_dagger::Vector{SparseMatrixCSC{HoppingEltype, Int64}}
  chkr_hop::Vector{SparseMatrixCSC{HoppingEltype, Int64}} # without prefactor 0.5 in matrix exponentials
  chkr_hop_inv::Vector{SparseMatrixCSC{HoppingEltype, Int64}}
  chkr_hop_dagger::Vector{SparseMatrixCSC{HoppingEltype, Int64}}
  chkr_mu_half::SparseMatrixCSC{HoppingEltype, Int64}
  chkr_mu_half_inv::SparseMatrixCSC{HoppingEltype, Int64}
  chkr_mu::SparseMatrixCSC{HoppingEltype, Int64}
  chkr_mu_inv::SparseMatrixCSC{HoppingEltype, Int64}


  DQMCStack{GreensEltype, HoppingEltype}() where {GreensEltype<:Number, HoppingEltype<:Number} = begin
    # @assert isleaftype(GreensEltype);
    # @assert isleaftype(HoppingEltype);
    @assert isconcretetype(GreensEltype);
    @assert isconcretetype(HoppingEltype);
    new()
  end
end

# type helpers
geltype(::Type{DQMCStack{G,H}}) where {G,H} = G
heltype(::Type{DQMCStack{G,H}}) where {G,H} = H
geltype(mc::DQMC{M, CB, CT, S}) where {M, CB, CT, S} = geltype(S)
heltype(mc::DQMC{M, CB, CT, S}) where {M, CB, CT, S} = heltype(S)

# type initialization
function initialize_stack(mc::DQMC)
  GreensEltype = geltype(mc)
  HoppingEltype = heltype(mc)
  N = mc.model.l.sites
  flv = mc.model.flv

  mc.s.eye_flv = Matrix{Float64}(I, flv,flv)
  mc.s.eye_full = Matrix{Float64}(I, flv*N,flv*N)
  mc.s.ones_vec = ones(flv*N)

  mc.s.n_elements = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

  mc.s.u_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)
  mc.s.d_stack = zeros(Float64, flv*N, mc.s.n_elements)
  mc.s.t_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)

  mc.s.greens = zeros(GreensEltype, flv*N, flv*N)
  mc.s.greens_temp = zeros(GreensEltype, flv*N, flv*N)

  mc.s.Ul = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Ur = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Tl = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Tr = Matrix{GreensEltype}(I, flv*N, flv*N)
  mc.s.Dl = ones(Float64, flv*N)
  mc.s.Dr = ones(Float64, flv*N)

  mc.s.U = zeros(GreensEltype, flv*N, flv*N)
  mc.s.D = zeros(Float64, flv*N)
  mc.s.T = zeros(GreensEltype, flv*N, flv*N)
  mc.s.u = zeros(GreensEltype, flv*N, flv*N)
  mc.s.d = zeros(Float64, flv*N)
  mc.s.t = zeros(GreensEltype, flv*N, flv*N)

  # # Global update backup
  # mc.s.gb_u_stack = zero(mc.s.u_stack)
  # mc.s.gb_d_stack = zero(mc.s.d_stack)
  # mc.s.gb_t_stack = zero(mc.s.t_stack)
  # mc.s.gb_greens = zero(mc.s.greens)
  # mc.s.gb_log_det = 0.
  # mc.s.gb_conf = zero(mc.conf)

  mc.s.ranges = UnitRange[]

  for i in 1:mc.s.n_elements - 1
    push!(mc.s.ranges, 1 + (i - 1) * mc.p.safe_mult:i * mc.p.safe_mult)
  end

  mc.s.curr_U = zero(mc.s.U)
  mc.s.eV = zeros(GreensEltype, flv*N, flv*N)

  # mc.s.hopping_matrix_exp = zeros(HoppingEltype, flv*N, flv*N)
  # mc.s.hopping_matrix_exp_inv = zeros(HoppingEltype, flv*N, flv*N)
  nothing
end

# hopping
function init_hopping_matrices(mc::DQMC{M,CB}, m::Model) where {M, CB<:Checkerboard}
  init_hopping_matrix_exp(mc, m)
  CB <: CheckerboardTrue && init_checkerboard_matrices(mc, m)
  nothing
end
function init_hopping_matrix_exp(mc::DQMC, m::Model)
  N = m.l.sites
  flv = m.flv
  dtau = mc.p.delta_tau

  T = hopping_matrix(mc, m)
  size(T) == (flv*N, flv*N) || error("Hopping matrix should have size "*
                                "$((flv*N, flv*N)) but has size $(size(T)) .")
  mc.s.hopping_matrix_exp = exp(-0.5 * dtau * T)
  mc.s.hopping_matrix_exp_inv = exp(0.5 * dtau * T)
  nothing
end

# checkerboard
rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)
function init_checkerboard_matrices(mc::DQMC, m::Model)
  s = mc.s
  l = m.l
  flv = m.flv
  H = heltype(mc)
  N = m.l.sites
  dtau = mc.p.delta_tau
  mu = m.mu

  s.checkerboard, s.groups, s.n_groups = build_checkerboard(l)
  n_grps = s.n_groups
  cb = s.checkerboard

  T = reshape(hopping_matrix(mc, m), (N, flv, N, flv))

  s.chkr_hop_half = Vector{SparseMatrixCSC{H, Int}}(n_grps)
  s.chkr_hop_half_inv = Vector{SparseMatrixCSC{H, Int}}(n_grps)
  s.chkr_hop = Vector{SparseMatrixCSC{H, Int}}(n_grps)
  s.chkr_hop_inv = Vector{SparseMatrixCSC{H, Int}}(n_grps)

  for (g, gr) in enumerate(s.groups)
    Tg = zeros(H, N, flv, N, flv)
    for i in gr
      src, trg = cb[1:2,i]
      for f1 in 1:flv, f2 in 1:flv
        Tg[trg, f1, src, f2] = T[trg, f1, src, f2]
      end
    end

    Tgg = reshape(Tg, (N*flv, N*flv))
    s.chkr_hop_half[g] = sparse(rem_eff_zeros!(exp(-0.5 * dtau * Tgg)))
    s.chkr_hop_half_inv[g] = sparse(rem_eff_zeros!(exp(0.5 * dtau * Tgg)))
    s.chkr_hop[g] = sparse(rem_eff_zeros!(exp(- dtau * Tgg)))
    s.chkr_hop_inv[g] = sparse(rem_eff_zeros!(exp(dtau * Tgg)))
  end

  s.chkr_hop_half_dagger = adjoint.(s.chkr_hop_half)
  s.chkr_hop_dagger = adjoint.(s.chkr_hop)

  mus = diag(reshape(T, (N*flv, N*flv)))
  s.chkr_mu_half = spdiagm(0 => exp.(-0.5 * dtau * mus))
  s.chkr_mu_half_inv = spdiagm(0 => exp.(0.5 * dtau * mus))
  s.chkr_mu = spdiagm(0 => exp.(-dtau * mus))
  s.chkr_mu_inv = spdiagm(0 => exp.(dtau * mus))

  # hop_mat_exp_chkr = foldl(*,s.chkr_hop_half) * sqrt.(s.chkr_mu)
  # r = effreldiff(s.hopping_matrix_exp,hop_mat_exp_chkr)
  # r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  # println("Checkerboard - Exact ≈ ", round(maximum(absdiff(s.hopping_matrix_exp,hop_mat_exp_chkr)), 4))
  nothing
end

# stack construction
"""
Build stack from scratch.
"""
function build_stack(mc::DQMC)
  mc.s.u_stack[:, :, 1] = mc.s.eye_full
  mc.s.d_stack[:, 1] = mc.s.ones_vec
  mc.s.t_stack[:, :, 1] = mc.s.eye_full

  @inbounds for i in 1:length(mc.s.ranges)
    add_slice_sequence_left(mc, i)
  end

  mc.s.current_slice = mc.p.slices + 1
  mc.s.direction = -1

  nothing
end
"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(mc::DQMC, idx::Int)
  copy!(mc.s.curr_U, mc.s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
  for slice in mc.s.ranges[idx]
    multiply_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
  end

  mc.s.curr_U *= spdiagm(0 => mc.s.d_stack[:, idx])
  mc.s.u_stack[:, :, idx + 1], mc.s.d_stack[:, idx + 1], T = decompose_udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx + 1] =  T * mc.s.t_stack[:, :, idx]
end
"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(mc::DQMC, idx::Int)
  copy!(mc.s.curr_U, mc.s.u_stack[:, :, idx + 1])

  for slice in reverse(mc.s.ranges[idx])
    multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
  end

  mc.s.curr_U *=  spdiagm(0 => mc.s.d_stack[:, idx + 1])
  mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], T = decompose_udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx] = T * mc.s.t_stack[:, :, idx + 1]
end

# Green's function calculation
"""
Calculates G(slice) using mc.s.Ur,mc.s.Dr,mc.s.Tr=B(slice)' ... B(M)' and
mc.s.Ul,mc.s.Dl,mc.s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::DQMC)

  tmp = mc.s.Tl * adjoint(mc.s.Tr)
  mc.s.U, mc.s.D, mc.s.T = decompose_udt(spdiagm(0 => mc.s.Dl) * tmp * spdiagm(0 => mc.s.Dr))
  mc.s.U = mc.s.Ul * mc.s.U
  mc.s.T *= adjoint(mc.s.Ur)

  mc.s.u, mc.s.d, mc.s.t = decompose_udt(adjoint(mc.s.U) * inv(mc.s.T) + spdiagm(0 => mc.s.D))

  mc.s.T = inv(mc.s.t * mc.s.T)
  mc.s.U *= mc.s.u
  mc.s.U = adjoint(mc.s.U)
  mc.s.d = 1. ./ mc.s.d

  mc.s.greens = mc.s.T * spdiagm(0 => mc.s.d) * mc.s.U
end
"""
Only reasonable immediately after calculate_greens()!
"""
function calculate_logdet(mc::DQMC)
  mc.s.log_det = real(log(complex(det(mc.s.U))) + sum(log.(mc.s.d)) + log(complex(det(mc.s.T))))
  # mc.s.log_det = real(logdet(mc.s.U) + sum(log.(mc.s.d)) + logdet(mc.s.T))
end

# Green's function propagation
@inline function wrap_greens!(mc::DQMC, gf::Matrix, curr_slice::Int, direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
    multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
    multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
  end
  nothing
end
# @inline function wrap_greens(mc::DQMC, gf::Matrix,slice::Int,direction::Int)
#   temp = copy(gf)
#   wrap_greens!(mc, temp, slice, direction)
#   return temp
# end
function propagate(mc::DQMC)
  if mc.s.direction == 1
    if mod(mc.s.current_slice, mc.p.safe_mult) == 0
      mc.s.current_slice +=1 # slice we are going to
      if mc.s.current_slice == 1
        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]
        mc.s.u_stack[:, :, 1] = mc.s.eye_full
        mc.s.d_stack[:, 1] = mc.s.ones_vec
        mc.s.t_stack[:, :, 1] = mc.s.eye_full
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]

        calculate_greens(mc) # greens_1 ( === greens_{m+1} )
        calculate_logdet(mc)

      elseif 1 < mc.s.current_slice <= mc.p.slices
        idx = Int((mc.s.current_slice - 1)/mc.p.safe_mult)

        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]
        add_slice_sequence_left(mc, idx)
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]

        if mc.p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        wrap_greens!(mc, mc.s.greens_temp, mc.s.current_slice - 1, 1)

        calculate_greens(mc) # greens_{slice we are propagating to}

        if mc.p.all_checks
          greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if greensdiff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.1e\n", mc.s.current_slice, greensdiff)
          end
        end

      else # we are going to mc.p.slices+1
        idx = mc.s.n_elements - 1
        add_slice_sequence_left(mc, idx)
        mc.s.direction = -1
        mc.s.current_slice = mc.p.slices+1 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, mc.s.greens, mc.s.current_slice, 1)
      mc.s.current_slice += 1
    end

  else # mc.s.direction == -1
    if mod(mc.s.current_slice-1, mc.p.safe_mult) == 0
      mc.s.current_slice -= 1 # slice we are going to
      if mc.s.current_slice == mc.p.slices
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]
        mc.s.u_stack[:, :, end] = mc.s.eye_full
        mc.s.d_stack[:, end] = mc.s.ones_vec
        mc.s.t_stack[:, :, end] = mc.s.eye_full
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]

        calculate_greens(mc) # greens_{mc.p.slices+1} === greens_1
        calculate_logdet(mc) # calculate logdet for potential global update

        # wrap to greens_{mc.p.slices}
        wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

      elseif 0 < mc.s.current_slice < mc.p.slices
        idx = Int(mc.s.current_slice / mc.p.safe_mult) + 1
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]
        add_slice_sequence_right(mc, idx)
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]

        if mc.p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        calculate_greens(mc)

        if mc.p.all_checks
          greensdiff = maximum(abs.(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if greensdiff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.1e\n", mc.s.current_slice, greensdiff)
          end
        end

        wrap_greens!(mc, mc.s.greens, mc.s.current_slice + 1, -1)

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(mc, idx)
        mc.s.direction = 1
        mc.s.current_slice = 0 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, mc.s.greens, mc.s.current_slice, -1)
      mc.s.current_slice -= 1
    end
  end
  nothing
end
