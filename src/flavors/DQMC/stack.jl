mutable struct DQMCStack{GreensType}
  # eye_flv = eye(p.flv,p.flv)
  # eye_full = eye(p.flv*l.sites,p.flv*l.sites)
  # ones_vec = ones(p.flv*l.sites)

  u_stack::Array{GreensType, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{GreensType, 3}

  Ul::Matrix{GreensType}
  Ur::Matrix{GreensType}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{GreensType}
  Tr::Matrix{GreensType}

  greens::Matrix{GreensType}
  greens_temp::Matrix{GreensType}
  log_det::Float64 # contains logdet of greens_{p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  U::Matrix{GreensType}
  D::Vector{Float64}
  T::Matrix{GreensType}
  u::Matrix{GreensType}
  d::Vector{Float64}
  t::Matrix{GreensType}

  delta_i::Matrix{GreensType}
  M::Matrix{GreensType}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:p.slices+1, where 0 and p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  # -------- Global update backup
  gb_u_stack::Array{GreensType, 3}
  gb_d_stack::Matrix{Float64}
  gb_t_stack::Array{GreensType, 3}

  gb_greens::Matrix{GreensType}
  gb_log_det::Float64

  gb_conf::Array{Float64, 3}
  # --------


  #### Array allocations
  curr_U::Matrix{GreensType}
  eV::Matrix{GreensType}
  eVop1::Matrix{GreensType}
  eVop2::Matrix{GreensType}

  Stack() = new()
end

# TODO constructor: takes mc simulation. initialize_stack and eye_full, eye_flv, and ones_vec

function initialize_stack(mc::DQMC, m::Model)
  const N = m.l.sites
  const flv = m.flv

  mc.s.n_elements = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

  mc.s.u_stack = zeros(GreensType, flv*N, flv*N, mc.s.n_elements)
  mc.s.d_stack = zeros(Float64, flv*N, mc.s.n_elements)
  mc.s.t_stack = zeros(GreensType, flv*N, flv*N, mc.s.n_elements)

  mc.s.greens = zeros(GreensType, flv*N, flv*N)
  mc.s.greens_temp = zeros(GreensType, flv*N, flv*N)

  mc.s.Ul = eye(GreensType, flv*N, flv*N)
  mc.s.Ur = eye(GreensType, flv*N, flv*N)
  mc.s.Tl = eye(GreensType, flv*N, flv*N)
  mc.s.Tr = eye(GreensType, flv*N, flv*N)
  mc.s.Dl = ones(Float64, flv*N)
  mc.s.Dr = ones(Float64, flv*N)

  mc.s.U = zeros(GreensType, flv*N, flv*N)
  mc.s.D = zeros(Float64, flv*N)
  mc.s.T = zeros(GreensType, flv*N, flv*N)
  mc.s.u = zeros(GreensType, flv*N, flv*N)
  mc.s.d = zeros(Float64, flv*N)
  mc.s.t = zeros(GreensType, flv*N, flv*N)

  mc.s.delta_i = zeros(GreensType, flv, flv)
  mc.s.M = zeros(GreensType, flv, flv)

  # Global update backup
  mc.s.gb_u_stack = zero(mc.s.u_stack)
  mc.s.gb_d_stack = zero(mc.s.d_stack)
  mc.s.gb_t_stack = zero(mc.s.t_stack)
  mc.s.gb_greens = zero(mc.s.greens)
  mc.s.gb_log_det = 0. 
  mc.s.gb_conf = zero(mc.conf)

  mc.s.ranges = UnitRange[]

  for i in 1:mc.s.n_elements - 1
    push!(mc.s.ranges, 1 + (i - 1) * mc.p.safe_mult:i * mc.p.safe_mult)
  end

  mc.s.curr_U = zero(mc.s.U)
  mc.s.eV = zeros(GreensType, flv*N, flv*N)
  mc.s.eVop1 = zeros(GreensType, flv, flv)
  mc.s.eVop2 = zeros(GreensType, flv, flv)

end


function build_stack(mc::DQMC, m::Model)
  mc.s.u_stack[:, :, 1] = eye_full
  mc.s.d_stack[:, 1] = ones_vec
  mc.s.t_stack[:, :, 1] = eye_full

  @inbounds for i in 1:length(mc.s.ranges)
    add_slice_sequence_left(s, p, l, i)
  end

  mc.s.current_slice = p.slices + 1
  mc.s.direction = -1

  nothing
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(s::Stack, p::Parameters, l::Lattice, idx::Int)
  
  copy!(mc.s.curr_U, mc.s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
  for slice in mc.s.ranges[idx]
    if p.chkr
      multiply_slice_matrix_left!(s, p, l, slice, mc.s.curr_U)
    else
      mc.s.curr_U = slice_matrix_no_chkr(s, p, l, slice) * mc.s.curr_U
    end
  end

  mc.s.curr_U *= spdiagm(mc.s.d_stack[:, idx])
  mc.s.u_stack[:, :, idx + 1], mc.s.d_stack[:, idx + 1], T = decompose_udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx + 1] =  T * mc.s.t_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(s::Stack, p::Parameters, l::Lattice, idx::Int)
  
  copy!(mc.s.curr_U, mc.s.u_stack[:, :, idx + 1])

  for slice in reverse(mc.s.ranges[idx])
    if p.chkr
      multiply_daggered_slice_matrix_left!(s, p, l, slice, mc.s.curr_U)
    else
      mc.s.curr_U = ctranspose(slice_matrix_no_chkr(s, p, l, slice)) * mc.s.curr_U
    end
  end

  mc.s.curr_U *=  spdiagm(mc.s.d_stack[:, idx + 1])
  mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], T = decompose_udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx] = T * mc.s.t_stack[:, :, idx + 1]
end


# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix_no_chkr(s::Stack, p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  if power > 0
    return l.hopping_matrix_exp * l.hopping_matrix_exp * interaction_matrix_exp(s, p, l, slice, power)
  else
    return interaction_matrix_exp(s, p, l, slice, power) * l.hopping_matrix_exp_inv * l.hopping_matrix_exp_inv
  end
end


@inline function wrap_greens_chkr!(s::Stack, p::Parameters, l::Lattice, gf::Matrix{GreensType}, curr_slice::Int,direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(s, p, l, curr_slice - 1, gf)
    multiply_slice_matrix_right!(s, p, l, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(s, p, l, curr_slice, gf)
    multiply_slice_matrix_inv_right!(s, p, l, curr_slice, gf)
  end
end

function wrap_greens_chkr(s::Stack, p::Parameters, l::Lattice, gf::Matrix{GreensType},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_chkr!(s, p, l, temp, slice, direction)
  return temp
end


function wrap_greens_no_chkr!(s::Stack, p::Parameters, l::Lattice, gf::Matrix{GreensType}, curr_slice::Int,direction::Int)
  if direction == -1
    gf[:] = slice_matrix_no_chkr(s, p, l, curr_slice - 1, -1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(s, p, l, curr_slice - 1, 1.)
  else
    gf[:] = slice_matrix_no_chkr(s, p, l, curr_slice, 1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(s, p, l, curr_slice, -1.)
  end
end

function wrap_greens_no_chkr(s::Stack, p::Parameters, l::Lattice, gf::Matrix{GreensType},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_no_chkr!(s, p, l, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using mc.s.Ur,mc.s.Dr,mc.s.Tr=B(slice)' ... B(M)' and mc.s.Ul,mc.s.Dl,mc.s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(s::Stack, p::Parameters, l::Lattice)

  tmp = mc.s.Tl * ctranspose(mc.s.Tr)
  mc.s.U, mc.s.D, mc.s.T = decompose_udt(spdiagm(mc.s.Dl) * tmp * spdiagm(mc.s.Dr))
  mc.s.U = mc.s.Ul * mc.s.U
  mc.s.T *= ctranspose(mc.s.Ur)

  mc.s.u, mc.s.d, mc.s.t = decompose_udt(ctranspose(mc.s.U) * inv(mc.s.T) + spdiagm(mc.s.D))

  mc.s.T = inv(mc.s.t * mc.s.T)
  mc.s.U *= mc.s.u
  mc.s.U = ctranspose(mc.s.U)
  mc.s.d = 1./mc.s.d

  mc.s.greens = mc.s.T * spdiagm(mc.s.d) * mc.s.U
end


"""
Only reasonable immediately after calculate_greens()!
"""
function calculate_logdet(s::Stack, p::Parameters, l::Lattice)
  if p.opdim == 1
    mc.s.log_det = real(log(complex(det(mc.s.U))) + sum(log.(mc.s.d)) + logdet(complex(det(mc.s.T))))
  else
    mc.s.log_det = real(logdet(mc.s.U) + sum(log.(mc.s.d)) + logdet(mc.s.T))
  end
end


################################################################################
# Propagation
################################################################################
function propagate(s::Stack, p::Parameters, l::Lattice)
  if mc.s.direction == 1
    if mod(mc.s.current_slice, mc.p.safe_mult) == 0
      mc.s.current_slice +=1 # slice we are going to
      if mc.s.current_slice == 1
        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]
        mc.s.u_stack[:, :, 1] = eye_full
        mc.s.d_stack[:, 1] = ones_vec
        mc.s.t_stack[:, :, 1] = eye_full
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, 1], mc.s.d_stack[:, 1], mc.s.t_stack[:, :, 1]

        calculate_greens(s, p, l) # greens_1 ( === greens_{m+1} )
        calculate_logdet(s, p, l)

      elseif 1 < mc.s.current_slice <= p.slices
        idx = Int((mc.s.current_slice - 1)/mc.p.safe_mult)

        mc.s.Ur[:, :], mc.s.Dr[:], mc.s.Tr[:, :] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]
        add_slice_sequence_left(s, p, l, idx)
        mc.s.Ul[:,:], mc.s.Dl[:], mc.s.Tl[:,:] = mc.s.u_stack[:, :, idx+1], mc.s.d_stack[:, idx+1], mc.s.t_stack[:, :, idx+1]

        if p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        if p.chkr
          wrap_greens_chkr!(s, p, l, mc.s.greens_temp, mc.s.current_slice - 1, 1)
        else
          wrap_greens_no_chkr!(s, p, l, mc.s.greens_temp, mc.s.current_slice - 1, 1)
        end

        calculate_greens(s, p, l) # greens_{slice we are propagating to}

        if p.all_checks
          diff = maximum(absdiff(mc.s.greens_temp, mc.s.greens))
          if diff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.4f\n", mc.s.current_slice, diff)
          end
        end

      else # we are going to p.slices+1
        idx = mc.s.n_elements - 1
        add_slice_sequence_left(s, p, l, idx)
        mc.s.direction = -1
        mc.s.current_slice = p.slices+1 # redundant
        propagate(s, p, l)
      end

    else
      # Wrapping
      if p.chkr
        wrap_greens_chkr!(s, p, l, mc.s.greens, mc.s.current_slice, 1)
      else
        wrap_greens_no_chkr!(s, p, l, mc.s.greens, mc.s.current_slice, 1)
      end
      mc.s.current_slice += 1
    end

  else # mc.s.direction == -1
    if mod(mc.s.current_slice-1, mc.p.safe_mult) == 0
      mc.s.current_slice -= 1 # slice we are going to
      if mc.s.current_slice == p.slices
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]
        mc.s.u_stack[:, :, end] = eye_full
        mc.s.d_stack[:, end] = ones_vec
        mc.s.t_stack[:, :, end] = eye_full
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, end], mc.s.d_stack[:, end], mc.s.t_stack[:, :, end]

        calculate_greens(s, p, l) # greens_{p.slices+1} === greens_1
        calculate_logdet(s, p, l) # calculate logdet for potential global update

        # wrap to greens_{p.slices}
        if p.chkr
          wrap_greens_chkr!(s, p, l, mc.s.greens, mc.s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(s, p, l, mc.s.greens, mc.s.current_slice + 1, -1)
        end

      elseif 0 < mc.s.current_slice < p.slices
        idx = Int(mc.s.current_slice / mc.p.safe_mult) + 1
        mc.s.Ul[:, :], mc.s.Dl[:], mc.s.Tl[:, :] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]
        add_slice_sequence_right(s, p, l, idx)
        mc.s.Ur[:,:], mc.s.Dr[:], mc.s.Tr[:,:] = mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], mc.s.t_stack[:, :, idx]

        if p.all_checks
          mc.s.greens_temp = copy(mc.s.greens)
        end

        calculate_greens(s, p , l)

        if p.all_checks
          diff = maximum(absdiff(mc.s.greens_temp, mc.s.greens))
          if diff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", mc.s.current_slice, diff)
          end
        end

        if p.chkr
          wrap_greens_chkr!(s, p, l, mc.s.greens, mc.s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(s, p, l, mc.s.greens, mc.s.current_slice + 1, -1)
        end

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(s, p, l, idx)
        mc.s.direction = 1
        mc.s.current_slice = 0 # redundant
        propagate(s,p,l)
      end

    else
      # Wrapping
      if p.chkr
        wrap_greens_chkr!(s, p, l, mc.s.greens, mc.s.current_slice, -1)
      else
        wrap_greens_no_chkr!(s, p, l, mc.s.greens, mc.s.current_slice, -1)
      end
      mc.s.current_slice -= 1
    end
  end
  # compare(mc.s.greens,calculate_greens_udv(p,l,mc.s.current_slice))
  # compare(mc.s.greens,calculate_greens_udv_chkr(p,l,mc.s.current_slice))
  nothing
end