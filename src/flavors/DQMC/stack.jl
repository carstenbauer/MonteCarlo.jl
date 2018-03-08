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

  # delta_i::Matrix{GreensEltype}
  # M::Matrix{GreensEltype}

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
  # eVop1::Matrix{GreensEltype}
  # eVop2::Matrix{GreensEltype}

  # hopping matrices
  # TODO so far not initialized
  hopping_matrix_exp::Matrix{HoppingEltype} # mu included
  hopping_matrix_exp_inv::Matrix{HoppingEltype} # mu included

  # checkerboard hopping matrices
  # TODO

  DQMCStack{GreensEltype, HoppingEltype}() where {GreensEltype<:Number, HoppingEltype<:Number} = begin
    @assert isleaftype(GreensEltype);
    @assert isleaftype(HoppingEltype);
    new()
  end
end

geltype(::Type{DQMCStack{G,H}}) where {G,H} = G
heltype(::Type{DQMCStack{G,H}}) where {G,H} = H
geltype(mc::DQMC{M, CB, ConfType, Stack}) where {M, CB, ConfType, Stack} = geltype(Stack)
heltype(mc::DQMC{M, CB, ConfType, Stack}) where {M, CB, ConfType, Stack} = heltype(Stack)

# TODO constructor: takes mc simulation.

function initialize_stack(mc::DQMC)
  const GreensEltype = geltype(mc)
  const N = mc.model.l.sites
  const flv = mc.model.flv

  mc.s.eye_flv = eye(flv,flv)
  mc.s.eye_full = eye(flv*N,flv*N)
  mc.s.ones_vec = ones(flv*N)

  mc.s.n_elements = convert(Int, mc.p.slices / mc.p.safe_mult) + 1

  mc.s.u_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)
  mc.s.d_stack = zeros(Float64, flv*N, mc.s.n_elements)
  mc.s.t_stack = zeros(GreensEltype, flv*N, flv*N, mc.s.n_elements)

  mc.s.greens = zeros(GreensEltype, flv*N, flv*N)
  mc.s.greens_temp = zeros(GreensEltype, flv*N, flv*N)

  mc.s.Ul = eye(GreensEltype, flv*N, flv*N)
  mc.s.Ur = eye(GreensEltype, flv*N, flv*N)
  mc.s.Tl = eye(GreensEltype, flv*N, flv*N)
  mc.s.Tr = eye(GreensEltype, flv*N, flv*N)
  mc.s.Dl = ones(Float64, flv*N)
  mc.s.Dr = ones(Float64, flv*N)

  mc.s.U = zeros(GreensEltype, flv*N, flv*N)
  mc.s.D = zeros(Float64, flv*N)
  mc.s.T = zeros(GreensEltype, flv*N, flv*N)
  mc.s.u = zeros(GreensEltype, flv*N, flv*N)
  mc.s.d = zeros(Float64, flv*N)
  mc.s.t = zeros(GreensEltype, flv*N, flv*N)

  # mc.s.delta_i = zeros(GreensEltype, flv, flv)
  # mc.s.M = zeros(GreensEltype, flv, flv)

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
  # mc.s.eVop1 = zeros(GreensEltype, flv, flv)
  # mc.s.eVop2 = zeros(GreensEltype, flv, flv)

end


function build_stack(mc::DQMC)
  mc.s.u_stack[:, :, 1] = eye_full
  mc.s.d_stack[:, 1] = ones_vec
  mc.s.t_stack[:, :, 1] = eye_full

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

  mc.s.curr_U *= spdiagm(mc.s.d_stack[:, idx])
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

  mc.s.curr_U *=  spdiagm(mc.s.d_stack[:, idx + 1])
  mc.s.u_stack[:, :, idx], mc.s.d_stack[:, idx], T = decompose_udt(mc.s.curr_U)
  mc.s.t_stack[:, :, idx] = T * mc.s.t_stack[:, :, idx + 1]
end


@inline function wrap_greens!(mc::DQMC, gf::Matrix, curr_slice::Int, direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(mc, mc.model, curr_slice - 1, gf)
    multiply_slice_matrix_right!(mc, mc.model, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(mc, mc.model, curr_slice, gf)
    multiply_slice_matrix_inv_right!(mc, mc.model, curr_slice, gf)
  end
end

@inline function wrap_greens(mc::DQMC, gf::Matrix,slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens!(mc, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using mc.s.Ur,mc.s.Dr,mc.s.Tr=B(slice)' ... B(M)' and mc.s.Ul,mc.s.Dl,mc.s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::DQMC)

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
function calculate_logdet(mc::DQMC)
  # TODO: How does this depend on model?
  if p.opdim == 1
    mc.s.log_det = real(log(complex(det(mc.s.U))) + sum(log.(mc.s.d)) + log(complex(det(mc.s.T))))
  else
    mc.s.log_det = real(logdet(mc.s.U) + sum(log.(mc.s.d)) + logdet(mc.s.T))
  end
end


################################################################################
# Propagation
################################################################################
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
          greensdiff = maximum(abs(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if diff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.4f\n", mc.s.current_slice, diff)
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
          greensdiff = maximum(abs(mc.s.greens_temp - mc.s.greens)) # OPT: could probably be optimized through explicit loop
          if greensdiff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", mc.s.current_slice, greensdiff)
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
