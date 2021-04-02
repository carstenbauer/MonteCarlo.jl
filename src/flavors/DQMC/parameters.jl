"""
    DQMCParameters([p::DQMCParameters]; [kwargs...])

Specifies the parameters of the determinant quantum Monte-Carlo (DQMC) 
simulation. 

If an old set of parameters `p` is passed a copy will be generated. Any 
parameter passed via kwargs will replace the respective parameter in the copy.

## Keyword Arguments:
- `thermalization = 100`: Number of thermalization sweeps. (First MC stage)
- `sweeps = 100`: Number of measurement sweeps. (Second MC stage)
- `silent = false`: Should error printing be supressed? (They are still recorded
 in DQMCAnalysis.)
- `check_sign_problem = true`: Should check for sign problems happen? (This fully 
enables/disables sign problem checks, including those for DQMCAnalysis!)
- `check_propagation_error = true`: Should checks for propagation errors happen? 
(This fully enables/disables propgation error checks, includign those for 
DQMCAnalysis!)
- `safe_mult = 10`: Number of safe matrix multiplications in `propagate`. If you
see frequent/large propagation errors you may want to lower this number. Note 
that the number of time slices must be divisable by `safe_mult`.
- `delta_tau`: The imaginary time discretization of the path integral. 
- `beta`: The inverse temperature.
- `slices`: The total number of time slices.
- `measure_rate = 10`: Perform measurements every `measure_rate` sweeps.
- `print_rate = 10`: Print general information every `print_rate` sweeps.

If you construct `DQMCParameters` from scratch you need to specify two of 
`delta_tau`, `beta` and `slices`. The following formula must hold: 
`beta = delta_tau * slices`.
"""
struct DQMCParameters
    thermalization::Int
    sweeps::Int
    
    silent::Bool
    check_sign_problem::Bool
    check_propagation_error::Bool
    
    safe_mult::Int
    delta_tau::Float64
    beta::Float64
    slices::Int
    
    measure_rate::Int
    print_rate::Int
end

function DQMCParameters(
        p::DQMCParameters;
        thermalization = p.thermalization,
        sweeps = p.sweeps,
        silent = p.silent, 
        check_sign_problem = p.check_sign_problem,
        check_propagation_error = p.check_propagation_error,
        safe_mult = p.safe_mult,
        delta_tau = p.delta_tau,
        measure_rate = p.measure_rate,
        print_rate = p.print_rate,
        kwargs...
    )
    if (!haskey(kwargs, :beta)) && (!haskey(kwargs, :slices))
        kwargs = Dict{Symbol, Any}(kwargs)
        push!(kwargs, :beta => p.beta)
    end
    DQMCParameters(
        thermalization = thermalization,
        sweeps = sweeps,
        silent = silent,
        check_sign_problem = check_sign_problem,
        check_propagation_error = check_propagation_error,
        safe_mult = safe_mult,
        delta_tau = delta_tau,
        measure_rate = measure_rate,
        print_rate = print_rate;
        kwargs...
    )
end

function DQMCParameters(;
        thermalization::Int = 100,
        sweeps::Int         = 100,
        silent::Bool        = false,
        check_sign_problem::Bool = true,
        check_propagation_error::Bool = true,
        safe_mult::Int      = 10,
        measure_rate::Int   = 10,
        warn_round::Bool    = true,
        print_rate::Int     = 10,
        kwargs...
    )
    nt = (;kwargs...)
    keys(nt) == (:beta,) && (nt = (;beta=nt.beta, delta_tau=0.1))
    @assert length(nt) >= 2 "Invalid keyword arguments to DQMCParameters: $nt"
    if (Set ∘ keys)(nt) == Set([:delta_tau, :beta, :slices])
        delta_tau, beta = nt.delta_tau, nt.beta
        slices = round(Int, beta/delta_tau)
        if slices != nt.slices
            error(
                "Given slices ($(nt.slices)) does not match calculated slices" * 
                " beta/delta_tau ≈ $(slices)"
            )
        end
    elseif (Set ∘ keys)(nt) == Set([:beta, :slices])
        beta, slices = nt.beta, nt.slices
        delta_tau = beta / slices
    elseif (Set ∘ keys)(nt) == Set([:delta_tau, :slices])
        delta_tau, slices = nt.delta_tau, nt.slices
        beta = delta_tau * slices
    elseif (Set ∘ keys)(nt) == Set([:delta_tau, :beta])
        delta_tau, beta = nt.delta_tau, nt.beta
        slices = round(beta/delta_tau)
        if warn_round && !(slices ≈ beta/delta_tau)
            @warn "beta/delta_tau = $(beta/delta_tau) not an integer. Rounded to $slices"
        end
    else
        error("Invalid keyword arguments to DQMCParameters $nt")
    end
    
    DQMCParameters(
        thermalization,
        sweeps,
        silent, 
        check_sign_problem,
        check_propagation_error,
        safe_mult,
        delta_tau,
        beta,
        slices,
        measure_rate,
        print_rate
    )
end
