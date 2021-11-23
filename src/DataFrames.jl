import .DataFrames

adjust_prediction(old, new) = 0.95old + 0.05new

function load(
        files::Vector{String}; 
        simplify = true, replace = identity, skip = Symbol[]
    )
    mc = load(first(files))
    param = parameters(mc)
    header = Dict{Symbol, Any}()
    for key in keys(param)
        key in skip && continue
        p = replace(key => param[key])
        push!(header, p[1] => typeof(p[2])[])
    end
    push!(header, :mc => DQMC[])
    df = DataFrames.DataFrame(header)

    N = length(files)
    Nstr = string(N)

    time_prediction = 1.0

    for (i, f) in enumerate(files)
        t = @sprintf("%3.2fs", time_prediction * (N - i + 1))
        print("\r[", lpad(i, length(Nstr), ' '), "/$N] ($t) Loading $f             ")
        t0 = time()
        mc = load(f)
        data = Dict{Symbol, Any}()

        # Parameters for this simulation
        param = parameters(mc)
        for key in keys(param)
            key in skip && continue
            push!(data, replace(key => param[key]))
        end
        
        # reduce measurements to values
        if simplify
            for key in keys(mc)
                to_value!(mc, Val(key))
            end
        end
        push!(data, :mc => mc)

        # commit row
        push!(df, data)
        dt = time() - t0
        if i == 1
            time_prediction = dt
        else
            time_prediction = adjust_prediction(time_prediction, dt)
        end
    end
    println()

    df
end