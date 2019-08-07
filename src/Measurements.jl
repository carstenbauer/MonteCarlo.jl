struct Measurement{DataT, ObsT <: MonteCarloObservable.AbstractObservable}
    obs::ObsT
    data::DataT
    init!::Function
    update_data!::Function
    measure!::Function
    finish!::Function
end

# this function will be skipped when generating measure!
donothing(args...) = nothing

function Measurement(
        obs;
        data = nothing,
        init! = donothing,
        update_data! = donothing,
        measure! = donothing,
        finish! = donothing
    )
    Measurement(obs, data, init!, update_data!, measure!, finish!)
end


function initialize!(measurements::Vector{M}, mc, model) where {M <: Measurement}
    indices = Int64[]
    for (i, m) in enumerate(measurements)
        # Skip if there is no data or no init!
        # This sort of assumes the user initializes if they don't give us an
        # init!, which may be reasonable in some situations
        # m.data == nothing && continue
        if m.data == nothing
            m.init! != donothing && @warn(
                "`init!` defined but no data to initialize for $(name(m.obs))"
            )
            m.update_data! != donothing && @warn(
                "`update_data!` defined but no data to update for $(name(m.obs))"
            )
            continue
        end
        m.init! == donothing && continue


        # Check if var has been initialized already
        # If the init! functions don't match, get angry
        found = false
        for j in indices
            if m.data == measurements[j].data
                found = true
                m.init! == measurements[j].init! && break
                throw(AssertionError(
                    "Measurements that use the same data must either use have" *
                    " `init! = donothing` or use the same `init!` function. " *
                    "This is not the case for measurements[$j]: " *
                    "$(name(measurements[j].obs)) with `init! = " *
                    "$(measurements[j].init!)` and measurements[$i]: " *
                    "$(name(m.obs)) with `init! = $(m.init!)`."
                ))
            end
        end

        # if data has not been initialized, do it
        if !found
            m.init!(m.obs, m.data, mc, model)
            push!(indices, i)
        end
    end

    # generate a concrete measure! function
    # more specifically a measure! function where m.update_data! and
    # m.measure! are concrete
    # Not doing this has a significant performance penalty (2x slower)
    code = quote
        function measure!(measurements, mc, model, args...)
            $([
                :($(m.update_data!)(
                    measurements[$i].data,
                    mc, model, args...
                ))
                for (i, m) in enumerate(measurements)
                if m.update_data! != donothing
            ]...)

            $([
                :($(m.measure!)(
                    measurements[$i].obs,
                    measurements[$i].data,
                    mc, model, args...
                ))
                for (i, m) in enumerate(measurements)
                if m.measure! != donothing
            ]...)
            nothing
        end
    end

    eval(code)

    code
    # nothing
end

function finish!(measurements, mc, model)
    for m in measurements
        m.finish!(m.obs, m.data, mc, model)
    end
    nothing
end
