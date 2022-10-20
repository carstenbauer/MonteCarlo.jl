################################################################################
### FileIO
################################################################################

to_tag(::Type{<: DQMC}) = Val(:DQMC)
to_tag(::Type{<: DQMCParameters}) = Val(:DQMCParameters)
to_tag(::Type{<: DQMCAnalysis}) = Val(:DQMCAnalysis)
to_tag(::Type{<: MagnitudeStats}) = Val(:MagnitudeStats)

function _save(file::FileLike, entryname::String, mc::DQMC)
    write(file, entryname * "/VERSION", 3)
    write(file, entryname * "/tag", "DQMC")
    _save(file, entryname * "/Parameters", mc.parameters)
    _save(file, entryname * "/Analysis", mc.analysis)
    _save(file, entryname * "/field", mc.field)
    _save(file, entryname * "/configs", mc.recorder)
    write(file, entryname * "/last_sweep", mc.last_sweep)
    save_measurements(file, entryname * "/Measurements", mc)
    _save(file, entryname * "/Model", mc.model)
    _save(file, entryname * "/Scheduler", mc.scheduler)
    nothing
end


function _load(data, ::Val{:DQMC})
    if data["VERSION"] > 3
        throw(ErrorException("Failed to load DQMC version $(data["VERSION"])"))
    end

    
    parameters = _load(data["Parameters"])
    if haskey(data, "CB")
        parameters = DQMCParameters(parameters, checkerboard = data["CB"])
    end
    tag = Val(Symbol(get(data["Analysis"], "tag", :DQMCAnalysis)))
    analysis = _load(data["Analysis"], tag)
    recorder = _load(data["configs"], to_tag(data["configs"]))
    last_sweep = data["last_sweep"]
    model = load_model(data["Model"], to_tag(data["Model"]))
    if haskey(data, "field")
        tag = Val(Symbol(get(data["field"], "tag", :Field)))
        field = _load(data["field"], tag, parameters, model)
    else
        conf = data["conf"]
        field = field_hint(model, to_tag(data["Model"]))(parameters, model)
        conf!(field, conf)
    end
    scheduler = if haskey(data, "Scheduler")
        _load(data["Scheduler"])
    else
        if haskey(data["Parameters"], "global_moves") && Bool(data["Parameters"]["global_moves"])
            rate = get(data["Parameters"], "global_rate", 10)
            @warn "Replacing `global_moves = true` with GlobalFlip"
            SimpleScheduler(LocalSweep(rate), GlobalFlip())
        else
            SimpleScheduler(LocalSweep())
        end
    end

    tag = Val(Symbol(data["Measurements"], "tag", :Measurements))
    combined_measurements = _load(data["Measurements"])
    thermalization_measurements = combined_measurements[:TH]
    measurements = combined_measurements[:ME]

    stack = DQMCStack(field, model, parameters.checkerboard)
    ut_stack = UnequalTimeStack{geltype(stack), gmattype(stack)}()
    
    DQMC(
        model, field, last_sweep, 
        stack, ut_stack, scheduler,
        parameters, analysis, 
        recorder, thermalization_measurements, measurements
    )
end


function _save(file::FileLike, entryname::String, p::DQMCParameters)
    write(file, entryname * "/VERSION", 3)
    write(file, entryname * "/tag", "DQMCParameters")

    write(file, entryname * "/thermalization", p.thermalization)
    write(file, entryname * "/sweeps", p.sweeps)
    write(file, entryname * "/silent", Int(p.silent))
    write(file, entryname * "/check_sign_problem", Int(p.check_sign_problem))
    write(file, entryname * "/check_propagation_error", Int(p.check_propagation_error))
    write(file, entryname * "/safe_mult", p.safe_mult)
    write(file, entryname * "/delta_tau", p.delta_tau)
    write(file, entryname * "/beta", p.beta)
    write(file, entryname * "/slices", p.slices)
    write(file, entryname * "/measure_rate", p.measure_rate)
    write(file, entryname * "/print_rate", p.print_rate)
    write(file, entryname * "/checkerboard", p.checkerboard)

    nothing
end


function _load(data, ::Val{:DQMCParameters})
    if !(data["VERSION"] in (1, 2, 3))
        throw(ErrorException("Failed to load DQMCParameters version $(data["VERSION"])"))
    end

    DQMCParameters(
        data["thermalization"],
        data["sweeps"],

        Bool(data["silent"]),
        Bool(data["check_sign_problem"]),
        Bool(data["check_propagation_error"]),

        data["safe_mult"],
        data["delta_tau"],
        data["beta"],
        data["slices"],

        data["measure_rate"],
        haskey(data, "print_rate") ? data["print_rate"] : 10,
        haskey(data, "checkerboard") ? data["checkerboard"] : false,
    )
end

function _save(file::FileLike, entryname::String, a::DQMCAnalysis)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "DQMCAnalysis")
    write(file, entryname * "/type", typeof(a))

    write(file, entryname * "/th_runtime", a.th_runtime)
    write(file, entryname * "/me_runtime", a.me_runtime)
    _save(file, entryname * "/imag_prob", a.imaginary_probability)
    _save(file, entryname * "/neg_prob", a.negative_probability)
    _save(file, entryname * "/propagation", a.propagation_error)
end
function _save(file::FileLike, entryname::String, ms::MagnitudeStats)
    write(file, entryname * "/max", ms.max)
    write(file, entryname * "/min", ms.min)
    write(file, entryname * "/sum", ms.sum)
    write(file, entryname * "/count", ms.count)
end

function _load(data, ::Val{:DQMCAnalysis})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load DQMCAnalysis version $(data["VERSION"])"))
    end

    DQMCAnalysis(
        th_runtime = get(data, "th_runtime", 0.0),
        me_runtime = get(data, "me_runtime", 0.0),
        imaginary_probability = load_stats(data["imag_prob"]),
        negative_probability  = load_stats(data["neg_prob"]),
        propagation_error     = load_stats(data["propagation"])
    )
end
function load_stats(data)
    MagnitudeStats(data["max"], data["min"], data["sum"], data["count"])
end