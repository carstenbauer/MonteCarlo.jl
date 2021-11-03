################################################################################
### FileIO
################################################################################

to_tag(::Type{<: DQMC}) = Val(:DQMC)
to_tag(::Type{<: DQMCParameters}) = Val(:DQMCParameters)
to_tag(::Type{<: DQMCAnalysis}) = Val(:DQMCAnalysis)
to_tag(::Type{<: MagnitudeStats}) = Val(:MagnitudeStats)


#     save_mc(filename, mc, entryname)
#
# Saves (minimal) information necessary to reconstruct a given `mc::DQMC` to a
# JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC`
function save_mc(file::JLDFile, mc::DQMC, entryname::String="DQMC")
    write(file, entryname * "/VERSION", 2)
    write(file, entryname * "/tag", "DQMC")
    write(file, entryname * "/CB", mc isa DQMC_CBTrue)
    save_parameters(file, mc.parameters, entryname * "/Parameters")
    save_analysis(file, mc.analysis, entryname * "/Analysis")
    write(file, entryname * "/conf", mc.conf)
    _save(file, mc.recorder, entryname * "/configs")
    write(file, entryname * "/last_sweep", mc.last_sweep)
    save_measurements(file, mc, entryname * "/Measurements")
    save_model(file, mc.model, entryname * "/Model")
    save_scheduler(file, mc.scheduler, entryname * "/Scheduler")
    nothing
end

CB_type(T::UnionAll) = T.body.parameters[2]
CB_type(T::DataType) = T.parameters[2]

#     load_mc(data, ::Type{<: DQMC})
#
# Loads a DQMC from a given `data` dictionary produced by `JLD.load(filename)`.
function _load(data, ::Val{:DQMC})
    if data["VERSION"] > 2
        throw(ErrorException("Failed to load DQMC version $(data["VERSION"])"))
    end

    CB = if haskey(data, "CB")
        data["CB"] ? CheckerboardTrue : CheckerboardFalse
    else CB_type(data["type"]) end
    @assert CB <: Checkerboard
    parameters = _load(data["Parameters"], Val(:DQMCParameters))
    analysis = _load(data["Analysis"], Val(:DQMCAnalysis))
    conf = data["conf"]
    recorder = _load(data["configs"], to_tag(data["configs"]))
    last_sweep = data["last_sweep"]
    model = try
        _load(data["Model"], to_tag(data["Model"]))
    catch e
        _load(data["Model"], Val(:DummyModel))
    end
    scheduler = if haskey(data, "Scheduler")
        _load(data["Scheduler"], to_tag(data["Scheduler"]))
    else
        if haskey(data["Parameters"], "global_moves") && Bool(data["Parameters"]["global_moves"])
            rate = get(data["Parameters"], "global_rate", 10)
            @warn "Replacing `global_moves = true` with GlobalFlip"
            SimpleScheduler(LocalSweep(rate), GlobalFlip())
        else
            SimpleScheduler(LocalSweep())
        end
    end

    combined_measurements = _load(data["Measurements"], Val(:Measurements))
    thermalization_measurements = combined_measurements[:TH]
    measurements = combined_measurements[:ME]

    HET = hoppingeltype(DQMC, model)
    GET = greenseltype(DQMC, model)
    HMT = hopping_matrix_type(DQMC, model)
    GMT = greens_matrix_type(DQMC, model)
    IMT = interaction_matrix_type(DQMC, model)
    stack = DQMCStack{GET, HET, GMT, HMT, IMT}()
    ut_stack = UnequalTimeStack{GET, GMT}()
    
    DQMC(
        CB, 
        model, conf, deepcopy(conf), last_sweep, 
        stack, ut_stack, scheduler,
        parameters, analysis, 
        recorder, thermalization_measurements, measurements
    )
end

#   save_parameters(file::JLDFile, p::DQMCParameters, entryname="Parameters")
#
# Saves (minimal) information necessary to reconstruct a given
# `p::DQMCParameters` to a JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC/Parameters`
function save_parameters(file::JLDFile, p::DQMCParameters, entryname::String="Parameters")
    write(file, entryname * "/VERSION", 1)
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

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function _load(data, ::Val{:DQMCParameters})
    if !(data["VERSION"] == 1)
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
    )
end

function save_analysis(file::JLDFile, a::DQMCAnalysis, entryname::String="Analysis")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(a))

    write(file, entryname * "/th_runtime", a.th_runtime)
    write(file, entryname * "/me_runtime", a.me_runtime)
    save_stats(file, a.imaginary_probability, entryname * "/imag_prob")
    save_stats(file, a.negative_probability, entryname * "/neg_prob")
    save_stats(file, a.propagation_error, entryname * "/propagation")
end
function save_stats(file::JLDFile, ms::MagnitudeStats, entryname::String="MStats")
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