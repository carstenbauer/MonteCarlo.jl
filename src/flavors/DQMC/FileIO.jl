################################################################################
### FileIO
################################################################################



#     save_mc(filename, mc, entryname)
#
# Saves (minimal) information necessary to reconstruct a given `mc::DQMC` to a
# JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC`
function save_mc(file::JLDFile, mc::DQMC, entryname::String="MC")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(mc))
    save_parameters(file, mc.p, entryname * "/Parameters")
    save_analysis(file, mc.a, entryname * "/Analysis")
    write(file, entryname * "/conf", mc.conf)
    _save(file, mc.configs, entryname * "/configs")
    write(file, entryname * "/last_sweep", mc.last_sweep)
    save_measurements(file, mc, entryname * "/Measurements")
    save_model(file, mc.model, entryname * "/Model")
    nothing
end

CB_type(T::UnionAll) = T.body.parameters[2]
CB_type(T::DataType) = T.parameters[2]

#     load_mc(data, ::Type{<: DQMC})
#
# Loads a DQMC from a given `data` dictionary produced by `JLD.load(filename)`.
function _load(data, ::Type{T}) where T <: DQMC
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    CB = CB_type(data["type"])
    @assert CB <: Checkerboard
    mc = DQMC(CB)
    mc.p = _load(data["Parameters"], data["Parameters"]["type"])
    mc.a = _load(data["Analysis"], data["Analysis"]["type"])
    mc.conf = data["conf"]
    mc.configs = _load(data["configs"], data["configs"]["type"])
    mc.last_sweep = data["last_sweep"]
    mc.model = _load(data["Model"], data["Model"]["type"])

    measurements = _load(data["Measurements"], Measurements)
    mc.thermalization_measurements = measurements[:TH]
    mc.measurements = measurements[:ME]
    HET = hoppingeltype(DQMC, mc.model)
    GET = greenseltype(DQMC, mc.model)
    HMT = hopping_matrix_type(DQMC, mc.model)
    GMT = greens_matrix_type(DQMC, mc.model)
    IMT = interaction_matrix_type(DQMC, mc.model)
    mc.s = DQMCStack{GET, HET, GMT, HMT, IMT}()
    mc.ut_stack = UnequalTimeStack{GET, GMT}()
    
    make_concrete!(mc)
end

#   save_parameters(file::JLDFile, p::DQMCParameters, entryname="Parameters")
#
# Saves (minimal) information necessary to reconstruct a given
# `p::DQMCParameters` to a JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC/Parameters`
function save_parameters(file::JLDFile, p::DQMCParameters, entryname::String="Parameters")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(p))

    write(file, entryname * "/global_moves", Int(p.global_moves))
    write(file, entryname * "/global_rate", p.global_rate)
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
function _load(data, ::Type{T}) where T <: DQMCParameters
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    data["type"](
        Bool(data["global_moves"]),
        data["global_rate"],
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

function _load(data, ::Type{T}) where T <: DQMCAnalysis
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    data["type"](
        imaginary_probability = load_stats(data["imag_prob"]),
        negative_probability = load_stats(data["neg_prob"]),
        propagation_error = load_stats(data["propagation"])
    )
end
function load_stats(data)
    MagnitudeStats(data["max"], data["min"], data["sum"], data["count"])
end