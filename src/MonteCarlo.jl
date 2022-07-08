module MonteCarlo

using LinearAlgebra: AbstractMatrix
using Reexport
# Loading the RNG will fail if Random is nto exported
@reexport using Random, BinningAnalysis
using Parameters, Requires
using TimerOutputs, StructArrays
using Printf, SparseArrays, LinearAlgebra, Dates, Statistics, Random, Distributed

if get(ENV, "MONTECARLO_USE_LOOPVECTORIZATION", "true") == "true"
    import LoopVectorization
    using LoopVectorization: @turbo
else
    printstyled(
        "Using MonteCarlo.jl without LoopVectorization. This should only be done for tests.",
        color = :red
    )
    macro turbo(code)
        esc(quote @inbounds @fastmath $code end)
    end
end

import JLD2, CodecZlib

include("helpers.jl")
export enable_benchmarks, disable_benchmarks, print_timer, reset_timer!

mutable struct FileData
    data::Dict{String, Any}
    subspace::String
    path::String
end
FileData(data, path) = FileData(data, "", path)
@bm function Base.getindex(f::FileData, key::String)
    subspace = f.subspace * key * '/' 
    if any(k -> startswith(k, subspace), keys(f.data))
        return FileData(f.data, subspace, f.path)
    elseif haskey(f.data, f.subspace * key)
        return getindex(f.data, f.subspace * key)
    else
        return getindex(f.data, key)
    end
end
@bm function Base.haskey(f::FileData, key::String)
    subspace = f.subspace * key
    haskey(f.data, key) || any(k -> startswith(k, subspace), keys(f.data))
end
@bm function Base.get(f::FileData, k::String, v)
    return haskey(f, k) ? getindex(f, k) : v
end
@bm function Base.keys(f::FileData)
    _keys = String[]
    i0 = length(f.subspace)
    for key in keys(f.data)
        if startswith(key, f.subspace)
            i1 = something(findnext(isequal('/'), key, i0+1), length(key)+1)
            _key = string(key[i0+1:i1-1])
            _key in _keys || push!(_keys, _key)
        end
    end
    return _keys
end

const FileLike = Union{JLD2.JLDFile, JLD2.Group, FileData}
filepath(f::FileLike) = f.path
filepath(g::JLD2.Group) = g.f.path


include("lattices/lattice.jl")
include("flavors/abstract.jl")
include("models/abstract.jl")

include("configurations.jl")
export Discarder, ConfigRecorder, BufferedConfigRecorder, RelativePath, AbsolutePath
include("Measurements.jl")
export measurements, observables

include("lattices/constructors.jl")
include("lattices/lattice_cache.jl")
include("lattices/lattice_iterators.jl")
include("lattices/directions.jl")

export AbstractLattice, Lattice, Bravais
export Chain, SquareLattice, Honeycomb, CubicLattice
# export AbstractLattice, Chain, SquareLattice, CubicLattice, TriangularLattice, ALPSLattice
export EachSite, EachSiteAndFlavor, OnSite, EachSitePair, EachSitePairByDistance, 
        EachLocalQuadByDistance, EachLocalQuadBySyncedDistance, 
        Sum, ApplySymmetries
export unitcell, positions, bonds, directions

include("flavors/MC/MC.jl")
include("flavors/DQMC/main.jl")
export Greens, GreensAt, CombinedGreensIterator, TimeIntegral
export GreensMatrix, swapop
export boson_energy_measurement, greens_measurement, occupation, magnetization
export charge_density, charge_density_correlation, charge_density_susceptibility
export spin_density, spin_density_correlation, spin_density_susceptibility
export pairing, pairing_correlation, pairing_susceptibility
export current_current_susceptibility, superfluid_density
export noninteracting_energy, interacting_energy, total_energy

export EmptyScheduler, SimpleScheduler, AdaptiveScheduler
export Adaptive, NoUpdate, LocalSweep
export GlobalFlip, GlobalShuffle, SpatialShuffle, TemporalShuffle, 
        Denoise, DenoiseFlip, StaggeredDenoise
export ReplicaExchange, ReplicaPull, connect, disconnect
# export mask, uniform_fourier, structure_factor, SymmetryWrapped, swave, eswave

include("models/Ising/IsingModel.jl")
include("models/HubbardModel.jl")
include("models/DummyModel.jl")
export IsingEnergyMeasurement, IsingMagnetizationMeasurement

include("FileIO.jl")
include("BinningAnalysis.jl")
export save, load, resume!


export reset!
export run!, resume!, replay!
export Model, IsingModel
# export RepulsiveGHQHubbardModel, AttractiveGHQHubbardModel
export HubbardModel, HubbardModelAttractive, HubbardModelRepulsive
export MonteCarloFlavor, MC, DQMC
export greens, greens!, lattice, model, parameters
export DensityHirschField, MagneticHirschField, DensityGHQField, MagneticGHQField

# For extending
export AbstractMeasurement, Model

import Git
const git = let 
    olddir = pwd()
    cd(pkgdir(MonteCarlo))
    if isdir(".git")
        branch = readchomp(`$(Git.git()) rev-parse --abbrev-ref HEAD`)
        commit = readchomp(`$(Git.git()) rev-parse HEAD`)
        dirty  = !isempty(readchomp(`$(Git.git()) diff --name-only`)) || # unstaged w/o new files
                 !isempty(readchomp(`$(Git.git()) diff --name-only --cached`)) # staged
        git = (branch = branch, commit = commit, dirty = dirty)
    else
        @warn "Cannot identify git information because MonteCarlo.jl is not a git repository. "
        git = (branch = "unknown", commit = "unknown", dirty = "false")
    end
    cd(olddir)
    git
end


function __init__()
    # @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
    @require LatticePhysics = "53011200-ee7a-11e8-39f1-5f3e57afe4fd" include("lattices/LatPhys.jl")
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        include("mpi.jl")
        export mpi_queue
        
        include("flavors/DQMC/updates/mpi_updates.jl")
        export MPIReplicaExchange, MPIReplicaPull
    end
end

end # module
