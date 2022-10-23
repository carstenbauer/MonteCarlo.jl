include("fallbacks.jl")

const FVec64 = Vector{Float64}
const FMat64 = Matrix{Float64}

# Our custom linalg has two jobs:
# 1. Make basic stuff faster via LoopVectorization
include("real.jl")
include("complex.jl")
include("UDT.jl")

# 2. Add wrappers for specific matrix shapes
include("blockdiagonal.jl")

# This is not as generic...
include("updates.jl")

# Used for measurements to differentiate [M 0; 0 M] from M
include("repeating.jl")

# Checkerboard decomposition
include("checkerboard2.jl")

# There is some additional glue code here

# Not sure if this should be here
generalized_eltype(::Type{Float64},    ::Type{Float64})    = Float64
generalized_eltype(::Type{ComplexF64}, ::Type{Float64})    = ComplexF64
generalized_eltype(::Type{Float64},    ::Type{ComplexF64}) = ComplexF64
generalized_eltype(::Type{ComplexF64}, ::Type{ComplexF64}) = ComplexF64


# This hopefully simiplies some stuff
vector_type(::Type{Float64}) = FVec64
vector_type(::Type{ComplexF64}) = CVec64
matrix_type(::Type{Float64}) = FMat64
matrix_type(::Type{ComplexF64}) = CMat64