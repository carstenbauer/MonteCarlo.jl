include("fallbacks.jl")

# Our custom linalg has two jobs:
# 1. Make basic stuff faster via LoopVectorization
include("real.jl")
include("complex.jl")
include("UDT.jl")

# 2. Add wrappers for specific matrix shapes
include("blockdiagonal.jl")
