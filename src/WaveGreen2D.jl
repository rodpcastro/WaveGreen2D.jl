module WaveGreen2D

export create_wave

# using StaticArrays: SVector, SMatrix, MMatrix, SArray, Size

# Submodules
include("Wave/Wave.jl")
include("FiniteDepth/FiniteDepth.jl")
include("InfiniteDepth.jl")

using .Wave
using .FiniteDepth
using .InfiniteDepth

end # module
