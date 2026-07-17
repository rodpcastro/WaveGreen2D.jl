module WaveGreen2D

export create_wave

using StaticArrays: SVector, SMatrix, MMatrix, SArray, Size


include("utils.jl")
include("wave.jl")

# Submodules
include("NearField.jl")
using .NearField

end # module
