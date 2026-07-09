module WaveGreen2D

export setwave!

using StaticArrays: SVector, SMatrix, MMatrix, SArray, Size


include("wave_params.jl")


# Submodules
include("NearField.jl")

using .NearField

end
