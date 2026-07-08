module WaveGreen2D

export setwave!

using StaticArrays: SVector, SMatrix, MMatrix, SArray, Size


include("wave_params.jl")


# Submodules
include("Chebyshev/Chebyshev.jl")
include("NearField.jl")

using .Chebyshev
using .NearField

end
