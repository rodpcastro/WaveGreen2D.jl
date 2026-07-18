module FiniteDepth

include("FarField.jl")
include("NearField/NearField.jl")

using .FarField
using .NearField

end # module
