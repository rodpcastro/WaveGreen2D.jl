module WaveGreen2D

include("Chebyshev/Chebyshev.jl")
using .Chebyshev

"""
    function testing(a::T, b::T) where T

Testing function.
"""
function testing(a::T, b::T) where T
    return a + b
end

end
