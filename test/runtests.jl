using WaveGreen2D
using Test
# using Aqua
# using JET

@testset "WaveGreen2D.jl" begin
    # @testset "Code quality (Aqua.jl)" begin
    #     Aqua.test_all(WaveGreen2D)
    # end
    # @testset "Code linting (JET.jl)" begin
    #     JET.test_package(WaveGreen2D; target_defined_modules=true)
    # end
    # Write your tests here.
    include("test_chebyshev.jl")
end
