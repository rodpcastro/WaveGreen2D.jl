using WaveGreen2D
using Test
using Aqua
using JET

@testset "WaveGreen2D.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(WaveGreen2D)
    end

    @testset "Code linting (JET.jl)" begin
        JET.test_package(WaveGreen2D; target_modules=(WaveGreen2D,))
    end

    include("test_utils.jl")
    include("test_near_field_integrals.jl")
end
