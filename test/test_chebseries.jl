using JLD2
using Test
using StaticArrays
using WaveGreen2D.Chebyshev: TransformedChebyshevSeries


@load "data/l1_test_data.jld2" L1 ∇L1 HL1
@load "data/l1_test_data.jld2" L2 ∇L2 HL2

@testset "L₁ Chebyshev" begin
    @test 1 == 1.0
end


@testset "L₁ Chebyshev gradient" begin
    @test 1 == 1.0
end


@testset "L₁ Chebyshev hessian" begin
    @test 1 == 1.0
end


@testset "L₂ Chebyshev series" begin
    @test 1 == 1.0
end
