# Test accessory functions stored in utils.jl

using Test
using WaveGreen2D: findroot


@testset "Root finding" begin
    f(x) = x^2 - 2
    f′(x) = 2x

    tol = 1e-12

    x₀ = findroot(f, f′, 1.0, tol)

    @test isapprox(x₀, √2; rtol=tol, atol=tol)
    @test isapprox(f(x₀), 0.0; atol=tol)
end


@testset "Non-convergent root finding" begin
    f(x) = tan(x) - x
    f′(x) = tan(x)^2

    @test_logs (
        :warn,
        "Reached maximum number of iterations without convergence"
    ) findroot(f, f′, π/2)
end
