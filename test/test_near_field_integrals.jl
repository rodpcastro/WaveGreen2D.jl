# Tests for the Chebyshev series that approximate the integrals L₁ and L₂.

using Chebyshaw: gradient, hessian
using Random
using StaticArrays
using Test
using WaveGreen2D.NearField: integrals, setintegrals!

include("integrals.jl")


# Testing points are generated randomly.
Random.seed!(18)

# Number of testing points
npts = 20

# Tolerances
tol_L = 10^2 * eps()
tol_∇L = 10^4 * eps()
tol_HL = 10^6 * eps()


@testset "L₁ Chebyshev" begin
    # Testing points
    lb = SVector{3, Float64}(0.0, 0.0, 1e-2)
    ub = SVector{3, Float64}(0.5, 1.0, 7.0)
    points = [lb .+ rand(SVector{3,Float64}) .* (ub .- lb) for _ in 1:npts]

    # Pre-allocating data to be obtained with QuadGK.jl
    L = Vector{Float64}(undef, npts)
    ∇L = Matrix{Float64}(undef, npts, 2)
    HL = Array{Float64,3}(undef, npts, 2, 2)

    # Pre-allocating data to be obtained with Chebyshaw.jl
    C = Vector{Float64}(undef, npts)
    Cg = Vector{Float64}(undef, npts)
    Ch = Vector{Float64}(undef, npts)

    ∇C = Matrix{Float64}(undef, npts, 2)
    ∇Ch = Matrix{Float64}(undef, npts, 2)

    HC = Array{Float64,3}(undef, npts, 2, 2)

    for (i, x) in enumerate(points)
        setintegrals!(x[3])

        # Data obtained with QuadGK.jl
        y, ∇y, Hy = HL₁(x)

        L[i] = y
        ∇L[i, :] .= ∇y[1:2]
        HL[i, :, :] .= Hy[1:2, 1:2]

        # Data obtained with Chebyshaw.jl
        u = SVector{2, Float64}(x[1], x[2])
        y₁ = integrals.L₁(u)
        y₂, ∇y₂ = gradient(integrals.L₁, u)
        y₃, ∇y₃, Hy₃ = hessian(integrals.L₁, u)

        C[i] = y₁
        Cg[i] = y₂
        Ch[i] = y₃
        ∇C[i, :] .= ∇y₂
        ∇Ch[i, :] .= ∇y₃
        HC[i, :, :] .= Hy₃
    end

    # Test if ChebyshevSeries, gradient and hessian return the same value of L₁
    @test C == Cg
    @test C == Ch

    # Test if gradient and hessian return the same value of ∇L₁
    @test ∇C == ∇Ch

    @test all(isapprox.(L, C; rtol=tol_L, atol=tol_L))
    @test all(isapprox.(∇L, ∇C; rtol=tol_∇L, atol=tol_∇L))
    @test all(isapprox.(HL, HC; rtol=tol_HL, atol=tol_HL))
end


@testset "L₂ Chebyshev" begin
    # Testing points
    lb = SVector{3, Float64}(0.0, 0.0, 1e-2)
    ub = SVector{3, Float64}(0.5, 2.0, 7.0)
    points = [lb .+ rand(SVector{3,Float64}) .* (ub .- lb) for _ in 1:npts]

    # Pre-allocating data to be obtained with QuadGK.jl
    L = Vector{Float64}(undef, npts)
    ∇L = Matrix{Float64}(undef, npts, 2)
    HL = Array{Float64,3}(undef, npts, 2, 2)

    # Pre-allocating data to be obtained with Chebyshaw.jl
    C = Vector{Float64}(undef, npts)
    Cg = Vector{Float64}(undef, npts)
    Ch = Vector{Float64}(undef, npts)

    ∇C = Matrix{Float64}(undef, npts, 2)
    ∇Ch = Matrix{Float64}(undef, npts, 2)

    HC = Array{Float64,3}(undef, npts, 2, 2)

    for (i, x) in enumerate(points)
        setintegrals!(x[3])

        # Data obtained with QuadGK.jl
        y, ∇y, Hy = HL₂(x)

        L[i] = y
        ∇L[i, :] .= ∇y[1:2]
        HL[i, :, :] .= Hy[1:2, 1:2]

        # Data obtained with Chebyshaw.jl
        u = SVector{2, Float64}(x[1], x[2])
        y₁ = integrals.L₂(u)
        y₂, ∇y₂ = gradient(integrals.L₂, u)
        y₃, ∇y₃, Hy₃ = hessian(integrals.L₂, u)

        C[i] = y₁
        Cg[i] = y₂
        Ch[i] = y₃
        ∇C[i, :] .= ∇y₂
        ∇Ch[i, :] .= ∇y₃
        HC[i, :, :] .= Hy₃
    end

    # Test if ChebyshevSeries, gradient and hessian return the same value of L₂
    @test C == Cg
    @test C == Ch

    # Test if gradient and hessian return the same value of ∇L₂
    @test ∇C == ∇Ch

    @test all(isapprox.(L, C; rtol=tol_L, atol=tol_L))
    @test all(isapprox.(∇L, ∇C; rtol=tol_∇L, atol=tol_∇L))
    @test all(isapprox.(HL, HC; rtol=tol_HL, atol=tol_HL))
end


@testset "Integrals DomainError" begin
    @test_throws DomainError setintegrals!(1e-2 * (1.0 - eps()))
    @test_throws DomainError setintegrals!(7.0 * (1.0 + eps()))
end
