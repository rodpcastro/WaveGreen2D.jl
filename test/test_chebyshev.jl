using JLD2
using Test
using StaticArrays
using WaveGreen2D.Chebyshev
using WaveGreen2D.Chebyshev: normalize, contains, clenshaw, gradient_clenshaw, hessian_clenshaw


# Note: Chebyshev series must have a least order 4 in each dimension, which means
# that the array of coefficients must have have at least size 5 in each dimension.
# The reason behind this limitation is the Clenshaw algorithm.


@testset "0-D Chebyshev series" begin
    coefs = Array{Float64,0}(undef)
    cs = ChebyshevSeries(coefs, SVector{0}(), SVector{0}())
    @test cs == coefs[]
end


@testset "1-D normalize" begin
    cs = ChebyshevSeries(zeros(2), SA[2.0], SA[5.0])
    @test normalize(cs, 2.0) == -1
    @test normalize(cs, 3.5) == 0
    @test normalize(cs, 5.0) == 1
end


@testset "2-D normalize" begin
    cs = ChebyshevSeries(zeros(2, 2), SA[-5.0, 8.0], SA[0.0, 20.0])
    @test normalize(cs, SA[-5.0, 8.0]) == [-1, -1]
    @test normalize(cs, SA[-1.25, 11.0]) == [0.5, -0.5]
    @test normalize(cs, SA[0.0, 20.0]) == [1, 1]
end


@testset "3-D normalize" begin
    cs = ChebyshevSeries(zeros(2, 2, 2), SA[-2.0, 3.0, 0.0], SA[-1.0, 5.0, 2.0])
    @test normalize(cs, SA[-2.0, 3.0, 0.0]) == [-1, -1, -1]
    @test normalize(cs, SA[-1.75, 4.0, 1.5]) == [-0.5, 0.0, 0.5]
    @test normalize(cs, SA[-1.0, 5.0, 2.0]) == [1, 1, 1]
end


@testset "1-D contains" begin
    cs = ChebyshevSeries(zeros(2), SA[-2.0], SA[4.0])
    @test contains(cs, 2.0)
    @test !contains(cs, 4.1)
end


@testset "2-D contains" begin
    cs = ChebyshevSeries(zeros(2, 2), SA[5.0, -3.5], SA[6.0, 0.5])
    @test contains(cs, SA[5.1, -1.5])
    @test contains(cs, [5.5, -3.0])
    @test !contains(cs, SA[4.9, 0.0])
end


@testset "3-D contains" begin
    cs = ChebyshevSeries(zeros(2, 2, 2), SA[0.5, -1.2, 3.0], SA[1.5, 0.3, 3.5])
    @test contains(cs, SA[1.0, 0.0, 3.2])
    @test contains(cs, [1.4, 0.2, 3.1])
    @test !contains(cs, SA[0.5, 0.3, 3.55])
end


@testset "Cluster contains" begin
    cs1 = ChebyshevSeries(zeros(2), SA[0.0], SA[1.0])
    cs2 = ChebyshevSeries(zeros(2), SA[1.0], SA[2.0])
    cc = ChebyshevCluster(cs1, cs2)

    check_in1, i1 = contains(cc, 0.5)
    check_in2, i2 = contains(cc, 1.5)
    check_in3, _ = contains(cc, 3.0)

    @test check_in1
    @test i1 == 1
    @test check_in2
    @test i2 == 2
    @test !check_in3
end


@testset "1-D polynomial function" begin
    coefs = collect(UnitRange(1.0, 6.0))
    cs = ChebyshevSeries(coefs, SA[-1.0], SA[1.0])
    x = 0.5
    @test clenshaw(cs, x) == -3
    @test gradient_clenshaw(cs, x) == (-3, -42)
    @test hessian_clenshaw(cs, x) == (-3, -42, -20)
end


@testset "2-D polynomial function" begin
    coefs = Float64[3 * i * (j + 1) for i in 1:6, j in 1:5]
    cs = ChebyshevSeries(coefs, SA[-1.0, -1.0], SA[1.0, 1.0])
    x = SA[-0.5, 0.5]
    @test clenshaw(cs, x) == 58.5
    @test gradient_clenshaw(cs, x) == (58.5, 273, 117)
    @test hessian_clenshaw(cs, x) == (58.5, 273, -2418, 117, 546, -1116)
end


@testset "3-D polynomial function" begin
    coefs = Float64[0.25 * (i - 1) * (j + 1) * (k + 3) for i in 0:6, j in 0:4, k in 0:5]
    cs = ChebyshevSeries(coefs, SA[-1.0, -1.0, -1.0], SA[1.0, 1.0, 1.0])
    x = SA[-0.5, 0.5, -0.5]
    @test clenshaw(cs, x) == 9
    @test gradient_clenshaw(cs, x) == (9, -45, 18, 54)
    @test hessian_clenshaw(cs, x) == (9, -45, -702, 18, -90, -150, 54, -270, 108, -492)
end


@testset "1-D transcendental function" begin
    # f(x) = sin(x) in [0.0, π/2]
    @load "test_chebyshev_1dtf.jld2" coefs
    cs = ChebyshevSeries(coefs, SA[0.0], SA[0.5*π])
    x = π / 3

    @test cs(x) ≈ sin(x)

    y, dy_dx = gradient(cs, x)
    @test y ≈ sin(x)
    @test dy_dx ≈ cos(x)

    y, dy_dx, d²y_dx² = hessian(cs, x)
    @test y ≈ sin(x)
    @test dy_dx ≈ cos(x)
    @test d²y_dx² ≈ -sin(x)
end


@testset "2-D transcendental function" begin
    # f(x) = cos(x*y/4) in [-0.05, 0.15]×[0.2, 0.4]
    @load "test_chebyshev_2dtf.jld2" coefs
    cs = ChebyshevSeries(coefs, SA[-0.05, 0.2], SA[0.15, 0.4])
    x̄ = [0.1, 0.3]
    x, y = x̄

    qxy = 0.25 * x * y

    f = cos(qxy)
    df_dx = -0.25 * y * sin(qxy)
    df_dy = -0.25 * x * sin(qxy)
    d²f_dx² = -0.0625 * y^2 * cos(qxy)
    d²f_dxy = -0.0625 * (x * y * cos(qxy) + 4 * sin(qxy))
    d²f_dy² = -0.0625 * x^2 * cos(qxy)

    @test cs(x̄) ≈ f

    z, ∇z = gradient(cs, x̄)
    @test z ≈ f
    @test ∇z[1] ≈ df_dx
    @test ∇z[2] ≈ df_dy

    z, ∇z, Hz = hessian(cs, x̄)
    @test z ≈ f
    @test ∇z[1] ≈ df_dx
    @test ∇z[2] ≈ df_dy
    @test Hz[1, 1] ≈ d²f_dx²
    @test Hz[1, 2] ≈ d²f_dxy
    @test Hz[2, 1] ≈ d²f_dxy
    @test Hz[2, 2] ≈ d²f_dy²
end


@testset "3-D transcendental function" begin
    # f(x) = exp(x*y) * cos(x + z/2) in [0.5, 0.7]×[-0.2, 0.0]×[1.0, 1.2]
    @load "test_chebyshev_3dtf.jld2" coefs
    cs = ChebyshevSeries(coefs, SA[0.5, -0.2, 1.0], SA[0.7, 0.0, 1.2])
    x̄ = [0.57, -0.02, 1.13]
    x, y, z = x̄

    exy = exp(x * y)
    sxz = sin(x + 0.5 * z)
    cxz = cos(x + 0.5 * z)

    f = exy * cxz
    df_dx = exy * (y * cxz - sxz)
    df_dy = x * exy * cxz
    df_dz = -0.5 * exy * sxz
    d²f_dx² = exy * (y^2 * cxz - 2 * y * sxz - cxz)
    d²f_dxy = exy * (x * y * cxz - x * sxz + cxz)
    d²f_dxz = -0.5 * exy * (y * sxz + cxz)
    d²f_dy² = x^2 * exy * cxz
    d²f_dyz = -0.5 * x * exy * sxz
    d²f_dz² = -0.25 * exy * cxz

    @test cs(x̄) ≈ f

    w, ∇w = gradient(cs, x̄)
    @test w ≈ f
    @test ∇w[1] ≈ df_dx
    @test ∇w[2] ≈ df_dy
    @test ∇w[3] ≈ df_dz

    w, ∇w, Hw = hessian(cs, x̄)
    @test w ≈ f
    @test ∇w[1] ≈ df_dx
    @test ∇w[2] ≈ df_dy
    @test ∇w[3] ≈ df_dz
    @test Hw[1, 1] ≈ d²f_dx²
    @test Hw[1, 2] ≈ d²f_dxy
    @test Hw[1, 3] ≈ d²f_dxz
    @test Hw[2, 1] ≈ d²f_dxy
    @test Hw[2, 2] ≈ d²f_dy²
    @test Hw[2, 3] ≈ d²f_dyz
    @test Hw[3, 1] ≈ d²f_dxz
    @test Hw[3, 2] ≈ d²f_dyz
    @test Hw[3, 3] ≈ d²f_dz²
end


@testset "1-D cluster" begin
    # f(x) = exp(cos(x/2)) in [0.0, 0.5]∪[0.5, 1.0]
    @load "test_chebyshev_1dcc.jld2" coefs1 coefs2
    cs₁ = ChebyshevSeries(coefs1, SA[0.0], SA[0.5])
    cs₂ = ChebyshevSeries(coefs2, SA[0.5], SA[1.0])
    cc = ChebyshevCluster(cs₁, cs₂)

    x₁, x₂ = 0.35, 0.9

    sx₁, sx₂ = sin.(0.5 .* [x₁, x₂])
    cx₁, cx₂ = cos.(0.5 .* [x₁, x₂])

    f₁ = exp(cx₁)
    f₂ = exp(cx₂)
    df₁_dx = -0.5 * f₁ * sx₁
    df₂_dx = -0.5 * f₂ * sx₂
    d²f₁_dx² = 0.25 * f₁ * (sx₁^2 - cx₁)
    d²f₂_dx² = 0.25 * f₂ * (sx₂^2 - cx₂)

    @test cc(x₁) ≈ f₁
    @test cc(x₂) ≈ f₂

    y₁, dy₁_dx = gradient(cc, x₁)
    y₂, dy₂_dx = gradient(cc, x₂)
    @test y₁ ≈ f₁
    @test y₂ ≈ f₂
    @test dy₁_dx ≈ df₁_dx
    @test dy₂_dx ≈ df₂_dx

    y₁, dy₁_dx, d²y₁_dx² = hessian(cc, x₁)
    y₂, dy₂_dx, d²y₂_dx² = hessian(cc, x₂)
    @test y₁ ≈ f₁
    @test y₂ ≈ f₂
    @test dy₁_dx ≈ df₁_dx
    @test dy₂_dx ≈ df₂_dx
    @test d²y₁_dx² ≈ d²f₁_dx²
    @test d²y₂_dx² ≈ d²f₂_dx²
end


@testset "1-D cluster DomainError" begin
    cs₁ = ChebyshevSeries(zeros(2), SA[-1.0], SA[-0.5])
    cs₂ = ChebyshevSeries(ones(2), SA[-0.5], SA[2.5])
    cc = ChebyshevCluster(cs₁, cs₂)

    @test_throws DomainError cc(-2.0)
    @test_throws DomainError gradient(cc, 3.0)
    @test_throws DomainError hessian(cc, 4.0)
end
