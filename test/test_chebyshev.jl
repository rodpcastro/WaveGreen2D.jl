using JLD2
using Test
using StaticArrays
using WaveGreen2D.Chebyshev: ChebyshevSeries, TransformedChebyshevSeries, ChebyshevCluster,
    gradient, hessian, normalize, contains, clenshaw, gradient_clenshaw, hessian_clenshaw


# Note: Chebyshev series must have a least order 4 in each dimension, which means
# that the array of coefficients must have have at least size 5 in each dimension.
# The reason behind this limitation is the Clenshaw algorithm.


@testset "0-D Chebyshev series" begin
    coefs = Array{Float64,0}(undef)
    cs = ChebyshevSeries(coefs, SVector{0,Float64}(), SVector{0,Float64}())
    @test cs == coefs[]
end


@testset "1-D normalize" begin
    cs = ChebyshevSeries(zeros(2), SA[2.0], SA[5.0])
    @test normalize(cs, SA[2.0]) == SA[-1]
    @test normalize(cs, SA[3.5]) == SA[0]
    @test normalize(cs, SA[5.0]) == SA[1]
end


@testset "2-D normalize" begin
    cs = ChebyshevSeries(zeros(2, 2), SA[-5.0, 8.0], SA[0.0, 20.0])
    @test normalize(cs, SA[-5.0, 8.0]) == SA[-1, -1]
    @test normalize(cs, SA[-1.25, 11.0]) == SA[0.5, -0.5]
    @test normalize(cs, SA[0.0, 20.0]) == SA[1, 1]
end


@testset "3-D normalize" begin
    cs = ChebyshevSeries(zeros(2, 2, 2), SA[-2.0, 3.0, 0.0], SA[-1.0, 5.0, 2.0])
    @test normalize(cs, SA[-2.0, 3.0, 0.0]) == SA[-1, -1, -1]
    @test normalize(cs, SA[-1.75, 4.0, 1.5]) == SA[-0.5, 0.0, 0.5]
    @test normalize(cs, SA[-1.0, 5.0, 2.0]) == SA[1, 1, 1]
end


@testset "1-D contains" begin
    cs = ChebyshevSeries(zeros(2), SA[-2.0], SA[4.0])
    @test contains(cs, SA[2.0])
    @test !contains(cs, SA[4.1])
end


@testset "2-D contains" begin
    cs = ChebyshevSeries(zeros(2, 2), SA[5.0, -3.5], SA[6.0, 0.5])
    @test contains(cs, SA[5.1, -1.5])
    @test contains(cs, SA[5.5, -3.0])
    @test !contains(cs, SA[4.9, 0.0])
end


@testset "3-D contains" begin
    cs = ChebyshevSeries(zeros(2, 2, 2), SA[0.5, -1.2, 3.0], SA[1.5, 0.3, 3.5])
    @test contains(cs, SA[1.0, 0.0, 3.2])
    @test contains(cs, SA[1.4, 0.2, 3.1])
    @test !contains(cs, SA[0.5, 0.3, 3.55])
end


@testset "Transformed contains" begin
    u(x::SVector{1,Float64}) = 2x
    ∇u(x::SVector{1,Float64}) = reshape(SA[2.0], Size(1, 1))
    Hu(x::SVector{1,Float64}) = reshape(SA[0.0], Size(1, 1, 1))

    cs = ChebyshevSeries(zeros(2), SA[1.0], SA[2.0])  # u ∈ [1.0, 2.0]
    ts = TransformedChebyshevSeries(cs, u, ∇u, Hu)    # x ∈ [0.5, 1.0]

    @test contains(ts, SA[0.6])
    @test contains(ts, SA[0.9])
    @test !contains(ts, SA[0.4])
    @test !contains(ts, SA[1.1])
end


@testset "Cluster contains" begin
    cs1 = ChebyshevSeries(zeros(2), SA[0.0], SA[1.0])
    cs2 = ChebyshevSeries(zeros(2), SA[1.0], SA[2.0])
    cc = ChebyshevCluster(cs1, cs2)

    i1 = contains(cc, SA[0.5])
    i2 = contains(cc, SA[1.5])
    i3 = contains(cc, SA[3.0])

    @test i1 == 1
    @test i2 == 2
    @test i3 == 0
end


@testset "1-D polynomial function" begin
    a = collect(UnitRange(1.0, 6.0))
    x = SA[0.5]
    @test clenshaw(a, x) == -3
    @test gradient_clenshaw(a, x) == (-3, -42)
    @test hessian_clenshaw(a, x) == (-3, -42, -20)
end


@testset "2-D polynomial function" begin
    a = Float64[3 * i * (j + 1) for i in 1:6, j in 1:5]
    x = SA[-0.5, 0.5]
    @test clenshaw(a, x) == 58.5
    @test gradient_clenshaw(a, x) == (58.5, 273, 117)
    @test hessian_clenshaw(a, x) == (58.5, 273, -2418, 117, 546, -1116)
end


@testset "3-D polynomial function" begin
    a = Float64[0.25 * (i - 1) * (j + 1) * (k + 3) for i in 0:6, j in 0:4, k in 0:5]
    x = SA[-0.5, 0.5, -0.5]
    @test clenshaw(a, x) == 9
    @test gradient_clenshaw(a, x) == (9, -45, 18, 54)
    @test hessian_clenshaw(a, x) == (9, -45, -702, 18, -90, -150, 54, -270, 108, -492)
end


@testset "1-D transcendental function" begin
    # f(x) = sin(x), x ∈ [0.0, π/2]
    @load "coefs/test_chebyshev_1dtf.jld2" coefs lb ub
    cs = ChebyshevSeries(coefs, lb, ub)
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
    # f(x) = cos(x*y/4), x ∈ [-0.05, 0.15]×[0.2, 0.4]
    @load "coefs/test_chebyshev_2dtf.jld2" coefs lb ub
    cs = ChebyshevSeries(coefs, lb, ub)
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
    # f(x) = exp(x*y) * cos(x + z/2), x ∈ [0.5, 0.7]×[-0.2, 0.0]×[1.0, 1.2]
    @load "coefs/test_chebyshev_3dtf.jld2" coefs lb ub
    cs = ChebyshevSeries(coefs, lb, ub)
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


@testset "1-D transformed series" begin
    # f(u) = sin(u²)
    # u(x) = √x
    # u ∈ [0.0, 1.5]
    @load "coefs/test_chebyshev_1dts.jld2" coefs lb ub

    u(x::SVector{1,Float64}) = x .^ 0.5
    ∇u(x::SVector{1,Float64}) = reshape(0.5 * x .^ -0.5, Size(1, 1))
    Hu(x::SVector{1,Float64}) = reshape(-0.25 * x .^ -1.5, Size(1, 1, 1))

    cs = ChebyshevSeries(coefs, lb, ub)
    ts = TransformedChebyshevSeries(cs, u, ∇u, Hu)

    x₀ = 0.71

    f₀ = sin(x₀)
    ∇f₀ = cos(x₀)
    Hf₀ = -sin(x₀)

    @test ts(x₀) ≈ f₀

    y₀, ∇y₀ = gradient(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀

    y₀, ∇y₀, Hy₀ = hessian(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀
    @test Hy₀ ≈ Hf₀
end


@testset "2-D transformed series" begin
    # f(u) = f(r, θ) = exp(r*cos(θ)) * cos(r*sin(θ))
    # u(x) = u(ξ, η) = (r=√(ξ² + η²), θ=atan(η/ξ))
    # u ∈ [0.1, 2.0]×[-0.7, 1.3]
    @load "coefs/test_chebyshev_2dts.jld2" coefs lb ub

    function u(x::SVector{2,Float64})
        ξ, η = x

        r = sqrt(ξ^2 + η^2)
        θ = atan(η / ξ)

        return SA[r, θ]
    end

    function ∇u(x::SVector{2,Float64})
        ξ, η = x

        r² = ξ^2 + η^2
        r = √r²

        r₁ = ξ / r
        r₂ = η / r
        θ₁ = -η / r²
        θ₂ = ξ / r²

        ∇u₁ = SA[r₁, r₂]
        ∇u₂ = SA[θ₁, θ₂]

        return vcat(∇u₁', ∇u₂')
    end

    function Hu(x::SVector{2,Float64})
        ξ, η = x

        ξ², η² = ξ^2, η^2

        r² = ξ² + η²
        r = √r²
        r³ = r² * r
        r⁴ = r³ * r

        r₁₁ = η² / r³
        r₁₂ = -ξ * η / r³
        r₂₁ = r₁₂
        r₂₂ = ξ² / r³

        θ₁₁ = 2 * ξ * η / r⁴
        θ₁₂ = (η² - ξ²) / r⁴
        θ₂₁ = θ₁₂
        θ₂₂ = -2 * ξ * η / r⁴

        Hu₁ = reshape([r₁₁ r₁₂;
                r₂₁ r₂₂], 1, 2, 2)

        Hu₂ = reshape([θ₁₁ θ₁₂;
                θ₂₁ θ₂₂], 1, 2, 2)

        hess_u = vcat(Hu₁, Hu₂)

        return SArray{Tuple{2,2,2},Float64}(hess_u)
    end

    cs = ChebyshevSeries(coefs, lb, ub)
    ts = TransformedChebyshevSeries(cs, u, ∇u, Hu)

    function f(x::SVector{2,Float64})
        ξ, η = x
        return exp(ξ) * cos(η)
    end

    function ∇f(x::SVector{2,Float64})
        ξ, η = x

        f₁ = exp(ξ) * cos(η)
        f₂ = -exp(ξ) * sin(η)

        return SA[f₁, f₂]
    end

    function Hf(x::SVector{2,Float64})
        ξ, η = x

        f₁₁ = exp(ξ) * cos(η)
        f₁₂ = -exp(ξ) * sin(η)
        f₂₁ = f₁₂
        f₂₂ = -exp(ξ) * cos(η)

        return SA[
            f₁₁ f₁₂;
            f₂₁ f₂₂
        ]
    end

    x₀ = SA[0.087, 0.110]

    f₀ = f(x₀)
    ∇f₀ = ∇f(x₀)
    Hf₀ = Hf(x₀)

    @test ts(x₀) ≈ f₀

    y₀, ∇y₀ = gradient(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀

    y₀, ∇y₀, Hy₀ = hessian(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀
    @test Hy₀ ≈ Hf₀
end


@testset "3-D transformed series" begin
    # f(u) = f(r, θ, ϕ) = r² * sin(ϕ) * cos(θ) * cos(ϕ) * exp(-r²)
    # u(x) = u(ξ, η, ζ) = (r=√(ξ² + η² + ζ²), θ=atan(η/ξ), ϕ=acos(ζ/r))
    # u ∈ [0.1, 2.2]×[0.2, 1.8]×[0.3, 1.6]
    @load "coefs/test_chebyshev_3dts.jld2" coefs lb ub

    function u(x::SVector{3,Float64})
        ξ, η, ζ = x

        ρ = sqrt(ξ^2 + η^2)

        r = sqrt(ξ^2 + η^2 + ζ^2)
        θ = atan(η / ξ)
        ϕ = atan(ρ / ζ)

        return SA[r, θ, ϕ]
    end

    function ∇u(x::SVector{3,Float64})
        ξ, η, ζ = x

        r² = ξ^2 + η^2 + ζ^2
        r = √r²
        ρ² = ξ^2 + η^2
        ρ = √ρ²

        r₁ = ξ / r
        r₂ = η / r
        r₃ = ζ / r
        θ₁ = -η / ρ²
        θ₂ = ξ / ρ²
        θ₃ = 0
        ϕ₁ = ξ * ζ / (ρ * r²)
        ϕ₂ = η * ζ / (ρ * r²)
        ϕ₃ = -ρ / r²

        ∇u₁ = SA[r₁, r₂, r₃]
        ∇u₂ = SA[θ₁, θ₂, θ₃]
        ∇u₃ = SA[ϕ₁, ϕ₂, ϕ₃]

        return vcat(∇u₁', ∇u₂', ∇u₃')
    end

    function Hu(x::SVector{3,Float64})
        ξ, η, ζ = x

        ξ² = ξ^2
        η² = η^2
        ζ² = ζ^2

        r² = ξ² + η² + ζ²
        r = √r²
        r³ = r² * r
        r⁴ = r³ * r

        ρ² = ξ² + η²
        ρ = √ρ²
        ρ³ = ρ² * ρ
        ρ⁴ = ρ³ * ρ

        r₁₁ = (η² + ζ²) / r³
        r₁₂ = -ξ * η / r³
        r₁₃ = -ξ * ζ / r³
        r₂₁ = r₁₂
        r₂₂ = (ξ² + ζ²) / r³
        r₂₃ = -η * ζ / r³
        r₃₁ = r₁₃
        r₃₂ = r₂₃
        r₃₃ = ρ² / r³

        θ₁₁ = 2 * ξ * η / ρ⁴
        θ₁₂ = (η² - ξ²) / ρ⁴
        θ₁₃ = 0
        θ₂₁ = θ₁₂
        θ₂₂ = -θ₁₁
        θ₂₃ = 0
        θ₃₁ = θ₁₃
        θ₃₂ = θ₂₃
        θ₃₃ = 0

        ϕ₁₁ = ζ * (-ξ² * ζ² - 3 * ξ² * ρ² + ρ² * r²) / (ρ³ * r⁴)
        ϕ₁₂ = -ξ * η * ζ * (3 * ξ² + 3 * η² + ζ²) / (ρ³ * r⁴)
        ϕ₁₃ = ξ * (ρ² - ζ²) / (ρ * r⁴)
        ϕ₂₁ = ϕ₁₂
        ϕ₂₂ = ζ * (-η² * ζ² - 3 * η² * ρ² + ρ² * r²) / (ρ³ * r⁴)
        ϕ₂₃ = η * (ρ² - ζ²) / (ρ * r⁴)
        ϕ₃₁ = ϕ₁₃
        ϕ₃₂ = ϕ₂₃
        ϕ₃₃ = 2ζ * ρ / r⁴

        Hu₁ = reshape([r₁₁ r₁₂ r₁₃;
                r₂₁ r₂₂ r₂₃;
                r₃₁ r₃₂ r₃₃], 1, 3, 3)

        Hu₂ = reshape([θ₁₁ θ₁₂ θ₁₃;
                θ₂₁ θ₂₂ θ₂₃;
                θ₃₁ θ₃₂ θ₃₃], 1, 3, 3)

        Hu₃ = reshape([ϕ₁₁ ϕ₁₂ ϕ₁₃;
                ϕ₂₁ ϕ₂₂ ϕ₂₃;
                ϕ₃₁ ϕ₃₂ ϕ₃₃], 1, 3, 3)

        hess_u = vcat(Hu₁, Hu₂, Hu₃)

        return SArray{Tuple{3,3,3},Float64}(hess_u)
    end

    cs = ChebyshevSeries(coefs, lb, ub)
    ts = TransformedChebyshevSeries(cs, u, ∇u, Hu)

    function f(x::SVector{3,Float64})
        ξ, η, ζ = x
        return ξ * ζ * exp(-ξ^2 - η^2 - ζ^2)
    end

    function ∇f(x::SVector{3,Float64})
        ξ, η, ζ = x

        ξ², η², ζ² = ξ^2, η^2, ζ^2
        r² = ξ² + η² + ζ²
        er = exp(-r²)

        f₁ = ζ * (1 - 2ξ²) * er
        f₂ = -2 * ξ * η * ζ * er
        f₃ = ξ * (1 - 2ζ²) * er

        return SA[f₁, f₂, f₃]
    end

    function Hf(x::SVector{3,Float64})
        ξ, η, ζ = x

        ξ², η², ζ² = ξ^2, η^2, ζ^2
        r² = ξ² + η² + ζ²
        er = exp(-r²)

        f₁₁ = 2 * ξ * ζ * (2ξ² - 3) * er
        f₁₂ = 2 * η * ζ * (2ξ² - 1) * er
        f₁₃ = (4 * ξ² * ζ² - 2ξ² - 2ζ² + 1) * er
        f₂₁ = f₁₂
        f₂₂ = 2 * ξ * ζ * (2η² - 1) * er
        f₂₃ = 2 * ξ * η * (2ζ² - 1) * er
        f₃₁ = f₁₃
        f₃₂ = f₂₃
        f₃₃ = 2 * ξ * ζ * (2ζ² - 3) * er

        return SA[
            f₁₁ f₁₂ f₁₃;
            f₂₁ f₂₂ f₂₃;
            f₃₁ f₃₂ f₃₃
        ]
    end

    x₀ = SA[0.391, 0.472, 0.607]

    f₀ = f(x₀)
    ∇f₀ = ∇f(x₀)
    Hf₀ = Hf(x₀)

    @test ts(x₀) ≈ f₀

    y₀, ∇y₀ = gradient(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀

    y₀, ∇y₀, Hy₀ = hessian(ts, x₀)
    @test y₀ ≈ f₀
    @test ∇y₀ ≈ ∇f₀
    @test Hy₀ ≈ Hf₀
end


@testset "Identity transformation" begin
    @load "coefs/test_chebyshev_1dtf.jld2" coefs
    cs = ChebyshevSeries(coefs, 0.0, 0.5 * π)
    ts = TransformedChebyshevSeries(cs)

    x = 1.42

    @test ts(x) ≈ sin(x)

    y, dy_dx = gradient(ts, x)
    @test y ≈ sin(x)
    @test dy_dx ≈ cos(x)

    y, dy_dx, d²y_dx² = hessian(ts, x)
    @test y ≈ sin(x)
    @test dy_dx ≈ cos(x)
    @test d²y_dx² ≈ -sin(x)
end


@testset "Transformed series return type error" begin
    x_type = SVector{1,Float64}
    u_type = SVector{1,Float64}
    ∇u_type = SMatrix{1,1,Float64,1}
    Hu_type = SArray{Tuple{1,1,1},Float64,3,1}

    u(x::x_type) = zero(u_type)
    ∇u(x::x_type) = zero(∇u_type)
    Hu(x::x_type) = zero(Hu_type)

    u_bad_input(x::Vector) = zero(u_type)
    ∇u_bad_input(x::Vector) = zero(u_type)
    Hu_bad_input(x::Vector) = zero(u_type)

    u_bad_output(x::x_type) = 0.0
    ∇u_bad_output(x::x_type) = Vector{Float64}(undef, 1)
    Hu_bad_output(x::x_type) = Matrix{Float64}(undef, 1, 1)

    u_boom(x::x_type) = error("boom")

    cs = ChebyshevSeries(zeros(2), SA[-1.5], SA[0.5])

    err_x_type = ErrorException(
        "The transformation function must accept an argument of type $x_type"
    )
    err_u_type = ErrorException(
        "The transformation function must return a $u_type"
    )
    err_∇u_type = ErrorException(
        "The gradient of the transformation function must return a $∇u_type"
    )
    err_Hu_type = ErrorException(
        "The hessian of the transformation function must return a $Hu_type"
    )
    err_boom = ErrorException("boom")

    @test_throws err_x_type TransformedChebyshevSeries(cs, u_bad_input, ∇u, Hu)
    @test_throws err_x_type TransformedChebyshevSeries(cs, u, ∇u_bad_input, Hu)
    @test_throws err_x_type TransformedChebyshevSeries(cs, u, ∇u, Hu_bad_input)

    @test_throws err_u_type TransformedChebyshevSeries(cs, u_bad_output, ∇u, Hu)
    @test_throws err_∇u_type TransformedChebyshevSeries(cs, u, ∇u_bad_output, Hu)
    @test_throws err_Hu_type TransformedChebyshevSeries(cs, u, ∇u, Hu_bad_output)

    @test_throws err_boom TransformedChebyshevSeries(cs, u_boom, ∇u, Hu_bad_output)
end


@testset "1-D cluster" begin
    # f(x) = exp(cos(x/2)), x ∈ [0.0, 0.5]∪[0.5, 1.0]
    @load "coefs/test_chebyshev_1dcc.jld2" coefs1 lb1 ub1 coefs2 lb2 ub2
    cs₁ = ChebyshevSeries(coefs1, lb1, ub1)
    cs₂ = ChebyshevSeries(coefs2, lb2, ub2)
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
