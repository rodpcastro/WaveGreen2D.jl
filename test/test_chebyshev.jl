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


@testset "1-D transcedental function" begin
    # Coefficients for f(x) = sin(x)
    coefs = [
        6.0219470125554653e-01, 5.1362516667910696e-01, -1.0354634426296377e-01,
        -1.3732034234358511e-02, 1.3586698380903617e-03, 1.0726309440600129e-04,
        -7.0462967934650502e-06, -3.9639025062457094e-07, 1.9499597787191747e-08,
        8.5229287361514171e-10, -3.3516569268963781e-11, -1.1979172283756631e-12,
        3.9247734881581288e-14, 1.1922754450388594e-15,
    ]
    cs = ChebyshevSeries(coefs, SA[0.0], SA[π/2])
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
