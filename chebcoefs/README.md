# Chebyshev coefficients
Coefficients for Chebyshev series defined in the main code are obtained by the [`wavegreen_coefficients.jl`][wavecoefs] script, while those created for testing are calculated by [`test_coefficients.jl`][testcoefs]. All coefficients are computed with the package [FastChebInterp.jl].

## Evaluating $L_1$ and $L_2$
The integrals $L_1$ and $L_2$ are the main objects of approximation by Chebyshev series in WaveGreen2D. Their analytical expressions are given in detail [here][wavegreen2d]. Their numerical evaluation in Julia, with the [QuadGK.jl] package, required some experiments due to the singular nature of the integrands.

The numerical experiment was performed with [BenchmarkTools.jl], and studied how the quadrature `order`, parameter of [`quadgk`][quadgk], influences execution time, memory allocated and number of allocations. Each quadrature was executed for 1000 random points in the domain of interest, and the averages are presented in the following image, created with [Makie].

<p align="center">
    <img src="images/quadrature_order.svg" alt="Quadrature order">
</p>

The optimal value of quadrature order ranges between 24 and 34. The integrals $L_1$ and $L_2$ are evaluated with the `order` parameter of `quadgk` varying in this range. If a particular choice of `order` does not make the integral converge to the desired accuracy, another value in the range is chosen and the computation is repeated. This slight change in the quadrature process proved itself successful in avoiding non-convergent calculations.

## Chebyshev coefficients for $L_1$ and $L_2$

## References
1. Steven Johnson. 2021. FastChebInterp.jl. https://github.com/JuliaMath/FastChebInterp.jl
2. Julia Math. 2016. QuadGK.jl. https://github.com/JuliaMath/QuadGK.jl


<!--Links-->
[wavegreen2d]: https://github.com/rodpcastro/WaveGreen2D.jl/tree/main/src
[wavecoefs]: https://github.com/rodpcastro/WaveGreen2D.jl/blob/main/chebcoefs/wavegreen_coefficients.jl
[testcoefs]: https://github.com/rodpcastro/WaveGreen2D.jl/blob/main/chebcoefs/test_coefficients.jl
[quadorder]: https://github.com/rodpcastro/WaveGreen2D.jl/blob/main/chebcoefs/quadrature_order.jl
[QuadGK.jl]: https://github.com/JuliaMath/QuadGK.jl
[quadgk]: https://juliamath.github.io/QuadGK.jl/stable/api/#quadgk
[FastChebInterp.jl]: https://github.com/JuliaMath/FastChebInterp.jl
[BenchmarkTools.jl]: https://github.com/JuliaCI/BenchmarkTools.jl
[Makie]: https://docs.makie.org/
