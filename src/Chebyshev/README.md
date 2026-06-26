# Chebyshev
`ChebyshevSeries` is the main type of the `Chebyshev` module and represents a n-dimensional Chebyshev series, defined by its coefficients and domain boundaries. The following expressions provide an overview of the algorithm used for its evaluation, using the one-dimensional case as a concrete example.

The 1D Chebyshev series of order $n$ is expressed as

$$
s_n = \sum_{k=0}^{n} a_k T_k(x),
$$

where $a_k$ is a coefficient and $$T_k(x)$ is the $k$-th order [Chebyshev polynomial][cheby] of the first kind, defined recursively as

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{k} = 2x T_{k-1}(x) - T_{k-2}(x).
$$

With the coefficients $a_k$ at hand[^1], 

## References
1. C. W. Clenshaw. 1955. A note on the summation of Chebyshev series. Math. Comp. 9 (July 1955), 118–120. https://doi.org/10.1090/S0025-5718-1955-0071856-0
2. M. R. Skrzipek. 1998. Polynomial evaluation and associated polynomials. Numer. Math. 79, 4 (June 1998), 601–613. https://doi.org/10.1007/s002110050354

[^1]: All Chebyshev coefficients used in `WaveGreen2D` were obtained with [`FastChebInterp.jl`].

<!--Links-->
[cheby]: https://en.wikipedia.org/wiki/Chebyshev_polynomials
[FastChebInterp.jl]: https://github.com/JuliaMath/FastChebInterp.jl
