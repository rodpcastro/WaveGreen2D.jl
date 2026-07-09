# Chebyshev
`ChebyshevSeries` is the main type of the `Chebyshev` module and represents a n-dimensional Chebyshev series, defined by its coefficients and domain boundaries. The following expressions provide an overview of the algorithm used for its evaluation, using the one-dimensional case as a concrete example.

The 1D Chebyshev series of order $n$ is expressed as

$$
s_n = \sum_{k=0}^{n} a_k T_k(x),
$$

where $a_k$ is a coefficient and $T_k(x)$ is the $k$-th order [Chebyshev polynomial][cheby] of the first kind, defined recursively as

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{k} = 2x T_{k-1}(x) - T_{k-2}(x).
$$

With the coefficients $a_k$ at hand[^1], the *Clenshaw* (1955) algorithm is the recommended method to obtain the value of $s_n$. The algorithm makes use of the following recurrence

$$
\begin{split}
& b_n = a_n, \quad b_{n-1} = a_{n-1} + 2 x b_n, \\
& b_k = a_k + 2 x b_{k+1} - b_{k+2}, \quad k = n-2, \ldots, 1.
\end{split}
$$

With that defined, $s_n$ is given by

$$
s_n = a_0 + x b_1 - b_2.
$$

An extension of this algorithm was obtained by *Skrzipek* (1998) to get the derivatives of any order. For the first and second order derivatives, we define the recurrence relations

$$
\begin{split}
& c_{n-1} = 2 b_n, \quad c_{n-2} = 2 b_{n-1} + 2 x c_{n-1}, \\
& c_k = 2 b_{k+1} + 2 x c_{k+1} - c_{k+2}, \quad k = n-3, \ldots, 1,
\end{split}
$$

$$
\begin{split}
& d_{n-2} = 2 c_{n-1}, \quad d_{n-3} = 2 c_{n-2} + 2 x d_{n-2}, \\
& d_k = 2 c_{k+1} + 2 x d_{k+1} - d_{k+2}, \quad k = n-4, \ldots, 1.
\end{split}
$$

Then, the first and second order derivatives are

$$
\frac{d s_n}{d x} = b_1 + x c_1 - c_2,
$$

$$
\frac{d^2 s_n}{d x^2} = 2(c_1 + x d_1 - d_2).
$$

The *Clenshaw* algorithm and its extension by *Skrzipek* are implemented in the `Chebyshev` Julia module. 

## References
1. C. W. Clenshaw. 1955. A note on the summation of Chebyshev series. Math. Comp. 9 (July 1955), 118–120. https://doi.org/10.1090/S0025-5718-1955-0071856-0
2. M. R. Skrzipek. 1998. Polynomial evaluation and associated polynomials. Numer. Math. 79, 4 (June 1998), 601–613. https://doi.org/10.1007/s002110050354

[^1]: All the Chebyshev coefficients used in `WaveGreen2D` were obtained with [FastChebInterp.jl].

<!--Links-->
[cheby]: https://en.wikipedia.org/wiki/Chebyshev_polynomials
[FastChebInterp.jl]: https://github.com/JuliaMath/FastChebInterp.jl
