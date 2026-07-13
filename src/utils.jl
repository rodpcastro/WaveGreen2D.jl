"""
    findroot(y, y′, x₀::Real, tol::Real=eps(), nmax::Int=50) -> Float64

Finds the root of `y` closer to `x₀`, if this root exists, using the
Newton-Raphson method.

# Arguments
- `y`: the function to find the root of
- `y′`: the derivative of `y`
- `x₀::Real`: initial estimate for the root of `y`
- `tol::Real=eps()`: tolerance for the absolute error
- `nmax::Int=50`: maximum number of iterations

# Returns
- `Float64`: final estimate for the root of `y`

# Warnings
- A warning is raised if the maximum number of iterations is reached without convergence.
"""
function findroot(y, y′, x₀::Real, tol::Real=1e-12, nmax::Int=50)
    xᵢ = x₀
    yᵢ = y(x₀)
    imax = 0
    converged = false

    for i in 1:nmax
        xᵢ = xᵢ - yᵢ / y′(xᵢ)
        yᵢ = y(xᵢ)
        imax = i
        if abs(yᵢ) ≤ tol
            converged = true
            break
        end
    end

    if !converged
        @warn "Reached maximum number of iterations ($nmax) without convergence (y = $yᵢ)"
    end

    return xᵢ
end
