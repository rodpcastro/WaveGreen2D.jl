# Integrals L₁ and L₂ and their derivatives computed with the QuadGK package.

using QuadGK


tol = eps()
imax = 1e4
qorder = 26


function mod_quadgk(f, a, b; rtol=sqrt(eps()), atol=0, maxevals=10^7, order=7)
    # Put 26 as the first try.
    qorder_vals = [[26, 25, 24]; collect(27:34)]

    if !(order in qorder_vals)
        pushfirst!(qorder_vals, order)
    end

    ∫f = err = nothing

    for qo in qorder_vals
        ∫f, err, count = quadgk_count(
            f, a, b; rtol=rtol, atol=atol, maxevals=maxevals, order=qo
        )
        if count < maxevals
            return ∫f, err
        end
    end

    @warn "Reached the maximum number of function evaluations" #maxlog=1

    return ∫f, err
end


function L₁(x::AbstractVector{<:Real})
    A, B, H = x

    f(u) = (u + H) / (u - H - (u + H) * exp(-2u))
    g(u) = exp(-u * (2 + B)) + exp(-u * (2 - B))
    h(u) = cos(u * A)
    p(u) = (f(u) * g(u) * h(u) + exp(-u)) / u

    path = (0.0, H + im, H + 1.0, Inf)

    y = mod_quadgk(p, path[1], path[2]; rtol=tol, atol=tol, maxevals=imax, order=qorder)[1]
    y += mod_quadgk(p, path[2], path[3]; rtol=tol, atol=tol, maxevals=imax, order=qorder)[1]
    y += mod_quadgk(p, path[3], path[4]; rtol=tol, atol=tol, maxevals=imax)[1]

    return real(y)
end


function L₂(x::AbstractVector{<:Real})
    A, B, H = x

    f(u) = (u + H) / (u - H - (u + H) * exp(-2u))
    g(u) = (u + H)^2 / ((u - H)^2 - (u^2 - H^2) * exp(-2u))
    p(u) = exp(-u * (2 + B))
    q(u) = exp(-u * (4 - B))
    r(u) = cos(u * A)

    h(u) = (f(u) * p(u) + g(u) * q(u)) * r(u) / u

    path = (0.0, H + im, H + 1.0, Inf)

    y = mod_quadgk(h, path[1], path[2]; rtol=tol, atol=tol, maxevals=imax, order=qorder)[1]
    y += mod_quadgk(h, path[2], path[3]; rtol=tol, atol=tol, maxevals=imax, order=qorder)[1]
    y += mod_quadgk(h, path[3], path[4]; rtol=tol, atol=tol, maxevals=imax)[1]

    return real(y)
end


function HL₁(x::AbstractVector{<:Real})
    A, B, H = x

    f(u) = (u + H) / (u - H - (u + H) * exp(-2u))
    f₃(u) = 2 / (u - H - (u + H) * exp(-2u))^2
    f₃₃(u) = 4 * (1 + exp(-2u)) / (u - H - (u + H) * exp(-2u))^3

    g(u) = exp(-u * (2 + B)) + exp(-u * (2 - B))
    g₂(u) = exp(-u * (2 - B)) - exp(-u * (2 + B))
    g₂₂(u) = exp(-u * (2 + B)) + exp(-u * (2 - B))

    h(u) = cos(u * A)
    h₁(u) = -sin(u * A)
    h₁₁(u) = -cos(u * A)

    p(u) = (f(u) * g(u) * h(u) + exp(-u)) / u

    p₁(u) = f(u) * g(u) * h₁(u)
    p₂(u) = f(u) * g₂(u) * h(u)
    p₃(u) = f₃(u) * g(u) * h(u)

    p₁₁(u) = f(u) * g(u) * h₁₁(u) * u
    p₁₂(u) = f(u) * g₂(u) * h₁(u) * u
    p₁₃(u) = f₃(u) * g(u) * h₁(u) * u

    p₂₁(u) = f(u) * g₂(u) * h₁(u) * u
    p₂₂(u) = f(u) * g₂₂(u) * h(u) * u
    p₂₃(u) = f₃(u) * g₂(u) * h(u) * u

    p₃₁(u) = f₃(u) * g(u) * h₁(u) * u
    p₃₂(u) = f₃(u) * g₂(u) * h(u) * u
    p₃₃(u) = f₃₃(u) * g(u) * h(u)

    path = (0.0, H + im, H + 1.0, Inf)
    keyword_args = (rtol=tol, atol=tol, maxevals=imax, order=qorder)

    y = 0.0
    y₁, y₂, y₃ = [0.0 for _ in 1:3]
    y₁₁, y₁₂, y₁₃, y₂₂, y₂₃, y₃₃ = [0.0 for _ in 1:6]

    for i in 1:length(path)-1
        y += mod_quadgk(p, path[i], path[i+1]; keyword_args...)[1]

        y₁ += mod_quadgk(p₁, path[i], path[i+1]; keyword_args...)[1]
        y₂ += mod_quadgk(p₂, path[i], path[i+1]; keyword_args...)[1]
        y₃ += mod_quadgk(p₃, path[i], path[i+1]; keyword_args...)[1]

        y₁₁ += mod_quadgk(p₁₁, path[i], path[i+1]; keyword_args...)[1]
        y₁₂ += mod_quadgk(p₁₂, path[i], path[i+1]; keyword_args...)[1]
        y₁₃ += mod_quadgk(p₁₃, path[i], path[i+1]; keyword_args...)[1]
        y₂₂ += mod_quadgk(p₂₂, path[i], path[i+1]; keyword_args...)[1]
        y₂₃ += mod_quadgk(p₂₃, path[i], path[i+1]; keyword_args...)[1]
        y₃₃ += mod_quadgk(p₃₃, path[i], path[i+1]; keyword_args...)[1]
    end

    y = real(y)

    ∇y = real([y₁, y₂, y₃])

    Hy = real([
        y₁₁ y₁₂ y₁₃;
        y₁₂ y₂₂ y₂₃;
        y₁₃ y₂₃ y₃₃
    ])

    return y, ∇y, Hy
end


function HL₂(x::AbstractVector{<:Real})
    A, B, H = x

    # Auxilary variables for f, g and its derivatives
    a(u) = u + H
    b(u) = u - H
    c(u) = (b(u) - a(u) * exp(-2u))
    d(u) = b(u)^2 - a(u) * b(u) * exp(-2u)
    e(u) = b(u)^2 + d(u)
    dH(u) = -2 * (b(u) - H * exp(-2u))

    f(u) = a(u) / c(u)
    f₃(u) = 2 / c(u)^2
    f₃₃(u) = 4 * (1 + exp(-2u)) / c(u)^3

    g(u) = a(u)^2 / d(u)
    g₃(u) = 2 * a(u) * (b(u)^2 + d(u)) / (b(u) * d(u)^2)
    g₃₃(u) = 2 * (-2a(u) * b(u) * e(u) * dH(u) + a(u) * d(u) * e(u)
                  + b(u) * d(u) * (e(u) - a(u) * (2b(u) - dH(u)))) / (b(u)^2 * d(u)^3)

    p(u) = exp(-u * (2 + B))
    p₂(u) = -exp(-u * (2 + B))
    p₂₂(u) = u * exp(-u * (2 + B))

    q(u) = exp(-u * (4 - B))
    q₂(u) = exp(-u * (4 - B))
    q₂₂(u) = u * exp(-u * (4 - B))

    r(u) = cos(u * A)
    r₁(u) = -sin(u * A)
    r₁₁(u) = -u * cos(u * A)

    h(u) = (f(u) * p(u) + g(u) * q(u)) * r(u) / u

    h₁(u) = (f(u) * p(u) + g(u) * q(u)) * r₁(u)
    h₂(u) = (f(u) * p₂(u) + g(u) * q₂(u)) * r(u)
    h₃(u) = (f₃(u) * p(u) + g₃(u) * q(u)) * r(u)

    h₁₁(u) = (f(u) * p(u) + g(u) * q(u)) * r₁₁(u)
    h₁₂(u) = (f(u) * p₂(u) + g(u) * q₂(u)) * r₁(u) * u
    h₁₃(u) = (f₃(u) * p(u) + g₃(u) * q(u)) * r₁(u) * u
    h₂₂(u) = (f(u) * p₂₂(u) + g(u) * q₂₂(u)) * r(u)
    h₂₃(u) = (f₃(u) * p₂(u) + g₃(u) * q₂(u)) * r(u) * u
    h₃₃(u) = (f₃₃(u) * p(u) + g₃₃(u) * q(u)) * r(u)

    path = (0.0, H + im, H + 1.0, Inf)
    keyword_args = (rtol=tol, atol=tol, maxevals=imax, order=qorder)

    y = 0.0
    y₁, y₂, y₃ = [0.0 for _ in 1:3]
    y₁₁, y₁₂, y₁₃, y₂₂, y₂₃, y₃₃ = [0.0 for _ in 1:6]

    for i in 1:length(path)-1
        y += mod_quadgk(h, path[i], path[i+1]; keyword_args...)[1]

        y₁ += mod_quadgk(h₁, path[i], path[i+1]; keyword_args...)[1]
        y₂ += mod_quadgk(h₂, path[i], path[i+1]; keyword_args...)[1]
        y₃ += mod_quadgk(h₃, path[i], path[i+1]; keyword_args...)[1]

        y₁₁ += mod_quadgk(h₁₁, path[i], path[i+1]; keyword_args...)[1]
        y₁₂ += mod_quadgk(h₁₂, path[i], path[i+1]; keyword_args...)[1]
        y₁₃ += mod_quadgk(h₁₃, path[i], path[i+1]; keyword_args...)[1]
        y₂₂ += mod_quadgk(h₂₂, path[i], path[i+1]; keyword_args...)[1]
        y₂₃ += mod_quadgk(h₂₃, path[i], path[i+1]; keyword_args...)[1]
        y₃₃ += mod_quadgk(h₃₃, path[i], path[i+1]; keyword_args...)[1]
    end

    y = real(y)

    ∇y = real([y₁, y₂, y₃])

    Hy = real([
        y₁₁ y₁₂ y₁₃;
        y₁₂ y₂₂ y₂₃;
        y₁₃ y₂₃ y₃₃
    ])

    return y, ∇y, Hy
end
