"""
    clenshaw(a::Array{T, N}, x::T) where {T, N} -> Array{T, N-1}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev
series with coefficients `a` at a normalized value `x` of its `N`-th dimension.
"""
function clenshaw(a::Array{T,N}, x::T) where {T,N}
    # m = n+1, where n is the Chebyshev series order along the N-th dimension.
    m = size(a, N)
    dx = 2x

    aₘ₋₁, aₘ = (selectdim(a, N, i) for i in m-1:m)
    bₖ, bₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)

    # bₖ used on the right-hand side actually represents bₖ₊₂.
    # bₖ₊₂ is ommited to reduce allocations.

    # k = m-2 to 2.
    @. bₖ = aₘ  # Here, bₖ is bₖ₊₂
    @. bₖ₊₁ = aₘ₋₁ + dx * bₖ

    for k in m-2:-1:2
        aₖ = selectdim(a, N, k)
        @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
        bₖ, bₖ₊₁ = bₖ₊₁, bₖ
    end

    # k = 1
    aₖ = selectdim(a, N, 1)
    @. bₖ = aₖ + x * bₖ₊₁ - bₖ

    return bₖ
end


"""
    clenshaw(a::Array{T, N}, x::SVector{N, T}) where {T, N} -> T

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev
series with coefficients `a` at a normalized point `x` in ``[-1, 1]^N``.
"""
function clenshaw(a::Array{T,N}, x::SVector{N,T}) where {T,N}
    b = clenshaw(a, x[N])
    xᴺ⁻¹ = pop(x)
    return clenshaw(b, xᴺ⁻¹)
end


function clenshaw(a::Array{T,1}, x::SVector{1,T}) where T
    b = clenshaw(a, x[1])
    return b[]
end


"""
    clenshaw(f::ChebyshevSeries{T,N}, x::T) where {T,N} -> ChebyshevSeries{T,N-1}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional
Chebyshev series at a normalized value `x` of its `N`-th dimension.
"""
function clenshaw(f::ChebyshevSeries{T,N}, x::T) where {T,N}
    coefs = clenshaw(f.coefs, x)
    lb = SVector(ntuple(i -> f.lb[i], Val(N - 1)))
    ub = SVector(ntuple(i -> f.ub[i], Val(N - 1)))
    return ChebyshevSeries(coefs, lb, ub)
end


"""
    (f::ChebyshevSeries{T, N})(x::SVector{N, T}) where {T, N} -> T

Evaluates the Chebyshev series `f` at a point `x`.
"""
function (f::ChebyshevSeries{T,N})(x::SVector{N,T}) where {T,N}
    x̄ = normalize(f, x)
    y = clenshaw(f.coefs, x̄)
    return y
end


"""
    (f::ChebyshevSeries{T, N})(x::SVector{N, T}) where {T, N} -> T

Evaluates the transformed Chebyshev series `g` at a point `x`, which
is equivalent to evaluating `g.series` at the point `g.u(x)`.
"""
function (g::TransformedChebyshevSeries{T,N})(x::SVector{N,T}) where {T,N}
    return g.series(g.u(x))
end


"""
    (h::ChebyshevCluster{T, N})(x::SVector{N, T}) where {T, N} -> T

Evaluates the Chebyshev cluster `h` at a point `x`.
"""
function (h::ChebyshevCluster{T,N})(x::SVector{N,T}) where {T,N}
    i = contains(h, x)
    i == 0 && throw(DomainError(x))
    return h.series[i](x)
end


"""
    (f::AbstractChebyshevSeries{T, N})(x::AbstractVector{T}) where {T, N} -> T

Simpler callable constructor for evaluating a Chebyshev series `f`
at a point `x`, where `x` is of any subtype of an `AbstractVector{T}`.
"""
function (f::AbstractChebyshevSeries{T,N})(x::AbstractVector{T}) where {T,N}
    return f(SVector{N,T}(x))
end


"""
    (f::AbstractChebyshevSeries{T, 1})(x::T) where T -> T

Simpler callable constructor for evaluating a one-dimensional
Chebyshev series `f` at a point `x`, where `x` is of type `T`.
"""
function (f::AbstractChebyshevSeries{T,1})(x::T) where T
    return f(SVector{1,T}(x))
end
