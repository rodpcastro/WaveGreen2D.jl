"""
    gradient_clenshaw(a::Array{T, N}, x::T) where {T, N} -> Array{T, N-1}, Array{T, N-1}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev series
with coefficients `a` and its gradient at a normalized value `x` of its `N`-th dimension.
"""
function gradient_clenshaw(a::Array{T,N}, x::T) where {T,N}
    # m = n+1, where n is the Chebyshev series order along the N-th dimension.
    m = size(a, N)
    dx = 2x

    aₖ, aₘ₋₁, aₘ = (selectdim(a, N, i) for i in m-2:m)
    bₖ, bₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)
    cₖ, cₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)

    # bₖ used on the right-hand side actually represents bₖ₊₂.
    # bₖ₊₂ is ommited to reduce allocations. Idem for cₖ₊₂.

    # k = m-2
    @. bₖ = aₘ  # Here, bₖ is bₖ₊₂
    @. bₖ₊₁ = aₘ₋₁ + dx * bₖ
    @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
    bₖ, bₖ₊₁ = bₖ₊₁, bₖ

    # k = m-3 to 2
    @. cₖ = 2aₘ  # Here, cₖ is cₖ₊₂
    @. cₖ₊₁ = 2bₖ + dx * cₖ

    for k in m-3:-1:2
        aₖ = selectdim(a, N, k)
        @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
        @. cₖ = 2bₖ₊₁ + dx * cₖ₊₁ - cₖ
        bₖ, bₖ₊₁ = bₖ₊₁, bₖ
        cₖ, cₖ₊₁ = cₖ₊₁, cₖ
    end

    # k = 1
    aₖ = selectdim(a, N, 1)
    @. bₖ = aₖ + x * bₖ₊₁ - bₖ
    @. cₖ = bₖ₊₁ + x * cₖ₊₁ - cₖ

    return bₖ, cₖ
end


"""
    gradient_clenshaw(a::Array{T, N}, x::SVector{N, T}) where {T, N} -> T, SVector{N, T}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev series
with coefficients `a` and its gradient at a normalized point `x` in ``[-1, 1]^N``.
"""
function gradient_clenshaw(a::Array{T,N}, x::SVector{N,T}) where {T,N}
    b, c = gradient_clenshaw(a, x[N])
    xᴺ⁻¹ = pop(x)
    return gradient_clenshaw(b, xᴺ⁻¹)..., clenshaw(c, xᴺ⁻¹)
end


function gradient_clenshaw(a::Array{T,1}, x::SVector{1,T}) where T
    b, c = gradient_clenshaw(a, x[1])
    return b[], c[]
end


"""
    gradient(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N} -> T, SVector{N, T}

Evaluates the Chebyshev series `f` and its gradient at a point `x`.
"""
function gradient(f::ChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    x̄ = normalize(f, x)
    dx̄_dx = @. 2.0 / (f.ub - f.lb)

    res = gradient_clenshaw(f.coefs, x̄)

    y = res[1]
    ∇y = SVector{N,T}(ntuple(i -> res[i+1], Val(N))) .* dx̄_dx

    return y, ∇y
end


"""
    gradient(
        g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}
    ) where {T, N} -> T, SVector{N, T}

Evaluates the transformed Chebyshev series `g` and its gradient at a point `x`.
"""
function gradient(g::TransformedChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    y, ∇ᵤy = gradient(g.series, g.u(x))

    ∇ₓu = g.∇u(x)

    # ∂y/∂x = ∂y/∂u ⋅ ∂u/∂x
    ∇y = ∇ₓu' * ∇ᵤy

    return y, ∇y
end


"""
    gradient(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N} -> T, SVector{N, T}

Evaluates the Chebyshev cluster `h` and its gradient at a point `x`.
"""
function gradient(h::ChebyshevCluster{T,N}, x::SVector{N,T}) where {T,N}
    i = contains(h, x)
    i == 0 && throw(DomainError(x))
    return gradient(h.series[i], x)
end


"""
    gradient(
        f::AbstractChebyshevSeries{T, N}, x::AbstractVector{T}
    ) where {T, N} -> T, SVector{N, T}

Simpler function for evaluating a Chebyshev series `f` and its gradient
at a point `x`, where `x` is of any subtype of an `AbstractVector{T}`.
"""
function gradient(f::AbstractChebyshevSeries{T,N}, x::AbstractVector{T}) where {T,N}
    return gradient(f, SVector{N,T}(x))
end


"""
    gradient(f::AbstractChebyshevSeries{T, 1}, x::T) where T -> T, T

Simpler function for evaluating a one-dimensional Chebyshev series
`f` and its gradient at a point `x`, where `x` is of type `T`.
"""
function gradient(f::AbstractChebyshevSeries{T,1}, x::T) where T
    y, ∇y = gradient(f, SVector{1,T}(x))
    return y, ∇y[]
end
