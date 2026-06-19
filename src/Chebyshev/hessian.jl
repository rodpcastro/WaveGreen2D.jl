"""
    hessian_clenshaw(
        a::Array{T, N}, x::T
    ) where {T, N} -> Array{T, N-1}, Array{T, N-1}, Array{T, N-1}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev series
with coefficients `a`, its gradient and hessian at a normalized value `x` of its `N`-th
dimension.
"""
function hessian_clenshaw(a::Array{T,N}, x::T) where {T,N}
    n = size(a, N)
    dx = 2x

    aₖ, aₙ₋₁, aₙ = (selectdim(a, N, i) for i in n-2:n)
    bₖ, bₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)
    cₖ, cₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)
    dₖ, dₖ₊₁ = (Array{T,N - 1}(undef, a.size[1:N-1]) for _ in 1:2)

    # bₖ used on the right-hand side actually represents bₖ₊₂.
    # bₖ₊₂ is ommited to reduce allocations. Idem for cₖ₊₂ and dₖ₊₂.

    # k = n - 2
    @. bₖ = aₙ  # Here, bₖ is bₖ₊₂
    @. bₖ₊₁ = aₙ₋₁ + dx * bₖ
    @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
    bₖ, bₖ₊₁ = bₖ₊₁, bₖ

    # k = n - 3
    @. cₖ = 2aₙ  # Here, cₖ is cₖ₊₂
    @. cₖ₊₁ = 2bₖ + dx * cₖ

    aₖ = selectdim(a, N, n - 3)
    @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
    @. cₖ = 2bₖ₊₁ + dx * cₖ₊₁ - cₖ
    bₖ, bₖ₊₁ = bₖ₊₁, bₖ
    cₖ, cₖ₊₁ = cₖ₊₁, cₖ

    # k = n-4 to 2
    @. dₖ = 4aₙ  # Here, dₖ is dₖ₊₂
    @. dₖ₊₁ = 2cₖ + dx * dₖ

    for k in n-4:-1:2
        aₖ = selectdim(a, N, k)
        @. bₖ = aₖ + dx * bₖ₊₁ - bₖ
        @. cₖ = 2bₖ₊₁ + dx * cₖ₊₁ - cₖ
        @. dₖ = 2cₖ₊₁ + dx * dₖ₊₁ - dₖ
        bₖ, bₖ₊₁ = bₖ₊₁, bₖ
        cₖ, cₖ₊₁ = cₖ₊₁, cₖ
        dₖ, dₖ₊₁ = dₖ₊₁, dₖ
    end

    # k = 1
    aₖ = selectdim(a, N, 1)
    @. bₖ = aₖ + x * bₖ₊₁ - bₖ
    @. cₖ = bₖ₊₁ + x * cₖ₊₁ - cₖ
    @. dₖ = 2.0 * (cₖ₊₁ + x * dₖ₊₁ - dₖ)

    return bₖ, cₖ, dₖ
end


"""
    hessian_clenshaw(
        a::Array{T, N}, x::SVector{N, T}
    ) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Implements the Clenshaw algorithm to evaluate the `N`-th dimensional Chebyshev series with
coefficients `a`, its gradient and hessian at a normalized point `x` in ``[-1, 1]^N``.
"""
function hessian_clenshaw(a::Array{T,N}, x::SVector{N,T}) where {T,N}
    b, c, d = hessian_clenshaw(a, x[N])
    xᴺ⁻¹ = pop(x)
    return hessian_clenshaw(b, xᴺ⁻¹)..., gradient_clenshaw(c, xᴺ⁻¹)..., clenshaw(d, xᴺ⁻¹)
end


function hessian_clenshaw(a::Array{T,1}, x::SVector{1,T}) where T
    b, c, d = hessian_clenshaw(a, x[1])
    return b[], c[], d[]
end


"""
    symmatrix(u::SVector{K, T}, ::Val{N}) where {T, N, K} -> SMatrix{N, N, T}

Converts a vector of `K` values representing the upper triangular matrix of
order `N`, stored in column-major order, into a symmetric matrix of order `N`.
It's necessary that `K = N*(N+1)÷2`.
"""
function symmatrix(u::SVector{K,T}, ::Val{N}) where {T,N,K}
    A = MMatrix{N,N,T}(undef)
    k = 1
    for j in 1:N
        for i in 1:j-1
            A[i, j] = u[k]
            A[j, i] = u[k]
            k += 1
        end
        A[j, j] = u[k]
        k += 1
    end

    return SMatrix{N,N,T}(A)
end


"""
    hessian(
        f::ChebyshevSeries{T, N}, x::SVector{N, T}
    ) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Evaluates the Chebyshev series `f`, its gradient and hessian at a point `x`.
"""
function hessian(f::ChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    x̄ = normalize(f, x)
    dx̄_dx = @. 2 / (f.ub - f.lb)
    K = N * (N + 1) ÷ 2
    gidx = [i * (i + 1) ÷ 2 + 1 for i in 1:N]
    hidx = [i * (i + 1) ÷ 2 + 1 + j for i in 1:N for j in 1:i]

    res = hessian_clenshaw(f.coefs, x̄)

    y = res[1]
    ∇y = SVector{N,T}(ntuple(i -> res[gidx[i]], Val(N))) .* dx̄_dx
    Hy_vec = SVector{K,T}(ntuple(i -> res[hidx[i]], Val(K)))
    Hy = symmatrix(Hy_vec, Val(N)) .* dx̄_dx .* dx̄_dx'

    return y, ∇y, Hy
end


"""
    hessian(
        g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}
    ) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Evaluates the transformed Chebyshev series `g`, its gradient and hessian at a point `x`.
"""
function hessian(g::TransformedChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    y, ∇ᵤy, Hᵤy = hessian(g.series, g.u(x))

    ∇ₓu = g.∇u(x)
    Hₓu = g.Hu(x)

    # ∂y/∂x = ∂y/∂u ⋅ ∂u/∂x
    ∇y = ∇ₓu' * ∇ᵤy

    # ∂²y/∂x² = ∂y/∂u ⋅ ∂²u/∂x² + ∂²y/∂u² ⋅ (∂u/∂x)²
    Hy = (reshape(reshape(Hₓu, Size(N, N^2))' * ∇ᵤy, Size(N, N)))' + ∇ₓu' * Hᵤy * ∇ₓu

    return y, ∇y, Hy
end


"""
    hessian(
        h::ChebyshevCluster{T, N}, x::SVector{N, T}
    ) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Evaluates the Chebyshev cluster `h`, its gradient and hessian at a point `x`.
"""
function hessian(h::ChebyshevCluster{T,N}, x::SVector{N,T}) where {T,N}
    i = contains(h, x)
    i == 0 && throw(DomainError(x))
    return hessian(h.series[i], x)
end


"""
    hessian(
        f::AbstractChebyshevSeries{T, N}, x::AbstractVector{T}
    ) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Simpler function for evaluating a Chebyshev series `f`, its gradient and
hessian at a point `x`, where `x` is of any subtype of an `AbstractVector{T}`.
"""
function hessian(f::AbstractChebyshevSeries{T,N}, x::AbstractVector{T}) where {T,N}
    return hessian(f, SVector{N,T}(x))
end


"""
    hessian(f::AbstractChebyshevSeries{T, 1}, x::T) where T -> T, T, T

Simpler function for evaluating a one-dimensional Chebyshev series `f`,
its gradient and hessian at a point `x`, where `x` is of type `T`.
"""
function hessian(f::AbstractChebyshevSeries{T,1}, x::T) where T
    y, ∇y, Hy = hessian(f, SVector{1,T}(x))
    return y, ∇y[], Hy[]
end
