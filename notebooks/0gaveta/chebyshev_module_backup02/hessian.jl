"""
    hessian_clenshaw(f::ChebyshevSeries{T, N}, x::T) where {T, N} -> ChebyshevSeries{T, N-1}

Implements the Clenshaw algorithm to evaluate the series `f` and its first and second
order partial derivatives with respect to the `N`-th dimension at a value `x` of its
`N`-th dimension.

# Arguments
- `f::ChebyshevSeries{T, N}`: `N`-dimensional series to be evaluated
- `x::T`: Value of the `N`-th coordinate in the domain [-1, 1]

# Returns
- `ChebyshevSeries{T, N-1}`: `f` evaluated at `x`
- `ChebyshevSeries{T, N-1}`: partial derivative of `f` with
  respect to the `N`-th dimension evaluated at `x`
- `ChebyshevSeries{T, N-1}`: second-order partial derivative of `f` with
  respect to the `N`-th dimension evaluated at `x`
"""
function hessian_clenshaw(a::Array{T, N}, x::T) where {T, N}
    n = size(a, N)
    dx = 2x

    aₖ, aₙ₋₁, aₙ = (selectdim(a, N, i) for i in n-2:n)
    bₖ, bₖ₊₁ = (Array{T, N-1}(undef, a.size[1:N-1]) for _ in 1:2)
    cₖ, cₖ₊₁ = (Array{T, N-1}(undef, a.size[1:N-1]) for _ in 1:2)
    dₖ, dₖ₊₁ = (Array{T, N-1}(undef, a.size[1:N-1]) for _ in 1:2)

    # bₖ used on the right-hand side actually represents bₖ₊₂.
    # bₖ₊₂ is ommited to reduce allocations. Idem for cₖ₊₂ and dₖ₊₂.
    
    # k = n - 2
    @. bₖ = aₙ  # Here, bₖ is bₖ₊₂
    @. bₖ₊₁ = aₙ₋₁ + dx*bₖ
    @. bₖ = aₖ + dx*bₖ₊₁ - bₖ
    bₖ, bₖ₊₁ = bₖ₊₁, bₖ
    
    # k = n - 3
    @. cₖ = 2aₙ  # Here, cₖ is cₖ₊₂
    @. cₖ₊₁ = 2bₖ + dx*cₖ
    
    aₖ = selectdim(a, N, n-3)
    @. bₖ = aₖ + dx*bₖ₊₁ - bₖ
    @. cₖ = 2bₖ₊₁ + dx*cₖ₊₁ - cₖ
    bₖ, bₖ₊₁ = bₖ₊₁, bₖ
    cₖ, cₖ₊₁ = cₖ₊₁, cₖ
    
    # k = n-4 to 2
    @. dₖ = 4aₙ  # Here, dₖ is dₖ₊₂
    @. dₖ₊₁ = 2cₖ + dx*dₖ
    
    for k in n-4:-1:2
        aₖ = selectdim(a, N, k)
        @. bₖ = aₖ + dx*bₖ₊₁ - bₖ
        @. cₖ = 2bₖ₊₁ + dx*cₖ₊₁ - cₖ
        @. dₖ = 2cₖ₊₁ + dx*dₖ₊₁ - dₖ
        bₖ, bₖ₊₁ = bₖ₊₁, bₖ
        cₖ, cₖ₊₁ = cₖ₊₁, cₖ
        dₖ, dₖ₊₁ = dₖ₊₁, dₖ
    end

    # k = 1
    aₖ = selectdim(a, N, 1)
    @. bₖ = aₖ + x*bₖ₊₁ - bₖ
    @. cₖ = bₖ₊₁ + x*cₖ₊₁ - cₖ
    @. dₖ = 2.0*(cₖ₊₁ + x*dₖ₊₁ - dₖ)
    
    return bₖ, cₖ, dₖ
end


function hessian_clenshaw(a::Array{T, N}, x::SVector{N, T}) where {T, N}
    b, c, d = hessian_clenshaw(a, x[N])
    xᴺ⁻¹ = pop(x)
    return hessian_clenshaw(b, xᴺ⁻¹)..., gradient_clenshaw(c, xᴺ⁻¹)..., clenshaw(d, xᴺ⁻¹)
end


function hessian_clenshaw(a::Array{T, 1}, x::SVector{1, T}) where T
    b, c, d = hessian_clenshaw(a, x[1])
    return b[], c[], d[]
end


"""
Converts a vector of `K` values representing the upper triangular matrix of
order `N`, stored in column-major order, into a symmetric matrix of order `N`.
It's necessary that `K = N*(N+1)÷2`.
"""
function symmatrix(u::SVector{K, T}, ::Val{N}) where {T, N, K}
    A = MMatrix{N, N, T}(undef)
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
    
    return SMatrix{N, N, T}(A)
end
 

"""
    hessian(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N} -> T, SVector{N, T}, SMatrix{N, N, T}

Evaluates the series `f`, its gradient and hessian at a point `x`.

# Arguments
- `x::SVector{N, T}`: evaluation point

# Returns
- `T`: `f` evaluated at `x`
- `SVector{N, T}`: gradient of `f` evaluated at `x`
- `SMatrix{N, N, T}`: hessian of `f` evaluated at `x`
"""
function hessian(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    x̄ = normalize(f, x)
    dx̄_dx = @. 2 / (f.ub - f.lb)
    K = N*(N+1)÷2
    gidx = [i*(i+1)÷2 + 1 for i in 1:N]
    hidx = [i*(i+1)÷2 + 1 + j for i in 1:N for j in 1:i]

    res = hessian_clenshaw(f.coefs, x̄)
    
    y = res[1]
    ∇ᵤy = SVector{N, T}(ntuple(i -> res[gidx[i]], Val(N))) .* dx̄_dx
    Hᵤy_vec = SVector{K, T}(ntuple(i -> res[hidx[i]], Val(K)))
    Hᵤy = symmatrix(Hᵤy_vec, Val(N)) .* dx̄_dx .* dx̄_dx'
    
    ∇ₓu = f.tf.∇u(x)
    Hₓu = f.tf.Hu(x)
    
    # ∂y/∂x = ∂y/∂u ⋅ ∂u/∂x
    ∇y = ∇ₓu' * ∇ᵤy
    
    # ∂²y/∂x² = ∂y/∂u ⋅ ∂²u/∂x² + ∂²y/∂u² ⋅ (∂u/∂x)²
    Hy = (reshape(reshape(Hₓu, Size(N, N^2))' * ∇ᵤy, Size(N, N)))' + ∇ₓu' * Hᵤy * ∇ₓu
    
    return y, ∇y, Hy
end


function hessian(g::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
    i = contains(g, x)
    i == 0 && throw(DomainError(x))
    return hessian(g.series[i], x)
end


function hessian(f::AbstractChebyshevSeries{T, N}, x::AbstractVector{T}) where {T, N}
    return hessian(f, SVector{N, T}(x))
end


function hessian(f::AbstractChebyshevSeries{T, 1}, x::T) where T
    y, ∇y, Hy = hessian(f, SVector{1, T}(x))
    return y, ∇y[], Hy[]
end
