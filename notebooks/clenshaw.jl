"""
    clenshaw(f::ChebyshevSeries{T, N}, x::T) where {T, N} -> ChebyshevSeries{T, N-1}

Implements the Clenshaw algorithm to evaluate the
series `f` at a value `x` of its `N`-th dimension.

# Arguments
- `f::ChebyshevSeries{T, N}`: `N`-dimensional series to be evaluated
- `x::T`: Value of the `N`-th coordinate in the domain [-1, 1]

# Returns
- `ChebyshevSeries{T, N-1}`: `f` evaluated at `x`
"""
function clenshaw(a::Array{T, N}, x::T) where {T, N}
    n = size(a, N)
    dx = 2x
    
    aₙ₋₁, aₙ = (selectdim(a, N, i) for i in n-1:n)
    bₖ, bₖ₊₁ = (Array{T, N-1}(undef, a.size[1:N-1]) for _ in 1:2)

    # bₖ used on the right-hand side actually represents bₖ₊₂.
    # bₖ₊₂ is ommited to reduce allocations.
    
    # k = n-2 to 2. 
    @. bₖ = aₙ  # Here, bₖ is bₖ₊₂
    @. bₖ₊₁ = aₙ₋₁ + dx*bₖ
    
    for k in n-2:-1:2
        aₖ = selectdim(a, N, k)
        @. bₖ = aₖ + dx*bₖ₊₁ - bₖ
        bₖ, bₖ₊₁ = bₖ₊₁, bₖ
    end

    # k = 1
    aₖ = selectdim(a, N, 1)
    @. bₖ = aₖ + x*bₖ₊₁ - bₖ
    
    return bₖ
end


function clenshaw(a::Array{T, N}, x::SVector{N, T}) where {T, N}
    b = clenshaw(a, x[N])
    xᴺ⁻¹ = pop(x)
    return clenshaw(b, xᴺ⁻¹)
end


function clenshaw(a::Array{T, 1}, x::SVector{1, T}) where T
    b = clenshaw(a, x[1])
    return b[]
end


"""
    (f::ChebyshevSeries{T, N})(x::SVector{N, T}) where {T, N} -> T

Evaluates the series `f` at a point `x`.

# Arguments
- `x::SVector{N, T}`: evaluation point

# Returns
- `T`: `f` evaluated at `x`
"""
function (f::ChebyshevSeries{T, N})(x::SVector{N, T}) where {T, N}
    x̄ = normalize(f, x)
    y = clenshaw(f.coefs, x̄)
    return y
end


function (h::ChebyshevCluster{T, N})(x::SVector{N, T}) where {T, N}
    i = contains(h, x)
    i == 0 && throw(DomainError(x))
    return h.series[i](x)
end


function (f::AbstractChebyshevSeries{T, N})(x::AbstractVector{T}) where {T, N}
    return f(SVector{N, T}(x))
end


function (f::AbstractChebyshevSeries{T, 1})(x::T) where T
    return f(SVector{1, T}(x))
end
