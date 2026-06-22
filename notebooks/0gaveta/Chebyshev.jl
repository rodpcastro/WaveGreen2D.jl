module Chebyshev

export ChebyshevSeries, TransformedChebyshevSeries, ChebyshevCluster, gradient, hessian

using StaticArrays: SVector, SMatrix, MMatrix, SArray, Size, pop  # TODO: send Size, pop and others to where they are used


abstract type AbstractChebyshevSeries{T,N} end


"""
    ChebyshevSeries{T, N}

The Chebyshev series approximation of a `N`-dimensional
function defined in a bounded domain.

# Fields
- `coefs::Array{T, N}`: coefficients
- `lb::SVector{N, T}`: domain lower bound
- `ub::SVector{N, T}`: domain upper bound
"""
struct ChebyshevSeries{T,N} <: AbstractChebyshevSeries{T,N}
    coefs::Array{T,N}
    lb::SVector{N,T}
    ub::SVector{N,T}
end


"""
    ChebyshevSeries(coefs::Array{T, 1}, lb::T, ub::T) where T -> ChebyshevSeries{T, 1}

Simpler constructor for one-dimensional Chebyshev series.
"""
function ChebyshevSeries(coefs::Array{T,1}, lb::T, ub::T) where T
    return ChebyshevSeries(coefs, SVector{1,T}(lb), SVector{1,T}(ub))
end


"""
    ChebyshevSeries(coefs::Array{T, 0}, lb::SVector{0, T}, ub::SVector{0, T}) where T -> T

A zero-dimensional Chebyshev series is just the coefficient.
"""
function ChebyshevSeries(coefs::Array{T,0}, lb::SVector{0,T}, ub::SVector{0,T}) where T
    return coefs[]
end


"""
    validate_transformation(::Type{T}, N::Int, u, ∇u, Hu) where T

Validates the input and return types of `u`, `∇u` and `Hu`.
"""
function validate_transformation(::Type{T}, N::Int, u, ∇u, Hu) where T
    # Expected types
    x_type = SVector{N,T}
    u_type = SVector{N,T}
    ∇u_type = SMatrix{N,N,T,N^2}
    Hu_type = SArray{Tuple{N,N,N},T,3,N^3}

    # Test input
    x₀ = zero(x_type)

    try
        u₀ = u(x₀)
        ∇u₀ = ∇u(x₀)
        Hu₀ = Hu(x₀)

        if !(u₀ isa u_type)
            error("The transformation function must return a $u_type")
        end

        if !(∇u₀ isa ∇u_type)
            error("The gradient of the transformation function must return a $∇u_type")
        end

        if !(Hu₀ isa Hu_type)
            error("The hessian of the transformation function must return a $Hu_type")
        end
    catch e
        if e isa MethodError
            error("The transformation function must accept an argument of type $x_type")
        else
            rethrow(e)
        end
    end
end


"""
    TransformedChebyshevSeries{T, N}

The transformation of a Chebyshev series in the domain `u` to the domain `x`.

# Fields
- `series::ChebyshevSeries{T, N}`: series in the domain `u`
- `u::F`: transformation function ``u(x)``
- `∇u::G`: transformation function gradient ``∇u(x)``
- `Hu::H`: transformation function hessian ``\\mathrm{H}u(x)``
"""
struct TransformedChebyshevSeries{T,N,F,G,H} <: AbstractChebyshevSeries{T,N}
    series::ChebyshevSeries{T,N}
    u::F
    ∇u::G
    Hu::H

    function TransformedChebyshevSeries(
        series::ChebyshevSeries{T,N}, u::F, ∇u::G, Hu::H
    ) where {T,N,F,G,H}
        validate_transformation(T, N, u, ∇u, Hu)
        return new{T,N,F,G,H}(series, u, ∇u, Hu)
    end
end


"""
    TransformedChebyshevSeries(series::ChebyshevSeries{T, N}) where {T, N} -> TransformedChebyshevSeries{T, N}

Identically transformed Chebyshev series.
"""
function TransformedChebyshevSeries(series::ChebyshevSeries{T,N}) where {T,N}
    # Identity transformation
    u(x::SVector{N,T}) = x
    ∇u(x::SVector{N,T}) = one(SMatrix{N,N,T,N^2})
    Hu(x::SVector{N,T}) = zero(SArray{Tuple{N,N,N},T,3,N^3})

    return TransformedChebyshevSeries(series, u, ∇u, Hu)
end


"""
    ChebyshevCluster{T, N, M}

A collection of `M` equidimensional `ChebyshevSeries` objects.

# Fields
- `series::NTuple{M, ChebyshevSeries{T, N}}`: `N`-dimensional Chebyshev series
"""
struct ChebyshevCluster{T,N,M} <: AbstractChebyshevSeries{T,N}
    series::NTuple{M,ChebyshevSeries{T,N}}
end


"""
    ChebyshevCluster(
        series::ChebyshevSeries{T, N}...
    ) where {T, N} -> ChebyshevCluster{T, N, M}

Simpler constructor for a Chebyshev cluster.
"""
function ChebyshevCluster(series::ChebyshevSeries{T,N}...) where {T,N}
    M = length(series)
    return ChebyshevCluster{T,N,M}(series)
end


"""
    normalize(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N} -> SVector{N, T}

Converts a point `x` to its normalized coordinates in ``[-1, 1]^N``.
"""
function normalize(f::ChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    @. (2x - f.lb - f.ub) / (f.ub - f.lb)
end


"""
    contains(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N} -> Bool

Checks if the point `x` is in the domain of `f`.
"""
function contains(f::ChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    return all(f.lb .≤ x .≤ f.ub)
end


"""
    contains(g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N} -> Bool

Checks if the point `x` is in the domain of `g`, which is equivalent
to checking if the point `g.u(x)` is in the domain of `g.series`.
"""
function contains(g::TransformedChebyshevSeries{T,N}, x::SVector{N,T}) where {T,N}
    return contains(g.series, g.u(x))
end


"""
    contains(h::ChebyshevCluster{T, N, M}, x::SVector{N, T}) where {T, N, M} -> Int

Finds the index of the series in the cluster `h` where the
point `x` is located. Returns `0` if `x` is not in `h`.
"""
function contains(h::ChebyshevCluster{T,N,M}, x::SVector{N,T}) where {T,N,M}
    for i in 1:M
        if contains(h.series[i], x)
            return i
        end
    end
    return 0
end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")
include("printing.jl")

end
