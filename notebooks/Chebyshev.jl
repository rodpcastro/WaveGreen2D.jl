module Chebyshev

export AbstractChebyshevSeries, ChebyshevSeries, TransformedChebyshevSeries, ChebyshevCluster, gradient, hessian, contains, normalize

using StaticArrays


abstract type AbstractChebyshevSeries{T, N} end


"""
    ChebyshevSeries{T, N}

The Chebyshev Series approximation of a `N`-dimensional
function defined in a bounded domain.

# Fields
- `coefs::Array{T, N}`: coefficients
- `lb::SVector{N, T}`: domain lower bound
- `ub::SVector{N, T}`: domain upper bound
"""
struct ChebyshevSeries{T, N} <: AbstractChebyshevSeries{T, N}
    coefs::Array{T, N}
    lb::SVector{N, T}
    ub::SVector{N, T}
end


function ChebyshevSeries(coefs::Array{T, 1}, lb::T, ub::T) where T
    return ChebyshevSeries(coefs, SVector{1, T}(lb), SVector{1, T}(ub))
end


function ChebyshevSeries(coefs::Array{T, 0}, lb::SVector{0}, ub::SVector{0}) where T
    return coefs[]
end


struct TransformedChebyshevSeries{T, N, F<:Function, G<:Function, H<:Function} <: AbstractChebyshevSeries{T, N}
    series::ChebyshevSeries{T, N}
    u::F
    ∇u::G
    Hu::H

    function TransformedChebyshevSeries(
        series::ChebyshevSeries{T, N}, u::F, ∇u::G, Hu::H,
    ) where {T, N, F<:Function, G<:Function, H<:Function}
        
        if !all(hasmethod.([u, ∇u, Hu], Tuple{SVector{N, T}}))
            error("The transformation function must accept an argument of type SVector{$N, $T}")
        end

        u_return_type = Core.Compiler.return_type(u, Tuple{SVector{N, T}})
        u_proper_type = SVector{N, T}
        if !(u_return_type <: u_proper_type)
            error("The transformation function must return a $u_proper_type")
        end
        
        ∇u_return_type = Core.Compiler.return_type(∇u, Tuple{SVector{N, T}})
        ∇u_proper_type = SMatrix{N, N, T, N^2}
        if !(∇u_return_type <: ∇u_proper_type)
            error("The gradient of the transformation function must return a $∇u_proper_type")
        end

        Hu_return_type = Core.Compiler.return_type(Hu, Tuple{SVector{N, T}})
        Hu_proper_type = SArray{Tuple{N, N, N}, T, 3, N^3}
        if !(Hu_return_type <: Hu_proper_type)
            error("The hessian of the transformation function must return a $Hu_proper_type")
        end

        return new{T, N, F, G, H}(series, u, ∇u, Hu)
    end
end


"""A collection of equidimensional `AbstractChebyshevSeries` objects."""
struct ChebyshevCluster{T, N, M} <: AbstractChebyshevSeries{T, N}
    series::NTuple{M, AbstractChebyshevSeries{T, N}}
end


function ChebyshevCluster(series::AbstractChebyshevSeries{T, N}...) where {T, N}
    M = length(series)
    return ChebyshevCluster{T, N, M}(series)
end


"""Converts a point `x` to its normalized coordinates in ``[-1, 1]^N``."""
function normalize(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    @. (2x - f.lb - f.ub) / (f.ub - f.lb)
end


function normalize(g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return normalize(g.series, g.u(x))
end


"""Checks if the point `x` is in the domain of `f`."""
function contains(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return all(f.lb .≤ x .≤ f.ub)
end


function contains(g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return contains(g.series, g.u(x))
end


function contains(h::ChebyshevCluster{T, N, M}, x::SVector{N, T}) where {T, N, M}
    for i in 1:M
        if contains(h.series[i], x)
            return true, i
        end
    end
    return false, nothing
end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")

end