module Chebyshev

export ChebyshevSeries, Transformation, ChebyshevCluster, gradient, hessian

using StaticArrays


"""
    ChebyshevSeries{T, N}

The Chebyshev Series approximation of a `N`-dimensional
function defined in a bounded domain.

# Fields
- `coefs::Array{T, N}`: coefficients
- `lb::SVector{N, T}`: domain lower bound
- `ub::SVector{N, T}`: domain upper bound
"""
struct ChebyshevSeries{T, N}
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


# TODO: Change How Transformation and ChebyshevCluster are combined. I believe I
# need T and N type parameters also included in the transformations (How to do this?).
struct Transformation{F<:Function, G<:Function, H<:Function}
    u::F
    ∇u::G
    Hu::H
end


"""A collection of `ChebyshevSeries` objects."""  # TODO: Improve this description
struct ChebyshevCluster{T, N, M}
    series::NTuple{M, ChebyshevSeries{T, N}}
    tforms::NTuple{M, Transformation}
end


# TODO: Modify this Cluster definition to make identity transfomrations by default.
function ChebyshevCluster(series::ChebyshevSeries{T, N}...) where {T, N}
    M = length(series)
    return ChebyshevCluster{T, N, M}(series)
end


"""Converts a point `x` to its normalized coordinates in ``[-1, 1]^N``."""
function normalize(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    @. (2x - f.lb - f.ub) / (f.ub - f.lb)
end


function normalize(f::ChebyshevSeries{T, 1}, x::T) where T
    return normalize(f, SVector{1, T}(x))[]
end


"""Checks if the point `x` is in the domain of `f`."""
function contains(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return all(f.lb .≤ x .≤ f.ub)
end


function contains(f::ChebyshevSeries{T, N}, x::AbstractVector{T}) where {T, N}
    return contains(f, SVector{N, T}(x))
end


function contains(f::ChebyshevSeries{T, 1}, x::T) where T
    return contains(f, SVector{1, T}(x))
end


function contains(g::ChebyshevCluster{T, N, M}, x::Union{AbstractVector{T}, T}) where {T, N, M}
    for i in 1:M
        u = g.tforms[i].u(x)
        if contains(g.series[i], u)
            return true, i
        end
    end
    return false, nothing
end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")

end