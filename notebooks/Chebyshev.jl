module Chebyshev

export AbstractChebyshevSeries, ChebyshevSeries, Transformation, ChebyshevCluster, gradient, hessian, contains, normalize

using StaticArrays


struct Transformation{F<:Function, G<:Function, H<:Function}
    u::F
    ∇u::G
    Hu::H
end


function validate_transformation(tf::Transformation, T, N)
    # Expected types.
    x_type = SVector{N, T}
    u_type = SVector{N, T}
    ∇u_type = SMatrix{N, N, T, N^2}
    Hu_type = SArray{Tuple{N, N, N}, T, 3, N^3}

    x = zero(x_type)  # test input
    
    try
        u = tf.u(x)
        ∇u = tf.∇u(x)
        Hu = tf.Hu(x)

        if !(u isa u_type)
            error("The transformation function must return a $u_type")
        end

        if !(∇u isa ∇u_type)
            error("The gradient of the transformation function must return a $∇u_type")
        end

        if !(Hu isa Hu_type)
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
struct ChebyshevSeries{T, N, U<:Transformation} <: AbstractChebyshevSeries{T, N}
    coefs::Array{T, N}
    lb::SVector{N, T}
    ub::SVector{N, T}
    tf::U

    function ChebyshevSeries(coefs::Array{T, N}, lb::SVector{N, T}, ub::SVector{N, T}, tf::U) where {T, N, U<:Transformation}
        validate_transformation(tf, T, N)
        return new{T, N, U}(coefs, lb, ub, tf)
    end
end


function ChebyshevSeries(coefs::Array{T, 1}, lb::T, ub::T, tf::Transformation) where T
    return ChebyshevSeries(coefs, SVector{1, T}(lb), SVector{1, T}(ub), tf)
end


function ChebyshevSeries(coefs::Array{T, N}, lb::SVector{N, T}, ub::SVector{N, T}) where {T, N}
    u(x::SVector{N, T}) = identity(x)
    ∇u(x::SVector{N, T}) = one(SMatrix{N, N, T, N^2})
    Hu(x::SVector{N, T}) = zero(SArray{Tuple{N, N, N}, T, 3, N^3})
    tf = Transformation(u, ∇u, Hu)
    return ChebyshevSeries(coefs, lb, ub, tf)
end


function ChebyshevSeries(coefs::Array{T, 1}, lb::T, ub::T) where T
    return ChebyshevSeries(coefs, SVector{1, T}(lb), SVector{1, T}(ub))
end


"""A collection of equidimensional `ChebyshevSeries` objects."""
struct ChebyshevCluster{T, N, M} <: AbstractChebyshevSeries{T, N}
    series::NTuple{M, ChebyshevSeries{T, N}}
end


function ChebyshevCluster(series::AbstractChebyshevSeries{T, N}...) where {T, N}
    M = length(series)
    return ChebyshevCluster{T, N, M}(series)
end


"""Converts a point `x` to its normalized coordinates in ``[-1, 1]^N``."""
function normalize(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    u = f.tf.u(x)
    @. (2u - f.lb - f.ub) / (f.ub - f.lb)
end


"""Checks if the point `x` is in the domain of `f`."""
function contains(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    u = f.tf.u(x)
    return all(f.lb .≤ u .≤ f.ub)
end


function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
    i = findfirst(s -> contains(s, x), h.series)
    return something(i, 0)
end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")

end