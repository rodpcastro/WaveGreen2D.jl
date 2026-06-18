module Chebyshev

export Transformation, ChebyshevSeries, TransformedChebyshevSeries, ChebyshevCluster,
       gradient, hessian, contains, normalize, clenshaw, gradient_clenshaw, hessian_clenshaw

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
struct ChebyshevSeries{T, N} <: AbstractChebyshevSeries{T, N}
    coefs::Array{T, N}
    lb::SVector{N, T}
    ub::SVector{N, T}
end


function ChebyshevSeries(coefs::Array{T, 1}, lb::T, ub::T) where T
    return ChebyshevSeries(coefs, SVector{1, T}(lb), SVector{1, T}(ub))
end


struct TransformedChebyshevSeries{T, N, U<:Transformation} <: AbstractChebyshevSeries{T, N}
    series::ChebyshevSeries{T, N}
    tf::U

    function TransformedChebyshevSeries(series::ChebyshevSeries{T, N}, tf::U) where {T, N, U<:Transformation}
        validate_transformation(tf, T, N)
        return new{T, N, U}(series, tf)
    end
end


function TransformedChebyshevSeries(series::ChebyshevSeries{T, N}) where {T, N}
    u(x::SVector{N, T}) = identity(x)
    ∇u(x::SVector{N, T}) = one(SMatrix{N, N, T, N^2})
    Hu(x::SVector{N, T}) = zero(SArray{Tuple{N, N, N}, T, 3, N^3})
    tf = Transformation(u, ∇u, Hu)
    return TransformedChebyshevSeries(series, tf)
end


"""A collection of equidimensional `ChebyshevSeries` objects."""
struct ChebyshevCluster{T, N, S<:Tuple{Vararg{AbstractChebyshevSeries{T, N}}}} <: AbstractChebyshevSeries{T, N}
    series::S
end


function ChebyshevCluster(series::AbstractChebyshevSeries{T, N}...) where {T, N}
    S = typeof(series)
    return ChebyshevCluster{T, N, S}(series)
end


"""Converts a point `x` to its normalized coordinates in ``[-1, 1]^N``."""
function normalize(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    @. (2x - f.lb - f.ub) / (f.ub - f.lb)
end


function normalize(g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return normalize(g.series, g.tf.u(x))
end


"""Checks if the point `x` is in the domain of `f`."""
function contains(f::ChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return all(f.lb .≤ x .≤ f.ub)
end


function contains(g::TransformedChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return contains(g.series, g.tf.u(x))
end


# function contains(g::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
#     i = findfirst(s -> contains(s, x), g.series)
#     return something(i, 0)
# end


function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
    for i in 1:length(h.series)
        if evaluate_contains(h.series[i], x)
            return i
        end
    end
    return 0
end


function evaluate_contains(f::AbstractChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
    return contains(f, x)
end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")

end