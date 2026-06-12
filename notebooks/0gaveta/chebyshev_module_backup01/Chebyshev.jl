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
# struct ChebyshevCluster{T, N, M} <: AbstractChebyshevSeries{T, N}
#     series::NTuple{M, AbstractChebyshevSeries{T, N}}
# end
# struct ChebyshevCluster{T, N, M, S <: NTuple{M, AbstractChebyshevSeries{T, N}}} <: AbstractChebyshevSeries{T, N}  # Works
#     series::S
# end
# struct ChebyshevCluster{T, N, S <: Tuple} <: AbstractChebyshevSeries{T, N}  # Doesn't work
#     series::S
# end
struct ChebyshevCluster{T, N, S <: Tuple{Vararg{AbstractChebyshevSeries{T, N}}}} <: AbstractChebyshevSeries{T, N}  # Works
    series::S
end
# struct ChebyshevCluster{T, N, M, S <: AbstractChebyshevSeries{T, N}} <: AbstractChebyshevSeries{T, N}  # Doesn't work
#     series::NTuple{M, S}
# end
# struct ChebyshevCluster{T, N, M} <: AbstractChebyshevSeries{T, N}  # Doesn't solve the issue
#     series::NTuple{M, Union{ChebyshevSeries{T, N}, TransformedChebyshevSeries{T, N}}}
# end


# function ChebyshevCluster(series::AbstractChebyshevSeries{T, N}...) where {T, N}
#     M = length(series)
#     return ChebyshevCluster{T, N, M}(series)
# end
function ChebyshevCluster(series::AbstractChebyshevSeries{T, N}...) where {T, N}  # Works
    S = typeof(series)
    return ChebyshevCluster{T, N, S}(series)
end
# function ChebyshevCluster(series::S...) where {T, N, S <: AbstractChebyshevSeries{T, N}}  # Doesn't work
#     return ChebyshevCluster{T, N, Tuple{Vararg{S}}}(series)
# end


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


# function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
#     for i in 1:length(h.series)
#         if contains(h.series[i], x)
#             return i
#         end
#     end
#     return 0
# end
# function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
#     checks = ntuple(i -> contains(h.series[i], x), length(h.series))
    
#     for i in 1:length(checks)
#         if checks[i]
#             return i
#         end
#     end

#     return 0
# end
# function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
#     checks = ntuple(i -> contains(h.series[i], x), length(h.series))
#     idx = findfirst(checks) 
#     return isnothing(idx) ? 0 : idx
# end
function contains(h::ChebyshevCluster{T, N}, x::SVector{N, T}) where {T, N}
    i = findfirst(s -> contains(s, x), h.series)
    return something(i, 0)
end

# function _evaluate_series(s::AbstractChebyshevSeries{T, N}, x::SVector{N, T}) where {T, N}
#     return s(x)
# end


include("clenshaw.jl")
include("gradient.jl")
include("hessian.jl")

end