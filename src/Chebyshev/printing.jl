using Printf: @sprintf


function order(f::ChebyshevSeries{T,N}) where {T,N}
    return ntuple(i -> f.coefs.size[i] - 1, Val(N))
end


function order(f::ChebyshevSeries{T,1}) where T
    return f.coefs.size[1] - 1
end


function domain(f::ChebyshevSeries{T,N}) where {T,N}
    intervals = ntuple(i -> @sprintf("[%.3f, %.3f]", f.lb[i], f.ub[i]), Val(N))
    return join(intervals, "×")
end


function Base.show(io::IO, ::MIME"text/plain", f::ChebyshevSeries{T,N}) where {T,N}
    print(io, "$N-dimensional Chebyshev series of order $(order(f)) for x ∈ $(domain(f))")
end


function Base.show(io::IO, f::ChebyshevSeries{T,N}) where {T,N}
    print(io, "$N-D Chebyshev series of order $(order(f))")
end


function Base.show(
    io::IO, ::MIME"text/plain", g::TransformedChebyshevSeries{T,N}
) where {T,N}
    print(io, "$N-dimensional transformed Chebyshev series of order " *
              "$(order(g.series)) for u(x) ∈ $(domain(g.series))")
end


function Base.show(io::IO, f::TransformedChebyshevSeries{T,N}) where {T,N}
    print(io, "$N-D transformed Chebyshev series of order $(order(f.series))")
end


function Base.show(io::IO, ::MIME"text/plain", h::ChebyshevCluster{T,N,M}) where {T,N,M}
    series = join(
        ntuple(i -> "\n$i. Order $(order(h.series[i])), x ∈ $(domain(h.series[i]))", Val(M))
    )
    print(io, "Cluster of $M $N-D Chebyshev series: $series")
end


function Base.show(io::IO, h::ChebyshevCluster{T,N,M}) where {T,N,M}
    print(io, "Cluster of $M $N-D Chebyshev series")
end
