#!/usr/bin/env julia

# The objective of this script is to generate the coefficients of the Chebyshev series
# used for the testing of the Chebyshev module.

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

println("Script start")

using StaticArrays
using FastChebInterp
using JLD2


coefs_dir = joinpath(dirname(@__DIR__), "test", "coefs")
mkpath(coefs_dir)


let filename = joinpath(coefs_dir, "test_chebyshev_1dtf.jld2")
    f(x) = sin(x)

    lb, ub = SA[0.0], SA[0.5π]

    xc = chebpoints(50, lb[], ub[])
    cb = chebinterp(f.(xc), lb[], ub[])
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_2dtf.jld2")
    f(x) = cos(0.25 * x[1] * x[2])

    lb = SA[-0.05, 0.2]
    ub = SA[0.15, 0.4]

    xc = chebpoints((50, 50), lb, ub)
    cb = chebinterp(f.(xc), lb, ub)
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_3dtf.jld2")
    f(x) = exp(x[1] * x[2]) * cos(x[1] + x[3] / 2)

    lb = SA[0.5, -0.2, 1.0]
    ub = SA[0.7, 0.0, 1.2]

    xc = chebpoints((50, 50, 50), lb, ub)
    cb = chebinterp(f.(xc), lb, ub)
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_1dts.jld2")
    f(u) = sin(u^2)

    lb, ub = SA[0.0], SA[1.5]

    uc = chebpoints(50, lb[], ub[])
    cb = chebinterp(f.(uc), lb[], ub[])
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_2dts.jld2")
    function f(u::SVector{2,Float64})
        r, θ = u
        return exp(r * cos(θ)) * cos(r * sin(θ))
    end

    lb = SA[0.1, -0.7]
    ub = SA[2.0, 1.3]

    uc = chebpoints((50, 50), lb, ub)
    cb = chebinterp(f.(uc), lb, ub)
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_3dts.jld2")
    function f(u::SVector{3,Float64})
        r, θ, ϕ = u
        r² = r^2
        return r² * sin(ϕ) * cos(θ) * cos(ϕ) * exp(-r²)
    end

    lb = SA[0.1, 0.2, 0.3]
    ub = SA[2.2, 1.8, 1.6]

    uc = chebpoints((50, 50, 50), lb, ub)
    cb = chebinterp(f.(uc), lb, ub)
    coefs = cb.coefs

    @save filename coefs lb ub
end


let filename = joinpath(coefs_dir, "test_chebyshev_1dcc.jld2")
    f(x) = exp(cos(0.5x))

    lb1, ub1 = SA[0.0], SA[0.5]
    lb2, ub2 = SA[0.5], SA[1.0]

    xc1 = chebpoints(50, lb1[], ub1[])
    cb1 = chebinterp(f.(xc1), lb1[], ub1[])
    coefs1 = cb1.coefs

    xc2 = chebpoints(50, lb2[], ub2[])
    cb2 = chebinterp(f.(xc2), lb2[], ub2[])
    coefs2 = cb2.coefs

    @save filename coefs1 lb1 ub1 coefs2 lb2 ub2
end

println("Coefficients saved at $coefs_dir")


println("Script end")
