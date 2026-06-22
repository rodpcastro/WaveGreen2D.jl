#!/usr/bin/env julia

# The objective of this script is to study the optimal quadrature order for L₁ and L₂.
# From the results, we conclude that the optimal quadrature order is between 24 and 34.

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using BenchmarkTools
using CairoMakie
using JLD2
using QuadGK
using Random


data_dir = joinpath(@__DIR__, "data")
img_dir = joinpath(@__DIR__, "images")

bench_file = joinpath(data_dir, "qorder_benchmarks.jld2")
img_file = joinpath(img_dir, "qorder_benchmarks.svg")

Random.seed!(18)
tol = eps()
imax = 1e4


function L₁(x::AbstractVector{<:Real}; qorder::Int=7)
    A, B, C = x
    H = 10.0^C  # C = log₁₀H

    f(u) = (u + H) / (u - H - (u + H) * exp(-2u))
    g(u) = exp(-u * (2 + B)) + exp(-u * (2 - B))
    h(u) = cos(u * A)
    p(u) = (f(u) * g(u) * h(u) + exp(-u)) / u

    path = (0.0, H + im, H + 1.0, Inf)
    y = 0.0

    for i in 1:length(path)-1
        y += quadgk(
            p, path[i], path[i+1]; rtol=tol, atol=tol, maxevals=imax, order=qorder
        )[1]
    end

    return real(y)
end


function L₂(x::AbstractVector{<:Real}; qorder::Int=7)
    A, B, C = x
    H = 10.0^C  # C = log₁₀H

    f(u) = (u + H) / (u - H - (u + H) * exp(-2u))
    g(u) = (u + H)^2 / ((u - H)^2 - (u^2 - H^2) * exp(-2u))
    p(u) = exp(-u * (2 + B))
    q(u) = exp(-u * (4 - B))
    r(u) = cos(u * A)

    h(u) = (f(u) * p(u) + g(u) * q(u)) * r(u) / u

    path = (0.0, H + im, H + 1.0, Inf)
    y = 0.0

    for i in 1:length(path)-1
        y += quadgk(
            h, path[i], path[i+1]; rtol=tol, atol=tol, maxevals=imax, order=qorder
        )[1]
    end

    return real(y)
end


if length(ARGS) > 0
    arg = ARGS[1]
    if arg == "benchmark" || arg == "bench"
        println("Running benchmark")

        x₁ = [[0.0, 0.0, -2.0] + rand(3) .* [0.5, 1.0, 4.0] for _ in 1:1000]
        x₂ = [[0.0, 0.0, -2.0] + rand(3) .* [0.5, 2.0, 4.0] for _ in 1:1000]

        qorder_vals = collect(5:100)

        L₁_bench = Dict{Int,BenchmarkTools.TrialEstimate}()
        L₂_bench = Dict{Int,BenchmarkTools.TrialEstimate}()

        for qo in qorder_vals
            L₁_benchmark = @benchmark L₁.(x₁; qorder=$qo)
            L₁_bench[qo] = mean(L₁_benchmark)

            L₂_benchmark = @benchmark L₂.(x₂; qorder=$qo)
            L₂_bench[qo] = mean(L₂_benchmark)

            println("L₁ and L₂ with quadrature order = $qo done")
        end

        L₁_bench = sort(L₁_bench)
        L₂_bench = sort(L₂_bench)

        mkpath(data_dir)
        @save bench_file qorder_vals L₁_bench L₂_bench

    else
        println("Unknown argument: $arg")
        exit()
    end
else
    if !isfile(bench_file)
        println("There is no benchmark file to load. Create the benchmark file by running:")
        println("   julia $(@__FILE__) benchmark")
        exit()
    end

    println("Loading benchmark")
    @load bench_file qorder_vals L₁_bench L₂_bench
end


println("Plotting")
mkpath(img_dir)

L₁_time = [bench.time * 1e-9 for bench in values(L₁_bench)]
L₁_memory = [bench.memory * 1e-6 for bench in values(L₁_bench)]
L₁_allocs = [bench.allocs * 1e-3 for bench in values(L₁_bench)]

L₂_time = [bench.time * 1e-9 for bench in values(L₂_bench)]
L₂_memory = [bench.memory * 1e-6 for bench in values(L₂_bench)]
L₂_allocs = [bench.allocs * 1e-3 for bench in values(L₂_bench)]

fig = Figure(size=(1200, 400))
axes = [Axis(fig[1, m], xlabel="Quadrature order") for m in 1:3]

axes[1].title = "Mean execution time (s)"
lines!(axes[1], qorder_vals, L₁_time, color=:green, linestyle=:solid, label=L"L_1")
lines!(axes[1], qorder_vals, L₂_time, color=:red, linestyle=:dash, label=L"L_2")
axislegend(axes[1])

axes[2].title = "Allocated memory (MB)"
lines!(axes[2], qorder_vals, L₁_memory, color=:green, linestyle=:solid, label=L"L_1")
lines!(axes[2], qorder_vals, L₂_memory, color=:red, linestyle=:dash, label=L"L_2")

axes[3].title = "Number of allocations (10³)"
lines!(axes[3], qorder_vals, L₁_allocs, color=:green, linestyle=:solid, label=L"L_1")
lines!(axes[3], qorder_vals, L₂_allocs, color=:red, linestyle=:dash, label=L"L_2")

save(img_file, fig)


println("Done")
