using JLD2
using WaveGreen2D.Chebyshev: ChebyshevSeries, gradient, hessian, contains, reduce


# Load Chebyshev series approximations for L₁ and L₂
cs_file = joinpath(@__DIR__, "chebyshev_series.jld2")
cs_jld2 = jldopen(cs_file)

const L₁_series = read(cs_jld2, "L₁_series")
const L₂_series = read(cs_jld2, "L₂_series")

close(cs_jld2)


mutable struct ReducedChebyshevSeries
    L₁::ChebyshevSeries{Float64,2}
    L₂::ChebyshevSeries{Float64,2}
end


# Reduced series initizalizer
const integrals = ReducedChebyshevSeries(
    ChebyshevSeries(
        Array{Float64,2}(undef, 1, 1),
        zero(SVector{2,Float64}),
        zero(SVector{2,Float64})
    ),
    ChebyshevSeries(
        Array{Float64,2}(undef, 1, 1),
        zero(SVector{2,Float64}),
        zero(SVector{2,Float64})
    ),
)


function setintegrals!(H::Float64)
    H̃ = log(H)

    if H < 0.01 || H > 7.0
        throw(DomainError(
            H,
            """The Chebyshev series for the integrals L₁ and\
               L₂ are accurate only for 0.01 ≤ H ≤ 7"""
        ))
    end

    if contains(L₁_series[1], H; dim=3)
        integrals.L₁ = reduce(L₁_series[1], H; dim=3)
    elseif contains(L₁_series[2], H̃; dim=3)
        integrals.L₁ = reduce(L₁_series[2], H̃; dim=3)
    elseif contains(L₁_series[3], H̃; dim=3)
        integrals.L₁ = reduce(L₁_series[3], H̃; dim=3)
    end

    if contains(L₂_series[1], H̃; dim=3)
        integrals.L₂ = reduce(L₂_series[1], H̃; dim=3)
    elseif contains(L₂_series[2], H̃; dim=3)
        integrals.L₂ = reduce(L₂_series[2], H̃; dim=3)
    elseif contains(L₂_series[3], H̃; dim=3)
        integrals.L₂ = reduce(L₂_series[3], H̃; dim=3)
    elseif contains(L₂_series[4], H̃; dim=3)
        integrals.L₂ = reduce(L₂_series[4], H̃; dim=3)
    end

    return nothing
end


function ∇Λ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ = gradient(L, u)

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    return λ, ∇λ
end


function HΛ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ, Hᵤλ = hessian(L, u)

    ∇uᵈ = SMatrix{2,2,Float64}([∇u[1] 0.0; 0.0 ∇u[2]])

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    # ∂²λ/∂x² = ∂²λ/∂u² ⋅ (∂u/∂x)²
    Hλ = ∇uᵈ * Hᵤλ * ∇uᵈ

    return λ, ∇λ, Hλ
end
