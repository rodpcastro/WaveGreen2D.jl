# Evaluates L
function Gᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64})
    return L(u)
end


# Evaluates L and ∇L at the field point with respect to the global coordinate system.
function ∇Gᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ = gradient(L, u)

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    return λ, ∇λ
end


# Evaluates L, ∇L and HL at the field point with respect to the global coordinate system.
function HGᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ, Hᵤλ = hessian(L, u)

    ∇uᵈ = SMatrix{2,2,Float64}([∇u[1] 0.0; 0.0 ∇u[2]])

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    # ∂²λ/∂x² = ∂²λ/∂u² ⋅ (∂u/∂x)²
    Hλ = ∇uᵈ * Hᵤλ * ∇uᵈ

    return λ, ∇λ, Hλ
end
