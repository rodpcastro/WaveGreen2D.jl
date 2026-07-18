# These three functions evaluate the integrals L₁ and L₂. The gradient and the hessian are
# computed with respect to the field points coordinates.


function Gᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64})
    return L(u)
end


function ∇Gᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ = gradient(L, u)

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    return λ, ∇λ
end


function HGᴸ(L::ChebyshevSeries{Float64,2}, u::SVector{2,Float64}, ∇u::SVector{2,Float64})
    λ, ∇ᵤλ, Hᵤλ = hessian(L, u)

    ∇uᵈ = SMatrix{2,2,Float64}([∇u[1] 0.0; 0.0 ∇u[2]])

    # ∂λ/∂x = ∂λ/∂u ⋅ ∂u/∂x
    ∇λ = ∇ᵤλ .* ∇u

    # ∂²λ/∂x² = ∂²λ/∂u² ⋅ (∂u/∂x)²
    Hλ = ∇uᵈ * Hᵤλ * ∇uᵈ

    return λ, ∇λ, Hλ
end
