module NearField

using Chebyshaw
using SpecialFunctions
using StaticArrays: SVector, SMatrix
using WaveGreen2D.Wave: FiniteDepthWave


include("rankine.jl")
include("fzerone.jl")
include("integrals.jl")


# Finite-depth free surface Green function for field and source points close to each other,
# which is defined by the dimensionless horizontal distance A ≤ 0.5.


function Gᴺ(
    wave::FiniteDepthWave, field_point::SVector{2,Float64}, source_point::SVector{2,Float64}
)
    # Dimensional parameters
    h = wave.h
    K = wave.K

    x, z = field_point
    ξ, ζ = source_point

    R = abs(x - ξ)
    R² = R^2

    v₁ = abs(z - ζ)
    v₃ = -z - ζ
    v₂ = 2h - v₃

    r₁ = √(R² + v₁^2)
    r₂ = √(R² + v₂^2)
    r₃ = √(R² + v₃^2)

    # Dimensionless parameters
    H = K * h

    A = R / h

    B₁ = v₁ / h
    B₂ = v₂ / h

    V₃ = K * v₃
    X = K * R

    # Rankine sources
    s₁ = r₁ / h
    s₂ = r₂ / h
    s₃ = r₃ / h

    Gᴿ₁ = Gᴿ(s₁)
    Gᴿ₂ = Gᴿ(s₂)
    Gᴿ₃ = Gᴿ(s₃)

    # Function F₀
    t = SVector{2,Float64}(X, V₃)

    F₀ = Φ₀(t)

    # Integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    u₂ = SVector{2,Float64}(A, B₂)

    L₁ = Λ(wave.L₁, u₁)
    L₂ = Λ(wave.L₂, u₂)

    # Combine components
    G = Gᴿ₁ + Gᴿ₂ + Gᴿ₃ - F₀ - L₁ - L₂ + 2*log(H)

    return G
end


function ∇Gᴺ(
    wave::FiniteDepthWave, field_point::SVector{2,Float64}, source_point::SVector{2,Float64}
)
    # Dimensional parameters
    h = wave.h
    K = wave.K

    x, z = field_point
    ξ, ζ = source_point

    R̄ = x - ξ
    R = abs(R̄)
    R² = R^2
    ∇R = sign(R̄)

    v̄₁ = z - ζ
    v₁ = abs(v̄₁)
    v₃ = -z - ζ
    v₂ = 2h - v₃

    r₁ = √(R² + v₁^2)
    r₁ˣ = R̄ / r₁
    r₁ᶻ = v̄₁ / r₁
    ∇r₁ = SVector{2, Float64}(r₁ˣ, r₁ᶻ)

    r₂ = √(R² + v₂^2)
    r₂ˣ = R̄ / r₂
    r₂ᶻ = v₂ / r₂
    ∇r₂ = SVector{2, Float64}(r₂ˣ, r₂ᶻ)

    r₃ = √(R² + v₃^2)
    r₃ˣ = R̄ / r₃
    r₃ᶻ = -v₃ / r₃
    ∇r₃ = SVector{2, Float64}(r₃ˣ, r₃ᶻ)

    # Dimensionless parameters
    H = K * h

    A = R / h
    ∇A = ∇R / h

    B₁ = v₁ / h
    B₂ = v₂ / h

    ∇B₁ = sign(v̄₁) / h
    ∇B₂ = 1 / h

    X = K * R
    ∇X = K * ∇R

    V₃ = K * v₃
    ∇V₃ = -K

    # Rankine sources
    s₁ = r₁ / h
    s₂ = r₂ / h
    s₃ = r₃ / h

    ∇s₁ = ∇r₁ ./ h
    ∇s₂ = ∇r₂ ./ h
    ∇s₃ = ∇r₃ ./ h

    Gᴿ₁, ∇Gᴿ₁ = ∇Gᴿ(s₁, ∇s₁)
    Gᴿ₂, ∇Gᴿ₂ = ∇Gᴿ(s₂, ∇s₂)
    Gᴿ₃, ∇Gᴿ₃ = ∇Gᴿ(s₃, ∇s₃)

    # Function F₀
    t = SVector{2,Float64}(X, V₃)
    ∇t = SVector{2,Float64}(∇X, ∇V₃)

    F₀, ∇F₀ = ∇Φ₀(t, ∇t)

    # Integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    u₂ = SVector{2,Float64}(A, B₂)

    ∇u₁ = SVector{2,Float64}(∇A, ∇B₁)
    ∇u₂ = SVector{2,Float64}(∇A, ∇B₂)

    L₁, ∇L₁ = ∇Λ(wave.L₁, u₁, ∇u₁)
    L₂, ∇L₂ = ∇Λ(wave.L₂, u₂, ∇u₂)

    # Combine components
    G = Gᴿ₁ + Gᴿ₂ + Gᴿ₃ - F₀ - L₁ - L₂ + 2*log(H)
    ∇G = ∇Gᴿ₁ + ∇Gᴿ₂ + ∇Gᴿ₃ -∇F₀ - ∇L₁ - ∇L₂

    return G, ∇G
end


function HGᴺ(
    wave::FiniteDepthWave, field_point::SVector{2,Float64}, source_point::SVector{2,Float64}
)
    # Dimensional parameters
    h = wave.h
    K = wave.K

    x, z = field_point
    ξ, ζ = source_point

    R̄ = x - ξ
    R = abs(R̄)
    R² = R^2
    ∇R = sign(R̄)

    v̄₁ = z - ζ
    v₁ = abs(v̄₁)
    v₃ = -z - ζ
    v₂ = 2h - v₃

    v₁² = v₁^2
    v₂² = v₂^2
    v₃² = v₃^2

    r₁² = R² + v₁²
    r₁ = √r₁²
    r₁³ = r₁² * r₁
    r₁ˣ = R̄ / r₁
    r₁ᶻ = v̄₁ / r₁
    ∇r₁ = SVector{2, Float64}(r₁ˣ, r₁ᶻ)
    Hr₁ˣˣ = v₁² / r₁³
    Hr₁ˣᶻ = -R̄ * v̄₁ / r₁³
    Hr₁ᶻᶻ = R² / r₁³
    Hr₁ = SMatrix{2, 2, Float64}([Hr₁ˣˣ Hr₁ˣᶻ; Hr₁ˣᶻ Hr₁ᶻᶻ])

    r₂² = R² + v₂²
    r₂ = √r₂²
    r₂³ = r₂² * r₂
    r₂ˣ = R̄ / r₂
    r₂ᶻ = v₂ / r₂
    ∇r₂ = SVector{2, Float64}(r₂ˣ, r₂ᶻ)
    Hr₂ˣˣ = v₂² / r₂³
    Hr₂ˣᶻ = -R̄ * v₂ / r₂³
    Hr₂ᶻᶻ = R² / r₂³
    Hr₂ = SMatrix{2, 2, Float64}([Hr₂ˣˣ Hr₂ˣᶻ; Hr₂ˣᶻ Hr₂ᶻᶻ])

    r₃² = R² + v₃²
    r₃ = √r₃²
    r₃³ = r₃² * r₃
    r₃ˣ = R̄ / r₃
    r₃ᶻ = -v₃ / r₃
    ∇r₃ = SVector{2, Float64}(r₃ˣ, r₃ᶻ)
    Hr₃ˣˣ = v₃² / r₃³
    Hr₃ˣᶻ = R̄ * v₃ / r₃³
    Hr₃ᶻᶻ = R² / r₃³
    Hr₃ = SMatrix{2, 2, Float64}([Hr₃ˣˣ Hr₃ˣᶻ; Hr₃ˣᶻ Hr₃ᶻᶻ])

    # Dimensionless parameters
    H = K * h

    A = R / h
    ∇A = ∇R / h

    B₁ = v₁ / h
    B₂ = v₂ / h

    ∇B₁ = sign(v̄₁) / h
    ∇B₂ = 1 / h

    X = K * R
    ∇X = K * ∇R

    V₃ = K * v₃
    ∇V₃ = -K

    # Rankine sources
    s₁ = r₁ / h
    s₂ = r₂ / h
    s₃ = r₃ / h

    ∇s₁ = ∇r₁ ./ h
    ∇s₂ = ∇r₂ ./ h
    ∇s₃ = ∇r₃ ./ h

    Hs₁ = Hr₁ ./ h
    Hs₂ = Hr₂ ./ h
    Hs₃ = Hr₃ ./ h

    Gᴿ₁, ∇Gᴿ₁, HGᴿ₁ = HGᴿ(s₁, ∇s₁, Hs₁)
    Gᴿ₂, ∇Gᴿ₂, HGᴿ₂ = HGᴿ(s₂, ∇s₂, Hs₂)
    Gᴿ₃, ∇Gᴿ₃, HGᴿ₃ = HGᴿ(s₃, ∇s₃, Hs₃)

    # Function F₀
    t = SVector{2,Float64}(X, V₃)
    ∇t = SVector{2,Float64}(∇X, ∇V₃)

    F₀, ∇F₀, HF₀ = HΦ₀(t, ∇t)

    # Integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    u₂ = SVector{2,Float64}(A, B₂)

    ∇u₁ = SVector{2,Float64}(∇A, ∇B₁)
    ∇u₂ = SVector{2,Float64}(∇A, ∇B₂)

    L₁, ∇L₁, HL₁ = HΛ(wave.L₁, u₁, ∇u₁)
    L₂, ∇L₂, HL₂ = HΛ(wave.L₂, u₂, ∇u₂)

    # Combine components
    G = Gᴿ₁ + Gᴿ₂ + Gᴿ₃ - F₀ - L₁ - L₂ + 2*log(H)
    ∇G = ∇Gᴿ₁ + ∇Gᴿ₂ + ∇Gᴿ₃ - ∇F₀ - ∇L₁ - ∇L₂
    HG = HGᴿ₁ + HGᴿ₂ + HGᴿ₃ - HF₀ - HL₁ - HL₂

    return G, ∇G, HG
end

end # module
