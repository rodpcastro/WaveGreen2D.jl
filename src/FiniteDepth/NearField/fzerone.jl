# These three functions evaluate the functions F₀ and F₁. The gradient and the hessian
# are computed with respect to the field points coordinates.


function Φ₀(t::SVector{2, Float64})
    X, V₃ = t

    Zₐ = sqrt(V₃^2 + X^2)

    F = Φ₁(t) + 2*log(Zₐ)

    return F
end


function ∇Φ₀(t::SVector{2, Float64}, ∇t::SVector{2, Float64})
    X, V₃ = t
    ∇X, ∇V₃ = ∇t

    X² = X^2
    V₃² = V₃^2

    Zₐ² = V₃² + X²
    Zₐ = sqrt(Zₐ²)

    g = log(Zₐ)
    gˣ = X * ∇X / Zₐ²
    gᶻ = V₃ * ∇V₃ / Zₐ²
    ∇g = SVector{2, Float64}(gˣ, gᶻ)

    F1, ∇F1 = ∇Φ₁(t, ∇t)

    F = F1 + 2*g
    ∇F = ∇F1 + 2*∇g

    return F, ∇F
end


function HΦ₀(t::SVector{2, Float64}, ∇t::SVector{2, Float64})
    X, V₃ = t
    ∇X, ∇V₃ = ∇t

    X² = X^2
    V₃² = V₃^2

    Zₐ² = V₃² + X²
    Zₐ = sqrt(Zₐ²)
    Zₐ⁴ = Zₐ² * Zₐ²

    g = log(Zₐ)
    gˣ = X * ∇X / Zₐ²
    gᶻ = V₃ * ∇V₃ / Zₐ²
    ∇g = SVector{2, Float64}(gˣ, gᶻ)
    gˣˣ = (V₃² - X²) * ∇X^2 / Zₐ⁴
    gˣᶻ = -2 * X * V₃ * ∇X * ∇V₃ / Zₐ⁴
    gᶻᶻ = (X² - V₃²) * ∇V₃^2 / Zₐ⁴
    Hg = SMatrix{2, 2, Float64}([gˣˣ gˣᶻ; gˣᶻ gᶻᶻ])

    F1, ∇F1, HF1 = HΦ₁(t, ∇t)

    F = F1 + 2*g
    ∇F = ∇F1 + 2*∇g
    HF = HF1 + 2*Hg

    return F, ∇F, HF
end


function Φ₁(t::SVector{2, Float64})
    X, V₃ = t
    Z = V₃ - im*X

    e = SpecialFunctions.expintx(-Z)
    f = exp(-V₃) * sin(X)

    F = 2 * (real(e) - π * f)

    return F
end


function ∇Φ₁(t::SVector{2, Float64}, ∇t::SVector{2, Float64})
    X, V₃ = t
    ∇X, ∇V₃ = ∇t

    Z = V₃ - im*X
    ∇Z = SVector{2, ComplexF64}(-im*∇X, ∇V₃)

    e = SpecialFunctions.expintx(-Z)
    e′ = -e - 1/Z
    ∇e = e′ .* ∇Z

    e⁻ⱽ³ = exp(-V₃)
    f = e⁻ⱽ³ * sin(X)
    fˣ = e⁻ⱽ³ * cos(X) * ∇X
    fᶻ = -f * ∇V₃
    ∇f = SVector{2, Float64}(fˣ, fᶻ)

    F = 2 * (real(e) - π * f)
    ∇F = 2 .* (real(∇e) .- π .* ∇f)

    return F, ∇F
end


function HΦ₁(t::SVector{2, Float64}, ∇t::SVector{2, Float64})
    X, V₃ = t
    ∇X, ∇V₃ = ∇t

    Z = V₃ - im*X
    ∇Z = SVector{2, ComplexF64}(-im*∇X, ∇V₃)

    e = SpecialFunctions.expintx(-Z)
    e′ = -e - 1/Z
    e′′ = -e′ + 1/Z^2
    ∇e = e′ .* ∇Z
    He = e′′ .* ∇Z * transpose(∇Z)

    e⁻ⱽ³ = exp(-V₃)
    f = e⁻ⱽ³ * sin(X)
    fˣ = e⁻ⱽ³ * cos(X) * ∇X
    fᶻ = -f * ∇V₃
    fˣˣ = -f * ∇X^2
    fˣᶻ = -fˣ * ∇V₃
    fᶻᶻ = f * ∇V₃^2
    ∇f = SVector{2, Float64}(fˣ, fᶻ)
    Hf = SMatrix{2, 2, Float64}([fˣˣ fˣᶻ; fˣᶻ fᶻᶻ])

    F = 2 * (real(e) - π * f)
    ∇F = 2 .* (real(∇e) .- π .* ∇f)
    HF = 2 .* (real(He) .- π .* Hf)

    return F, ∇F, HF
end
