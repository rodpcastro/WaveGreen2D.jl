using JLD2
using WaveGreen2D.Chebyshev: ChebyshevSeries, gradient, hessian, contains


cs_file = joinpath(@__DIR__, "chebyshev_series.jld2")
cs_jld2 = jldopen(cs_file)

const L₁_series = read(cs_jld2, "L₁_series")
const L₂_series = read(cs_jld2, "L₂_series")

close(cs_jld2)


function get_L₁_series_index(x::SVector{3,Float64})
    u = SVector{3,Float64}(x[1], x[2], log(x[3]))

    if contains(L₁_series[1], x)
        return 1
    elseif contains(L₁_series[2], u)
        return 2
    elseif contains(L₁_series[3], u)
        return 3
    else
        throw(DomainError(x, "Point not in the domain of L₁ Chebyshev series."))
    end
end


function get_L₂_series_index(x::SVector{3,Float64})
    u = SVector{3,Float64}(x[1], x[2], log(x[3]))

    if contains(L₂_series[1], u)
        return 1
    elseif contains(L₂_series[2], u)
        return 2
    elseif contains(L₂_series[3], u)
        return 3
    elseif contains(L₂_series[4], u)
        return 4
    else
        throw(DomainError(x, "Point not in the domain of L₂ Chebyshev series."))
    end
end


function L₁(A::Float64, B₁::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₁, H)
    u = SVector{3,Float64}(A, B₁, log(H))
    i = get_L₁_series_index(x)

    if i == 1
        return L₁_series[i](x)
    end

    return L₁_series[i](u)
end


function ∇L₁(A::Float64, B₁::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₁, H)
    u = SVector{3,Float64}(A, B₁, log(H))
    i = get_L₁_series_index(x)

    if i == 1
        return gradient(L₁_series[i], x)
    end

    L, ∇ᵤL = gradient(L₁_series[i], u)

    # ∂L/∂x = ∂L/∂u ⋅ ∂u/∂x
    ∇L = SVector{3,Float64}(∇ᵤL[1], ∇ᵤL[2], ∇ᵤL[3] / H)

    return L, ∇L
end


function HL₁(A::Float64, B₁::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₁, H)
    u = SVector{3,Float64}(A, B₁, log(H))
    i = get_L₁_series_index(x)

    if i == 1
        return hessian(L₁_series[i], x)
    end

    L, ∇ᵤL, HᵤL = hessian(L₁_series[i], u)

    # ∂L/∂x = ∂L/∂u ⋅ ∂u/∂x
    ∇L = SVector{3,Float64}(∇ᵤL[1], ∇ᵤL[2], ∇ᵤL[3] / H)

    # ∂²L/∂x² = ∂L/∂u ⋅ ∂²u/∂x² + ∂²L/∂u² ⋅ (∂u/∂x)²
    HL¹ = zero(MMatrix{3,3,Float64,9})
    HL¹[3, 3] = -∇ᵤL[3] / H^2

    HL² = MMatrix{3,3,Float64,9}(HᵤL)
    HL²[3, :] .*= 1 / H
    HL²[:, 3] .*= 1 / H

    HL = SMatrix{3,3,Float64,9}(HL¹ + HL²)

    return L, ∇L, HL
end


function L₂(A::Float64, B₂::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₂, H)
    u = SVector{3,Float64}(A, B₂, log(H))
    i = get_L₂_series_index(x)
    return L₂_series[i](u)
end


function ∇L₂(A::Float64, B₂::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₂, H)
    u = SVector{3,Float64}(A, B₂, log(H))
    i = get_L₂_series_index(x)
    L, ∇ᵤL = gradient(L₂_series[i], u)

    # ∂L/∂x = ∂L/∂u ⋅ ∂u/∂x
    ∇L = SVector{3,Float64}(∇ᵤL[1], ∇ᵤL[2], ∇ᵤL[3] / H)

    return L, ∇L
end


function HL₂(A::Float64, B₂::Float64, H::Float64)
    x = SVector{3,Float64}(A, B₂, H)
    u = SVector{3,Float64}(A, B₂, log(H))
    i = get_L₂_series_index(x)
    L, ∇ᵤL, HᵤL = hessian(L₂_series[i], u)

    # ∂L/∂x = ∂L/∂u ⋅ ∂u/∂x
    ∇L = SVector{3,Float64}(∇ᵤL[1], ∇ᵤL[2], ∇ᵤL[3] / H)

    # ∂²L/∂x² = ∂L/∂u ⋅ ∂²u/∂x² + ∂²L/∂u² ⋅ (∂u/∂x)²
    HL¹ = zero(MMatrix{3,3,Float64,9})
    HL¹[3, 3] = -∇ᵤL[3] / H^2

    HL² = MMatrix{3,3,Float64,9}(HᵤL)
    HL²[3, :] .*= 1 / H
    HL²[:, 3] .*= 1 / H

    HL = SMatrix{3,3,Float64,9}(HL¹ + HL²)

    return L, ∇L, HL
end
