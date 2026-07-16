module NearField

using WaveGreen2D: wave
using StaticArrays: SVector, SMatrix


include("near_field_integrals.jl")


"""
    Gᴺ(
        field_point::SVector{2,Float64}, source_point::SVector{2,Float64}
    ) -> Float64, SVector{2,Float64}

Finite-depth free surface Green function for field and source points close to each other,
which is defined by the dimensionless horizontal distance A ≤ 0.5.
"""
function Gᴺ(field_point::SVector{2,Float64}, source_point::SVector{2,Float64})
    # Define variables
    x, z = field_point
    ξ, ζ = source_point

    R̄ = x - ξ
    R = abs(R̄)
    A = R / wave.h

    v̄₁ = z - ζ
    v₁ = abs(v̄₁)
    B₁ = v₁ / wave.h

    v₂ = 2 * wave.h + z + ζ
    B₂ = v₂ / wave.h

    # Compute integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    u₂ = SVector{2,Float64}(A, B₂)

    L₁ = integrals.L₁(u₁)
    L₂ = integrals.L₂(u₂)

    # Combine components
    G = -L₁ - L₂

    return G
end


function ∇Gᴺ(field_point::SVector{2,Float64}, source_point::SVector{2,Float64})
    # Define variables
    x, z = field_point
    ξ, ζ = source_point

    R̄ = x - ξ
    R = abs(R̄)
    A = R / wave.h
    dA_dx = sign(R̄) / wave.h

    v̄₁ = z - ζ
    v₁ = abs(v̄₁)
    B₁ = v₁ / wave.h
    dB₁_dz = sign(v̄₁) / wave.h

    v₂ = 2 * wave.h + z + ζ
    B₂ = v₂ / wave.h
    dB₂_dz = 1 / wave.h

    # Compute integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    ∇u₁ = SVector{2,Float64}(dA_dx, dB₁_dz)

    u₂ = SVector{2,Float64}(A, B₂)
    ∇u₂ = SVector{2,Float64}(dA_dx, dB₂_dz)

    L₁, ∇L₁ = ∇Λ(integrals.L₁, u₁, ∇u₁)
    L₂, ∇L₂ = ∇Λ(integrals.L₂, u₂, ∇u₂)

    # Combine components
    G = -L₁ - L₂
    ∇G = -∇L₁ - ∇L₂

    return G, ∇G
end


function HGᴺ(field_point::SVector{2,Float64}, source_point::SVector{2,Float64})
    # Define variables
    x, z = field_point
    ξ, ζ = source_point

    R̄ = x - ξ
    R = abs(R̄)
    A = R / wave.h
    dA_dx = sign(R̄) / wave.h

    v̄₁ = z - ζ
    v₁ = abs(v̄₁)
    B₁ = v₁ / wave.h
    dB₁_dz = sign(v̄₁) / wave.h

    v₂ = 2 * wave.h + z + ζ
    B₂ = v₂ / wave.h
    dB₂_dz = 1 / wave.h

    # Compute integrals L₁ and L₂
    u₁ = SVector{2,Float64}(A, B₁)
    ∇u₁ = SVector{2,Float64}(dA_dx, dB₁_dz)

    u₂ = SVector{2,Float64}(A, B₂)
    ∇u₂ = SVector{2,Float64}(dA_dx, dB₂_dz)

    L₁, ∇L₁, HL₁ = HΛ(integrals.L₁, u₁, ∇u₁)
    L₂, ∇L₂, HL₂ = HΛ(integrals.L₂, u₂, ∇u₂)

    # Combine components
    G = -L₁ - L₂
    ∇G = -∇L₁ - ∇L₂
    HG = -HL₁ - HL₂

    return G, ∇G, HG
end

end # module
