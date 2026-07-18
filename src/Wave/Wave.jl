module Wave

export create_wave

using Chebyshaw
using JLD2


include("root.jl")


abstract type AbstractWaterWave end


# Number of evanescent modes
const _nevamodes = 12


"""
    FiniteDepthWave

Parameters that define a wave at waters of finite depth.

# Fields
- `h::Float64`: water depth (m)
- `ω::Float64`: wave frequency (rad/s)
- `g::Float64`: acceleration of gravity (m/s²)
- `K::Float64`: infinite-depth wavenumber (1/m⁻¹)
- `k₀::Float64`: wavenumber (1/m⁻¹)
- `kₙ::NTuple{n, Float64}`: n evanescent wavenumbers (1/m⁻¹)
- `L₁::ChebyshevSeries{Float64, 2}`: Near field integral L₁
- `L₂::ChebyshevSeries{Float64, 2}`: Near field integral L₂
"""
struct FiniteDepthWave <: AbstractWaterWave
    h::Float64
    ω::Float64
    g::Float64
    K::Float64
    k₀::Float64
    kₙ::NTuple{_nevamodes, Float64}
    L₁::ChebyshevSeries{Float64, 2}
    L₂::ChebyshevSeries{Float64, 2}
end


"""
    InfiniteDepthWave

Parameters that define a wave at waters of infinite depth.

# Fields
- `h::Float64`: water depth (m)
- `ω::Float64`: wave frequency (rad/s)
- `g::Float64`: acceleration of gravity (m/s²)
- `K::Float64`: wavenumber (1/m⁻¹)
"""
struct InfiniteDepthWave <: AbstractWaterWave
    h::Float64
    ω::Float64
    g::Float64
    K::Float64
end


# Avoid non-physical values for the wave parameters.
function validate_wave(depth::Real, frequency::Real, gravity::Real)
    if depth ≤ 0
        throw(DomainError(depth, "The depth must be positive."))
    elseif frequency < 0
        throw(DomainError(frequency, "The frequency must be non-negative."))
    elseif gravity ≤ 0
        throw(DomainError(gravity, "The acceleration of gravity must be positive."))
    end
end


"""
    create_wave(; depth::Real, frequency::Real, gravity::Real=9.80665) -> AbstractWaterWave

Creates the wave by defining its frequency and the environmental conditions.
"""
function create_wave(; depth::Real, frequency::Real, gravity::Real=9.80665)
    h = Float64(depth)
    ω = Float64(frequency)
    g = Float64(gravity)

    validate_wave(h, ω, g)

    K = ω^2 / g
    H = K * h

    if 0.01 ≤ H ≤ π
        k₀ = find_k₀(h, ω, g)
        kₙ = ntuple(i -> find_kₙ(i, h, ω, g), _nevamodes)
        L₁, L₂ = get_integrals(H)
        return FiniteDepthWave(h, ω, g, K, k₀, kₙ, L₁, L₂)
    else
        return InfiniteDepthWave(h, ω, g, K)
    end
end


"""
    find_k₀(h::Real, ω::Real, g::Real=9.80665) -> Float64

Finds the wavenumber `k₀` from the dispersion relation ``ω^2 = k g \\tanh(k h)``.
"""
function find_k₀(h::Real, ω::Real, g::Real=9.80665)
    f(k::Real) = k*g * tanh(k*h) - ω^2
    f′(k::Real) = g * tanh(k*h) + k*g*h * sech(k*h)^2

    # Initial estimate for k₀
    κ₁ = ω/√(g*h)  # shallow water
    κ₂ = ω^2/g  # deep water
    k̄₀ = √(κ₁*κ₂)  # geometric mean

    return findroot(f, f′, k̄₀)
end


"""
    find_kₙ(n::Int, h::Real, ω::Real, g::Real=9.80665) -> Float64

Finds the evanescent wavenumber `kₙ` from the dispersion relation ``ω^2 = -k g \\tan(k h)``.
"""
function find_kₙ(n::Int, h::Real, ω::Real, g::Real=9.80665)
    f(k::Real) = k*g * tan(k*h) + ω^2
    f′(k::Real) = g * tan(k*h) + k*g*h * sec(k*h)^2

    # Initial estimate for kₙ
    κ₁ = (2n-1)*π/2h
    κ₂ = n*π/h
    k̄ₙ = 0.5*(κ₁+κ₂)

    return findroot(f, f′, k̄ₙ)
end


# Load Chebyshev series approximations for L₁ and L₂
cs_file = joinpath(@__DIR__, "chebyshev_series.jld2")
cs_jld2 = JLD2.jldopen(cs_file)

const L₁_series = read(cs_jld2, "L₁_series")
const L₂_series = read(cs_jld2, "L₂_series")

close(cs_jld2)


"""
    get_integrals(H::Float64) -> ChebyshevSeries{Float64, 2}, ChebyshevSeries{Float64, 2}

Gets the Chebyshev series approximations of ``L₁`` and ``L₂`` for a fixed parameter
``0.01 ≤ H ≤ 7``, where ``H = h ω^2 / g``. Finite-depth waters are defined by ``H ≤ π``, and
``H = 0.01`` is the minimum value that was used to compute the analytical expressions of
``L₁`` and ``L₂``. The range ``π < H ≤ 7`` is used solely for testing and not used to
compute the near field finite-depth Green function.
"""
function get_integrals(H::Float64)
    H̃ = log(H)

    if Chebyshaw.contains(L₁_series[1], H; dim=3)
        L₁ = Chebyshaw.reduce(L₁_series[1], H; dim=3)
    elseif Chebyshaw.contains(L₁_series[2], H̃; dim=3)
        L₁ = Chebyshaw.reduce(L₁_series[2], H̃; dim=3)
    elseif Chebyshaw.contains(L₁_series[3], H̃; dim=3)
        L₁ = Chebyshaw.reduce(L₁_series[3], H̃; dim=3)
    end

    if Chebyshaw.contains(L₂_series[1], H̃; dim=3)
        L₂ = Chebyshaw.reduce(L₂_series[1], H̃; dim=3)
    elseif Chebyshaw.contains(L₂_series[2], H̃; dim=3)
        L₂ = Chebyshaw.reduce(L₂_series[2], H̃; dim=3)
    elseif Chebyshaw.contains(L₂_series[3], H̃; dim=3)
        L₂ = Chebyshaw.reduce(L₂_series[3], H̃; dim=3)
    elseif Chebyshaw.contains(L₂_series[4], H̃; dim=3)
        L₂ = Chebyshaw.reduce(L₂_series[4], H̃; dim=3)
    end

    return L₁, L₂
end


function Base.show(io::IO, ::MIME"text/plain", w::FiniteDepthWave)
    print(
        io,
        "Finite-depth wave: ",
        "h = $(round(w.h; digits=2)) m, ",
        "ω = $(round(w.ω; digits=2)) rad/s, ",
        "g = $(round(w.g; digits=2)) m/s²",
    )
end


function Base.show(io::IO, ::MIME"text/plain", w::InfiniteDepthWave)
    print(
        io,
        "Infinite-depth wave: ",
        "ω = $(round(w.ω; digits=2)) rad/s, ",
        "g = $(round(w.g; digits=2)) m/s²",
    )
end


function Base.show(io::IO, w::AbstractWaterWave)
    Base.show(io, MIME"text/plain"(), w)
end

end # module
