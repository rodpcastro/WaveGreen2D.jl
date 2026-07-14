# Number of evanescent modes
const nevamodes = 20


"""
    WaveParameters

Dimensional parameters that define the environmental conditions. The wavenumber and
evanescent wavenumbers must be computed from the others via the function `setwave!`.

# Fields
- `depth::Float64`: water depth (m)
- `frequency::Float64`: wave frequency (rad/s)
- `gravity::Float64`: acceleration of gravity (m/s²)
- `wavenumber::Float64`: wavenumber (1/m⁻¹)
- `evanumbers::NTuple{n, Float64}`: n evanescent wavenumbers (1/m⁻¹)
"""
mutable struct WaveParameters
    depth::Float64
    frequency::Float64
    gravity::Float64
    wavenumber::Float64
    evanumbers::NTuple{nevamodes, Float64}

    function WaveParameters(depth::Real, frequency::Real, gravity::Real)
        validate_wave(depth, frequency, gravity)

        h = Float64(depth)
        ω = Float64(frequency)
        g = Float64(gravity)
        k₀ = NaN
        kₙ = ntuple(i -> NaN, nevamodes)

        return new(h, ω, g, k₀, kₙ)
    end
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


# Wave initializer
const wave = WaveParameters(NaN, NaN, NaN)


"""
    setwave!(depth::Real, frequency::Real, gravity::Real=9.80665) -> Nothing

Sets the parameters that define the environmental conditions.
"""
function setwave!(; depth::Real, frequency::Real, gravity::Real=9.80665)
    h = Float64(depth)
    g = Float64(gravity)
    ω = Float64(frequency)

    validate_wave(h, ω, g)

    wave.depth = h
    wave.frequency = ω
    wave.gravity = g
    wave.wavenumber = find_k₀(h, ω, g)

    # Dimensionless depth.
    H = h * ω^2 / g

    # H ≤ π defines shallow and intermediate waters.
    if 0.01 ≤ H ≤ π
        wave.evanumbers = ntuple(i -> find_kₙ(i, h, ω, g), nevamodes)

        # Set Chebyshev series approximations of L₁ and L₂ for a fixed H.
        NearField.setintegrals!(H)
    end

    @info "$wave"

    return nothing
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


# Reading aliases
function Base.getproperty(obj::WaveParameters, sym::Symbol)
    if sym === :h
        return getfield(obj, :depth)
    elseif sym === :ω
        return getfield(obj, :frequency)
    elseif sym === :g
        return getfield(obj, :gravity)
    elseif sym === :k₀
        return getfield(obj, :wavenumber)
    elseif sym === :kₙ
        return getfield(obj, :evanumbers)
    else
        return getfield(obj, sym)  # Default fieldname
    end
end


function Base.show(io::IO, ::MIME"text/plain", w::WaveParameters)
    print(
        io,
        "Wave parameters ",
        "h = $(round(w.h; digits=2)) m, ",
        "ω = $(round(w.ω; digits=2)) rad/s, ",
        "g = $(round(w.g; digits=2)) m/s²",
    )
end


function Base.show(io::IO, w::WaveParameters)
    Base.show(io, MIME"text/plain"(), w)
end
