"""
    WaveParameters

Dimensional parameters that define the environmental conditions.

# Fields
- `depth::Float64`: water depth (m)
- `frequency::Float64`: wave frequency (rad/s)
- `wavenumber::Float64`: wavenumber (1/m⁻¹)
- `gravity::Float64`: acceleration of gravity (m/s²)
"""
mutable struct WaveParameters
    depth::Float64
    frequency::Float64
    wavenumber::Float64
    gravity::Float64

    function WaveParameters(
        depth::Real, frequency::Real, wavenumber::Real, gravity::Real
    )
        validate_wave(depth, frequency, wavenumber, gravity)
        h = Float64(depth)
        ω = Float64(frequency)
        k₀ = Float64(wavenumber)
        g = Float64(gravity)
        return new(h, ω, k₀, g)
    end
end


# Avoid non-physical values for the wave parameters.
function validate_wave(
    depth::Real, frequency::Real, wavenumber::Real, gravity::Real
)
    if depth ≤ 0
        throw(DomainError(depth, "The depth must be positive."))
    elseif frequency < 0
        throw(DomainError(frequency, "The frequency must be non-negative."))
    elseif wavenumber < 0
        throw(DomainError(wavenumber, "The wavenumber must be non-negative."))
    elseif gravity ≤ 0
        throw(DomainError(gravity, "The acceleration of gravity must be positive."))
    end
end


# Wave initializer
const wave = WaveParameters(NaN, NaN, NaN, NaN)


"""
    function setwave!(; depth::Real, frequency::Real, gravity::Real=9.80665) -> Nothing

Sets the parameters that define the environmental conditions.
"""
function setwave!(; depth::Real, frequency::Real, gravity::Real=9.80665)
    h = Float64(depth)
    g = Float64(gravity)
    ω = Float64(frequency)
    k₀ = find_k₀(h, ω, g)

    validate_wave(h, ω, k₀, g)

    # Set Chebyshev series approximations of L₁ and L₂ for a fixed parameter H.
    H = h * ω^2 / g

    if 0.01 ≤ H ≤ π
        # H ≤ π defines shallow and intermediate waters. H = 0.01 is the minimum value
        # that could be used to compute the anlytical expressions of L₁ and L₂.
        NearField.setintegrals!(H)
    end

    wave.depth = h
    wave.frequency = ω
    wave.wavenumber = k₀
    wave.gravity = g

    @info wave

    return nothing
end


"""
    find_k₀(h::Real, ω::Real, g::Real) -> Float64

Finds the wavenumber `k₀` from the dispersion relation ``ω^2 = k g \\tanh(k h)``.
"""
function find_k₀(h::Real, ω::Real, g::Real=9.80665)
    f(k::Real) = k*g * tanh(k*h) - ω^2
    f′(k::Real) = g * tanh(k*h) + k*g*h * sech(k*h)^2

    # initial estimate for k₀
    κ₁ = ω/√(g*h)  # shallow water
    κ₂ = ω^2/g  # deep water
    k̄₀ = √(κ₁*κ₂)  # geometric mean

    return findroot(f, f′, k̄₀)
end


# Reading aliases
function Base.getproperty(obj::WaveParameters, sym::Symbol)
    if sym === :h
        return getfield(obj, :depth)
    elseif sym === :ω
        return getfield(obj, :frequency)
    elseif sym === :k₀
        return getfield(obj, :wavenumber)
    elseif sym === :g
        return getfield(obj, :gravity)
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
