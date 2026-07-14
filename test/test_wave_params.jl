using WaveGreen2D: nevamodes, wave, validate_wave, find_k₀, find_kₙ


@testset "Find wavenumber" begin
    h = 10.0
    k = 1.0
    g = 10.0
    ω = sqrt(k*g*tanh(k*h))

    @test find_k₀(h, ω, g) ≈ k
end


@testset "Check evanescent wavenumbers" begin
    h = 40.0
    ω = 0.8
    g = 10.0

    for n in 1:nevamodes
        kₙ = find_kₙ(n, h, ω, g)
        @test (n-0.5)*π/h < kₙ < n*π/h
    end
end


@testset "Wave setting" begin
    @test all(isnan.(wave.kₙ))
    @test WaveGreen2D.NearField.integrals.L₁.coefs.size == (1, 1)

    setwave!(depth=30.0, frequency=0.7, gravity=9.8)
    @test !all(isnan.(wave.kₙ))
    @test WaveGreen2D.NearField.integrals.L₁.coefs.size != (1, 1)
end


@testset "Wave info" begin
    @test_logs (
        :info,
        "Wave parameters h = 30.0 m, ω = 0.7 rad/s, g = 9.8 m/s²",
    ) setwave!(depth=30.0, frequency=0.7, gravity=9.8)
end


@testset "Wave show" begin
    io = IOBuffer()
    setwave!(depth=30.0, frequency=0.7, gravity=9.8)

    @test sprint(show, wave) == "Wave parameters h = 30.0 m, ω = 0.7 rad/s, g = 9.8 m/s²"
    @test sprint(show, MIME"text/plain"(), wave) == """Wave parameters h = 30.0 m, \
                                                       ω = 0.7 rad/s, g = 9.8 m/s²"""
end


@testset "Non-physical wave" begin
    @test_throws DomainError validate_wave(-1.0, 1.0, 1.0)
    @test_throws DomainError validate_wave(1.0, -1.0, 1.0)
    @test_throws DomainError validate_wave(1.0, 1.0, -1.0)
end


@testset "Field aliases" begin
    setwave!(depth=1.0, frequency=2.0, gravity=3.0)

    @test wave.depth === wave.h
    @test wave.frequency === wave.ω
    @test wave.gravity === wave.g
    @test wave.wavenumber === wave.k₀
    @test wave.evanumbers === wave.kₙ
end
