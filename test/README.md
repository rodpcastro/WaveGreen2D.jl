# Tests
Unit testing is performed with the Julia Language Standard Library [Test] submodule, code quality is checked by [Aqua] and type stability analysis is performed with [JET]. Chebyshev coefficients loaded for testing are created by the [`test_coefficients.jl`][testcoefs] script.

## Refereces
1. JuliaLang. 2026. Test - The Julia Language Standard Library. https://docs.julialang.org/en/v1/stdlib/Test/
2. Julia Testing. 2020. Aqua.jl. https://github.com/JuliaTesting/Aqua.jl
3. Shuhei Kadowaki. 2021. JET.jl. https://aviatesk.github.io/JET.jl/

[Test]: https://docs.julialang.org/en/v1/stdlib/Test/
[Aqua]: https://juliatesting.github.io/Aqua.jl/
[JET]: https://aviatesk.github.io/JET.jl/
[testcoefs]: https://github.com/rodpcastro/WaveGreen2D.jl/blob/main/chebcoefs/test_coefficients.jl
