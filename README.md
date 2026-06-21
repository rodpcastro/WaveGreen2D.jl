<p align="center" width="100%">
    <img src="raw-github-link-to-logo" alt="WaveGreen2D Logo" width="500">
</p>

---

<p align="center">

<!--[![Docs-stable Badge]][Docs-stable]-->
[![Docs-dev Badge]][Docs-dev]
[![Coverage Badge]][Coverage]
[![Aqua Badge]][Aqua]
[![JET Badge]][JET]

</p>

# WaveGreen2D (Work in Progress)

This package was created with the following Julia commands:

```julia
using PkgTemplates

t = Template(
    authors=["Rodrigo Castro <code@rpc.aleeas.com>"],
    user="rodpcastro",
    dir="~/repos",
    host="github.com",
    julia=v"1.12",
    plugins=[
        ProjectFile(; version=v"0.0.1"),
        Git(; manifest=true, ssh=true),
        GitHubActions(),
        Documenter{GitHubActions}(),
        Codecov(),
        Develop(),
        Dependabot(),
        TagBot(),
        Tests(; project=true, aqua=true, jet=true),
    ]
)

t("WaveGreen2D")
```

<!--[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://rodpcastro.github.io/WaveGreen2D.jl/stable/)-->

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rodpcastro.github.io/WaveGreen2D.jl/dev/)
[![Build Status](https://github.com/rodpcastro/WaveGreen2D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rodpcastro/WaveGreen2D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rodpcastro/WaveGreen2D.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rodpcastro/WaveGreen2D.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)


[Docs-stable]: https://rodpcastro.github.io/WaveGreen2D.jl/stable/
[Docs-stable Badge]: https://img.shields.io/badge/docs-stable-blue.svg
[Docs-dev]: https://rodpcastro.github.io/WaveGreen2D.jl/dev/
[Docs-dev Badge]: https://img.shields.io/badge/docs-dev-blue.svg
[Coverage]: https://codecov.io/gh/rodpcastro/WaveGreen2D.jl
[Coverage Badge]: https://codecov.io/gh/rodpcastro/WaveGreen2D.jl/branch/main/graph/badge.svg
[Aqua]: https://github.com/JuliaTesting/Aqua.jl
[Aqua Badge]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[JET]: https://github.com/aviatesk/JET.jl
[JET Badge]: https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a
