using WaveGreen2D
using Documenter

DocMeta.setdocmeta!(WaveGreen2D, :DocTestSetup, :(using WaveGreen2D); recursive=true)

makedocs(;
    modules=[WaveGreen2D],
    authors="Rodrigo Castro <code@rpc.aleeas.com>",
    sitename="WaveGreen2D.jl",
    format=Documenter.HTML(;
        canonical="https://rodpcastro.github.io/WaveGreen2D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/rodpcastro/WaveGreen2D.jl",
    devbranch="main",
)
