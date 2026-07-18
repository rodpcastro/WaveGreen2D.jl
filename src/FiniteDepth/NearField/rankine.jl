# These three functions evaluate the Rankine source influence. The gradient and the hessian
# are computed with respect to the field points coordinates.


function Gᴿ(s::Float64)
    return log(s)
end


function ∇Gᴿ(s::Float64, ∇s::SVector{2, Float64})
    G = log(s)
    ∇G = ∇s ./ s
    return G, ∇G
end


function HGᴿ(s::Float64, ∇s::SVector{2, Float64}, Hs::SMatrix{2, 2, Float64})
    G = log(s)
    ∇G = ∇s ./ s
    HG = -(∇s * ∇s') ./ s^2 .+ Hs ./ s
    return G, ∇G, HG
end
