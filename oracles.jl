using Distributed
using LinearAlgebra
using SparseArrays
using ProximalOperators


function euclidean_proj_simplex(v, s=1) 
    @assert s > 0 "Radius s must be strictly positive ($(s) ≤ 0)"
    n = length(v)
    if sum(v) == s && all(v .≥ 0)
        # best projection: itself!
        return v
    end
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = sum(u .* collect(1:n) .> (cssv .- s))
    theta = (cssv[rho] .- s) ./ (rho)
    w = max.(v .- theta, 0)
    return w
end 

function euclidean_proj_l1ball(v, s=1)
    @assert s > 0 "Radius s must be strictly positive ($(s) ≤ 0)"
    n = length(v)
    if norm(v, 1) <= s
        return v
    end
    w = euclidean_proj_simplex(v, s)
    w = sign.(v) .* w
    return w
end

function projection_l1(v, s=1)
    f = IndBallL1(s)
    y, fy = prox(f,v)
    return y
end

function lmo_fn(v, s=1)
    index = argmax(abs.(v))
    dim =size(v)
    x = zeros(dim)
    x[index] = -sign(v[index]) * s
    return x
end
