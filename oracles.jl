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
end; 

function euclidean_proj_l1ball(v, s=1)
    @assert s > 0 "Radius s must be strictly positive ($(s) ≤ 0)"
    n = length(v)
    if norm(v, 1) <= s
        return v
    end
    w = euclidean_proj_simplex(v, s)
    w = sign.(v) .* w
    return w
end;

function projection_l1(v, s=1)
    f = IndBallL1(s)
    y, fy = prox(f,v)
    return y
end;

function lmo_fn(v, s=1)
    index = argmax(abs.(v))
    dim =size(v)
    x = spzeros(dim)
    x[index] = -sign(v[index]) * s
    return x
end;


function generate_linear_prog_function(d::Vector{Float64}, cardinality::Int64)
    function linear_prog(x0) # linear programming
        dim = length(x0);
        ret = spzeros(dim);
        max_indices = partialsortperm(x0, 1:cardinality)
        ret[max_indices] = ones(cardinality);
        return ret;
    end
    return linear_prog;
end

# function generate_linear_prog_function1(d::Vector{Float64}, R)
#     function linear_prog(x0)
#         dim = length(x0)
        
#         # Create a JuMP model
#         model = Model()
        
#         # Define the decision variables
#         @variable(model, x[1:dim])
        
#         # Objective function: <gradient, x>
#         @objective(model, Min, dot(d, x))
        
#         # Constraint: norm(x) < R
#         @constraint(model, sum(abs(x)) <= R)
#         set_optimizer(model, GLPK.Optimizer)
#         # Solve the optimization problem
#         optimize!(model)
        
#         # Get the optimal solution
#         x_opt = value.(x)
        
#         # Return the sparse vector
#         return sparse(x_opt)
#     end
#     return linear_prog
# end;

# using Convex
# using JuMP
# using GLPK
# using SparseArrays
# function generate_linear_prog_function1(d::Vector{Float64}, R)
#     function linear_prog(x0)
#         dim = length(x0)
        
#         # Define the decision variables
#         x = Variable(dim)
#         objective = dot(x0,x)
#         # Define the 1-norm constraint
#         constraint = norm(x,1) <= R
        
#         # Solve the optimization problem subject to the 1-norm constraint
#         problem = minimize(objective, [constraint])
#         solve!(problem, GLPK.Optimizer)
#         # Get the optimal solution
#         x_opt = evaluate(x)
        
#         # Return the sparse vector
#         return sparse(x_opt)
#     end
#     return linear_prog
# end;

# lmo1 = generate_linear_prog_function1(rand(100),10);
# x0 = rand(100);
# @time sol = lmo1(x0)


# @time lmo_fn(x0,10)
