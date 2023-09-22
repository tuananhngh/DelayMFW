using MAT, FileIO
@everywhere using Distributed

@everywhere function get_delay(delay, max_delay, t)
    f_delay = []
    for s = max(1,t-max_delay):t
        if s + delay[s] - 1 == t
            push!(f_delay, s);
        end
    end
    return f_delay
end

@everywhere function retrieve_delay(f_delay,max_delay, cpt, t)
    f_delay_ = []
    for s in f_delay
        if cpt <= max_delay
            s_ = s;
            push!(f_delay_,s_) 
        else
            s_ = s-(t-max_delay) ;
            push!(f_delay_,s_);
        end
    end
    return f_delay_
end



@everywhere function f_batch(xt, data)
    f = dot(xt, data) + norm(xt,2)^2;
    return f
end

@everywhere function gradient_batch(xt, data)
    grad = 2*xt + data;
    return grad
end


@everywhere function delay_hold(cpt, max_delay, curr_delay_hold, x)
    if cpt <= max_delay
        push!(curr_delay_hold, x)
        @assert curr_delay_hold[end]==x;
    else 
        popfirst!(curr_delay_hold)
        push!(curr_delay_hold,x)
        @assert curr_delay_hold[end]==x;
    end
    return curr_delay_hold
end 


function load_network(network_type="er", num_agents=50)
    ROOT = "./data/";
    filename = "$(ROOT)weights_$(network_type)_$(num_agents).mat";
    file = matopen(filename);
    weights = read(file, "weights");
    close(file);
    # find the first and second largest (in magnitude) eigenvalues
    dim = size(weights, 1);
    eigvalues = (LinearAlgebra.eigen(weights)).values;
    if abs(eigvalues[dim] - 1.0) > 1e-8
        error("the largest eigenvalue of the weight matrix is $(eigvalues[dim]), but it must be 1");
    end
    beta = max(abs(eigvalues[1]), abs(eigvalues[dim - 1]));
    if beta < 1e-8
        beta = 0.0;
    end
    return (weights, beta);
end


function lmo_fn_dec(V, radius)
    num_rows, num_cols = size(V)
    v = zeros(size(V))
    # Find row indices corresponding to the maximum absolute values in each column
    idx = argmax(abs.(V), dims=1)
    for col in 1:num_cols
        row = idx[col][1]
        v[row, col] = -radius * sign(V[row, col])
    end
    return v
end

# myV = rand(20,10)
# ok = zeros(size(myV))
# idx = argmax(abs.(myV), dims=1)
# for col in 1:10
#     row = idx[col][1]
#     ok[row, col] = -radius .* sign.(myV[row, col])
# end
# radius = 8
# lmo_fn(myV, radius)