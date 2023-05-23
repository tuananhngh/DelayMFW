using Flux

function loss_a(x, data)
    return norm(x,2)^2 + dot(data, x)
end

function gradient_a(x, data)
    return Flux.gradient(loss_a, x, data)[1]
end

function d_ofw(dim, num_iters, max_delay, eta, R, data_cell, delay)
    x = ones(dim, 1)
    x = (R * x) / sum(x)
    x_1 = copy(x)
    loss = zeros(num_iters)
    G = zeros(dim)
    rec_G = zeros(dim, num_iters)

    for t in 1:num_iters
        b = data_cell[t, :];
        loss[t] = loss_a(x,b) 
        rec_G[:, t] = gradient_a(x,b) 
        for iner_iter in max(1, t - max_delay):t
            if iner_iter + delay[iner_iter] - 1 == t
                G += rec_G[:, iner_iter]
                DF = eta * G + 2 * (x - x_1)
                index = argmax(abs.(DF))
                v = zeros(dim, 1)
                v[index] = -sign(DF[index]) * R
                delta = v - x
                sigma = min(-0.5 * dot(delta, DF) / dot(delta, delta), 1)
                x = (1 - sigma) * x + sigma * v
            end
        end
    end
    return loss
end

function bold_ofw(dim, num_iters, max_delay, eta, R, data_cell, delay)
    x = ones(dim,1)
    x = (R * x) / sum(x)
    x_1 = copy(x)

    X = zeros(dim, max_delay+1)
    is_free = zeros(max_delay+1)
    for i = 1:max_delay+1
        X[:,i] = x_1
    end
    G = zeros(dim, max_delay+1)

    loss = zeros(num_iters)

    for t = 1:num_iters
        b = data_cell[t,:]
        base_id = 1
        while is_free[base_id] != 0
            base_id += 1
        end
        x = X[:,base_id]
        is_free[base_id] = delay[t]
        loss[t] = loss_a(x, b)
        G[:,base_id] += gradient_a(x,b)
        DF = eta * G[:,base_id] + 2 * (x - x_1)
        # linear optimization
        index = argmax(abs.(DF))
        v = zeros(dim,1)
        v[index] = -sign(DF[index]) * R
        delta = v - x
        sigma = min(-0.5 * dot(delta, DF) / dot(delta, delta), 1) # Line Search
        X[:,base_id] = (1 - sigma) * x + sigma * v
        for iner_id = 1:max_delay+1
            if is_free[iner_id] > 0
                is_free[iner_id] -= 1
            end
        end
    end
    return loss
end;


function delay_mfw(dim, data_cell, lmo, num_iters, max_delay, delay, zeta, R)
    function update_projection(oracle_, gradient, zeta_)
        oracle_.zeta = zeta_
        x_ = oracle_.x - oracle_.zeta * gradient
        oracle_.x = lmo(x_, R)
    end
    function update_fpl(oracle_, gradient)
        oracle_.accumulated_gradient += gradient;
        sol = lmo(oracle_.accumulated_gradient*oracle_.eta .+ oracle_.n0);
        oracle_.x = sol
    end
    function get_vector(oracle_)
        return oracle_.x
    end
    x0 = ones(dim);
    x0 = (R * x0) / sum(x0);
    K = Int(ceil(sqrt(num_iters)));
    println("K = $(K)");
    oracles = [ProjectionOracle(x0, zeta) for k in 1:K];
    
    rewards = zeros(num_iters);
    x_delay = [];
    st = time();
    for t in 1:num_iters
        cpt = t;
        f_delay = [];
        for s = max(1,t-max_delay):t
            if s + delay[s] - 1 == t
                push!(f_delay, s);
            end
        end
        xs = zeros(dim, K+1);
        gs = zeros(dim, K+1);
        v = [get_vector(oracles[k]) for k in 1:K];
        for k in 1:K
            eta_k = 1/k;
            xs[:,k+1] = xs[:,k] + eta_k*(v[k]-xs[:,k]);
        end
        if cpt <= max_delay
            push!(x_delay,xs);
            @assert x_delay[end]==xs;
        else
            popfirst!(x_delay);
            push!(x_delay,xs);
            @assert x_delay[end]==xs;
        end
        @assert length(x_delay) <= max_delay+1;
        xt = xs[:,K+1];
        rewards[t] = loss_a(xt, data_cell[t,:])
        epsilon = 0.01;
        @assert norm(xt,1) <= R+epsilon "Constraint violated";
        f_delay_ = [];
        for s in f_delay
            if cpt <= max_delay
                s_ = s;
                push!(f_delay_,s_) 
            else
                s_ = s-(t-max_delay) ;
                push!(f_delay_,s_);
            end
        end
        for k in 1:K
            for (s_,s) in zip(f_delay_,f_delay)
                gs[:,k] += gradient_a(x_delay[s_][:,k], data_cell[s,:]);
            end
            zeta_ = zeta
            update_projection(oracles[k], gs[:,k], zeta_);
        end
    end
    println("Time taken: $(time()-st)");
    return rewards;
end;

