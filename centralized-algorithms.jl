
Random.seed!(1234)
using Logging

function loss_a(x, data)
    return norm(x,2)^2 + dot(data, x)
end

function gradient_a(x, data)
    #return 2*x + data
    return Flux.gradient(loss_a, x, data)[1]
end


function d_ofw(dim, num_iters, lmo, max_delay, eta, R, data_cell, delay)
    x = ones(dim)
    x = (R * x) / sum(x)
    x_1 = copy(x)
    loss = zeros(num_iters)
    G = zeros(dim)
    rec_G = zeros(dim, num_iters)

    for t in 1:num_iters
        b = data_cell[t, :];
        loss[t] = loss_a(x,b) 
        if t%1000==0
            println("loss $(t): $(loss[t])")
        end
        rec_G[:, t] = gradient_a(x,b) 
        for iner_iter in max(1, t - max_delay):t
            if iner_iter + delay[iner_iter] - 1 == t
                G += rec_G[:, iner_iter]
                DF = eta * G + 2 * (x - x_1)
                # index = argmax(abs.(DF))
                # v = zeros(dim, 1)
                # v[index] = -sign(DF[index]) * R
                v = lmo(DF, R)
                delta = v - x
                sigma = min(-0.5 * dot(delta, DF) / dot(delta, delta), 1)
                #println("sigma $(sigma)")
                #sigma = min(1, 2/t^(1/2))
                x = (1 - sigma) * x + sigma * v
            end
        end
    end
    return loss
end

function online_frank_wolfe(dim, num_iters, lmo, max_delay, eta, R, data_cell, delay)
    x = ones(dim)
    x = (R * x) / sum(x)
    x_1 = copy(x)
    G = zeros(dim)
    rec_G = zeros(dim, num_iters)
    loss = zeros(num_iters)
    for t in 1:num_iters
        b = data_cell[t, :]
        loss[t] = loss_a(x, b)
        if t%1000==0
            println("loss $(t): $(loss[t])")
        end
        rec_G[:, t] = gradient_a(x, b)
        G += rec_G[:, t]
        DF = eta * G + 2 * (x - x_1)
        v = lmo(DF, R)
        delta = v - x
        #sigma = min(-0.5 * dot(delta, DF) / dot(delta, delta), 1)
        sigma = min(1, 2/t^(1/2))
        x = (1 - sigma) * x + sigma * v
        #x_1 = copy(x)
    end

    return loss
end

function bold_ofw(dim, num_iters, lmo, max_delay, eta, R, data_cell, delay)
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
        # index = argmax(abs.(DF))
        # v = zeros(dim,1)
        # v[index] = -sign(DF[index]) * R
        v = lmo(DF, R)
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

function bold_mfw(dim, num_iters, lmo, max_delay, eta, R, data_cell, delay)
    function update_fpl(oracle_, gradient)
        #oracle_.n0 = rand(dim);
        oracle_.accumulated_gradient += gradient;
        sol = lmo(oracle_.accumulated_gradient*oracle_.eta .+ rand(dim), R);
        oracle_.x = sol
    end

    function get_vector(oracle_)
        return oracle_.x
    end

    x = ones(dim, 1)
    x = (R * x) / sum(x)
    is_free = zeros(max_delay+1)
    n0 = rand(dim)
    K = Int(ceil(sqrt(num_iters)));
    oracles = [[FPL(x,eta, zeros(dim), n0, R) for k in 1:K] for d in 1:max_delay+1];
    loss = zeros(num_iters)
    st = time()
    for t = 1:num_iters
        b = data_cell[t,:]
        base_id = 1
        while is_free[base_id] != 0
            base_id += 1
        end
        is_free[base_id] = delay[t]
        xs = zeros(dim, K+1)
        v = [get_vector(oracles[base_id][k]) for k in 1:K];
        for k in 1:K
            eta_k = 1/k;
            xs[:,k+1] = (1-eta_k)xs[:,k] + eta_k*v[k]
        end
        x = xs[:,K+1]
        loss[t] = loss_a(x, b)
        for k in 1:K
            grad = gradient_a(xs[:,k],b)
            update_fpl(oracles[base_id][k], grad)
        end
        for iner_id = 1:max_delay+1
            if is_free[iner_id] > 0
                is_free[iner_id] -= 1
            end
        end
        #println("loss $(loss[t])")
    end
    println("Time taken: $(time()-st)");
    return loss
end;

function delay_mfw(dim, data_cell, projection, num_iters, max_delay, delay, zeta, R)
    function update_fpl(oracle_, gradient)
        #oracle_.n0 = rand(dim);
        oracle_.accumulated_gradient += oracle_.eta*gradient;
        sol = lmo_fn(oracle_.accumulated_gradient .+ oracle_.n0, R);
        oracle_.x = sol
    end
    # function update_projection(oracle_, gradient, zeta_)
    #     oracle_.zeta = zeta_
    #     x_ = oracle_.x - oracle_.zeta * gradient
    #     oracle_.x = projection(x_, R)
    # end
    function get_vector(oracle_)
        return oracle_.x
    end
    x = ones(dim)
    x = (R * x) / sum(x)
    K = Int(ceil(sqrt(num_iters)));
    #K = 3000
    println("K = $(K)");
    #oracles = [ProjectionOracle(x,zeta) for k in 1:K];
    n0 = [rand(dim) for k in 1:K]
    oracles = [FPL(projection(-n0[k],R), zeta, zeros(dim), n0[k], R) for k in 1:K];
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
        xs[:,1] = x;
        gs = zeros(dim, K+1);
        v = [get_vector(oracles[k]) for k in 1:K]
        for k in 1:K
            eta_k = 1/k;
            xs[:,k+1] = (1-eta_k)*xs[:,k] + eta_k*v[k]
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
        if t%1000 == 0
            println("loss $(t): $(rewards[t])")
        end
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
            #update_projection(oracles[k], gs[:,k], zeta);
            update_fpl(oracles[k], gs[:,k]);
        end
    end
    println("Time taken: $((time()-st)/60) minutes");
    return rewards;
end;


function mfw(dim, data_cell, lmo, num_iters, eta, R)
    K = num_iters
    #fpl = [FPL(x0, eta, zeros(dim), n0, R) for k in 1:K]
    rewards = zeros(num_iters)
    fpl_grad = zeros(dim, K)
    x = ones(dim, 1)
    x = (R * x) / sum(x)
    x_1 = copy(x)
    for t in 1:num_iters
        b = data_cell[t,:]
        xs = zeros(dim)
        for k in 1:K 
            eta_k = 1/k
            fb = eta * fpl_grad[:,k] + rand(dim) 
            v = lmo_fn(fb, R)
            fpl_grad[:,k] .+= gradient_a(xs,b)
            xs = xs + eta_k * (v - xs)
        end
        rewards[t] = loss_a(xs, b)
        # for k in 1:K
        #     fpl_grad[:,k] .+= gradient_a(xs[:,k],b)
        # end
    end
    return rewards
end



function delay_mfw2(dim, data_cell, lmo, num_iters,max_delay, delay, eta, R)
    function update_fpl(oracle_, gradient)
        oracle_.accumulated_gradient += gradient
        #oracle_.n0 = rand(dim);
        sol = lmo(oracle_.eta * oracle_.accumulated_gradient + oracle_.n0, oracle_.R)
        oracle_.x = sol
    end
    function get_vector(oracle_)
        return oracle_.x
    end
    K = Int(num_iters/2)
    println("K = $(K)")
    n0 = [rand(dim) for k in 1:K]
    n01 = rand(dim)
    #fpl = [FPL(lmo(-n0[k],R), eta, zeros(dim), n0[k], R) for k in 1:K]
    fpl = [FPL(lmo(-n01,R), eta, zeros(dim), n01, R) for k in 1:K]
    loss= zeros(num_iters)
    st = time()
    #compute ahead the released time for each time step
    released_time = zeros(Int, num_iters);
    for (i, ds) in enumerate(delay)
        released_time[i] = i+ds-1 
    end
    gs_ = zeros(dim, K, num_iters)
    epsilon = 0.01
    for t in 1:num_iters
        b = data_cell[t,:]
        xs = zeros(dim, K+1)
        #xs[:,1] = x
        v = [get_vector(fpl[k]) for k in 1:K]
        for k in 1:K 
            sigma = min(1,10/(k));
            delta = v[k] - xs[:,k]
            xs[:,k+1] = xs[:,k] + sigma * delta
        end
        #println(norm(xs[:,K+1],1))
        @assert norm(xs[:,K+1],1) <= R+epsilon "Constraint violated";
        loss[t] = loss_a(xs[:,K+1], b)
        if t%1000 == 0
            println("loss $(t) $(loss[t])")
        end
        #println("Release time of $(t) is $(released_time[t])")
        for k in 1:K
            if released_time[t] <= num_iters
                gs_[:,k,released_time[t]] += gradient_a(xs[:,k],b)
            end
            #update_fpl(oracles[k], gs_[:,k,t]);
            update_fpl(fpl[k],gs_[:,k,t]);
        end
    end
    println("Time taken: $((time()-st)/60) minutes")
    return loss
end;

function delay_mfw3(dim, data_cell, lmo, num_iters, max_delay, delay, eta, R)
    function update_fpl(oracle_, gradient)
        oracle_.accumulated_gradient += gradient
        #oracle_.n0 = rand(dim) 
        sol = lmo(oracle_.eta * oracle_.accumulated_gradient + oracle_.n0, oracle_.R)
        oracle_.x = sol
    end
    function get_vector(oracle_)
        return oracle_.x
    end
    K = Int(num_iters/2)
    #K = num_iters
    println("K = $(K)")
    n0 = [rand(dim) for k in 1:K]
    n01 = rand(dim)
    fpl = [FPL(lmo(n0[k],R), eta, zeros(dim), n0[k], R) for k in 1:K]
    #fpl = [FPL(lmo(n01,R), eta, zeros(dim), n01, R) for k in 1:K]
    loss= zeros(num_iters)
    st = time()
    gs_ = zeros(dim, K, num_iters)
    epsilon = 0.01
    norm_x = zeros(num_iters)
    norm_g = zeros(num_iters)
    x = ones(dim)
    x = (R * x) / sum(x)
    grad_cell = zeros(dim,K)
    for t in 1:num_iters
        b = data_cell[t,:]
        xs = zeros(dim, K+1)
        xs[:,1] = zeros(dim)
        #v = [get_vector(fpl[k]) for k in 1:K]
        for k in 1:K 
            sigma = min(1,1/(k+3))
            gs_[:,k,t] = gradient_a(xs[:,k],b)
            #update oracle
            injected_noise = rand(dim) .- 0.5
            v = lmo(grad_cell[:,k] + eta*injected_noise, R) 
            #lmo(eta*grad_cell[:,k] +  rand_point_around(xs[:,k],1), R)
            xs[:,k+1] = (1-sigma)*xs[:,k] + sigma * v
            @assert norm(xs[:,k],k) <= R+epsilon "Constraint violated";
        end
        @assert norm(xs[:,K+1],1) <= R+epsilon "Constraint violated";
        # if 150 < t <200
        #     println("value of x$(t) $(norm(xs[:,K+1],1))")
        # end
        loss[t] = loss_a(xs[:,K+1], b)
        norm_x[t] = norm(xs[:,K+1],1)
        if t%1 == 0
            println("loss $(t) $(loss[t])")
        end
        #println("Release time of $(t) is $(released_time[t])")
        cpt = 0
        # for iner_iter in max(1, t - max_delay):t
        #     if iner_iter + delay[iner_iter] - 1 == t
        #         for k in 1:K
        #             grad_cell[:,k] += gs_[:,k,iner_iter]
        #         end
        #     end
        # end
        for k in 1:K
            fb_tmp = zeros(dim)
            for iner_iter in max(1, t-max_delay):t
                if iner_iter + delay[iner_iter] - 1 == t
                    #push!(idx_tmp,iner_iter)
                    fb_tmp += gs_[:,k,iner_iter] #(1-1/)fb_tmp + (1/2)*gs_[:,k,iner_iter]
                    cpt += 1
                end
            end
            if cpt != 0
                grad_cell[:,k] += fb_tmp
            end
        end
        # norm_g[t] = norm(gs,1)
        # fb_reg = min(1,1/cpt)
        # grad_cell[:,k] += gs_[:,k,t]*fb_reg
        # #update_fpl(fpl[k],gs*fb_reg);
    end
    plotnorm = plot(1:num_iters, [norm_x ],
                    label=["Norm of x" "Norm of gradient"], 
                    xlabel="Iterations t - Max Delay $(max_delay)",
                    ylabel="Norm of x",
                    legend=:topright)
    display(plotnorm)
    println("Time taken: $((time()-st)/60) minutes")
    return loss
end;

function delay_mfw_ml(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R)
    
    K = 1 #Int(num_iters/2)
    @warn "Value of K: $K"
    @warn "Value of eta: $eta"
    @warn "Value of max_delay: $max_delay"
    
    loss_val= zeros(num_iters)
    st = time()
    gs_ = zeros(num_iters, K, dim...)
    grad_cell = zeros(K, dim...)
    norm_x = zeros(num_iters)
    for t in 1:num_iters
        dt, lb = data[t,:,:], label[t,:]
        xs = zeros(K+1, dim...)
        for k in 1:K 
            sigma = min(1,1/(k+3))
            gs_[t,k,:,:] = gradient_enp(xs[k,:,:],dt,lb)
            #update oracle
            injected_noise = rand(dim...) .- 0.5
            v = lmo(eta * grad_cell[k, :, :] + injected_noise, R) 
            xs[k+1,:,:] = (1-sigma)*xs[k,:,:] + sigma * v
        end
        xt = xs[K+1,:,:]
        norm_x[t] = norm(xt,1)
        loss_val[t] = loss_enp(xt, dt, lb)
        if t%100 == 0
            @info "loss $(t): $(loss_val[t])"
        end
        cpt = 0
        #for k in 1:K
        fb_tmp = zeros(dim...)
        for iner_iter in max(1, t-max_delay):t
            if iner_iter + delay[iner_iter] - 1 == t
                for k in 1:K
                    grad_cell[k,:,:]+= gs_[iner_iter,k,:,:] 
                end
            end
        end
    end
    plotnorm = plot(1:num_iters, [norm_x],
                    label=["Norm of x" "Norm of gradient"], 
                    xlabel="Iterations t - Max Delay $(max_delay)",
                    ylabel="Norm of x",
                    legend=:topright)
    display(plotnorm)
    @info "Time taken: $((time()-st)/60) minutes"
    return loss_val
end;

function d_ofw_ml(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R)
    x = ones(dim...)
    x = (R * x) / sum(x)
    x_1 = copy(x)
    loss_val = zeros(num_iters)
    G = zeros(dim...)
    rec_G = zeros(num_iters, dim...)

    for t in 1:num_iters
        dt, lb = data[t,:,:], label[t,:]
        loss_val[t] = loss_enp(x,dt,lb) 
        if t%100==0
            @info "loss $(t): $(loss_val[t])"
        end
        rec_G[t,:,:] = gradient_enp(x,dt,lb) 
        for iner_iter in max(1, t - max_delay):t
            if iner_iter + delay[iner_iter] - 1 == t
                G += rec_G[iner_iter,:,:]
                DF = eta * G + 2 * (x - x_1)
                v = lmo(DF, R)
                delta = v - x
                sigma = min(-0.5 * dot(delta, DF) / dot(delta, delta), 1)
                x = (1 - sigma) * x + sigma * v
            end
        end
    end
    return loss_val
end
