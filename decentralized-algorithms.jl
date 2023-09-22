include("utils-decentralized.jl")
using Random


function dec_delay_sgd(dim, data, label, num_agents, weights, projection, f_batch, gradient_batch, num_iters, max_delay, radius)
    Random.seed!(1234);
    function f_sum(x, data_, label_)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x[:,:,i], data_, label_)
        end
        return f_x;
    end
    println("Data size $(size(data))")
    x0 = @sync @distributed (hcat) for i in 1:num_agents
        projection(rand(dim...), radius)
    end
    x0 = reshape(x0, (dim..., num_agents))
    xt = zeros(num_iters+1, dim..., num_agents)
    xt[1,:,:,:] = x0
    println("xt shape $(size(xt))")
    delay_agents = fill(max_delay, num_agents)
    loss = zeros(num_iters)
    t_start = time()
    function update_worker(x, data_, label_, timestep, delay, eta, radius)
        if timestep < delay
            x_tmp = x[timestep+1,:,:]
        else
            x_delay = x[timestep - delay+1,:,:]
            grad_delay = gradient_batch(x_delay, data_[timestep - delay+1,:,:], label_[timestep - delay+1,:])
            x_tmp = x[timestep+1,:,:] - eta*grad_delay
            x_tmp = projection(x_tmp, radius)
        end
        return x_tmp
    end
    avg_delay = mean(delay_agents)
    for t in 1:num_iters
        eta_tmp = 1/sqrt(max_delay*t)
        xt[t+1,:,:,:] = reshape(reshape(xt[t,:,:,:],(dim[1]*dim[2],nb_agents))*weights, (dim..., nb_agents))
        updated_x = @sync @distributed (hcat) for i in 1:num_agents
            update_worker(xt[:,:,:,i], data[:,:,:,i], label[:,:,i], t, delay_agents[i], eta_tmp, radius)
        end
        xt[t+1,:,:,:] = reshape(updated_x, (dim..., num_agents))
        tmp_loss = zeros(num_agents)
        for i in 1:num_agents
            tmp_loss[i] = f_sum(xt[t,:,:,:],data[t,:,:,i], label[t,:,i])/num_agents
        end
        loss[t] = maximum(tmp_loss)
        if t%10==0
            println("Iteration $(t), Loss $(loss[t])")
        end
    end
    println("Time taken: $(time()-t_start)");
    return loss
end

function dec_delay_mfw(dim, data, label, num_agents, weights, projection, f_batch, gradient_batch, num_iters, zeta, delay, max_delay, radius)
    Random.seed!(1234);
    function get_delay_cat(delay, max_delay, ite)
        f_delays = distribute([[] for _ in 1:num_agents])
        @sync @distributed for i in 1:num_agents
            append!(localpart(f_delays)[i],get_delay(delay[:,i], max_delay, ite))
        end; 
        return f_delays
    end
    function retrieve_delay_cat(f_delays, max_delay, cpt, ite)
        f_delays_ = distribute([[] for _ in 1:num_agents])
        @sync @distributed for i in 1:num_agents
            append!(localpart(f_delays_)[i], retrieve_delay(f_delays[i], max_delay, cpt, ite))
        end;
        return f_delays_
    end;
    function projection_cat(d, radius)
        res = @sync @distributed (hcat) for i in 1:num_agents
            projection(d[:,:,i], radius)
        end
        res = reshape(res, (dim..., num_agents))
        return res;
    end
    function delay_hold_cat(curr_delay_hold, cpt, max_delay, x)
        @sync @distributed for i in 1:num_agents
            delay_hold(cpt, max_delay, curr_delay_hold[i], x[:,:,:,i])
            @assert length(curr_delay_hold[i]) <= max_delay+1 
        end
    end

    function f_sum(x, data_, label_)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x[:,:,i], data_, label_)
        end
        return f_x;
    end

    function get_vector(oracle_)
        return oracle_.x
    end

    function update_projection(oracle_, gradient, zeta_)
        oracle_.zeta = zeta_
        x_ = oracle_.x - oracle_.zeta * gradient
        oracle_.x = projection_cat(x_, radius)
    end
    K = floor(Int, sqrt(num_iters));
    println("K $(K)")
    x0 = @sync @distributed (hcat) for i in 1:num_agents
        projection(rand(dim...), radius)
    end
    x0 = reshape(x0, (dim..., num_agents))
    println("x shape $(size(x0))")
    oracles = [ProjectionOracle(x0, zeta) for _ in 1:K];
    
    rewards = zeros(num_iters);
    t_start = time();
    xt = zeros(dim..., num_agents)
    x_delays = distribute([[] for i in 1:num_agents]);

    for t in 1:num_iters
        cpt = t;
        f_delays = get_delay_cat(delay, max_delay, t)
        xs = zeros(dim..., K+1, num_agents);
        gs_ = zeros(dim..., K+1, num_agents);
        v = [get_vector(oracles[k]) for k in 1:K];
    
        for k in 1:K
            eta_k = 1/k;
            ys_ = reshape(reshape(xs[:,:,k,:],(dim[1]*dim[2],nb_agents))*weights, (dim..., nb_agents))
            xs[:,:,k+1,:] = (1-eta_k)*ys_ + eta_k*v[k];
        end

        
        delay_hold_cat(x_delays, cpt, max_delay, xs)
        
        xt = xs[:,:,K+1,:];
        if t==100
            println(sum(xt[:,:,1]))
            println(sum(xt[:,:,2]))
            println(sum(xt[:,:,3]))
        end
        epsilon = 0.01;
        @assert norm(xt[:,:,1],1) <= radius+epsilon "Constraint violated";
        f_delays_ = retrieve_delay_cat(f_delays, max_delay, cpt, t)
        tmp_rw = zeros(num_agents)
        for i in 1:num_agents
            tmp_rw[i] = f_sum(xt,data[t,:,:,i], label[t,:,i])/num_agents;
        end
        rewards[t] = maximum(tmp_rw);
        if t%100==0
            println("Iteration $(t), Loss $(rewards[t])")
        end
        
        function compute_delay_grad(f1, f2, x, dt,lb,i)
            tmp_grad = zeros(dim...)
            for (s_, s) in zip(f1, f2)
                tmp_grad += gradient_batch(x[s_][:,:,1], dt[s,:,:,i],lb[s,:,i])
            end
            return tmp_grad
            println(size(tmp_grad))
        end

        function compute_delay_grad1(f1, f2, x, dt, lb, i, k)
            tmp_grad = zeros(dim...)
            for (s_, s) in zip(f1, f2)
                tmp_grad += gradient_batch(x[s_][:,:,k+1], dt[s,:,:,i], lb[s,:,i]) - gradient_batch(x[s_][:,:,k], dt[s,:,:,i], lb[s,:,i])
            end
            return tmp_grad
        end

        update_1 = @sync @distributed (hcat) for i in 1:num_agents
                   compute_delay_grad(f_delays_[i], f_delays[i], x_delays[i],data, label,i)
        end
        update_1 = reshape(update_1, (dim..., num_agents))
        gs_[:,:,1,:] = update_1
        for k in 1:K
            ds = reshape(reshape(gs_[:,:,k,:],(dim[1]*dim[2], nb_agents))*weights, (dim..., nb_agents))
            update_grad = @sync @distributed (hcat) for i in 1:num_agents 
                compute_delay_grad1(f_delays_[i], f_delays[i], x_delays[i], data, label, i, k)
            end 
            update_grad = reshape(update_grad, (dim..., num_agents))
            gs_[:,:,k+1,:] = update_grad + ds
            zeta_ = zeta 
            update_projection(oracles[k], ds, zeta_)
        end
    end
    println("Time taken: $(time()-t_start)");
    return rewards
end;


function dec_delay_mfw2(dim, data, label, num_agents, weights, lmo, f_batch, gradient_batch, num_iters, eta, delay, radius)
    Random.seed!(1234);
    function f_sum(x, data_, label_)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x[:,:,i], data_, label_)
        end
        return f_x;
    end
    # Get oracle's output
    function get_vector(oracle_)
        return oracle_.x
    end
    # Oracle's update
    function update_fpl(oracle_, gradient)
        oracle_.accumulated_gradient += gradient;
        sol = lmo(oracle_.accumulated_gradient*oracle_.eta .+ oracle_.n0, radius);
        oracle_.x = sol
    end
    # Initializa agent's oracle values
    x0 = ones(dim..., num_agents)
    for i in 1:num_agents
        tmp = x0[:,:,i]
        x0[:,:,i] = R*tmp/sum(tmp)
    end
    println("norm x0 $(norm(x0[:,:,1],1))")
    n0 = rand(dim..., num_agents)
    K = floor(Int, sqrt(num_iters));
    println("K $(K)")
    oracles = [FPL(x0,eta, zeros(dim...,num_agents), n0) for _ in 1:K];
    rewards = zeros(num_iters);
    t_start = time();
    xt = zeros(dim..., num_agents)

    # Compute feedback time of each agent
    released_time = zeros(Int, num_iters, num_agents);
    for i in 1:num_agents
        for (it, ds) in enumerate(delay[:,i])
            released_time[it,i] = it+ds-1 
        end
    end
    gs_ = zeros(dim..., K+1, num_iters, num_agents);
    # Main Loops
    for t in 1:num_iters
        xs = zeros(dim..., K+1, num_agents);
        v = [get_vector(oracles[k]) for k in 1:K];
        for k in 1:K
            eta_k = 1/k;
            ys_ = reshape(reshape(xs[:,:,k,:],(dim[1]*dim[2],nb_agents))*weights, (dim..., nb_agents))
            xs[:,:,k+1,:] = (1-eta_k)*ys_ + eta_k*v[k];
        end
        
        xt = xs[:,:,K+1,:];
        #epsilon = 0.01;
        #@assert norm(xt[:,:,1],1) <= radius+epsilon "Constraint violated";
        tmp_rw = zeros(num_agents)
        
        # Compute loss
        for i in 1:num_agents
            tmp_rw[i] = f_sum(xt,data[t,:,:,i], label[t,:,i])/num_agents;
        end
        rewards[t] = maximum(tmp_rw);
        if t%100==0
            println("Iteration $(t), Loss $(rewards[t])")
        end
        # Get feedback_time at time t of all agents
        feedback_at_t = released_time[t,:]
        # gs_ is of dimension (dim..., K+1, num_iters, num_agents)
        function add_delay_gradient1(agent_i, time_t, feedback_time)
        # Function compute gs_[...,k=1,...] for all agents 
        # the gradient is then keep at corresponding feedback time
            x = xs[:,:,1,agent_i]
            feat = data[time_t,:,:,agent_i]
            lab = label[time_t,:,agent_i]
            fb= feedback_time[agent_i]
            if fb <= num_iters
                gs_[:,:,1, fb, agent_i] += gradient_batch(x, feat,lab)
            else 
                println("From add_delay_gradient1 : Feedback time $(fb) of time $(time_t) of agent $(agent_i) is greater than iterations $(num_iters)")
            end
        end

        function add_delay_gradient2(agent_i, k ,time_t, feedback_time)
        # Function compute gs_[...,k,...] for all agents 
        # the gradient is then keep at corresponding feedback time
            x1 = xs[:,:,k, agent_i]
            x2 = xs[:,:,k+1, agent_i]
            feat = data[time_t, :, :, agent_i]
            lab = label[time_t, :, agent_i]
            fb = feedback_time[agent_i]
            if fb <= num_iters
                gs_[:,:, k+1, fb, agent_i] += gradient_batch(x2, feat,lab) - gradient_batch(x1, feat, lab)
            else 
                println("From add_delay_gradient2 : Feedback time $(fb) of time $(time_t) of agent $(agent_i) is greater than iterations $(num_iters)")
            end
        end  
        # Run in parallel initialization of gs_ for k = 1
        @sync @distributed for i in 1:num_agents
            add_delay_gradient1(i, t, feedback_at_t)
        end
        # #Update gradient oracle
        for k in 1:K
            ds = reshape(reshape(gs_[:,:,k,t,:],(dim[1]*dim[2], nb_agents))*weights, (dim..., nb_agents))
            @sync @distributed for i in 1:num_agents 
                add_delay_gradient2(i, k, t, feedback_at_t)
            end
            gs_[:,:,k+1,t,:] += ds
            update_fpl(oracles[k], ds)
        end
    end
    println("Time taken: $(time()-t_start)");
    return rewards
end;



