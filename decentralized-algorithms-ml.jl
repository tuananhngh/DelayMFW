#Random.seed!(1234);

# Function to compute weighted average
function weighted_average(x, w)
    dim1, dim2, nb_agent = size(x)
    # Reshape x to a 2D array (dim1*dim2, nb_agent)
    x_reshaped = reshape(x, dim1*dim2, nb_agent)
    result = x_reshaped * w    
    return reshape(result, (dim1, dim2, nb_agent))
    #return result
end

# Function to compute LMO in parallel
function distribute_lmo(grad, radius, lmo_function)
    #storage = zeros(size(grad))
    dim1, dim2, nb_agents = size(grad)
    #nb_agents = size(grad, 3)
    storage = @sync @distributed (hcat) for i in 1:nb_agents
        lmo_function(grad[:,:,i], radius)
    end
    return reshape(storage, (dim1, dim2, nb_agents))
end


function dec_delay_mfw2(dim, data, label, num_agents, weights, lmo, f_batch, gradient_batch, num_iters, eta, delay, radius, K)
    function f_sum(x, data_, label_)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x[:,:,i], data_, label_)
        end
        return f_x;
    end
    @info "Running for $(num_iters) Iterations and $(K) Subiterations"
    t_start = time();
    rewards = zeros(num_iters);
    # Compute feedback time of each agent
    released_time = zeros(Int, num_iters, num_agents);
    @sync @distributed for i in 1:num_agents
        for (it, ds) in enumerate(delay[:,i])
            released_time[it,i] = it+ds-1 
        end
    end
    gs_ = zeros(num_iters, K+1, dim..., num_agents)
    grad_cell = zeros(K, dim..., num_agents)
    #injected_noise = rand(dim..., num_agents) 
    # Main Loops
    @showprogress for t in 1:num_iters
        dt, lb = data[t,:,:,:], label[t,:,:]
        xs = zeros(K+1, dim..., num_agents);
        for k in 1:K
            sigma = min(1,1/(k+3))
            #assume that we know the gradient function to save computation
            #gs_[t,k,:,:,:] = gradient_batch(xs[k,:,:,:],dt,lb)
            injected_noise = rand(dim..., num_agents) .- 0.5
            obj_lmo = eta*grad_cell[k,:,:,:]+injected_noise
            v = distribute_lmo(obj_lmo, radius, lmo)
            ys_ = weighted_average(xs[k,:,:,:], weights)
            xs[k+1,:,:,:] = (1-sigma)*ys_ + sigma*v
        end
        xt = xs[K+1,:,:,:]
        tmp_rw = zeros(num_agents)
        # Compute loss
        for i in 1:num_agents
            tmp_rw[i] = f_sum(xt,dt[:,:,i], lb[:,i])/num_agents;
        end
        rewards[t] = maximum(tmp_rw);
        #Get feedback_time at time t of all agents
        feedback_at_t = released_time[t,:]

        # gs_ is of dimension (dim..., K+1, num_iters, num_agents)
        function add_delay_gradient1(agent_i, time_t, feedback_time)
        # Function compute gs_[...,k=1,...] for all agents 
        # the gradient is then keep at corresponding feedback time
            x = xs[1,:,:,agent_i]
            feat = dt[:,:,agent_i]
            lab = lb[:,agent_i]
            fb= feedback_time[agent_i]
            if fb <= num_iters
                tmp_grad = gradient_batch(x, feat,lab)
                gs_[fb,1,:,:,agent_i] += tmp_grad
            end
        end
            
        # Function compute gs_[...,k,...] for all agents 
        # the gradient is then keep at corresponding feedback time
        function add_delay_gradient2(agent_i, k ,time_t, feedback_time, ds)
            x1 = xs[k,:,:,agent_i]
            x2 = xs[k+1,:,:,agent_i]
            feat = dt[:, :, agent_i]
            lab = lb[:, agent_i]
            fb = feedback_time[agent_i]
            if fb <= num_iters
                tmp_grad1 = gradient_batch(x1, feat,lab)
                tmp_grad2 = gradient_batch(x2, feat,lab)
                gs_[fb,k+1,:,:,agent_i] += tmp_grad2-tmp_grad1 + ds[:,:,agent_i]
            end
        end  
        #Run in parallel initialization of gs_ for k = 1
        @sync @distributed for i in 1:num_agents
            add_delay_gradient1(i, t, feedback_at_t)
        end
        #Update gradient oracle
        for k in 1:K
            ds = weighted_average(gs_[t,k,:,:,:], weights)
            @sync @distributed for i in 1:num_agents 
                add_delay_gradient2(i, k, t, feedback_at_t, ds)
            end
            grad_cell[k,:,:,:] += ds
        end
    end
    @info "Time taken: $((time()-t_start)/60 ) minutes"
    return rewards
end;



function dec_delay_mfw3(dim, data, label, num_agents, weights, lmo, f_batch, gradient_batch, num_iters, eta, delay, max_delay, radius)
    function f_sum(x, data_, label_)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_[:,:,i], label_[:,i])
        end
        return f_x
    end
    function gradient_cat(x, data_, label_)
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:,:,i], data_[:,:,i], label_[:,i])
        end
        return reshape(grad_x, (dim..., num_agents))
    end
    K = Int(ceil(sqrt(num_iters)))
    @info "Running for $(num_iters) Iterations and $(K) Subiterations"
    t_start = time()
    rewards = zeros(num_iters) 
    @info "Initialization"
    grad_cell = zeros(K, dim..., num_agents)
    gradient_hold = [[] for _ in 1:num_agents]
    injected_noise = rand(dim..., num_agents) 
    @info "Main Loops"
    @showprogress for t in 1:num_iters
        dt, lb = data[t,:,:,:], label[t,:,:]
        xs = zeros(dim..., num_agents);
        # Assume that we know the gradient to reduce computation
        tmp_grad = zeros(K+1, dim..., num_agents)
        for k in 1:K
            sigma = min(1,1/(k+3))
            tmp_grad[k,:,:,:] = gradient_cat(xs,dt,lb) 
            obj_lmo = eta*grad_cell[k,:,:,:] + injected_noise  #lmo for linear loss function
            v = distribute_lmo(obj_lmo, radius, lmo)
            ys_ = weighted_average(xs, weights)
            xs = (1-sigma)*ys_ + sigma*v
        end
        #tmp_grad[K+1,:,:,:] = gradient_cat(xs[K+1,:,:,:],dt,lb)
        tmp_grad[K+1,:,:,:] = gradient_cat(xs,dt,lb)
        #xt = xs[K+1,:,:,:]
        xt = copy(xs)
        tmp_rw = zeros(num_agents)
        # Compute loss
        rw = 0
        for i in 1:num_agents
            tmp_rw[i] += f_sum(xt[:,:,i],dt, lb)/num_agents;
            #rw += f_sum(xt,dt[:,:,i], lb[:,i]);
        end
        #rewards[t] = rw/num_agents;
        rewards[t] = maximum(tmp_rw);

        # Push the gradient at corresponding time for delay feedback
        @sync @distributed for i in 1:num_agents
            push!(gradient_hold[i], (t,tmp_grad[:,:,:,i]))
        end

        # @sync @distributed for i in 1:num_agents
        #     for ele in gradient_hold[i]
        #         @info "iteration $(t) in gradient_hold $(i) - $(ele[1])"
        #     end
        # end

        # Find delay gradient for feedback at current time
        sum_delay_grad = zeros(K+1, dim..., num_agents)
        @sync @distributed for i in 1:num_agents
            for (idx_s,s) in enumerate(gradient_hold[i])
                if s[1] + delay[s[1],i] - 1 == t
                    sum_delay_grad[:,:,:,i] += s[2]
                    deleteat!(gradient_hold[i], idx_s)
                    #@info "iteration deleted $(s[1]), idx $(idx_s)"
                    #@info "length of gradient_hold $(length(gradient_hold[i]))"
                end
            end
        end
        # Update gradient oracle
        gs_ = zeros(K+1, dim..., num_agents)
        gs_[1,:,:,:] = sum_delay_grad[1,:,:,:]
        for k in 1:K
            ds = weighted_average(gs_[k,:,:,:], weights)
            grad_cell[k,:,:,:] += ds
            @sync @distributed for i in 1:num_agents 
                gs_[k+1,:,:,i] = sum_delay_grad[k+1,:,:,i] - sum_delay_grad[k,:,:,i] + ds[:,:,i]
            end
        end
    end
    @info "Time taken: $((time()-t_start)/60 ) minutes"
    return rewards
end;
