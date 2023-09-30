#Random.seed!(1234);

# Function to compute weighted average
@everywhere function weighted_average(x, w)
    dim1, dim2, nb_agent = size(x)
    
    # Reshape x to a 2D array (dim1*dim2, nb_agent)
    x_reshaped = reshape(x, dim1*dim2, nb_agent)
    
    # Compute the result using dot and broadcast
    result = x_reshaped * w    
    #result = zeros(dim1, dim2, nb_agent)
    # Reshape the result back to the original shape
    #@sync @distributed for i in 1:nb_agent
    #      result[:,:,i] = sum([x[:,:,j] * w[i,j] for j in 1:nb_agent])
    #end
    return reshape(result, dim1, dim2, nb_agent)
    #return result
end

# Function to compute LMO in parallel
function distribute_lmo(grad, radius, lmo_function)
    storage = zeros(size(grad))
    nb_agents = size(grad, 3)
    @sync @distributed for i in 1:nb_agents
        storage[:,:,i] = lmo_function(grad[:,:,i], radius)
    end
    return storage
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
        # if t%10==0
        #     @info "Iteration $(t), Loss $(rewards[t])"
        # end
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
                #@info "size of gs_ $(size(gs_[fb,1,:,:,agent_i]))]))"
                #@info "size of tmp_grad $(size(tmp_grad))"
                gs_[fb,1,:,:,agent_i] += tmp_grad
            # else 
            #     @warn "From add_delay_gradient1 : Feedback time $(fb) of time $(time_t) of agent $(agent_i) is greater than iterations $(num_iters)"
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
            # else 
            #     @warn "From add_delay_gradient2 : Feedback time $(fb) of time $(time_t) of agent $(agent_i) is greater than iterations $(num_iters)"
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


