#Random.seed!(1234);

# Function to compute weighted average
function weighted_average(x, w)
    dim1, dim2, nb_agent = size(x)
    # Reshape x to a 2D array (dim1*dim2, nb_agent)
    x_reshaped = reshape(x, dim1*dim2, nb_agent)
    result = x_reshaped * w    
    return reshape(result, (dim1, dim2, nb_agent))
end

# Function to compute LMO in parallel
function distribute_lmo(grad, radius, lmo_function)
    dim1, dim2, nb_agents = size(grad)
    storage = @sync @distributed (hcat) for i in 1:nb_agents
        lmo_function(grad[:,:,i], radius)
    end
    return reshape(storage, (dim1, dim2, nb_agents))
end


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
        tmp_grad[K+1,:,:,:] = gradient_cat(xs,dt,lb)
        xt = copy(xs)
        tmp_rw = zeros(num_agents)
        # Compute loss
        rw = 0
        for i in 1:num_agents
            tmp_rw[i] += f_sum(xt[:,:,i],dt, lb)/num_agents;
        end
        rewards[t] = maximum(tmp_rw);

        # Push the gradient at corresponding time for delay feedback
        @sync @distributed for i in 1:num_agents
            push!(gradient_hold[i], (t,tmp_grad[:,:,:,i]))
        end

        # Find delay gradient for feedback at current time
        sum_delay_grad = zeros(K+1, dim..., num_agents)
        @sync @distributed for i in 1:num_agents
            for (idx_s,s) in enumerate(gradient_hold[i])
                if s[1] + delay[s[1],i] - 1 == t
                    sum_delay_grad[:,:,:,i] += s[2]
                    deleteat!(gradient_hold[i], idx_s)
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
