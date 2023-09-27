#Random.seed!(1234);
function bold_mfw_ml(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R, K)
    loss_val = zeros(num_iters)
    st = time()
    grad_cell = zeros(max_delay+1, K, dim...)
    is_free = zeros(max_delay+1)
    for t = 1:num_iters
        dt, lb = data[t,:,:], label[t,:]
        xs = zeros(K+1, dim...)
        base_id = 1
        while is_free[base_id] != 0
            base_id += 1
        end
        tmp_grad = zeros(K, dim...)
        #@info "base_id: $(base_id)"
        is_free[base_id] = delay[t]
        for k in 1:K
            sigma = min(1, 1/(k+3))
            #tmp_grad[k,:,:] = gradient_enp(xs[k,:,:], dt, lb)
            injected_noise = rand(dim...) .- 0.5
            v = lmo(eta*grad_cell[base_id,k,:,:] + injected_noise, R)
            xs[k+1,:,:] = (1-sigma)*xs[k,:,:] + sigma*v
        end
        xt = xs[K+1,:,:]
        loss_val[t] = loss_enp(xt, dt, lb)
        if t%100 == 0
            @info "Loss $(t): $(loss_val[t])"
        end
        for k in 1:K
            grad_cell[base_id,k,:,:] += gradient_enp(xs[k,:,:], dt, lb)
        end
        for iner_id = 1:max_delay+1
            if is_free[iner_id] > 0
                is_free[iner_id] -= 1
            end
        end
    end
    @info "Time taken: $((time()-st)/60) minutes"
    return loss_val
end;

function delay_mfw_ml(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R, K)
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
            #assume that we know the gradient function to save computation
            gs_[t,k,:,:] = gradient_enp(xs[k,:,:],dt,lb)
            #update oracle
            injected_noise = rand(dim...) .- 0.5
            v = lmo(eta * grad_cell[k, :, :] + injected_noise, R) 
            xs[k+1,:,:] = (1-sigma)*xs[k,:,:] + sigma * v
        end
        xt = xs[K+1,:,:]
        #norm_x[t] = norm(xt,1)
        loss_val[t] = loss_enp(xt, dt, lb)
        if t%100 == 0
            @info "Loss $(t): $(loss_val[t])"
        end
        for iner_iter in max(1, t-max_delay):t
            if iner_iter + delay[iner_iter] - 1 == t
                for k in 1:K
                    grad_cell[k,:,:]+= gs_[iner_iter,k,:,:] 
                end
            end
        end
    end
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
    st = time()
    for t in 1:num_iters
        dt, lb = data[t,:,:], label[t,:]
        loss_val[t] = loss_enp(x,dt,lb) 
        if t%100==0
            @info "Loss $(t): $(loss_val[t])"
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
    @info "Time taken: $((time()-st)/60) minutes"
    return loss_val
end
