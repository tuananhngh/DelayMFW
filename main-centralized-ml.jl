using MLDatasets
using LinearAlgebra
using Logging
using Plots
using Random
using Flux
using JLD
using FileIO
using SparseArrays

include("centralized-algorithms-ml.jl")
Random.seed!(1234);
train_x, train_y = MNIST.traindata(Float32);

function data_processing(train_x, train_y, nb_agents, batch_size)
    data_shape = size(train_x)
    @info "Data Original Size $(data_shape)"
    features = prod(data_shape[1:end-1])
    chunk_size = Int(data_shape[end]/nb_agents)
    train_x = reshape(train_x, (features, data_shape[end]))
    shuffle_idx = randperm(data_shape[end])
    train_x, train_y = train_x[:,shuffle_idx], train_y[shuffle_idx]
    @info "Data Shuffled"
    agents_data = Array{Float32}(undef, features, chunk_size, nb_agents)
    agents_label = Array{Int32}(undef, chunk_size, nb_agents)
    for i in 1:nb_agents
        agents_data[:,:,i] = train_x[:,(i-1)*chunk_size+1:i*chunk_size]
        agents_label[:,i] = train_y[(i-1)*chunk_size+1:i*chunk_size]
    end
    nb_batches = Int(chunk_size/batch_size)
    train_data = Array{Float32}(undef, features, batch_size, nb_batches, nb_agents)
    train_label = Array{Int32}(undef, batch_size, nb_batches, nb_agents)
    for t in 1:nb_batches
        train_data[:,:,t,:] = agents_data[:,(t-1)*batch_size+1:t*batch_size,:]
        train_label[:,t,:] = agents_label[(t-1)*batch_size+1:t*batch_size,:] .+ 1
    end
    train_data = permutedims(train_data, [3,2,1,4])
    train_label = permutedims(train_label, [2,1,3])
    return features, train_data, train_label
end

#data size = (num_iters, batch_size, features, nb_agents)
nb_classes = 10
batch_size = 60
flat_dim, data_cell, label_cell = data_processing(train_x, train_y, 1, batch_size)
dim = (flat_dim, nb_classes)
data_cell = dropdims(data_cell, dims=length(size(data_cell)))
label_cell = dropdims(label_cell, dims=length(size(label_cell)))
num_iters, batch_size, _ = size(data_cell)
@info "Feature: $(dim), Data size:  $(size(data_cell)), Label Size: $(size(label_cell))"

function loss_fn(weights, data, label)
    prediction = Flux.softmax(weights'*data')
    labels = Flux.onehotbatch(label, 1:10)
    return Flux.crossentropy(prediction, labels)
end


function grad_fn(weights, data, labels)
    return Flux.gradient(loss_fn, weights, data, labels)[1]
end


function lmo_2dim_fn(v, s=1)
    dim = size(v)
    row, col = size(v)
    max_idx = argmax(abs.(v), dims=1)
    sol = spzeros(dim)
    for i in 1:col
        sol[max_idx[i][1],i] = -s * sign.(v[max_idx[i][1],i])
    end
    return sol
end

function setting(dim, data_x, data_y, loss_fn, gradient_fn, lmo, num_iters, R, D, max_delay)
    K = 50
    @info "Running for $(num_iters) Iterations and $(K) Subiterations"
    for md in max_delay
        @info "Running for Max Delay $(md)"
        delay = ceil(Int,0.1*md).*ones(Int,num_iters).-1 .+ rand(1:md-ceil(Int,0.1*md)+1,num_iters);
        eta = 1/((md*num_iters)^(1/2)) # lr of delayed O-PGD, small learning does not converge
        zeta = 1/sqrt(md*num_iters); #lr of projection
        eta_dofw = 1/(sqrt(2)*(num_iters+2)^(3/4)); #lr of d-ofw
        @info "------Running DMFW------"
        dmfw = delay_mfw_ml(dim, data_x, data_y, loss_fn, gradient_fn, lmo, num_iters, md, delay, eta, R, K)
        @info "------Running Bold-MFW------"
        bmfw = bold_mfw_ml(dim, data_x, data_y, loss_fn, gradient_fn, lmo, num_iters, md, delay, eta, R, K)
        @info "------Running DOFW------"
        dofw = d_ofw_ml(dim, data_x, data_y, loss_fn, gradient_fn, lmo, num_iters, md, delay, eta, R)
        save("./result-centralized-ml/$(md)-delay.jld", Dict( "dmfw" => dmfw,"bmfw"=>bmfw,"dofw"=>dofw))
    end
    #return dmfw, dofw
end

function plot_loss(path, num_iters)
    list_result_files = readdir(path)
    algos = ["Delay-MFW", "Delay-MFW2","Delay-OFW","Bold-OFW","Bold-MFW"]
    for ele in list_result_files
        a = load(joinpath(path,ele))
        md = split(ele, "-")[1]
        dmfw = a["dmfw"]
        bmfw = a["bmfw"]
        dofw = a["dofw"]
        compare_reg = plot(1:num_iters, [dmfw bmfw dofw], label=["Delay-MFW" "Bold-MFW" "Delay-OFW"], xlabel="Iterations t - Max Delay $(md)", ylabel="Cumulative Loss", legend=:topleft)
        display(compare_reg)
    end
end

function plot_result(path, num_iters)
    list_result_files = readdir(path)
    algos = ["Delay-MFW","MFW","Delay-MFW2","Delay-OFW","Bold-OFW","Bold-MFW"]
    for ele in list_result_files
        a = load(joinpath(path,ele))
        md = split(ele, "-")[1]  
        dmfw = a["dmfw"]
        bmfw = a["bmfw"]
        dofw = a["dofw"]
        compare_reg = plot(1:num_iters, [cumsum(dmfw) cumsum(bmfw) cumsum(dofw)], 
                    label=["Delay-mfw" "Bold-mfw" "Delay-ofw"], 
                    xlabel="Iterations t - Max Delay $(md)",
                    ylabel="Cumulative Loss",
                    legend=:topleft)
        
        display(compare_reg)
    end
end

R = 8
D = 2*R
max_delay = [1, 11, 21, 31, 41, 51]
setting(dim, data_cell, label_cell, loss_fn, grad_fn, lmo_2dim_fn, num_iters, R, D, max_delay)
plot_loss("./result-centralized-ml", num_iters)
plot_result("./result-centralized-ml", num_iters)

#a = load(joinpath("./result-centralized-ml/","1-delay.jld"))

#rm("./result-centralized-ml", recursive=true)