using MLDatasets
using LinearAlgebra
using Statistics
using Distributed, DistributedArrays
using Plots
using Random
using Flux
using JLD
include("oracles.jl")
include("decentralized-algorithms.jl")
include("utils-decentralized.jl")
Random.seed!(1234);
train_x, train_y = FashionMNIST.traindata(Float32);
#train_x, train_y = train_x[:,:,:,1:70000], train_y[1:70000];


nb_agents = 10;
batch_size = 6;


function data_processing(train_x, train_y, nb_agents, batch_size)
    data_shape = size(train_x)
    println("Data Size $(data_shape)")
    features = prod(data_shape[1:end-1])
    chunk_size = Int(data_shape[end]/nb_agents)
    train_x = reshape(train_x, (features, data_shape[end]))
    println("Shuffle Data")
    shuffle_idx = randperm(data_shape[end])
    train_x, train_y = train_x[:,shuffle_idx], train_y[shuffle_idx]
    println("Data Shuffled")
    println("Data Flat $(size(train_x))")
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
    println("Data reshape $(size(train_data))")
    println("Label reshape $(size(train_label))")
    
    return features, train_data, train_label
end

flat_dim, train_data, train_label = data_processing(train_x, train_y, nb_agents, batch_size);

mutable struct ProjectionOracle
    x
    zeta
end;

@everywhere function loss(weights, data, label)
    prediction = Flux.softmax(weights'*data')
    labels = Flux.onehotbatch(label, 1:10)
    return Flux.crossentropy(prediction, labels)
end

@everywhere function grad(weights, data, labels)
    return Flux.gradient(loss, weights, data, labels)[1]
end

max_delay = 11;
num_iters = size(train_data,1)
radius = 8; # or 50 ,radius of constraint set K
zeta = 1/sqrt(max_delay*num_iters)
delays = ceil(Int,0.1*max_delay).*ones(Int,num_iters, nb_agents).-1 .+ rand(1:max_delay-ceil(Int,0.1*max_delay)+1,num_iters, nb_agents);
w_er,_ = load_network("er", nb_agents);
dim = (flat_dim,10)

#ER
println("-----Running DDSGD on ER-----")
#ddsgd_er = dec_delay_sgd(dim, train_data,train_label,nb_agents,w_er, projection_l1,loss, grad, num_iters, max_delay, radius)
println("-----Running DDMFW on ER-----")
#ddmfw_er = dec_delay_mfw(dim, train_data, train_label, nb_agents, w_er, projection_l1, loss, grad, num_iters, zeta, delays, max_delay, radius)
println("-----Running DDMFW2 on ER-----")
ddmfw_er2 = dec_delay_mfw2(dim, train_data, train_label, nb_agents, w_er, lmo_fn_dec, loss, grad, num_iters, zeta, delays, radius)
#save("./result-decentralized/fashionmnist/$(num_iters)-$(max_delay)-staticlr-comp.jld", Dict("ddmfw" => ddmfw_er, "ddmfw2" => ddmfw_er2, "ddsgd" => cifar_ddsgd_er));


