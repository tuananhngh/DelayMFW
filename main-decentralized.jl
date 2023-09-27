using MLDatasets
using LinearAlgebra
using Statistics
using Distributed, DistributedArrays
using Plots
using Random
using JLD
using Logging
using Flux

@everywhere using SparseArrays
# Add worker processes
#addprocs(4)

include("decentralized-algorithms.jl")
include("oracles.jl")
#include("utils-decentralized.jl")
include("data-handler.jl")

Random.seed!(1234);
train_x, train_y = FashionMNIST.traindata(Float32);


nb_agents = 10;
batch_size = 6;
nb_classes = 10
flat_dim, train_data, train_label = data_processing(train_x, train_y, nb_agents, batch_size);
dim = (flat_dim, nb_classes)
num_iters, batch_size, _, num_agents = size(train_data)
@info "Number of agents: $(num_agents), Batch size: $(batch_size)"
@info "Weight_Dim : $(dim), Feature: $(flat_dim), , Data size:  $(size(train_data)), Label Size: $(size(train_label))"

mutable struct ProjectionOracle
    x
    zeta
end;


@everywhere function loss_fn(weights, data, label)
    prediction = Flux.softmax(weights'*data')
    labels = Flux.onehotbatch(label, 1:10)
    return Flux.crossentropy(prediction, labels)
end


@everywhere function grad_fn(weights, data, labels)
    return Flux.gradient(loss_fn, weights, data, labels)[1]
end

@everywhere function lmo_fn_dec(V, radius)
    num_rows, num_cols = size(V)
    v = spzeros(size(V))
    # Find row indices corresponding to the maximum absolute values in each column
    idx = argmax(abs.(V), dims=1)
    for col in 1:num_cols
        row = idx[col][1]
        v[row, col] = -radius * sign(V[row, col])
    end
    return v
end

@everywhere function lmo_2dim_fn(v, s=1)
    dim = size(v)
    row, col = size(v)
    max_idx = argmax(abs.(v), dims=1)
    sol = spzeros(dim)
    for i in 1:col
        sol[max_idx[i][1],i] = -s * sign.(v[max_idx[i][1],i])
    end
    return sol
end


max_delay = 1;
num_iters = size(train_data,1)
radius = 8; # or 50 ,radius of constraint set K
zeta = 1/sqrt(max_delay*num_iters)
delays = ceil(Int,0.1*max_delay).*ones(Int,num_iters, nb_agents).-1 .+ rand(1:max_delay-ceil(Int,0.1*max_delay)+1,num_iters, nb_agents);
w_er,_ = load_network("er", nb_agents);


#ER
@info "Iterations: $(num_iters), Max Delay: $(max_delay), Radius: $(radius), Zeta: $(zeta)"
@info "-----Running DDMFW2 on ER-----"
ddmfw_er2 = dec_delay_mfw2(dim, train_data, train_label, nb_agents, w_er, lmo_2dim_fn, loss_fn, grad_fn, num_iters, zeta, delays, radius)
@info "-----Running DDSGD on ER-----"
ddsgd_er = dec_delay_sgd(dim, train_data,train_label,nb_agents,w_er, projection_l1,loss_fn, grad_fn, num_iters, max_delay, radius)
#println("-----Running DDMFW on ER-----")
#ddmfw_er = dec_delay_mfw(dim, train_data, train_label, nb_agents, w_er, projection_l1, loss, grad, num_iters, zeta, delays, max_delay, radius)
save("./result-decentralized/fashionmnist/$(num_iters)-$(max_delay)-staticlr-comp.jld", Dict("ddmfw2" => ddmfw_er2, "ddsgd" => ddsgd_er));

#function to plot result from path
function plot_result(path, title)
    result = load(path)
    ddmfw2 = result["ddmfw2"]
    ddsgd = result["ddsgd"]
    #ddmfw2 = ddmfw2[1:1000]
    #ddsgd = ddsgd[1:1000]
    plot(1:num_iters, [ddmfw2 ddsgd], label=["DD-MFW" "DDSGD"], title=title, xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Loss", linewidth=1)
    #savefig("./result-decentralized/fashionmnist/$(title).png")
end

#function to plot result from path
function plot_reg(path, title)
    result = load(path)
    ddmfw2 = result["ddmfw2"]
    ddsgd = result["ddsgd"]
    #ddmfw2 = ddmfw2[1:1000]
    #ddsgd = ddsgd[1:1000]
    plot(1:num_iters, [cumsum(ddmfw2) cumsum(ddsgd)], label=["DD-MFW" "DDSGD"], title=title, xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Loss", linewidth=1, legend=:topleft)
end


plot_result("./result-decentralized/fashionmnist/1000-11-staticlr-comp.jld", "FashionMNIST")
plot_reg("./result-decentralized/fashionmnist/1000-11-staticlr-comp.jld", "FashionMNIST")