using MLDatasets
using LinearAlgebra
using Statistics
using Distributed
using Plots
using Random
using JLD
using Logging
using Flux
using LightGraphs
using ProgressMeter
using Profile
using SparseArrays

# Add worker processes
#addprocs(4)
#@everywhere using DistributedArrays

include("decentralized-algorithms-ml.jl")
include("data-handler.jl")
include("graph_handler.jl")

# Load network
function generate_graph(type="er", pl=0.4)
    if type == "er"
        graph = erdos_renyi(nb_agents, pl);
    elseif type == "complete"
        graph = complete_graph(nb_agents);
    elseif type == "cycle"
        graph = cycle_graph(nb_agents);
    elseif type == "star"
        graph = star_graph(nb_agents);
    end
    weight = generate_weighted_adjacency_matrix(graph)
    spectral_gap = compute_spectral_gap(weight)
    return graph, weight, spectral_gap
end


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
    sol = zeros(dim...)
    for i in 1:col
        sol[max_idx[i][1],i] = -s * sign.(v[max_idx[i][1],i])
    end
    return sol
end


# Main function to run the algorithm on different network
function run_main_exp(dim, graph_type, train_data, train_label, nb_agents, graph_gen, lmo, loss_function, gradient_function, num_iters, max_delay, radius, K, path="./result-decentralized/")
    # generate weight
    p = 0.5
    weight_list, spectral_gap_list = [], []
    for g_type in graph_type
        _, g_weight, g_spect = graph_gen(g_type, p)
        push!(weight_list, g_weight)
        push!(spectral_gap_list, g_spect)
    end
    @info "Spectral Gap: $(spectral_gap_list)"
    # generate delays
    delays = ceil(Int,0.1*max_delay).*ones(Int,num_iters, nb_agents).-1 .+ rand(1:max_delay-ceil(Int,0.1*max_delay)+1,num_iters, nb_agents)
    eta = 1/sqrt(max_delay*num_iters)
    # Main loop
    for (idx, w) in enumerate(weight_list)
        @info "--------Running for $(graph_type[idx])--------"
        ddmfw = dec_delay_mfw2(dim, train_data, train_label, nb_agents, w, lmo, loss_function, gradient_function, num_iters, eta, delays, radius, K)
        save(path*"$(graph_type[idx]).jld", Dict("ddmfw" => ddmfw));
    end
end 


# Function to plot result from path
function plot_loss(path, graph_type)
    p = plot(title="FashionMNIST", xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Loss", linewidth=1)
    for g in graph_type
        result = load(path*"$(g).jld")
        ddmfw = result["ddmfw"]
        plot!(1:num_iters, ddmfw, label=g, title="FashionMNIST", xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Loss", linewidth=1, legend=:topleft)
    end
    display(p)
end

function plot_cumsum(path, graph_type)
    p = plot(title="FashionMNIST", xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Cumulative Loss", linewidth=1)
    for g in graph_type
        result = load(path*"$(g).jld")
        ddmfw = cumsum(result["ddmfw"])
        plot!(1:num_iters, ddmfw, label=g, title="FashionMNIST", xlabel="Iterations MaxDelay Value $(max_delay)", ylabel="Cumulative Loss", linewidth=1, legend=:topleft)
    end
    display(p)
end

Random.seed!(1234);

# Parameters
train_x, train_y = MNIST.traindata(Float32);
data_name = "mnist";

nb_agents = 50;
batch_size = 12;
nb_classes = 10
K = 10
flat_dim, train_data, train_label = data_processing(train_x, train_y, nb_agents, batch_size);
dim = (flat_dim, nb_classes)
num_iters, batch_size, _, num_agents = size(train_data)
@info "Number of agents: $(num_agents), Batch size: $(batch_size)"
@info "Weight_Dim : $(dim), Feature: $(flat_dim), , Data size:  $(size(train_data)), Label Size: $(size(train_label))"

max_delay = 1;
radius = 8;
graph_list = ["er", "complete", "cycle"] #, "star"]
path = "./result-decentralized/$(data_name)/$(num_iters)-$(K)-$(max_delay)-$(nb_agents)-"
# Run the main experiment
@profile run_main_exp(dim, graph_list, train_data, train_label, nb_agents, generate_graph, lmo_2dim_fn, loss_fn, grad_fn, num_iters, max_delay, radius, K, path)
plot_loss(path, graph_list)
plot_cumsum(path, graph_list)

