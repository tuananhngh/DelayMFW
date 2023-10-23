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
using ArgParse


# Add worker processes
@everywhere using DistributedArrays

include("decentralized-algorithms-ml.jl")
include("data-handler.jl")
include("graph_handler.jl")
include("utils-decentralized.jl")
Random.seed!(1234);
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
    elseif type == "grid"
        graph = Grid([5,6]);
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
function run_main_exp(dim, graph_type, train_data, train_label, nb_agents, graph_gen, lmo, loss_function, gradient_function, num_iters, max_delay, radius, path="./result-decentralized/")
    # generate weight
    p = 0.3
    weight_list, spectral_gap_list = [], []
    for g_type in graph_type
        @info "Generating $(g_type) graph"
        _, g_weight, g_spect = graph_gen(g_type, p)
        push!(weight_list, (g_type,g_weight))
        push!(spectral_gap_list, (g_type,g_spect))
    end
    @info "Spectral Gap: $(spectral_gap_list)"
    # generate delays
    delays = ceil(Int,0.1*max_delay).*ones(Int,num_iters, nb_agents).-1 .+ rand(1:max_delay-ceil(Int,0.1*max_delay)+1,num_iters, nb_agents)
    eta = 1/sqrt(max_delay*num_iters)
    # Main loop
    for (idx, w) in enumerate(weight_list)
        g_type, g_weight = w
        @info "--------Running for $(g_type)--------"
        ddmfw = dec_delay_mfw3(dim, train_data, train_label, nb_agents, g_weight, lmo, loss_function, gradient_function, num_iters, eta, delays, max_delay, radius)
        save(path*"$(g_type).jld", Dict("ddmfw" => ddmfw));
    end
end 

function run_main_exp_select_delay(dim, graph_type, train_data, train_label, nb_agents, graph_gen, lmo, loss_function, gradient_function, num_iters, max_delay_total,  max_delay_agent, nb_agent_delay, radius, path="./result-decentralized/")
    # generate weight
    p = 0.3
    weight_list, spectral_gap_list = [], []
    for g_type in graph_type
        @info "Generating $(g_type) graph"
        _, g_weight, g_spect = graph_gen(g_type, p)
        push!(weight_list, (g_type,g_weight))
        push!(spectral_gap_list, (g_type,g_spect))
    end
    @info "Spectral Gap: $(spectral_gap_list)"
    # generate delays
    delays = ceil(Int,0.1*max_delay_total).*ones(Int,num_iters, nb_agents).-1 .+ rand(1:max_delay_total-ceil(Int,0.1*max_delay_total)+1,num_iters, nb_agents)
    idx_agent_delay = rand(1:nb_agents, nb_agent_delay)
    for i in 1:nb_agent_delay
        #delays[:,idx_agent_delay[i]] = ceil(Int,0.1*max_delay_agent).*ones(Int,num_iters, 1).-1 .+ rand(1:max_delay_agent-ceil(Int,0.1*max_delay_agent)+1,num_iters, 1)
        delays[:,idx_agent_delay[i]] = max_delay_agent .* ones(Int,num_iters, 1)
    end
    
    eta = 1/sqrt(max_delay*num_iters)
    # Main loop
    for (idx, w) in enumerate(weight_list)
        g_type, g_weight = w
        @info "--------Running for $(g_type)--------"
        ddmfw = dec_delay_mfw3(dim, train_data, train_label, nb_agents, g_weight, lmo, loss_function, gradient_function, num_iters, eta, delays, max_delay, radius)
        save(path*"select_$(nb_agent_delay)-$(max_delay_agent)-amaxdelay-$(g_type).jld", Dict("ddmfw" => ddmfw));
    end
end 


# Define command-line argument parser
s = ArgParseSettings()

# Add command-line arguments
@add_arg_table! s begin
    "--data_name"
    help = "Dataset name"
    arg_type = String
    default = "mnist"
    
    "--nb_agents"
    help = "Number of agents"
    arg_type = Int
    default = 25
    
    "--batch_size"
    help = "Batch size"
    arg_type = Int
    default = 4
    
    "--nb_classes"
    help = "Number of classes"
    arg_type = Int
    default = 10
    
    "--max_delay"
    help = "Maximum delay"
    arg_type = Int
    default = 1

    "--max_delay_agent"
    help = "Maximum delay for selected agent"
    arg_type = Int
    default = 10

    "--nb_agent_delay"
    help = "Number of agents with delay"
    arg_type = Int
    default = 25
    
    "--radius"
    help = "Radius"
    arg_type = Int
    default = 32

    "--runall"
    help = "Delay for all agent or only selected agent"
    arg_type = Bool
    default = false
end

# Parse command-line arguments
args = parse_args(s)


println(args)

# Access the parsed arguments
data_name = args["data_name"]
nb_agents = args["nb_agents"]
batch_size = args["batch_size"]
nb_classes = args["nb_classes"]
max_delay = args["max_delay"]
max_delay_agent = args["max_delay_agent"]
nb_agent_delay = args["nb_agent_delay"]
radius = args["radius"]
runall = args["runall"]


if data_name == "mnist"
    train_x, train_y = MNIST.traindata(Float64);
elseif data_name == "fashionmnist"
    train_x, train_y = FashionMNIST.traindata(Float64);
elseif data_name == "cifar10"
    train_x, train_y = CIFAR10.traindata(Float64);
end;


flat_dim, train_data, train_label = data_processing(train_x, train_y, nb_agents, batch_size);
dim = (flat_dim, nb_classes)
num_iters, batch_size, _, num_agents = size(train_data)
@info "Number of agents: $(num_agents), Batch size: $(batch_size)"
@info "Weight_Dim : $(dim), Feature: $(flat_dim), , Data size:  $(size(train_data)), Label Size: $(size(train_label))"


graph_list = ["er","complete","grid", "cycle"]
if !isdir("../result-decentralized-ml2/$(data_name)")
    @info "Creating Directory $(data_name)"
    mkdir("../result-decentralized-ml2/$(data_name)");
end;

path = "../result-decentralized-ml2/$(data_name)/$(num_iters)-$(max_delay)-$(nb_agents)-$(radius)-staticdelay-"
# Run the main experiment
if runall
    @info "--------Delay for all agents--------"
    run_main_exp(dim, graph_list, train_data, train_label, nb_agents, generate_graph, lmo_2dim_fn, loss_fn, grad_fn, num_iters, max_delay, radius, path)
else
    @info "------DELAY FOR RANDOMLY SELECTED $(nb_agent_delay) AGENTS------"
    run_main_exp_select_delay(dim, graph_list, train_data, train_label, nb_agents, generate_graph, lmo_2dim_fn, loss_fn, grad_fn, num_iters, max_delay, max_delay_agent, nb_agent_delay, radius, path)
end

#rm("./result-decentralized-ml/$(data_name)",recursive=true)
