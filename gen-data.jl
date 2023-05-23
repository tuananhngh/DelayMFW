using Random
using JLD
include("params.jl");

Random.seed!(1234);
data_cell_centralized = zeros(T, dim);
data_cell_decentralized = zeros(T, dim, num_agents);
for t in 1:T
    data_cell_centralized[t,:] = 2*rand(dim).-1;
    data_cell_decentralized[t,:,:] = 2*rand(dim,num_agents).-1;
end;

save("data_cell.jld", Dict("centralized"=>data_cell_centralized, "decentralized"=>data_cell_decentralized));



