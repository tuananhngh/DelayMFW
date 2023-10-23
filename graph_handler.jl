using LightGraphs
using LinearAlgebra

function generate_weighted_adjacency_matrix(graph::AbstractGraph)
    if  is_connected(graph)
        num_nodes = nv(graph)
        adj_mat = Matrix(adjacency_matrix(graph)) / 1.0
        weight = zeros(Float64, num_nodes, num_nodes)
        #adjacency_matrix = zeros(Float64, num_nodes, num_nodes)

        # Calculate the degrees of each vertex
        degrees = [degree(graph, i) for i in 1:num_nodes]
        # Populate the adjacency matrix with weights based on your formula
        for i in 1:num_nodes
            for j in 1:num_nodes
                if adj_mat[i, j] == 0
                    continue
                end
                weight[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            end
        end
        sum_weight = vec(sum(weight, dims=2))
        weight = weight + diagm(1 .- sum_weight)
        return weight
    else
        @error("Graph is not connected, Retry !")
        Base.exit()
    end
end

function compute_spectral_gap(weight_matrix)
    eig_vals = eigen(weight_matrix).values
    sorted = sort(eig_vals, rev=true)
    spec = sorted[1] - sorted[2]
    return spec
end

# Example usage:
# Create a graph using LightGraphs (replace with your own graph creation)
# graph = erdos_renyi(10, 0.5);  # Example: Erdős-Rényi graph with 10 nodes and edge probability 0.2
# #graph = complete_graph(10)
# circle = cycle_graph(10);
# cmp = complete_graph(25);
# star = star_graph(10);
# weight_gr = generate_weighted_adjacency_matrix(graph);
# weight_cmp = generate_weighted_adjacency_matrix(cmp)
# weight_cycle = generate_weighted_adjacency_matrix(circle);
# weight_star = generate_weighted_adjacency_matrix(star);
#compute_spectral_gap(weight_star)

