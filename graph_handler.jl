using LightGraphs
using LinearAlgebra
function generate_weighted_adjacency_matrix(graph::AbstractGraph)
    if is_connected(graph)
        num_nodes = nv(graph)
        adj_mat = Matrix(adjacency_matrix(graph)) / 1.0
        weight = zeros(Float64, num_nodes, num_nodes)
        #adjacency_matrix = zeros(Float64, num_nodes, num_nodes)

        # Calculate the degrees of each vertex
        degrees = [degree(graph, i) for i in 1:num_nodes]
        println(degrees)
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
        println("Graph is not connected")
    end
end

# Example usage:
# Create a graph using LightGraphs (replace with your own graph creation)
graph = erdos_renyi(2, 0.1)  # Example: Erdős-Rényi graph with 10 nodes and edge probability 0.2
#graph = complete_graph(10)
circle = cycle_graph(10)
adjacency_matrix(graph)
weight_gr = generate_weighted_adjacency_matrix(graph)
