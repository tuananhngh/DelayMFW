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


function load_network(network_type="er", num_agents=50)
    ROOT = "./data/";
    filename = "$(ROOT)weights_$(network_type)_$(num_agents).mat";
    file = matopen(filename);
    weights = read(file, "weights");
    close(file);
    # find the first and second largest (in magnitude) eigenvalues
    dim = size(weights, 1);
    eigvalues = (LinearAlgebra.eigen(weights)).values;
    if abs(eigvalues[dim] - 1.0) > 1e-8
        error("the largest eigenvalue of the weight matrix is $(eigvalues[dim]), but it must be 1");
    end
    beta = max(abs(eigvalues[1]), abs(eigvalues[dim - 1]));
    if beta < 1e-8
        beta = 0.0;
    end
    return (weights, beta);
end