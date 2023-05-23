using FrankWolfe
using MLDatasets
using Flux
using Plots
using JLD
include("oracles.jl")

Random.seed!(1234);
train_x, train_y = SVHN2.traindata(Float32);
train_x, train_y = train_x[:,:,:,1:70000], train_y[1:70000].-1;

batch_size = 70

function data_processing(train_x, train_y, batch_size)
    data_shape = size(train_x)
    println("Data Size $(data_shape)")
    features = prod(data_shape[1:end-1])
    train_x = reshape(train_x, (features, data_shape[end]))
    println("Shuffle Data")
    shuffle_idx = randperm(data_shape[end])
    train_x, train_y = train_x[:,shuffle_idx], train_y[shuffle_idx]
    println("Data Shuffled")
    println("Data Flat $(size(train_x))")

    nb_batches = Int(data_shape[end]/batch_size)
    train_data = Array{Float32}(undef, features, batch_size, nb_batches)
    train_label = Array{Int32}(undef, batch_size, nb_batches)
    for t in 1:nb_batches
        train_data[:,:,t,] = train_x[:,(t-1)*batch_size+1:t*batch_size]
        train_label[:,t] = train_y[(t-1)*batch_size+1:t*batch_size] .+ 1
    end
    train_data = permutedims(train_data, [3,2,1])
    train_label = permutedims(train_label, [2,1])
    println("Data reshape $(size(train_data))")
    println("Label reshape $(size(train_label))")
    
    return features, train_data, train_label
end

flat_dim, train_data, train_label = data_processing(train_x, train_y,batch_size);

function loss(weights, data, label)
    prediction = Flux.softmax(weights'*data')
    labels = Flux.onehotbatch(label, 1:10)
    return Flux.crossentropy(prediction, labels)
end

function grad(weights, data, labels)
    return Flux.gradient(loss, weights, data, labels)[1]
end


function frank_wolfe(dim, gradient, lmo, K)
    x = zeros(dim...)
    for k in 1:K
        println("Iteration $(k)")
        grad_x = gradient(x)
        v = collect(FrankWolfe.compute_extreme_point(lmo, grad_x))
        v = reshape(v, (dim...))
        eta = 2/(k+2)
        x = (1-eta)*x + eta*v
    end
    return x
end

function offline(dim, train_data, train_label, f, gradient, lmo, K)
    T = size(train_data,1);
    reward = zeros(T);
    gradient_cumul = x->sum([gradient(x, train_data[t,:,:], train_label[t,:]) for t in 1:T]);
    #v = projected_gradient_descent(dim, gradient_cumul, K, projection, radius);
    v = frank_wolfe(dim, gradient_cumul, lmo, K)
    for i in 1:T
        reward[i] = f(v,train_data[i,:,:], train_label[i,:]);
    end
    return reward;
end

dim = (flat_dim, 10)
K = 100
radius = 32
lmo = FrankWolfe.LpNormLMO{Float64,1}(radius)
@time loss_off = offline(dim, train_data, train_label, loss, grad, lmo, K)
save("./result-decentralized/svhn2/$(num_iters)-optimal.jld", Dict("opt" => loss_off));

