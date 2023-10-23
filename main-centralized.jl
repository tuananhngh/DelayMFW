using LinearAlgebra
using ProximalOperators
using JLD
using Random
using Plots
using FileIO

include("oracles.jl");
include("params-centralized.jl");
include("centralized-algorithms.jl")

data = load("./data_cell.jld")["centralized"];
#max_delay = [1, 51, 101, 201, 301, 401, 501, 601];
max_delay = [1];
seed_value = 1234
Random.seed!(seed_value)

function setting(dim, data_cell, projection, lmo, num_iters, R, D, G, max_delay)
    for md in max_delay
        delay = ceil(Int,0.1*md).*ones(Int,num_iters).-1 .+ rand(1:md-ceil(Int,0.1*md)+1,num_iters);
        M = D*G/md + D^2*G + D*G^2;
        eta = 1/sqrt(md*num_iters) # lr of delayed O-PGD, small learning does not converge
        zeta = 1/sqrt(md*num_iters); #lr of projection
        eta_dofw = D/(sqrt(2)*G*(num_iters+2)^(3/4)); #lr of d-ofw
        eta_bofw = D/(sqrt(2)*G*(num_iters/md+2)^(3/4)); #lr of bold-ofw
        println("------Running DMFW------")
        dmfw = delay_mfw(dim, data_cell, projection, num_iters, md, delay, zeta, R)
        println("------Running DMFW 2------")
        dmfw2 = delay_mfw2(dim, data_cell, lmo, num_iters, delay, eta, R)
        println("------Running DOFW------")
        dofw = d_ofw(dim, num_iters, lmo, md, eta_dofw, R, data_cell, delay)
        println("------Running Bold-MFW------")
        bmfw = bold_mfw(dim, num_iters, lmo, md, eta, R, data_cell, delay) 
        println("------Running Bold-OFW------")
        bofw = bold_ofw(dim, num_iters, lmo, md, eta_bofw, R, data_cell, delay)

        save("./result-centralized-staticlr/$(md)-delay.jld", Dict("dmfw" => dmfw, "dmfw2" => dmfw2,"dofw"=>dofw,"bmfw"=>bmfw,"bofw"=>bofw));
    end
end

function plot_result(path, num_iters)
    list_result_files = readdir(path)
    algos = ["Delay-MFW", "Delay-MFW2","Delay-OFW","Bold-OFW","Bold-MFW"]
    for ele in list_result_files
        a = load(joinpath(path,ele))
        md = split(ele, "-")[1]
        dmfw = a["dmfw"]
        dmfw2 = a["dmfw2"]
        #bmfw = a["bmfw"]
        bofw = a["bofw"]
        dofw = a["dofw"]
        compare_reg = plot(1:num_iters, [cumsum(dmfw) cumsum(dmfw2) cumsum(dofw) cumsum(bofw)], 
                    label=["Delay-mfw" "Delay-mfw2" "Delay-ofw" "Bold-ofw"], 
                    xlabel="Iterations t - Max Delay $(md)",
                    ylabel="Cumulative Loss",
                    legend=:topleft)
        display(compare_reg)
    end
end

setting(dim, data, projection_l1, lmo_fn, T, R, D, G_norm, max_delay)
plot_result("./result-centralized-staticlr", T)

#delete directory
rm("./result-centralized-staticlr", recursive=true)