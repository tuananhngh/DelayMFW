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
max_delay = [51, 101, 201, 301, 401, 501, 601];

function setting(dim, data_cell, projection, num_iters, R, D, G, max_delay)
    for md in max_delay
        delay = ceil(Int,0.1*md).*ones(Int,num_iters).-1 .+ rand(1:md-ceil(Int,0.1*md)+1,num_iters);
        zeta = 1/(sqrt(md*num_iters)); # lr of delayed O-PGD, small learning does not converge
        eta_dofw = D/(sqrt(2)*G*(num_iters+2)^(3/4)); #lr of d-ofw
        eta_bofw = D/(sqrt(2)*G*(num_iters/md+2)^(3/4)); #lr of bold-ofw
        println("------Running Algorithms------")
        dmfw = delay_mfw(dim, data_cell, projection, num_iters, md, delay, zeta, R)
        dofw = d_ofw(dim, num_iters, md, eta_dofw, R, data_cell, delay)
        bofw = bold_ofw(dim, num_iters, md, eta_bofw, R, data_cell, delay)

        save("./result-centralized-staticlr/$(md)-delay.jld", Dict("dmfw" => dmfw,"dofw"=>dofw,"bofw"=>bofw));
    end
end

function plot_result(path, num_iters)
    list_result_files = readdir(path)
    algos = ["Delay-MFW","Delay-OFW", "Bold-OFW"]
    for ele in list_result_files
        a = load(joinpath(path,ele))
        md = split(ele, "-")[1]
        dmfw = a["dmfw"]
        bofw = a["bofw"]
        dofw = a["dofw"]
        compare_reg = plot(1:num_iters, [cumsum(dmfw) cumsum(dofw) cumsum(bofw)], 
                    label=["Delay-mfw" "Delay-ofw" "Bold-ofw"], 
                    xlabel="Iterations t - Max Delay $(md)",
                    ylabel="Cumulative Loss",
                    legend=:topright)
        display(compare_reg)
    end
end


setting(dim, data, projection_l1, T, R, D, G_norm, max_delay)
