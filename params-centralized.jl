using Random

mutable struct ProjectionOracle
    x
    zeta
end;

mutable struct FPL
    x
    eta
    accumulated_gradient
    n0
    R
end


#Random.seed!(1234);
T = 500; #iterations
dim = 100; #dimension
R = 20; #radius of constraint set K
D = 2*R; #diameter of K
max_delay = 201;
num_agents = 50;
G_norm = 2*R+sqrt(dim);
#K = Int(ceil(sqrt(T)));
K = T;


