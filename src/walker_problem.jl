import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
using LinearAlgebra
using PyPlot
using ForwardDiff
using RobotZoo
using RobotDynamics
using Ipopt
using MathOptInterface
using Random
using Blink
using SparseArrays
const MOI = MathOptInterface;
include("quadratic_cost.jl")
include("walker.jl")
include("utils.jl")

vis = Visualizer()
set_mesh!(vis, SimpleWalker())
open(vis, Blink.Window())

##
model = SimpleWalker()
Nx = 12         # number of state
Nu = 3          # number of controls
Tfinal = 4.4    # final time
h = 0.1         # 10 hz
Nm = 5          # number of steps in each mode
Nt = Int(ceil(Tfinal/h)+1)   # number of time steps
Nmodes = Int(ceil(Nt/Nm))
thist = Array(range(0,h*(Nt-1), step=h));
n_nlp = Nx*Nt + Nu*(Nt-1) # number of decision variables
c_init_inds = 1:Nx                                                   # initial constraint
c_term_inds = (c_init_inds[end]+1):(c_init_inds[end]+Nx)             # terminal constraint
c_dyn_inds = (c_term_inds[end]+1):(c_term_inds[end]+Nx*(Nt-1))       # dynamics constraints
c_stance_inds = (c_dyn_inds[end]+1):(c_dyn_inds[end]+Nt)             # stance constraint (1 per time step)
c_length_inds = (c_stance_inds[end]+1):(c_stance_inds[end]+(2*Nt))   # length bounds     (2 per time step)
m_nlp = c_length_inds[end]                                           # total constraints
cinds = [c_init_inds, c_term_inds, c_dyn_inds, c_stance_inds, c_length_inds]

## Objective

# Cost weights
Q = Diagonal([1.0*ones(6); 1.0*ones(6)]);
Q = Diagonal([1; 10; fill(1.0, 4); 1; 10; fill(1.0, 4)]);
R = 0.001;
R = Diagonal(fill(1e-3,3))
Qn = Q;

## Reference Trajectory
uref = [0.5*mb*g; 0.5*mb*g; 0.0]
xref = zeros(Nx,Nt)
xref[1,:] .= LinRange(-1.5,1.5,Nt)
xref[2,:] .= ones(Nt)
xref[3,:] .= LinRange(-1.5,1.5,Nt)
xref[5,:] .= LinRange(-1.5,1.5,Nt)
xref[7,2:end-1] .= (3.0/Tfinal)*ones(Nt-2)
xref[9,2:end-1] .= (3.0/Tfinal)*ones(Nt-2)
xref[11,2:end-1] .= (3.0/Tfinal)*ones(Nt-2);

## Solve
Random.seed!(1)
xguess = xref + 0.1*randn(Nx,Nt)
uguess = kron(ones(Nt-1)', uref) + 0.1*randn(Nu,Nt-1)
z0 = [reshape([xguess[:,1:(Nt-1)]; uguess],(Nx+Nu)*(Nt-1),1); xguess[:,end]];

prob = ProblemMOI(n_nlp,m_nlp)
z_sol, solver = solve(z0,prob) # solve
xtraj = [z[1:Nx,:] z_sol[end-(Nx-1):end]]
utraj = z[(Nx+1):(Nx+Nu),:];

Xsol = [SVector{12}(x) for x in eachcol(xtraj)]
visualize!(vis, model, Tfinal, Xsol)

###########################
###        NLP          ###
###########################
include("hybrid_nlp.jl")

# Discretization
model = SimpleWalker()
tf = 4.4
dt = 0.1
T = Int(ceil(tf/dt)) + 1
M = 5
times = range(0,tf, length=T)

# Reference Trajectory
Xref,Uref = reference_trajectory(model, times);

## Objective
Q = Diagonal([1.0*ones(6); 1.0*ones(6)]);
Q = Diagonal([1; 10; fill(1.0, 4); 1; 10; fill(1.0, 4)]);
R = Diagonal(fill(1e-3,3))
Qf = Q
obj = map(1:T-1) do k
    LQRCost(Q,R,Xref[k],Uref[k])
end
push!(obj, LQRCost(Qf, R*0, Xref[T], Uref[1]))

# NLP
nlp = HybridNLP(model, obj, tf, T, M, Xref[1], Xref[end]);

Random.seed!(1)
Xguess = [x + 0.1*randn(12) for x in Xref]
Uguess = [u + 0.1*randn(3) for u in Uref]
# Xguess = [SVector{12}(x) for x in eachcol(xguess)]
# Uguess = [SVector{3}(u) for u in eachcol(uguess)]
Z0 = packZ(nlp, Xguess, Uguess);

Zsol = solve(Z0, nlp, c_tol=1e-4, tol=1e-2)
visualize!(vis, model, tf, unpackZ(nlp, Zsol[1])[1])