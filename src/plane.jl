import Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using TrajectoryOptimization
using Altro
using RobotZoo
using RobotDynamics
using StaticArrays
using LinearAlgebra
using Rotations
using Plots
using TrajOptPlots
using MeshCat
using Blink
using Colors
using ForwardDiff
using RobotZoo: YakPlane
const RD = RobotDynamics
const TO = TrajectoryOptimization

include("costfuns.jl")
include("ilqr.jl")
include("utils.jl")


function YakProblems(;
        N = 101,
        vecstate=false,
        scenario=:barrellroll, 
        heading=0.0,  # deg
        Qpos=1.0,
        kwargs...
    )
    model = RobotZoo.YakPlane(UnitQuaternion)

    n,m = size(model)
    ip = SA[1,2,3]
    iq = SA[4,5,6,7]
    iv = SA[8,9,10]
    iw = SA[11,12,13]

    # Discretization
    tf = 1.25
    if scenario == :fullloop 
        tf *= 2
        N = (N-1)*2 + 1
    end
    dt = tf/(N-1)

    # Initial and final orientation 
    vel = 5.0

    if scenario ∈ (:halfloop, :fullloop) 
        ey = @SVector [0,1,0.]

        # Heading
        dq = expm(SA[0,0,1]*deg2rad(heading))
        # pf = pf * dq

        # Initial state
        p0 = MRP(0.997156, 0., 0.075366) # initial orientation (level flight)
        x0 = RD.build_state(model, [-3,0,1.5], p0, [vel,0,0], [0,0,0])

        # Climb
        pm = expm(SA[1,0,0]*deg2rad(180))*expm(SA[0,1,0]*deg2rad(90))
        xm = RD.build_state(model, [0,0,3.], pm, pm * [vel,0,0.], [0,0,0])

        # Top of loop
        pf = MRP(0., -0.0366076, 0.) * dq # final orientation (upside down)
        xf = RD.build_state(model, dq*[3,0,6.], pf, pf * [vel,0,0.], [0,0,0])
        pf2 = RotZ(deg2rad(heading-180))

        # Dive
        xm2 = RD.build_state(model, [-3,3,4.], pm * RotY(pi), [0,0,-vel], [0,0,0])

        # Terminal state
        xf2 = RD.build_state(model, [0,4,1.5], p0, [vel,0,0], [0,0,0])

        t_flat = 5 / (xf[2] - xf[1])
        N_flat = Int(round(t_flat/dt))

        # Xref trajectory
        x̄0 = RBState(model, x0)
        x̄m = RBState(model, xm)
        x̄m2 = RBState(model, xm2)
        x̄f = RBState(model, xf)
        x̄f2 = RBState(model, xf2)
        Xref = map(1:N) do k
            t = (k-1)/(N-1)
            Nmid = N ÷ 4
            if scenario == :fullloop
                if k < Nmid
                    x1 = x̄0
                    x2 = x̄m
                    t = (k-1)/Nmid
                elseif k < 2Nmid
                    t = (k-Nmid)/Nmid
                    x1 = x̄m
                    x2 = x̄f 
                elseif k < 3Nmid
                    t = (k-2Nmid)/Nmid
                    x1 = x̄f
                    x2 = x̄m2
                else
                    t = (k-3Nmid)/Nmid
                    x1 = x̄m2
                    x2 = x̄f2
                end
            else
                if k < 2Nmid
                    t = (k-1)/2Nmid
                    x1 = x̄0
                    x2 = x̄m
                else
                    t = (k-2Nmid)/(2Nmid+1)
                    x1 = x̄m
                    x2 = x̄f
                end
            end
            RBState(
                x1.r + (x2.r - x1.r)*t,
                slerp(x1.q, x2.q, t),
                x1.v + (x2.v - x1.v)*t,
                SA[0,pi/1.25,0]
            )
        end
    else
        throw(ArgumentError("$scenario isn't a known scenario"))
    end

    # Get trim condition
    utrim = get_trim(model, x0, fill(124, 4))

    # Objective
    Qf_diag = RD.fill_state(model, 10, 500*0, 100, 100.)
    Q_diag = RD.fill_state(model, Qpos*0.1, 0.1*0, 0.1, 1.1)
    R = Diagonal(@SVector fill(1e-3,4))
    costs = map(1:N-1) do k
        RigidBodyCost(
            Diagonal(Q_diag[ip])*dt,
            10.0*dt, 
            Diagonal(Q_diag[iv])*dt,
            Diagonal(Q_diag[iw])*dt,
            R*dt,
            Xref[k][ip],
            Xref[k][iq],
            Xref[k][iv],
            Xref[k][iw],
            utrim
        )
    end
    costterm = RigidBodyCost(
            Diagonal(Qf_diag[ip]),
            200.0, 
            Diagonal(Qf_diag[iv]),
            Diagonal(Qf_diag[iw]),
            R*0,
            Xref[N][ip],
            Xref[N][iq],
            Xref[N][iv],
            Xref[N][iw],
            utrim
        )
    push!(costs, costterm)

    # Build problem
    prob = iLQRProblem(model, costs, tf, x0)

    return prob, Xref, utrim
end

function get_trim(model, x_trim, u_guess;
        verbose = false,
        tol = 1e-4,
        iters = 100,
    )

    a = zeros(3)       # linear acceleration
    α = zeros(3)       # angular acceleration
    n,m = size(model)

    iu = SVector{m}(1:m)           # control indices
    ic = SVector{6}(m .+ (1:6))    # constraint indices
    ia = SVector{6}(1:6) .+ 7      # acceleration indices

    # Jacobian of the constraint
    ∇f(u) = ForwardDiff.jacobian(u_->dynamics(model, x_trim, u_), u)[ia,:]

    # Residual function
    r(z) = [z[iu] - u_guess + ∇f(z[iu])'z[ic]; dynamics(model, x_trim, z[iu])[ia] ]

    # Jacobian of the the residual function
    ∇r(z) = begin
        u,λ = z[iu], z[ic] 
        B = ∇f(u)
        [I B'; B -I(6)*1e-6]
    end

    # Initial guess
    λ = @SVector zeros(6)
    z = [u_guess; λ]

    # Newton solve
    for i = 1:iters
        # Check convergence
        res = r(z)
        verbose && println(norm(res))
        if norm(res) < tol 
            verbose && println("converged in $i iterations")
            break
        end

        # Compute Newton step
        R = ∇r(z) 
        dz = -(R \ res)
        z += dz
    end

    # Return the trim controls
    return z[iu]
end

function state_parts(model::YakPlane)
    ip = SA[1,2,3]
    iq = SA[4,5,6,7]
    iv = SA[8,9,10]
    iw = SA[11,12,13]
    return ip, iq, iv, iw
end

function lmult(q)
    SA[
        q[1] -q[2] -q[3] -q[4];
        q[2]  q[1] -q[4]  q[3];
        q[3]  q[4]  q[1] -q[2];
        q[4] -q[3]  q[2]  q[1];
    ]
end

function state_error(model::YakPlane, x, x0)
    ip,iq,iv,iw = state_parts(model)
    q  = UnitQuaternion(x[iq])
    q0 = UnitQuaternion(x0[iq])
    dq = cayleymap(lmult(x0[iq])'x[iq]) 
    return [x[ip] - x0[ip]; dq; x[iv] - x0[iv]; x[iw] - x0[iw]]
end
cayleymap(q) = SA[q.x,q.y,q.z] / q.w
cayleymap(q::StaticVector{4}) = SA[q[2],q[3],q[4]] / q[1]

function state_error_jacobian(model, x)
    iq = state_parts(model)[2]
    q = x[iq] 
    G = attitude_jacobian(q)
    SA[
        1 0 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0 0;
        0 0 0 G[1] G[5] G[9]  0 0 0 0 0 0;
        0 0 0 G[2] G[6] G[10] 0 0 0 0 0 0;
        0 0 0 G[3] G[7] G[11] 0 0 0 0 0 0;
        0 0 0 G[4] G[8] G[12] 0 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 0 1;
    ]
end

function attitude_jacobian(q)
    SA[
        -q[2] -q[3] -q[4];
         q[1] -q[4]  q[3];
         q[4]  q[1] -q[2];
        -q[3]  q[2]  q[1];
    ]
end

function rollout(model, x0::StaticVector{n}, U, times) where n
    N = length(U0) + 1
    X = [@SVector zeros(n) for k = 1:N]
    X[1] = x0
    for k = 1:N-1
        dt = times[k+1] - times[k]
        X[k+1] = discrete_dynamics(RK4, model, X[k], U[k], times[k], dt)
    end
    return X
end

## Launch Visualizer
vis = Visualizer()
open(vis, Blink.Window())
delete!(vis)
model = RobotZoo.YakPlane(UnitQuaternion)
TrajOptPlots.set_mesh!(vis, model, color=colorant"yellow")

## Solve half-loop
prob_half, Xref_half, utrim = YakProblems(costfun=:QuatLQR, scenario=:halfloop, heading=130)
U0 = [copy(utrim) for k = 1:prob_half.N-1]
X0 = rollout(prob_half.model, prob_half.x0, U0, prob_half.times)
plot(X0, inds=1:3)
visualize!(vis, prob_half.model, prob_half.tf, Xref_half)

Xhalf, Uhalf = solve_ilqr(prob_half, X0, U0, verbose=1, eps=1e-4, reg_min=1e-6)
visualize!(vis, prob_half.model, prob_half.tf, Xhalf)

## Full Loop
prob, Xref2 = YakProblems(costfun=:QuatLQR, scenario=:fullloop, heading=130, Qpos=100)
visualize!(vis, prob.model, prob.tf, Xref2)

# Design initial control
U0 = [copy(utrim) for k = 1:prob.N-1]
for k = 1:length(Uhalf)
    U0[k] = Uhalf[k]
end
U0[1+length(Uhalf):end] .= reverse(Uhalf)
X0 = rollout(prob.model, prob.x0, U0, prob.times)
visualize!(vis, prob.model, prob.tf, X0)

# Solve
Xfull, Ufull = solve_ilqr(prob, X0, U0, verbose=1, eps=1e-2, reg_min=1e-8, iters=500)
visualize!(vis, prob.model, prob.tf, Xfull)
