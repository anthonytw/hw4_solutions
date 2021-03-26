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
const RD = RobotDynamics
const TO = TrajectoryOptimization

function slerp(qa::UnitQuaternion{T}, qb::UnitQuaternion{T}, t::T) where {T}
    # http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/
    coshalftheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;

    if coshalftheta < 0
        qm = -qb
        coshalftheta = -coshalftheta
    else
        qm = qb
    end
    abs(coshalftheta) >= 1.0 && return qa

    halftheta    = acos(coshalftheta)
    sinhalftheta = sqrt(one(T) - coshalftheta * coshalftheta)

    if abs(sinhalftheta) < 0.001
        return Quaternion(
            T(0.5) * (qa.w + qb.w),
            T(0.5) * (qa.x + qb.x),
            T(0.5) * (qa.y + qb.y),
            T(0.5) * (qa.z + qb.z),
        )
    end

    ratio_a = sin((one(T) - t) * halftheta) / sinhalftheta
    ratio_b = sin(t * halftheta) / sinhalftheta

    UnitQuaternion(
        qa.w * ratio_a + qm.w * ratio_b,
        qa.x * ratio_a + qm.x * ratio_b,
        qa.y * ratio_a + qm.y * ratio_b,
        qa.z * ratio_a + qm.z * ratio_b,
    )
end

function YakProblems(;
        integration=RD.RK4,
        N = 101,
        vecstate=false,
        scenario=:barrellroll, 
        costfun=:Quadratic, 
        termcon=:goal,
        quatnorm=:none,
        heading=0.0,  # deg
        Qpos=1.0,
        kwargs...
    )
    model = RobotZoo.YakPlane(UnitQuaternion)

    opts = SolverOptions(
        cost_tolerance_intermediate = 1e-1,
        penalty_scaling = 10.,
        penalty_initial = 10.;
        kwargs...
    )

    s = RD.LieState(model)
    n,m = size(model)
    rsize = size(model)[1] - 9
    vinds = SA[1,2,3,8,9,10,11,12,13]

    # Discretization
    tf = 1.25
    if scenario == :fullloop 
        tf *= 2
        @show N
        N = (N-1)*2 + 1
        @show N
    end
    dt = tf/(N-1)

    # Initial and final orientation 
    p0 = MRP(0.997156, 0., 0.075366) # initial orientation
    pf = MRP(0., -0.0366076, 0.) # final orientation (upside down)
    vel = 5.0
    utrim  = @SVector  [41.6666, 106, 74.6519, 106]

    if scenario ∈ (:halfloop, :fullloop) 
        ey = @SVector [0,1,0.]

        dq = expm(SA[0,0,1]*deg2rad(heading))
        pf = pf * dq
        pm = expm(SA[1,0,0]*deg2rad(180))*expm(SA[0,1,0]*deg2rad(90))

        x0 = RD.build_state(model, [-3,0,1.5], p0, [vel,0,0], [0,0,0])
        xm = RD.build_state(model, [0,0,3.], pm, pm * [vel,0,0.], [0,0,0])
        xm2 = RD.build_state(model, [-3,3,4.], pm * RotY(pi), [0,0,-vel], [0,0,0])
        xf = RD.build_state(model, dq*[3,0,6.], pf, pf * [vel,0,0.], [0,0,0])
        pf2 = RotZ(deg2rad(heading-180))
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
                    @show k-2Nmid
                    x1 = x̄m
                    x2 = x̄f
                end
            end
            RBState(
                x1.r + (x2.r - x1.r)*t,
                slerp(x1.q, x2.q, t),
                x1.v + (x2.v - x1.v)*t,
                # x1.ω + (x2.ω - x1.ω)*t,
                SA[0,pi/1.25,0]
            )
        end
    elseif scenario == :corkscrew
        rad = 1.0
        rev = 1.0 
        h_offset = rad - rad*cos(2pi*rev)
        @show h_offset

        x0 = RD.build_state(model, [-5,0,1.5], p0, [vel,0,0], [0,0,0])
        xf = RD.build_state(model, [5,0,1.5+h_offset], pf, pf * [vel,0,0.], [0,0,0])

        vel = (xf[1] - x0[1]) / tf
        N_roll = (N-1) ÷ 2 
        N_start = (N - N_roll) ÷ 2
        t_roll = N_roll * dt
        tstart = N_start * dt
        times = range(0, tf, length=N)
        ϕ = [zeros(N_start); range(0,-2pi*rev, length=N_roll); fill(-2pi*rev, N-N_start-N_roll)]
        f = 2pi*rev/t_roll
        Xref = map(1:N) do k
            t = times[k]
            do_roll = N_start < k < N_start + N_roll
            RBState(
                [-3 + vel*t, -rad*sin((t-tstart)*f)*do_roll, x0[3] - rad*(cos((t-tstart)*f)-1)*do_roll + h_offset*(k >= N_start + N_roll)],
                expm([1,0,0]*ϕ[k])*p0,
                [vel, -rad*f*cos((t-tstart)*f)*do_roll, rad*f*sin((t-tstart)*f)*do_roll],
                [2pi*rev/t_roll*do_roll,0,0]
            )
        end
    else
        throw(ArgumentError("$scenario isn't a known scenario"))
    end
    # Objective
    Qf_diag = RD.fill_state(model, 10, 500, 100, 100.)
    Q_diag = RD.fill_state(model, Qpos*0.1, 0.1, 0.1, 1.1)
    Qf = Diagonal(Qf_diag)
    Q = Diagonal(Q_diag)
    R = Diagonal(@SVector fill(1e-3,4))
    if quatnorm == :slack
        m += 1
        R = Diagonal(push(R.diag, 1e-6))
        utrim = push(utrim, 0)
    end
    if costfun == :Quadratic
        costfuns = map(Xref) do xref
            LQRCost(Q, R, xref, utrim)
        end
        costfun = LQRCost(Q, R, xf, utrim)
        costterm = LQRCost(Qf, R, xf, utrim)
        costfuns[end] = costterm
    elseif costfun == :Quadratic2
        Gf = zeros(n,n-1)
        costfuns = map(Xref) do xref
            RobotDynamics.state_diff_jacobian!(Gf, model, xref)
            Q̄ = Gf*Diagonal(deleteat(Q_diag,4))*Gf'
            LQRCost(Q̄, R, xref, utrim)
        end
        RobotDynamics.state_diff_jacobian!(Gf, model, Xref[end])
        Q̄f = Gf*Diagonal(deleteat(Qf_diag,4))*Gf'
        costterm = LQRCost(Q̄f, R, Xref[end], utrim)
        costfuns[end] = costterm
    elseif costfun == :QuatLQR
        costfuns = map(Xref) do xref
            TO.QuatLQRCost(Q, R, xref, utrim, w=0.1)
        end
        costterm = TO.QuatLQRCost(Qf, R, Xref[end], utrim; w=200.0)
        costfuns[end] = costterm
    end
    obj = Objective(costfuns)

    # Initialization
    U0 = [copy(utrim) for k = 1:N-1]

    # Build problem
    prob = Problem(model, obj, xf, tf, x0=x0, integration=integration)
    initial_controls!(prob, U0)
    prob, opts, Xref, x̄f
end

function get_trim(model, x_trim, u_guess;
        tol = 1e-4,
        iters = 100,
    )
    x̄ = RBState(model, x_trim)
    ṙ = x̄.v 
    q̇ = Rotations.kinematics(x̄.q, x̄.ω)
    a = zeros(3)
    α = zeros(3)
    v_trim = [ṙ; q̇; a; α]
    n,m = size(model)
    G = zeros(n,n-1)
    RobotDynamics.state_diff_jacobian!(G, model, x_trim)

    iu = SVector{m}(1:m)
    ic = SVector{6}(m .+ (1:6))
    ia = SVector{6}(1:6) .+ 7
    ∇f(u) = ForwardDiff.jacobian(u_->dynamics(model, x_trim, u_), u)[ia,:]

    r(z) = [z[iu] - u_guess + ∇f(z[iu])'z[ic]; dynamics(model, x_trim, z[iu])[ia] ]
    ∇r(z) = begin
        u,λ = z[iu], z[ic] 
        B = ∇f(u)
        [I B'; B -I(6)*1e-6]
    end
    # r([u_guess; @SVector zeros(n-1)])
    λ = @SVector zeros(6)
    z = [u_guess; λ]

    for i = 1:iters
        res = r(z)
        println(norm(res))
        if norm(res) < tol 
            println("converged in $i iterations")
            break
        end
        R = ∇r(z) 
        # display(R)
        dz = -(R \ res)
        z += dz
    end
    return z[iu]
end
u = get_trim(prob.model, x0, fill(100,4), tol=1e-4)

p0 = MRP(0.997156, 0., 0.075366) # initial orientation
pf = MRP(0., -0.0366076, 0.) # final orientation (upside down)
vel = 5.0
utrim  = @SVector  [41.6666, 106, 74.6519, 106]

##
prob, opts, Xref = YakProblems(costfun=:QuatLQR, scenario=:halfloop, heading=130)
x0 = prob.x0
∇f = RobotDynamics.DynamicsJacobian(prob.model)
dynamics(prob.model, x0, utrim)
visualize!(vis, prob.model, prob.tf, Xref)
solver = ALTROSolver(prob, opts, verbose=2, infeasible=false)
solve!(solver)
visualize!(vis, solver)
Uhalf = controls(solver)

prob2, opts, Xref2, xf = YakProblems(costfun=:QuatLQR, scenario=:fullloop, heading=130, Qpos=100)
visualize!(vis, prob2.model, prob2.tf, Xref2)

U0 = controls(prob2)
u_hover = U0[1]
for k = 1:length(U0)
    k2 = min(k, length(Uhalf))
    U0[k] = Uhalf[k2]
end
U0[1+length(Uhalf):end] .= reverse(Uhalf)
initial_controls!(prob2, U0)
rollout!(prob2)
visualize!(vis, prob2)
# initial_states!(prob2, Xref2)
solver = ALTROSolver(prob2, opts, verbose=2, infeasible=false, gradient_tolerance=0.1)
solve!(solver)
visualize!(vis, solver)

prob2, opts, Xref2 = YakProblems(costfun=:QuatLQR, scenario=:fullloop, heading=180)
initial_controls!(prob2, controls(solver))
solver = ALTROSolver(prob2, opts, verbose=2, infeasible=false, gradient_tolerance=0.1)
solve!(solver)
visualize!(vis, solver)

plot(controls(solver))
plot(Xref2, inds=11:13)

##
vis = Visualizer()
open(vis, Blink.Window())
delete!(vis)
model = RobotZoo.YakPlane(UnitQuaternion)
TrajOptPlots.set_mesh!(vis, model, color=colorant"yellow")