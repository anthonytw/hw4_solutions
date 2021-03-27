using Printf
"""
    iLQRProblem{n,m,L}

Describes a trajectory optimization problem with `n` states, `m` controls, and 
a model of type `L`. 

# Constructor
    Problem(model::L, obj::Vector{<:QuadraticCost{n,m}}, tf, x0, xf) where {n,m,L}

where `tf` is the final time, and `x0` is the initial state. 
"""
struct iLQRProblem{n,m,L,O}
    model::L
    obj::Vector{O}
    N::Int
    tf::Float64
    x0::MVector{n,Float64}
    times::Vector{Float64}
    function iLQRProblem(model::L, obj::Vector{O}, tf, x0) where {L,O}
        n,m = size(model)
        @assert length(x0) == n
        T = length(obj)
        times = range(0, tf, length=T)
        new{n,m,L,O}(model, obj, T, tf, x0, times)
    end
end
Base.size(prob::iLQRProblem{n,m}) where {n,m} = (n,m,prob.N)

"""
    backwardpass!(prob, P, p, K, d, X, U)

Evaluate the iLQR backward pass at state and control trajectories `X` and `U`, 
storing the cost-to-go expansion in `P` and `p` and the gains in `K` and `d`.

Should return ΔJ, expected cost reduction.
"""
function backwardpass!(prob::iLQRProblem{n,m}, P, p, K, d, X, U; 
        β=1e-6, ddp::Bool=false
    ) where {n,m}
    N = prob.N
    obj = prob.obj
    ΔJ = 0.0
    failed = false
    
    # TODO: Implement the backward pass

    # SOLUTION
    ∇f = RobotDynamics.DynamicsJacobian(prob.model) 
    ∇jac = zeros(n+m,n+m) 
    iq = state_parts(prob.model)[2]
    Iq = Diagonal(SA[0,0,0,1,1,1, 0,0,0, 0,0,0])

    G2 = state_error_jacobian(prob.model, X[N])
    Q, = hessian(prob.obj[N], X[N], 0*U[1])
    q, = gradient(prob.obj[N], X[N], 0*U[1])
    p[N] = G2'q
    b = q[iq]'X[N][iq]
    P[N] = G2'Q*G2 - Iq*(q[iq]'X[N][iq])
    
    #Backward Pass
    failed = false
    for k = (N-1):-1:1

        # Cost Expansion
        Q,R =  hessian(prob.obj[k], X[k], U[k])
        q,r = gradient(prob.obj[k], X[k], U[k])

        # Dynamics derivatives
        dt = prob.times[k+1] - prob.times[k]
        z = KnotPoint(SVector{n}(X[k]), SVector{m}(U[k]), dt, prob.times[k])
        discrete_jacobian!(RK4, ∇f, model, z)
        A = RobotDynamics.get_static_A(∇f)
        B = RobotDynamics.get_static_B(∇f)

        # Convert to error state
        G1 = state_error_jacobian(prob.model, X[k])
        q = G1'q
        Q = G1'Q*G1 - Iq*(q[iq]'X[k][iq])
        A = G2'A*G1
        B = G2'B
    
        gx = q + A'*p[k+1]
        gu = r + B'*p[k+1]
    
        Gxx = Q + A'*P[k+1]*A
        Guu = R + B'*P[k+1]*B
        Gux = B'*P[k+1]*A
        
        if ddp 
            # #Add full Newton terms
            RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[k+1])
            Gxx .+= ∇jac[1:n, 1:n]
            Guu .+= ∇jac[n+1:end, n+1:end]
            Gux .+= ∇jac[n+1:end, 1:n]
        end
    
        # Regularization
        Guu_reg = Guu + B'*β*I*B
        Gux_reg = Gux 
        Guu_reg = SMatrix{m,m}(Guu) + β*Diagonal(@SVector ones(m))
        
        # Calculate Gains
        d[k] .= Guu_reg\gu
        K[k] .= Guu_reg\Gux_reg
    
        # Cost-to-go Recurrence
        p[k] .= gx - K[k]'*gu + K[k]'*Guu*d[k] - Gux'*d[k]
        P[k] .= Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux
        ΔJ += gu'*d[k]

        G2 = G1        
    end
    return ΔJ
end

"""
    forwardpass!(prob, X, U, K, d, ΔJ, J)

Evaluate the iLQR forward pass at state and control trajectories `X` and `U`, using
the gains `K` and `d` to simulate the system forward. The new cost should be less than 
the current cost `J` together with the expected cost decrease `ΔJ`.

Should return the new cost `Jn` and the step length `α`.
"""
function forwardpass!(prob::iLQRProblem{n,m}, X, U, K, d, ΔJ, J,
        Xbar = deepcopy(X), Ubar = deepcopy(U);
        max_iters=10,
    ) where {n,m}
    N = prob.N
    model = prob.model

    # TODO: Implement the forward pass w/ line search
    Jn = J
    α = 0.0
    
    # SOLUTION
    # Line Search
    Xbar[1] = X[1]
    α = 1.0
    Jn = Inf
    for i = 1:max_iters
        
        # Forward Rollout
        for k = 1:(N-1)
            t = prob.times[k]
            dt = prob.times[k+1] - prob.times[k]
            dx = state_error(model, Xbar[k], X[k])
            Ubar[k] = U[k] - α*d[k] - K[k]*dx
            Xbar[k+1] = discrete_dynamics(RK4, model, Xbar[k], Ubar[k], t, dt) 
        end
        
        # Calculate the new cost
        Jn = evalcost(prob.obj, Xbar, Ubar)

        # Check Armijo condition
        if Jn <= J - 1e-2*α*ΔJ
            break
        else
            # Backtrack
            α *= 0.5  
        end
        if i == max_iters 
            @warn "Line Search failed"
            α = 0
        end
    end
    
    # Accept direction
    for k = 1:N-1
        X[k] = Xbar[k]
        U[k] = Ubar[k]
    end
    X[N] = Xbar[N]
    
    return Jn, α
end

"""
    solve_ilqr(prob, X, U; kwargs...)

Solve the trajectory optimization problem specified by `prob` using iterative LQR.
Returns the optimized state and control trajectories, as well as the local control gains,
`K` and `d`.

Should return the optimized state and control trajectories `X` and `U`, and the 
list of feedback gains `K` and cost-to-go hessians `P`.
"""
function solve_ilqr(prob::iLQRProblem{n,m}, X0, U0; 
        iters=100,     # max iterations
        ls_iters=10,   # max line search iterations
        reg_min=1e-6,  # minimum regularizatio for the backwardpass
        verbose=0,     # print verbosity
        eps=1e-5,      # termination tolerance
        eps_ddp=eps    # tolerance to switch to ddp
    ) where {n,m}
    t_start = time_ns()
    Nx,Nu,Nt = size(prob)

    # Initialization
    N = prob.N
    p = [zeros(n-1) for k = 1:N]              # ctg gradient
    P = [zeros(n-1,n-1) for k = 1:N]          # ctg hessian
    d = [zeros(m) for k = 1:N-1]              # feedforward gains
    K = [zeros(m,n-1) for k = 1:N-1]          # feedback gains
    Xbar = [@SVector zeros(n) for k = 1:N]    # line search trajectory
    Ubar = [@SVector zeros(m) for k = 1:N-1]  # line search trajectory
    ΔJ = 0.0

    # Don't modify the trajectories that are passed in
    X = deepcopy(X0)
    U = deepcopy(U0)

    # Initial cost
    J = evalcost(prob.obj, X, U)
    
    # Initialize parameters
    Jn = Inf
    iter = 0
    tol = 1.0
    β = reg_min
    while tol > eps 
        iter += 1
        
        # TODO: Implement iLQR
        
        # SOLUTION
        
        # Backward Pass
        ddp = tol < eps_ddp
        ΔJ, = backwardpass!(prob, P, p, K, d, X, U, ddp=ddp, β=β)

        # Forward Pass
        Jn, α = forwardpass!(prob, X, U, K, d, ΔJ, J, Xbar, Ubar, max_iters=ls_iters)

        if α === zero(α) 
            β = max(β*10, 1.0)
            # β *= 10 
        # elseif α === one(α)
        #     β = reg_min
        else 
            β = max(β/2, reg_min)
        end

        # Update parameters
        tol = maximum(norm.(d, Inf))
        β = max(0.9*β, reg_min)
        
        # END SOLUTION

        # Output
        if verbose > 0
            @printf("Iter: %3d, Cost: % 6.2f → % 6.2f (% 7.2e), res: % .2e, β= %.2e, α = %.3f, ΔJ = %.3e\n",
                iter, J, Jn, J-Jn, tol, β, α, ΔJ
            )
        end
        J = Jn

        if iter >= iters
            @warn "Reached max iterations"
            break
        end

    end
    println("Total Time: ", (time_ns() - t_start)*1e-6, " ms")
    return X,U,K,P
end