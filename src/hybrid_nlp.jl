"""
    HybridNLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`, 
and initial and final state `x0`, `xf`.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

# Constructor
    NLP(model, obj, tf, T, x0, xf, [integration])

# Basic Methods
    Base.size(nlp)    # returns (n,m,T)
    num_ineq(nlp)     # number of inequality constraints
    num_eq(nlp)       # number of equality constraints
    num_primals(nlp)  # number of primal variables
    num_duals(nlp)    # total number of dual variables
    packZ(nlp, X, U)  # Stacks state `X` and controls `U` into one vector `Z`

# Evaluating the NLP
The NLP supports the following API for evaluating various pieces of the NLP:

    eval_f(nlp, Z)         # evaluate the objective
    grad_f!(nlp, grad, Z)  # gradient of the objective
    hess_f!(nlp, hess, Z)  # Hessian of the objective
    eval_c!(nlp, c, Z)     # evaluate the constraints
    jac_c!(nlp, c, Z)      # constraint Jacobian
"""
struct HybridNLP{n,m,L,Q} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    T::Int                                   # number of knot points
    M::Int                                   # number of steps in each mode
    Nmodes::Int                              # number of modes
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    times::Vector{Float64}                   # vector of times
    modes::Vector{Int}                       # mode ID
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    cinds::Vector{UnitRange{Int}}
    lb::Vector{Float64}
    ub::Vector{Float64}
    zL::Vector{Float64}
    zU::Vector{Float64}
    rows::Vector{Int}
    cols::Vector{Int}
    function HybridNLP(model, obj::Vector{<:QuadraticCost{n,m}},
            tf::Real, T::Integer, M::Integer, x0::AbstractVector, xf::AbstractVector, integration::Type{<:QuadratureRule}=RK4
        ) where {n,m}
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:T]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:T-1]
        times = collect(range(0, tf, length=T))
        modes = map(1:T) do k
            isodd((k-1) ÷ M + 1) ? 1 : 2
        end
        Nmodes = Int(ceil(T/M))
        
        c_init_inds = 1:n                                                    # initial constraint
        c_term_inds = (c_init_inds[end]+1):(c_init_inds[end]+n)              # terminal constraint
        c_dyn_inds = (c_term_inds[end]+1):(c_term_inds[end]+n*(T-1)  )       # dynamics constraints
        c_stance_inds = (c_dyn_inds[end]+1):(c_dyn_inds[end]+T)              # stance constraint (1 per time step)
        c_length_inds = (c_stance_inds[end]+1):(c_stance_inds[end]+(2*T))    # length bounds     (2 per time step)
        m_nlp = c_length_inds[end]                                           # total number of constraints
        
        lb = fill(0.0,m_nlp)
        ub = fill(0.0,m_nlp)
        lb[c_length_inds] .= model.ℓ_min
        ub[c_length_inds] .= model.ℓ_max

        cinds = [c_init_inds, c_term_inds, c_dyn_inds, c_stance_inds, c_length_inds]
        
        n_nlp = n*T + (T-1)*m
        zL = fill(-Inf, n_nlp)
        zU = fill(+Inf, n_nlp)
        rows = Int[]
        cols = Int[]
        
        new{n,m,typeof(model), integration}(
            model, obj,
            T, M, Nmodes, tf, x0, xf, times, modes,
            xinds, uinds, cinds, lb, ub, zL, zU, rows, cols
        )
    end
end
Base.size(nlp::HybridNLP{n,m}) where {n,m} = (n,m,nlp.T)
num_primals(nlp::HybridNLP{n,m}) where {n,m} = n*nlp.T + m*(nlp.T-1)
num_duals(nlp::HybridNLP) = nlp.cinds[end][end]

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.T-1
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    Z[nlp.xinds[end]] = X[end]
    return Z
end

"""
    unpackZ(nlp, Z)

Take a vector of all the states and controls and return a vector of state vectors `X` and
controls `U`.
"""
function unpackZ(nlp, Z)
    X = [Z[xi] for xi in nlp.xinds]
    U = [Z[ui] for ui in nlp.uinds]
    return X, U
end

"""
    eval_f(nlp, Z)

Evaluate the objective, returning a scalar.
"""
function eval_f(nlp::HybridNLP, Z)
    # TASK: compute the objective value (cost)
    J = 0.0
    
    # SOLUTION
    xi,ui = nlp.xinds, nlp.uinds
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        J += stagecost(nlp.obj[k], x, u)
    end
    J += termcost(nlp.obj[end], Z[xi[end]])
    return J
end

"""
    grad_f!(nlp, grad, Z)

Evaluate the gradient of the objective at `Z`, storing the result in `grad`.
"""
function grad_f!(nlp::HybridNLP{n,m}, grad, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        # TASK: Compute the cost gradient
        grad[xi[k]] .= 0
        grad[ui[k]] .= 0
        
        # SOLUTION
        grad[xi[k]] = obj[k].Q*x + obj[k].q
        grad[ui[k]] = obj[k].R*u + obj[k].r
    end
    grad[xi[end]] = obj[end].Q*Z[xi[end]] + obj[end].q
    return nothing
end

"""
    hess_f!(nlp, hess, Z)

Evaluate the Hessian of the objective at `Z`, storing the result in `hess`.
Should work with `hess` sparse.
"""
function hess_f!(nlp::HybridNLP{n,m}, hess, Z, rezero=true) where {n,m}
    # TASK: Compute the objective hessian
    # HINT: It's a diagonal matrix
    if rezero
        for i = 1:size(hess,1)
            hess[i,i] = 0
        end
    end
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    i = 1
    for k = 1:nlp.T
        for j = 1:n
            hess[i,i] += nlp.obj[k].Q[j,j]
            i += 1
        end
        if k < nlp.T
            for j = 1:m
                hess[i,i] += nlp.obj[k].R[j,j]
                i += 1
            end
        end
    end
end

function dynamics_constraint!(nlp::HybridNLP{n,m}, c, ztraj) where {n,m}
    Nt = nlp.T
    Nx,Nu = n,m
    Nmodes = nlp.Nmodes
    model = nlp.model
    xi,ui = nlp.xinds, nlp.uinds
    Z = ztraj
    dt = nlp.times[2]
    Nm = nlp.M

    d = reshape(view(c, nlp.cinds[3]),Nx,Nt-1)
    for k = 1:(Nmodes-1)
        for j = 1:(Nm-1)
            s = (k-1)*Nm + j
            x,u = Z[xi[s]], Z[ui[s]]
            x2 = Z[xi[s+1]]
            if mod(k,2) == 1
                d[:,s] = stance1_dynamics_rk4(model, x, u, dt) - x2 
            else
                d[:,s] = stance2_dynamics_rk4(model, x, u, dt) - x2 
            end
        end
        s = k*Nm
        x,u = Z[xi[s]], Z[ui[s]]
        x2 = Z[xi[s+1]]
        if mod(k,2) == 1
            d[:,s] = jump2_map(stance1_dynamics_rk4(model, x,u, dt)) - x2
        else
            d[:,s] = jump1_map(stance2_dynamics_rk4(model, x,u, dt)) - x2
        end

    end
    for j = 1:(Nm-1)
        s = (Nmodes-1)*Nm + j
        x,u = Z[xi[s]], Z[ui[s]]
        x2 = Z[xi[s+1]]
        if mod(Nmodes,2) == 1
            d[:,s] = stance1_dynamics_rk4(model, x,u, dt) - x2
        else
            d[:,s] = stance2_dynamics_rk4(model, x,u, dt) - x2
        end
    end
    
    return nothing
end

function dynamics_jacobian!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    D = view(jac, nlp.cinds[3], :)
    Nt = nlp.T
    Nx,Nu = n,m
    Nmodes = nlp.Nmodes
    model = nlp.model
    xi,ui = nlp.xinds, nlp.uinds
    dt = nlp.times[2]
    Nm = nlp.M

    ic = 1:n
    for k = 1:(Nmodes-1)
        for j = 1:(Nm-1)
            s = (k-1)*Nm + j
            x,u = Z[xi[s]], Z[ui[s]]
            zi = [xi[s]; ui[s]]
            F = view(D, ic, zi)
            F2 = view(D, ic, xi[s+1])

            if mod(k,2) == 1
                F .= stance1_jacobian(model, x, u, dt)
            else
                F .= stance2_jacobian(model, x, u, dt)
            end
            for i = 1:n
                F2[i,i] = -1
            end

            ic = ic .+ n
        end
        s = k*Nm
        x,u = Z[xi[s]], Z[ui[s]]
        zi = [xi[s]; ui[s]]
        F = view(D, ic, zi)
        F2 = view(D, ic, xi[s+1])
        if mod(k,2) == 1
            F .= jump2_jacobian()*stance1_jacobian(model, x,u,dt)
        else
            F .= jump1_jacobian()*stance2_jacobian(model, x,u,dt)
        end
        for i = 1:n
            F2[i,i] = -1
        end
        ic = ic .+ n
    end
    for j = 1:(Nm-1)
        s = (Nmodes-1)*Nm + j
        x,u = Z[xi[s]], Z[ui[s]]
        zi = [xi[s]; ui[s]]
        F = view(D, ic, zi)
        F2 = view(D, ic, xi[s+1])

        if mod(Nmodes,2) == 1
            F .= stance1_jacobian(model, x, u, dt)
        else
            F .= stance2_jacobian(model, x, u, dt)
        end
        for i = 1:n
            F2[i,i] = -1
        end
        ic = ic .+ n
    end
    
    return nothing
end


function stance_constraint!(nlp::HybridNLP{n,m}, c,ztraj) where {n,m}
    Nt = nlp.T
    Nx,Nu = n,m
    Nmodes = nlp.Nmodes
    Nm = nlp.M
    model = nlp.model
    xi,ui = nlp.xinds, nlp.uinds
    Z = ztraj

    d = view(c, nlp.cinds[4])
    # z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    # xtraj = [z[1:Nx,:] ztraj[(end-(Nx-1)):end]]
    t = 1
    for k = 1:Nmodes
        if mod(k,2) == 1
            for j = 1:Nm
                s = (k-1)*Nm + j
                x = Z[xi[s]]
                d[t] = x[4]  # keep foot on the floor
                t += 1
            end
        else
            for j = 1:Nm
                s = (k-1)*Nm + j
                x = Z[xi[s]]
                d[t] = x[6]
                t += 1
            end
        end
            
    end
    return d 
end

function length_constraint!(nlp::HybridNLP{n,m}, c, ztraj) where {n,m}
    Nt = nlp.T
    Nx,Nu = n,m
    Nmodes = nlp.Nmodes
    model = nlp.model
    xi = nlp.xinds
    Z = ztraj

    d = view(c, nlp.cinds[5])
    # z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    # xtraj = [z[1:Nx,:] ztraj[(end-(Nx-1)):end]]
    for k = 1:Nt
        x = Z[xi[k]]
        d[2*(k-1)+1] = norm(x[1:2] - x[3:4])
        d[2*(k-1)+2] = norm(x[1:2] - x[5:6])
    end
    return d
end

function eval_c!(nlp::HybridNLP, c, ztraj)
    # c[c_init_inds] .= ztraj[1:Nx] - xref[:,1] #initial state constraint
    # c[c_term_inds] .= ztraj[(end-(Nx-1)):end] - xref[:,end] #terminal state constraint
    xi = nlp.xinds
    c[nlp.cinds[1]] .= ztraj[xi[1]] - nlp.x0
    c[nlp.cinds[2]] .= ztraj[xi[end]] - nlp.xf
    dynamics_constraint!(nlp, c, ztraj)
    stance_constraint!(nlp, c, ztraj)
    length_constraint!(nlp, c, ztraj)
end


function jac_c!(nlp::HybridNLP{n,m}, jac, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    c = zeros(eltype(Z), num_duals(nlp))
    # ForwardDiff.jacobian!(jac, con!, c, Z)

    dynamics_jacobian!(nlp, jac, Z)

    jac[nlp.cinds[1], xi[1]] .= I(n)
    jac[nlp.cinds[2], xi[end]] .= I(n)
    jac_stance = view(jac, nlp.cinds[4], :)
    jac_length = view(jac, nlp.cinds[5], :)

    # c = zeros(size(jac,1))
    # jac_stance = zero(jac)
    # ForwardDiff.jacobian!(jac_stance, (c,x)->stance_constraint!(nlp,c,x), c, Z)
    # jac .+= jac_stance
    # return
    
    t = 1
    for k = 1:nlp.T
        x = Z[xi[k]]
        foot_ind = nlp.modes[k] == 1 ? 4 : 6
        jac_stance[t, xi[k][foot_ind]] = 1
        
        d1 = x[1:2] - x[3:4]
        d2 = x[1:2] - x[5:6]
        # if abs(norm(d1)) > 1e-10
        jac_length[2*(k-1)+1, xi[k][1:2]] = +d1 / norm(d1)
        jac_length[2*(k-1)+1, xi[k][3:4]] = -d1 / norm(d1)
        # end
        # if abs(norm(d2)) > 1e-10
        jac_length[2*(k-1)+2, xi[k][1:2]] = +d2 / norm(d2)
        jac_length[2*(k-1)+2, xi[k][5:6]] = -d2 / norm(d2)
        # end
        t += 1
    end
end

function reference_trajectory(model::SimpleWalker, times)
    n,m = size(model)
    tf = times[end]
    T = length(times)
    mb,g = model.mb, model.g
    uref = [0.5*mb*g; 0.5*mb*g; 0.0]
    xref = zeros(n,T)
    xref[1,:] .= LinRange(-1.5,1.5,T)
    xref[2,:] .= ones(T)
    xref[3,:] .= LinRange(-1.5,1.5,T)
    xref[5,:] .= LinRange(-1.5,1.5,T)
    xref[7,2:end-1] .= (3.0/tf)*ones(T-2)
    xref[9,2:end-1] .= (3.0/tf)*ones(T-2)
    xref[11,2:end-1] .= (3.0/tf)*ones(T-2);  
    Uref = [SVector{m}(uref) for k = 1:T-1]
    Xref = [SVector{n}(x) for x in eachcol(xref)]
    return Xref, Uref
end

using SparseArrays
function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end
function jacobian_structure(nlp::HybridNLP{n,m}) where {n,m}
    n_nlp, m_nlp = num_primals(nlp), num_duals(nlp)
    jac = spzeros(m_nlp, n_nlp)
    
    xi,ui = nlp.xinds, nlp.uinds
    m_nlp = size(jac,1)
    Finit = view(jac, nlp.cinds[1], xi[1])
    Fterm = view(jac, nlp.cinds[2], xi[end])
    cnt = 1
    for i = 1:n
        Finit[i,i] = cnt
        Fterm[i,i] = cnt + 1
        cnt += 2
    end
    
    # Dynamics
    D = view(jac, nlp.cinds[3],:)
    nblk = (n+m)*n
    ci = 1:n
    for k = 1:nlp.T-1
        zi = [xi[k]; ui[k]]
        F = view(D, ci, zi)
        F2 = view(D, ci, xi[k+1])
        F .= LinearIndices(zeros(n,n+m)) .+ cnt
        cnt += n*(n+m)
        F2 .= LinearIndices(zeros(n,n)) .+ cnt
        cnt += n*n
        ci = ci .+ n
    end
   
    # Stance and length constraints
    jac_stance = view(jac, nlp.cinds[4], :)
    
    t = 1
    for k = 1:nlp.T
        foot_ind = nlp.modes[k] == 1 ? 4 : 6
        jac_stance[t, xi[k][foot_ind]] = cnt
        t += 1
        cnt += 1
    end
    
    jac_length = view(jac, nlp.cinds[5], :)
    for k = 1:nlp.T
        jac_length[2*(k-1)+1, xi[k]] .= (1:n) .+ cnt
        cnt += n
        
        jac_length[2*(k-1)+2, xi[k]] .= (1:n) .+ cnt
        cnt += n
    end
    
    return jac
end

include("moi.jl")