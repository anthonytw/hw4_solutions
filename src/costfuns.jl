struct RigidBodyCost{m,T}
    Qp::Diagonal{T,SVector{3,T}}
    w::T
    Qv::Diagonal{T,SVector{3,T}}
    Qω::Diagonal{T,SVector{3,T}}
    R::Diagonal{T,SVector{m,T}}
    p_ref::SVector{3,T}
    q_ref::SVector{4,T}
    v_ref::SVector{3,T}
    ω_ref::SVector{3,T}
    u_ref::SVector{m,T}
end

function stagecost(cost::RigidBodyCost, x, u)
    J = termcost(cost, x)
    du = u - cost.u_ref
    J += 0.5*du'cost.R*du
    return J
end
function termcost(cost::RigidBodyCost, x)
    p = SA[x[1], x[2], x[3]] - cost.p_ref
    q = SA[x[4],x[5],x[6],x[7]]
    v = SA[x[8], x[9], x[10]] - cost.v_ref
    ω = SA[x[11], x[12], x[13]] - cost.ω_ref

    J = 0.5*(p'cost.Qp*p + v'cost.Qv*v + ω'cost.Qω*ω)
    dq = cost.q_ref'q
    J += cost.w * min(1+dq, 1-dq)
    return J
end

function gradient(cost::RigidBodyCost, x, u)
    p = SA[x[1], x[2], x[3]] - cost.p_ref
    q = SA[x[4],x[5],x[6],x[7]]
    v = SA[x[8], x[9], x[10]] - cost.v_ref
    ω = SA[x[11], x[12], x[13]] - cost.ω_ref

    dq = cost.q_ref'q
    s = dq < 0 ? 1 : -1
    grad = [
        cost.Qp*p;
        cost.w*cost.q_ref*s;
        cost.Qv*v;
        cost.Qω*ω;
    ]
    return grad, cost.R*(u - cost.u_ref)
end

function hessian(cost::RigidBodyCost, x, u)
    Q = Diagonal([diag(cost.Qp); (@SVector zeros(4)); diag(cost.Qv); diag(cost.Qω)])
    return Q, cost.R
end

function evalcost(obj, X, U)
    J = zero(eltype(X[1]))
    for k = 1:length(U)
        J += stagecost(obj[k], X[k], U[k])
    end
    J += termcost(obj[end], X[end])
    return J
end