#Walker Dynamics
Base.@kwdef struct Walker
    g::Float64 = 9.81
    mb::Float64 = 5.0
    mf::Float64 = 1.0
    ℓ_min::Float64 = 0.5
    ℓ_max::Float64 = 1.5
end
RobotDynamics.state_dim(::Walker) = 12
RobotDynamics.control_dim(::Walker) = 3
g = 9.81
mb = 5.0 #body mass
mf = 1.0 #foot mass
ℓ_min = 0.5 #minimum length
ℓ_max = 1.5 #maximum length

function stance1_dynamics(x,u)
    #Foot 1 is in contact
    M = Diagonal([mb mb mf mf mf mf])
    
    rb  = x[1:2]   # position of the body
    rf1 = x[3:4]   # position of foot 1
    rf2 = x[5:6]   # position of foot 2
    v   = x[7:12]  # velocities
    
    ℓ1x = (rb[1]-rf1[1])/norm(rb-rf1)
    ℓ1y = (rb[2]-rf1[2])/norm(rb-rf1)
    ℓ2x = (rb[1]-rf2[1])/norm(rb-rf2)
    ℓ2y = (rb[2]-rf2[2])/norm(rb-rf2)
    
    B = [ℓ1x  ℓ2x  ℓ1y-ℓ2y;
         ℓ1y  ℓ2y  ℓ2x-ℓ1x;
          0    0     0;
          0    0     0;
          0  -ℓ2x  ℓ2y;
          0  -ℓ2y -ℓ2x]
    
    v̇ = [0; -g; 0; 0; 0; -g] + M\(B*u)
    
    ẋ = [v; v̇]
end

function stance2_dynamics(x,u)
    #Foot 2 is in contact
    M = Diagonal([mb mb mf mf mf mf])
    
    rb  = x[1:2]
    rf1 = x[3:4]
    rf2 = x[5:6]
    v   = x[7:12]
    
    ℓ1x = (rb[1]-rf1[1])/norm(rb-rf1)
    ℓ1y = (rb[2]-rf1[2])/norm(rb-rf1)
    ℓ2x = (rb[1]-rf2[1])/norm(rb-rf2)
    ℓ2y = (rb[2]-rf2[2])/norm(rb-rf2)
    
    B = [ℓ1x  ℓ2x  ℓ1y-ℓ2y;
         ℓ1y  ℓ2y  ℓ2x-ℓ1x;
        -ℓ1x   0  -ℓ1y;
        -ℓ1y   0   ℓ1x;
          0    0    0;
          0    0    0]
    
    v̇ = [0; -g; 0; -g; 0; 0] + M\(B*u)
    
    ẋ = [v; v̇]
end

function stance1_dynamics_rk4(x,u)
    #RK4 integration with zero-order hold on u
    f1 = stance1_dynamics(x, u)
    f2 = stance1_dynamics(x + 0.5*h*f1, u)
    f3 = stance1_dynamics(x + 0.5*h*f2, u)
    f4 = stance1_dynamics(x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function stance2_dynamics_rk4(x,u)
    #RK4 integration with zero-order hold on u
    f1 = stance2_dynamics(x, u)
    f2 = stance2_dynamics(x + 0.5*h*f1, u)
    f3 = stance2_dynamics(x + 0.5*h*f2, u)
    f4 = stance2_dynamics(x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function jump1_map(x)
    #Foot 1 experiences inelastic collision
    xn = [x[1:8]; 0.0; 0.0; x[11:12]]
    return xn
end

function jump2_map(x)
    #Foot 2 experiences inelastic collision
    xn = [x[1:10]; 0.0; 0.0]
    return xn
end

function stage_cost(x,u,k)
    return 0.5*((x-xref[:,k])'*Q*(x-xref[:,k])) + 0.5*(u-uref)'*R*(u-uref)
end
function terminal_cost(x)
    return 0.5*((x-xref[:,end])'*Qn*(x-xref[:,end]))
end

function cost(ztraj)
    z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    xtraj = [z[1:Nx,:] ztraj[end-(Nx-1):end]]
    utraj = z[(Nx+1):(Nx+Nu),:]
    J = 0.0
    for k = 1:(Nt-1)
        J += stage_cost(xtraj[:,k],utraj[:,k],k)
    end
    J += terminal_cost(xtraj[:,end])
    return J
end

function dynamics_constraint!(c,ztraj)
    d = reshape(view(c,c_dyn_inds),Nx,Nt-1)
    z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    xtraj = [z[1:Nx,:] ztraj[end-(Nx-1):end]]
    utraj = z[(Nx+1):(Nx+Nu),:]
    for k = 1:(Nmodes-1)
        if mod(k,2) == 1
            for j = 1:(Nm-1)
                s = (k-1)*Nm + j
                d[:,s] = stance1_dynamics_rk4(xtraj[:,s],utraj[:,s]) - xtraj[:,s+1]
            end
            s = k*Nm
            d[:,s] = jump2_map(stance1_dynamics_rk4(xtraj[:,s],utraj[:,s])) - xtraj[:,s+1]
        else
            for j = 1:(Nm-1)
                s = (k-1)*Nm + j
                d[:,s] = stance2_dynamics_rk4(xtraj[:,s],utraj[:,s]) - xtraj[:,s+1]
            end
            s = k*Nm
            d[:,s] = jump1_map(stance2_dynamics_rk4(xtraj[:,s],utraj[:,s])) - xtraj[:,s+1]
        end
    end
    if mod(Nmodes,2) == 1
        for j = 1:(Nm-1)
            s = (Nmodes-1)*Nm + j
            d[:,s] = stance1_dynamics_rk4(xtraj[:,s],utraj[:,s]) - xtraj[:,s+1]
        end
    else
        for j = 1:(Nm-1)
            s = (Nmodes-1)*Nm + j
            d[:,s] = stance2_dynamics_rk4(xtraj[:,s],utraj[:,s]) - xtraj[:,s+1]
        end
    end
    
    return nothing
end

function stance_constraint!(c,ztraj)
    d = view(c,c_stance_inds)
    z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    xtraj = [z[1:Nx,:] ztraj[(end-(Nx-1)):end]]
    t = 1
    for k = 1:Nmodes
        if mod(k,2) == 1
            for j = 1:Nm
                s = (k-1)*Nm + j
                d[t] = xtraj[4,s]  # keep foot on the floor
                t += 1
            end
        else
            for j = 1:Nm
                s = (k-1)*Nm + j
                d[t] = xtraj[6,s]
                t += 1
            end
        end
            
    end
    return nothing
end

function length_constraint!(c,ztraj)
    d = view(c,c_length_inds)
    z = reshape(ztraj[1:(end-Nx)],Nx+Nu,Nt-1)
    xtraj = [z[1:Nx,:] ztraj[(end-(Nx-1)):end]]
    for k = 1:Nt
        d[2*(k-1)+1] = norm(xtraj[1:2,k] - xtraj[3:4,k])
        d[2*(k-1)+2] = norm(xtraj[1:2,k] - xtraj[5:6,k])
    end
end

function con!(c,ztraj)
    c[c_init_inds] .= ztraj[1:Nx] - xref[:,1] #initial state constraint
    c[c_term_inds] .= ztraj[(end-(Nx-1)):end] - xref[:,end] #terminal state constraint
    @views dynamics_constraint!(c,ztraj)
    @views stance_constraint!(c,ztraj)
    @views length_constraint!(c,ztraj)
end
