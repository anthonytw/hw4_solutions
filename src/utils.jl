function visualize!(vis, model::AbstractModel, x::StaticVector, addrobot::Bool=true)
    visualize!(vis, RBState(model, x), addrobot)
end

function visualize!(vis, x::RBState{<:Real}, addrobot::Bool=true)
    p = position(x)
    q = orientation(x)
    robot = addrobot ? vis["robot"] : vis
    settransform!(robot, compose(Translation(p), LinearMap(UnitQuaternion(q))))
end

function visualize!(vis, prob, X)
    visualize!(vis, prob.model, prob.times[end], X)
end

function visualize!(vis, model::AbstractModel, tf::Real, X)
    fps = Int(round((length(X)-1)/tf))
    anim = MeshCat.Animation(fps)
    n = state_dim(model)
    for (k,x) in enumerate(X)
        atframe(anim, k) do
            x = X[k]
            visualize!(vis, model, SVector{n}(x)) 
        end
    end
    setanimation!(vis, anim)
end


function slerp(qa::UnitQuaternion{T}, qb::UnitQuaternion{T}, t::T) where {T}
    # Borrowed from Quaternions.jl
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