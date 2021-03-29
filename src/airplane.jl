using MeshCat
using CoordinateTransformations
using GeometryBasics
using Colors

function set_mesh!(vis, ::YakPlane; color=nothing, scale=0.15)
    meshfile = joinpath(@__DIR__,"..","data","piper","piper_pa18.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","cirrus","Cirrus.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","piper","piper_scaled.obj")
    jpg = joinpath(@__DIR__,"..","data","piper","piper_diffuse.jpg")
    if isnothing(color)
        img = PngImage(jpg)
        texture = Texture(image=img)
        # mat = MeshLambertMaterial(map=texture) 
        mat = MeshPhongMaterial(map=texture) 
    else
        mat = MeshPhongMaterial(color=color)
    end
    obj = MeshFileGeometry(meshfile)
    setobject!(vis["robot"]["geom"], obj, mat)
    settransform!(vis["robot"]["geom"], compose(Translation(0,0,0.07),LinearMap( RotY(pi/2)*RotZ(-pi/2) * scale)))
end