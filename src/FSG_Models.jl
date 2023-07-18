module FSG_Models

    using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays, MutableNamedTuples, IterativeSolvers

    function JacobianFSG(nc)
        Jacobian = (;
        ξ = (∂x = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)),
             ∂y = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)) ),
        η = (∂x = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)),
             ∂y = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)) ),
        )
        return Jacobian
    end
    export JacobianFSG

    @views avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])
    export avWESN

    include("FSG_Solvers.jl")
    export KSP_GCR!
    include("FSG_Poisson.jl")
    export SetBoundaryConditions!
    export EvaluateConductivity, ComputeConductivity!
    export Poisson2D, ResidualFSG!
    include("FSG_Gradient.jl")
    export GradientLoc1
    include("FSG_MeshDeformation.jl")
    export InverseJacobian!, CopyJacobian!
    export ComputeForwardTransformation!
    export FreeSurfaceDiscretisation, dhdx_num
    export Mesh_x, Mesh_y

end # module FSG_Models
