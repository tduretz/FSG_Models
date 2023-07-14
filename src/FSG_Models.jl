module FSG_Models

    using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays, MutableNamedTuples, IterativeSolvers

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
