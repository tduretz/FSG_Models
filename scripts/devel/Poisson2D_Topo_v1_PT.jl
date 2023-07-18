using FSG_Models
using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays, MutableNamedTuples, IterativeSolvers, Makie.GeometryBasics, MathTeXEngine, JLD2
import Statistics: mean
Makie.update_theme!( fonts = ( regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Flux BC OK for centroids
# Add ghost nodes to vertex arrays

function JacobianFSG(nc)
    Jacobian = (;
    ξ = (∂x = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)),
         ∂y = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)) ),
    η = (∂x = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)),
         ∂y = (ex=zeros(nc.x+3, nc.y+2), ey=zeros(nc.x+2, nc.y+3), v=zeros(nc.x+3, nc.y+3), c=zeros(nc.x+2, nc.y+2)) ),
    )
return Jacobian
end

#--------------------------------------------------------------------#

function u_anal(x, y) 
    return abs(1.0*x*(y+0.45))
end

function q_anal(x, y) 
    qx = 1.
    qy = 1.
    return qx, qy
end

function b_anal(x, y)
    sx, sy = 0.1, 0.1
    return 0*exp.( -(x-0.5)^2/sx^2 - (y-0.5)^2/sy)
end

function main()
    
    xmin       =  -0.5
    xmax       =  0.5
    ymin       = -1.0
    ymax       =  0.0
    params     = (k0=1.0, n=0.0, kBC=1.0)
    BC         = (W=:Dirichlet, E=:Dirichlet, S=:Dirichlet, N=:Dirichlet)
    adapt_mesh = true
    options    = (; 
        free_surface = false,
        swiss_x      = false,
        swiss_y      = false,
        topo         = true,
    )
    nc   = (x=40, y=40)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    nve  = (x=nv.x+2, y=nv.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    x    = (c=LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nce.x), v=LinRange(xmin-Δ.ξ, xmax+Δ.ξ, nve.x))
    y    = (c=LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nce.y), v=LinRange(ymin-Δ.η, ymax+Δ.η, nve.y))
    xxv, yyv = LinRange(xmin-Δ.ξ, xmax+Δ.ξ, 2nc.x+5), LinRange(ymin-Δ.η, ymax+Δ.η, 2nc.y+5)
    (xn, yn) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])     
    msh0 = (; ξ = copy(xn), η = copy(yn))
    T    = (c=zeros(nce...), v=zeros(nve...))
    f    = (c=zeros(nce...), v=zeros(nve...))
    b    = (c=zeros(nce...), v=zeros(nve...))
    T_an = (c=zeros(nce...), v=zeros(nve...))
    val  = (c=zeros(nce...), v=zeros(nve...))
    num  = (c=zeros(Int64, nce...), v=zeros(Int64, nve...))
    typ  = (c=zeros(Int64, nce...), v=zeros(Int64, nve...))
    k    = (ex=zeros(nve.x, nce.y), ey=zeros(nce.x, nve.y))
    q    = (x=(ex=zeros(nve.x, nce.y), ey=zeros(nce.x, nve.y)), y=(ex=zeros(nve.x, nce.y), ey=zeros(nce.x, nve.y)))
    ∂    = JacobianFSG(nc)
    # Mesh
    if adapt_mesh
        @load "/Users/tduretz/REPO/FullStagerredGrid/TinyTests/PoissonFSE_40x40.jld2" xn1 yn1 ∂1 ∂x1 ∂y1
        
        x0     = (xmax + xmin)/2.
        m      = ymin 
        σx     = 0.1
        σy     = 1.
        Amp    = 0.25
        hn     = Amp.*exp.(-(xn.-x0).^2 ./ σx^2) 
        hn[1:4,:] .= 0.;  hn[end-3:end,:] .= 0.         # Quick fix to allow E/W Neumann with topo /  cancel deformation on the sides else different ghost points treatment is needed 
        # Deform mesh
        X_msh  = zeros(2)
        for i in eachindex(msh0.ξ)          
            X_msh[1] = msh0.ξ[i]
            X_msh[2] = msh0.η[i] 
            xn[i]    = Mesh_x( X_msh, x0, x0, m, xmin, xmax, Amp, σx, σy, options )
            yn[i]    = Mesh_y( X_msh, x0, x0, m, ymin, ymax, Amp, σx, σy, options )
        end
        # Compute slope
        if options.topo 
            hx = -dhdx_num(xn, yn, Δ)
        end
        # Compute forward transformation
        ∂x     = (∂ξ=zeros(size(xn)), ∂η=zeros(size(yn)) )
        ∂y     = (∂ξ=zeros(size(xn)), ∂η=zeros(size(yn)) )
        # xgrad   = zeros(2)
        # ygrad   = zeros(2)
        # for i in eachindex(msh0.ξ)          
        #     X_msh[1] = msh0.ξ[i]
        #     X_msh[2] = msh0.η[i]
        #     xgrad .= 0.
        #     ygrad .= 0.
        #     autodiff(Enzyme.Reverse, Mesh_x, Duplicated(X_msh, xgrad), Const(x0), Const(x0), Const(m), Const(ymin), Const(ymax), Const(Amp), Const(σx), Const(σy), Const(options) )
        #     autodiff(Enzyme.Reverse, Mesh_y, Duplicated(X_msh, ygrad), Const(x0), Const(x0), Const(m), Const(ymin), Const(ymax), Const(Amp), Const(σx), Const(σy), Const(options) )
        #     ∂x.∂ξ[i] = xgrad[1]
        #     ∂x.∂η[i] = xgrad[2]
        #     ∂y.∂ξ[i] = ygrad[1]
        #     ∂y.∂η[i] = ygrad[2]
        # end
        
        ComputeForwardTransformation!(∂x, ∂y, xn, yn, Δ)

        
    ∂y.∂ξ[2:end-1,2:end-1] .= ∂y1.∂ξ
    ∂y.∂ξ[2:end-1,2:end-1] .= ∂y1.∂ξ
    ∂y.∂ξ[2:end-1,2:end-1] .= ∂y1.∂ξ
    ∂y.∂ξ[2:end-1,2:end-1] .= ∂y1.∂ξ

    ∂x.∂ξ[2:end-1,2:end-1] .= ∂x1.∂ξ
    ∂x.∂ξ[2:end-1,2:end-1] .= ∂x1.∂ξ
    ∂x.∂ξ[2:end-1,2:end-1] .= ∂x1.∂ξ
    ∂x.∂ξ[2:end-1,2:end-1] .= ∂x1.∂ξ

    ∂y.∂η[2:end-1,2:end-1] .= ∂y1.∂η
    ∂y.∂η[2:end-1,2:end-1] .= ∂y1.∂η
    ∂y.∂η[2:end-1,2:end-1] .= ∂y1.∂η
    ∂y.∂η[2:end-1,2:end-1] .= ∂y1.∂η

    ∂x.∂η[2:end-1,2:end-1] .= ∂x1.∂η
    ∂x.∂η[2:end-1,2:end-1] .= ∂x1.∂η
    ∂x.∂η[2:end-1,2:end-1] .= ∂x1.∂η
    ∂x.∂η[2:end-1,2:end-1] .= ∂x1.∂η




        @show minimum(∂x.∂ξ)
        @show maximum(∂x.∂ξ)
        @show minimum(∂x.∂η)
        @show maximum(∂x.∂η)
        @show minimum(∂y.∂ξ)
        @show maximum(∂y.∂η)
        @show minimum(∂y.∂η)
        @show maximum(∂y.∂η)
        # Solve for inverse transformation
        ∂ξ = (∂x=zeros(size(yn)), ∂y=zeros(size(yn))); ∂η = (∂x=zeros(size(yn)), ∂y=zeros(size(yn)))
        InverseJacobian!(∂ξ, ∂η, ∂x, ∂y)
        CopyJacobian!(∂, ∂ξ, ∂η)
    else
        # Coordinate transformation
        ∂.ξ.∂x.ex .=  ones(nc.x+3, nc.y+2)
        ∂.ξ.∂x.ey .=  ones(nc.x+2, nc.y+3)
        ∂.ξ.∂x.c  .=  ones(nc.x+2, nc.y+2)
        ∂.ξ.∂x.v  .=  ones(nc.x+3, nc.y+3)
        ∂.η.∂y.ex .=  ones(nc.x+3, nc.y+2)
        ∂.η.∂y.ey .=  ones(nc.x+2, nc.y+3)
        ∂.η.∂y.c  .=  ones(nc.x+2, nc.y+2)
        ∂.η.∂y.v  .=  ones(nc.x+3, nc.y+3)
    end


    # println( norm(xn[2:end-1,2:end-1]       .- xn1))
    # println( norm(yn[2:end-1,2:end-1]       .- yn1))


    # println( norm(∂x.∂ξ[2:end-1,2:end-1] .- ∂x1.∂ξ))
    # println( norm(∂x.∂ξ[2:end-1,2:end-1] .- ∂x1.∂ξ))
    # println( norm(∂x.∂ξ[2:end-1,2:end-1] .- ∂x1.∂ξ))
    # println( norm(∂x.∂ξ[2:end-1,2:end-1] .- ∂x1.∂ξ))

    # println( norm(∂x.∂η[2:end-1,2:end-1] .- ∂x1.∂η))
    # println( norm(∂x.∂η[2:end-1,2:end-1] .- ∂x1.∂η))
    # println( norm(∂x.∂η[2:end-1,2:end-1] .- ∂x1.∂η))
    # println( norm(∂x.∂η[2:end-1,2:end-1] .- ∂x1.∂η))

    # println( norm(∂y.∂ξ[2:end-1,2:end-1] .- ∂y1.∂ξ))
    # println( norm(∂y.∂ξ[2:end-1,2:end-1] .- ∂y1.∂ξ))
    # println( norm(∂y.∂ξ[2:end-1,2:end-1] .- ∂y1.∂ξ))
    # println( norm(∂y.∂ξ[2:end-1,2:end-1] .- ∂y1.∂ξ))

    # println( norm(∂y.∂η[2:end-1,2:end-1] .- ∂y1.∂η))
    # println( norm(∂y.∂η[2:end-1,2:end-1] .- ∂y1.∂η))
    # println( norm(∂y.∂η[2:end-1,2:end-1] .- ∂y1.∂η))
    # println( norm(∂y.∂η[2:end-1,2:end-1] .- ∂y1.∂η))

    # println( norm(∂.ξ.∂y.ex[2:end-1,:]      .- ∂1.ξ.∂y.x))
    # println( norm(∂.ξ.∂y.ey[:,2:end-1]      .- ∂1.ξ.∂y.y))
    # println( norm(∂.ξ.∂y.v[2:end-1,2:end-1] .- ∂1.ξ.∂y.v))
    # println( norm(∂.ξ.∂y.c[:,:]             .- ∂1.ξ.∂y.c))

    println( norm(∂.η.∂x.ex[2:end-1,:]      .- ∂1.η.∂x.x))
    println( norm(∂.η.∂x.ey[:,2:end-1]      .- ∂1.η.∂x.y))
    println( norm(∂.η.∂x.v[2:end-1,2:end-1] .- ∂1.η.∂x.v))
    println( norm(∂.η.∂x.c[:,:]             .- ∂1.η.∂x.c))

    # println( norm(∂.η.∂y.ex[2:end-1,:]      .- ∂1.η.∂y.x))
    # println( norm(∂.η.∂y.ey[:,2:end-1]      .- ∂1.η.∂y.y))
    # println( norm(∂.η.∂y.v[2:end-1,2:end-1] .- ∂1.η.∂y.v))
    # println( norm(∂.η.∂y.c[:,:]             .- ∂1.η.∂y.c))


    # println( norm(∂.ξ.∂x.ex[2:end-1,:]      .- ∂1.ξ.∂x.x))
    # println( norm(∂.ξ.∂x.ey[:,2:end-1]      .- ∂1.ξ.∂x.y))
    # println( norm(∂.ξ.∂x.v[2:end-1,2:end-1] .- ∂1.ξ.∂x.v))
    # println( norm(∂.ξ.∂x.c[:,:]             .- ∂1.ξ.∂x.c))

    # println( norm(∂.ξ.∂y.ex[2:end-1,:]      .- ∂1.ξ.∂y.x))
    # println( norm(∂.ξ.∂y.ey[:,2:end-1]      .- ∂1.ξ.∂y.y))
    # println( norm(∂.ξ.∂y.v[2:end-1,2:end-1] .- ∂1.ξ.∂y.v))
    # println( norm(∂.ξ.∂y.c[:,:]             .- ∂1.ξ.∂y.c))

    println( norm(∂.η.∂x.ex[2:end-1,:]      .- ∂1.η.∂x.x))
    println( norm(∂.η.∂x.ey[:,2:end-1]      .- ∂1.η.∂x.y))
    println( norm(∂.η.∂x.v[2:end-1,2:end-1] .- ∂1.η.∂x.v))
    println( norm(∂.η.∂x.c[:,:]             .- ∂1.η.∂x.c))
    @show size(f.c)
    @show size(∂.η.∂x.c)
    @show size(T.c)

    # println( norm(∂.η.∂y.ex[2:end-1,:]      .- ∂1.η.∂y.x))
    # println( norm(∂.η.∂y.ey[:,2:end-1]      .- ∂1.η.∂y.y))
    # println( norm(∂.η.∂y.v[2:end-1,2:end-1] .- ∂1.η.∂y.v))
    # println( norm(∂.η.∂y.c[:,:]             .- ∂1.η.∂y.c))

    # Initialise fields
    msh  = (v=(x=xn[1:2:end,1:2:end], y=yn[1:2:end,1:2:end]), c=(x=xn[2:2:end-1,2:2:end-1], y=yn[2:2:end-1,2:2:end-1]), ex=(x=xn[1:2:end,2:2:end-1], y=yn[1:2:end,2:2:end-1]), ey=(x=xn[2:2:end-1,1:2:end], y=yn[2:2:end-1,1:2:end]))
    # @show size(msh.ex.x)
    # @show size(k.ex)
    # k.ex .= 0.0001
    # k.ey .= 0.0001
    # k.ex[((msh.ex.x).^2 .+ (msh.ex.y .+ 0.5).^2) .< 0.01] .= 10.
    # k.ey[((msh.ey.x).^2 .+ (msh.ey.y .+ 0.5).^2) .< 0.01] .= 10.
    # k.ex[msh.ex.x.>0] .= 1
    # k.ey[msh.ey.x.>0] .= 1
    # @show maximum(k.ex)
    SetBoundaryConditions!(typ, val, BC, msh, nce, nve, u_anal, q_anal)
    b.c                    .= b_anal.(msh.c.x, msh.c.y)
    b.v                    .= b_anal.(msh.v.x, msh.v.y)
    T_an.c                 .= u_anal.(msh.c.x, msh.c.y)
    T_an.v                 .= u_anal.(msh.v.x, msh.v.y)
    T.c                    .= T_an.c
    T.v                    .= T_an.v
    num.c[2:end-1,2:end-1] .= reshape(1:(nc.x*nc.y),nc.x,nc.y)
    num.v[2:end-1,2:end-1] .= reshape(1:(nv.x*nv.y),nv.x,nv.y) .+ maximum(num.c)
    # Initialise sparse matrices and vectors
    ndof = nv.x*nv.y + nc.x*nc.y
    L    = ExtendableSparseMatrix(ndof, ndof)
    K    = ExtendableSparseMatrix(ndof, ndof)
    F    = zeros(ndof)
    δT   = zeros(ndof)
    Lc   = 0.
    # display( reverse(num.c', dims=1))
    # display( reverse(num.v', dims=1))
    # display( reverse(typ.c', dims=1))
    # display( reverse(typ.v', dims=1))
    nitmax = 20
    NLerrs = zeros(nitmax) 
    NLiter = 0; NLerr  = 1.0

    θ = nc.x/nc.x
    dt =  max(Δ...)^2/2.0 /1000
    dTdt    = (c=zeros(nce...), v=zeros(nve...))

    dt = 7.812499999999945e-5 
    θ = 0.025

    for iter=1:10000
        ResidualFSG!(K, f, T, b, k, q, ∂, Δ, num, val, typ, params, true, false)
        dTdt.c .= (1-θ) .* dTdt.c .+ f.c 
        dTdt.v .= (1-θ) .* dTdt.v .+ f.v
        T.c[2:end-1,2:end-1]   .-= dt.*dTdt.c[2:end-1,2:end-1]
        T.v[2:end-1,2:end-1]   .-= dt.*dTdt.v[2:end-1,2:end-1]
        if mod(iter, 1000)==0
            println(norm(f.c), " ", norm(f.v))
        end
    end


    # while ( NLerr>1e-10 && NLiter<nitmax)
    #     NLiter += 1
    #     # ComputeConductivity!( k, q, T, typ, val, params, nc, ∂, Δ)
    #     # ResidualFSG!(L, f, T, b, k, q, ∂, Δ, num, val, typ, params, true, true)
    #     ResidualFSG!(K, f, T, b, k, q, ∂, Δ, num, val, typ, params, true, false)
    #     NLerr = mean([norm(f.c)/sqrt(length(f.c)), norm(f.v)/sqrt(length(f.v))]); NLerrs[NLiter] = NLerr
    #     @printf("Iter. %03d: %1.6e (%1.6e --- %1.6e)\n", NLiter, NLerr, norm(f.c)/sqrt(length(f.c)), norm(f.v)/sqrt(length(f.v)))
    #     F[1:nc.x*nc.y]     .= f.c[2:end-1,2:end-1][:]
    #     F[nc.x*nc.y+1:end] .= f.v[2:end-1,2:end-1][:]
    #     KJ  = dropzeros!(K.cscmatrix)
    #     # LJ  = dropzeros!(L.cscmatrix)        
    #     # if NLiter==1
    #     #     Lc  = cholesky(Hermitian(LJ))
    #     # else
    #     #     cholesky!(Lc, LJ; check = false)
    #     # end
    #     # if (NLerr<1e-10) break end
    #     # tol = NLerr/500
    #     # KSP_GCR!( δT, KJ, F, tol, 1, Lc; restart=30 )
    #     # Kc = cholesky(KJ'*KJ)
    #     # δT =  (Kc\(KJ'*F))
    #     δT =  K\F
    #     T.c[2:end-1,2:end-1] .-= δT[num.c[2:end-1,2:end-1]]
    #     T.v[2:end-1,2:end-1] .-= δT[num.v[2:end-1,2:end-1]]
    # end

    # ------------------------------------- #
    # Errors
    @show norm(T.c[2:end-1,2:end-1] .- T_an.c[2:end-1,2:end-1])/norm(T.c[2:end-1,2:end-1])
    @show norm(T.v[2:end-1,2:end-1] .- T_an.v[2:end-1,2:end-1])/norm(T.v[2:end-1,2:end-1])
    # ------------------------------------- #
    # Post-process
    X = (v=(x=xn[3:2:end-2,3:2:end-2], y=yn[3:2:end-2,3:2:end-2]), c=(x=xn[2:2:end-1,2:2:end-1], y=yn[2:2:end-1,2:2:end-1]))
    cell_vertx = [  X.v.x[1:end-1,1:end-1][:]  X.v.x[2:end-0,1:end-1][:]  X.v.x[2:end-0,2:end-0][:]  X.v.x[1:end-1,2:end-0][:] ] 
    cell_verty = [  X.v.y[1:end-1,1:end-1][:]  X.v.y[2:end-0,1:end-1][:]  X.v.y[2:end-0,2:end-0][:]  X.v.y[1:end-1,2:end-0][:] ] 
    node_vertx = [  X.c.x[1:end-1,1:end-1][:]  X.c.x[2:end-0,1:end-1][:]  X.c.x[2:end-0,2:end-0][:]  X.c.x[1:end-1,2:end-0][:] ] 
    node_verty = [  X.c.y[1:end-1,1:end-1][:]  X.c.y[2:end-0,1:end-1][:]  X.c.y[2:end-0,2:end-0][:]  X.c.y[1:end-1,2:end-0][:] ] 
    pc = [Polygon( Point2f0[ (cell_vertx[i,j], cell_verty[i,j]) for j=1:4] ) for i in 1:nc.x*nc.y]
    pv = [Polygon( Point2f0[ (node_vertx[i,j], node_verty[i,j]) for j=1:4] ) for i in 1:nv.x*nv.y]
    # Visu
    res = 800
    fig = Figure(resolution = (res, res), fontsize=25)
    # ----
    ax  = Axis(fig[1, 1], title = L"$T$ - centroids", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    v = -f.c[2:end-1,2:end-1]
    minmax = (minimum(v), maximum(v)+1e-2)
    # poly!(ax, pc, color = v[:], colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    # Colorbar(fig[1, 2], colormap = :turbo, flipaxis = true, size = 10, colorrange=minmax )    
    # xlims!(ax, xmin, xmax); ylims!(ax, ymin, ymax)
    hm  = heatmap!(ax, x.c[2:end-1], y.c[2:end-1], v .- 0.0.*T_an.c[2:end-1,2:end-1], colormap=:jet)
    Colorbar(fig[1, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    # ----
    # ax  = Axis(fig[1, 3], title = L"$T$ - vertices", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    # v = -f.v[2:end-1,2:end-1]
    # minmax = (minimum(v), maximum(v)+1e-2)
    # poly!(ax, pv, color = v[:], colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    # Colorbar(fig[1, 4], colormap = :turbo, flipaxis = true, size = 10, colorrange=minmax )    
    # # xlims!(ax, xmin, xmax); ylims!(ax, ymin, ymax)
    # # hm  = heatmap!(ax, x.v[2:end-1], y.v[2:end-1], v .- 0.0.*T_an.v[2:end-1,2:end-1], colormap=:jet)
    # # Colorbar(fig[1, 4], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    # # ----
    # # ComputeConductivity!( k, q, T, typ, val, params, nc, ∂, Δ)
    # keff = avWESN( k.ey[2:end-1,2:end-1], k.ex[2:end-1,2:end-1])
    # ax  = Axis(fig[2, 1], title = L"$k$ - avg", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    # minmax = (minimum(keff), maximum(keff)+1e-2)
    # poly!(ax, pc, color = keff[:], colormap = :turbo, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    # Colorbar(fig[2, 2], colormap = :turbo, flipaxis = true, size = 10, colorrange=minmax )    
    # # hm  = heatmap!(ax, x.v, y.v, keff, colormap=:jet)
    # # Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    # # ----
    # ax  = Axis(fig[2, 3], title = L"$$Convergence", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    # hm  = lines!(ax, 1:NLiter, log10.(NLerrs[1:NLiter]))
    display(fig)
end 

@time main()