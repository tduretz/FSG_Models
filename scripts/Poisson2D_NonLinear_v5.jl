using FSG_Models
using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays, MutableNamedTuples, IterativeSolvers, Makie.GeometryBasics, MathTeXEngine
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
    α = 0.1; β = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return exp(α*sin(a*x + b*y) + β*(cos(c*x + d*y)))
end

function q_anal(x, y)
    A = 0.1; B = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    qx = (-A .* a .* cos(a .* x + b .* y) + B .* c .* sin(c .* x + d .* y)) .* exp(A .* sin(a .* x + b .* y) + B .* cos(c .* x + d .* y))
    qy = (-A .* b .* cos(a .* x + b .* y) + B .* d .* sin(c .* x + d .* y)) .* exp(A .* sin(a .* x + b .* y) + B .* cos(c .* x + d .* y))
    return qx, qy
end

function b_anal(x, y)
    A = 0.1; B = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return (-A .* a .^ 2 .* sin(a .* x + b .* y) - A .* b .^ 2 .* sin(a .* x + b .* y) - B .* c .^ 2 .* cos(c .* x + d .* y) - B .* d .^ 2 .* cos(c .* x + d .* y) + (A .* a .* cos(a .* x + b .* y) - B .* c .* sin(c .* x + d .* y)) .^ 2 + (A .* b .* cos(a .* x + b .* y) - B .* d .* sin(c .* x + d .* y)) .^ 2) .* exp(A .* sin(a .* x + b .* y) + B .* cos(c .* x + d .* y))
end

function main()
    
    xmin       =  0.0
    xmax       =  1.0
    ymin       =  0.0
    ymax       =  1.0
    params     = (k0=1.0, n=1.0, kBC=1.0)
    BC         = (W=:Neumann, E=:Neumann, S=:Dirichlet, N=:Dirichlet)
    adapt_mesh = true
    options    = (; 
        free_surface = false,
        swiss_x      = true,
        swiss_y      = true,
        topo         = false,
    )
    nc   = (x=20, y=10)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    nve  = (x=nv.x+2, y=nv.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    x    = (c=LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nce.x), v=LinRange(xmin-Δ.ξ, xmax+Δ.ξ, nve.x))
    y    = (c=LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nce.y), v=LinRange(ymin-Δ.η, ymax+Δ.η, nve.y))
    xxv, yyv    = LinRange(xmin-Δ.ξ, xmax+Δ.ξ, 2nc.x+5), LinRange(ymin-Δ.η, ymax+Δ.η, 2nc.y+5)
    (xn, yn) = ([x for x=xxv,y=yyv], [y for x=xxv,y=yyv])     
    msh0 = (; ξ = copy(xn), η = copy(yn))
    msh  = (v=(x=xn[1:2:end,1:2:end], y=yn[1:2:end,1:2:end]), c=(x=xn[2:2:end-1,2:2:end-1], y=yn[2:2:end-1,2:2:end-1]), ex=(x=xn[1:2:end,2:2:end-1], y=yn[1:2:end,2:2:end-1]), ey=(x=xn[2:2:end-1,1:2:end], y=yn[2:2:end-1,1:2:end]))
    T    = (c=zeros(nce...), v=zeros(nve...))
    f    = (c=zeros(nce...), v=zeros(nve...))
    b    = (c=zeros(nce...), v=zeros(nve...))
    T_an = (c=zeros(nce...), v=zeros(nve...))
    num  = (c=zeros(Int64, nce...), v=zeros(Int64, nve...))
    typ  = (c=zeros(Int64, nce...), v=zeros(Int64, nve...))
    val  = (c=zeros(nce...), v=zeros(nve...))
    k    = (ex=zeros(nv.x,nc.y+2), ey=zeros(nc.x+2, nv.y))
    q    = (x=(ex=zeros(nv.x,nc.y+2), ey=zeros(nc.x+2, nv.y)), y=(ex=zeros(nv.x,nc.y+2), ey=zeros(nc.x+2, nv.y)))
    ∂    = JacobianFSG(nc)
    # Mesh
    if adapt_mesh
        x0     = (xmax + xmin)/2.
        m      = ymin 
        σx     = 4.
        σy     = 4.
        Amp    = 0.0
        hn     = Amp.*exp.(-(xn.-x0).^2 ./ σx^2) 
        # Deform mesh
        X_msh  = zeros(2)
        for i in eachindex(msh0.ξ)          
            X_msh[1] = msh0.ξ[i]
            X_msh[2] = msh0.η[i] 
            xn[i]    = Mesh_x( X_msh, x0, m, xmin, xmax, σx, options; h=hn[i] )
            yn[i]    = Mesh_y( X_msh, x0, m, ymin, ymax, σy, options; h=hn[i] )
        end
        # Compute slope
        if options.topo 
            hx = -dhdx_num(xn, yn, Δ)
        end
        # # Compute forward transformation
        ∂x     = (∂ξ= ones(size(xn)), ∂η = zeros(size(yn)) )
        ∂y     = (∂ξ=zeros(size(xn)), ∂η =  ones(size(yn)) )
        ComputeForwardTransformation!(∂x, ∂y, xn, yn, Δ)
        # Solve for inverse transformation
        ∂ξ = (∂x=ones(size(yn)), ∂y=zeros(size(yn))); ∂η = (∂x=zeros(size(yn)), ∂y=ones(size(yn)))
        InverseJacobian!(∂ξ,∂η,∂x,∂y)
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
    # Initialise fields
    SetBoundaryConditions!(typ, val, BC, msh, nce, nve, u_anal, q_anal)
    typ.v
    b.c                    .= b_anal.(x.c, y.c')
    b.v                    .= b_anal.(x.v, y.v')
    T_an.c                 .= u_anal.(x.c, y.c')
    T_an.v                 .= u_anal.(x.v, y.v')
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
    while ( NLerr>1e-10 && NLiter<nitmax)
        NLiter += 1
        ComputeConductivity!( k, q, T, typ, val, params, nc, ∂, Δ)
        ResidualFSG!(L, f, T, b, k, q, ∂, Δ, num, val, typ, params, true, true)
        ResidualFSG!(K, f, T, b, k, q, ∂, Δ, num, val, typ, params, true, false)
        NLerr = mean([norm(f.c)/sqrt(length(f.c)), norm(f.v)/sqrt(length(f.v))]); NLerrs[NLiter] = NLerr
        @printf("Iter. %03d: %1.6e (%1.6e --- %1.6e)\n", NLiter, NLerr, abs(mean(f.c)), abs(mean(f.v)))
        F[1:nc.x*nc.y]     .= f.c[2:end-1,2:end-1][:]
        F[nc.x*nc.y+1:end] .= f.v[2:end-1,2:end-1][:]
        KJ  = dropzeros!(K.cscmatrix)
        LJ  = dropzeros!(L.cscmatrix)        
        if NLiter==1
            Lc  = cholesky(Hermitian(LJ))
        else
            cholesky!(Lc, LJ; check = false)
        end
        if (NLerr<1e-10) break end
        tol = NLerr/500
        KSP_GCR!( δT, KJ, F, tol, 1, Lc; restart=10 )
        # δT =  K\F
        T.c[2:end-1,2:end-1] .-= δT[num.c[2:end-1,2:end-1]]
        T.v[2:end-1,2:end-1] .-= δT[num.v[2:end-1,2:end-1]]
    end

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
    minmax = (minimum(T.c[2:end-1,2:end-1]), maximum(T.c[2:end-1,2:end-1]))
    poly!(ax, pc, color = T.c[2:end-1,2:end-1][:], colormap = :jet, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    Colorbar(fig[1, 2], colormap = :jet, flipaxis = true, size = 10, colorrange=minmax )    
    # xlims!(ax, xmin, xmax); ylims!(ax, ymin, ymax)
    # hm  = heatmap!(ax, x.c[2:end-1], y.c[2:end-1], T.c[2:end-1,2:end-1] .- 0.0.*T_an.c[2:end-1,2:end-1], colormap=:jet)
    # Colorbar(fig[1, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    # ----
    ax  = Axis(fig[1, 3], title = L"$T$ - vertices", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    minmax = (minimum(T.v[2:end-1,2:end-1]), maximum(T.v[2:end-1,2:end-1]))
    # hm  = heatmap!(ax, x.v[2:end-1], y.v[2:end-1], T.v[2:end-1,2:end-1] .- 0.0.*T_an.v[2:end-1,2:end-1], colormap=:jet)
    # Colorbar(fig[1, 4], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    poly!(ax, pv, color = T.v[2:end-1,2:end-1][:], colormap = :jet, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    Colorbar(fig[1, 4], colormap = :jet, flipaxis = true, size = 10, colorrange=minmax )    
    # xlims!(ax, xmin, xmax); ylims!(ax, ymin, ymax)
    # ----
    ComputeConductivity!( k, q, T, typ, val, params, nc, ∂, Δ)
    keff = avWESN( k.ey[2:end-1,:], k.ex[:,2:end-1])
    ax  = Axis(fig[2, 1], title = L"$k$ - avg", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    minmax = (minimum(keff), maximum(keff)+1e-2)
    poly!(ax, pc, color = keff[:], colormap = :jet, strokewidth = 0, strokecolor = :white, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect=:image, colorrange=minmax)#, colorrange=limits
    Colorbar(fig[2, 2], colormap = :jet, flipaxis = true, size = 10, colorrange=minmax )    
    # hm  = heatmap!(ax, x.v, y.v, keff, colormap=:jet)
    # Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    # ----
    ax  = Axis(fig[2, 3], title = L"$$Convergence", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    hm  = lines!(ax, 1:NLiter, log10.(NLerrs[1:NLiter]))
    display(fig)
end 

@time main()