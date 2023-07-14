using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays, MutableNamedTuples, IterativeSolvers
import Statistics: mean

#--------------------------------------------------------------------#

function dotavx(A, B)
    s = zero(promote_type(eltype(A), eltype(B)))
     for i in eachindex(A,B)
        s += A[i] * B[i]
    end
    s
end

function KSP_GCR!( x::Vector{Float64}, M::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Int64, Kf::SparseArrays.CHOLMOD.Factor{Float64}; restart=20)
    # KSP GCR solver
    norm_r, norm0 = 0.0, 0.0
    N               = length(x)
    maxit           = 10*restart
    ncyc, its       = 0, 0
    i1, i2, success = 0, 0, 0
    #
    f   = zeros(N) 
    v   = zeros(N) 
    s   = zeros(N)  
    val = zeros(N) 
    VV  = zeros(N,restart) 
    SS  = zeros(N,restart)    
    # Initial residual
    f     .= b .- M*x 
    norm_r = sqrt(dotavx( f, f ) )#norm(v)norm(f)
    norm0  = norm_r;
    # Solving procedure
     while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            s     .= Kf \ f
            # Action of Jacobian on s: v = J*s
             mul!(v, M, s)
            # Approximation of the Jv product
            for i2=1:i1
                val[i2] = dotavx( v, view(VV, :, i2 ) )   
            end
            # Scaling
            for i2=1:i1
                 v .-= val[i2] .* view(VV, :, i2 )
                 s .-= val[i2] .* view(SS, :, i2 )
            end
            # -----------------
            nrm_inv = 1.0 / sqrt(dotavx( v, v ) )
            r_dot_v = dotavx( f, v )  * nrm_inv
            # -----------------
             v     .*= nrm_inv
             s     .*= nrm_inv
            # -----------------
             x     .+= r_dot_v.*s
             f     .-= r_dot_v.*v
            # -----------------
            norm_r  = sqrt(dotavx( f, f ) )
            if norm_r/sqrt(length(f)) < eps 
                @printf("Converged @ GCR Iter. %04d: res. = %2.2e\n", its, norm_r/sqrt(length(f)))
                success = 1
                break
            end
            # Store 
             VV[:,i1] .= v
             SS[:,i1] .= s
            its      += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end
export KSP_GCR!

@views avWESN(A,B)  = 0.25.*(A[:,1:end-1] .+ A[:,2:end-0] .+ B[1:end-1,:] .+ B[2:end-0,:])

function u_anal(x, y)
    α = 0.1; β = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return exp(α*sin(a*x + b*y) + β*(cos(c*x + d*y)))
end

function b_anal(x, y)
    A = 0.1; B = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return (-A .* a .^ 2 .* sin(a .* x + b .* y) - A .* b .^ 2 .* sin(a .* x + b .* y) - B .* c .^ 2 .* cos(c .* x + d .* y) - B .* d .^ 2 .* cos(c .* x + d .* y) + (A .* a .* cos(a .* x + b .* y) - B .* c .* sin(c .* x + d .* y)) .^ 2 + (A .* b .* cos(a .* x + b .* y) - B .* d .* sin(c .* x + d .* y)) .^ 2) .* exp(A .* sin(a .* x + b .* y) + B .* cos(c .* x + d .* y))
end

function ComputeConductivity!( k, T, typ, val, k0, nc, Δ)
    for j in axes(k.ex,2), i in axes(k.ex,1)
        vBC = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
        tBC = (W=0.,        E=0.,        S=0.,        N=0.)
        if j>1 && j<nc.y+2
            TW = T.c[i,j];   TE = T.c[i+1,j]
            TS = T.v[i,j-1]; TN = T.v[i,j]
            if typ.c[i,j]==2
                tBC = (W=:Wc,        E=:Internal,    S=:Internal, N=:Internal)
                vBC = (W=val.c[i,j], E=0.,           S=0.,        N=0.)
            elseif typ.c[i+1,j]==2
                tBC = (W=:Internal,  E=:Ec,          S=:Internal, N=:Internal)
                vBC = (W=0.,         E=val.c[i+1,j], S=0.,        N=0.)
            end
            ∂T∂x, ∂T∂y = Gradient(TW, TE, TS, TN, Δ, tBC, vBC)
            k.ex[i,j]  = k0*sqrt(∂T∂x^2 + ∂T∂y^2)
        end
    end

    for j in axes(k.ey,2), i in axes(k.ey,1)
        vBC = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
        tBC = (W=0.,        E=0.,        S=0.,        N=0.)
        if i>1 && i<nc.x+2 
            TW = T.v[i-1,j]; TE = T.v[i,j]
            TS = T.c[i,j];   TN = T.c[i,j+1]
            if typ.c[i,j]==2
                tBC = (W=:Internal, E=:Internal, S=:Sc,          N=:Internal)
                vBC = (W=0.,        E=0.,        S=val.c[i,j],   N=0.)
            elseif typ.c[i,j+1]==2 
                tBC = (W=:Internal, E=:Internal, S=:Internal, N=:Nc)
                vBC = (W=0.,        E=0.,        S=0.,        N=val.c[i,j+1])
            end
            ∂T∂x, ∂T∂y = Gradient(TW, TE, TS, TN, Δ, tBC, vBC)
            k.ey[i,j]  = k0*sqrt(∂T∂x.^2 + ∂T∂y.^2)
        end
    end
    return
end

function Gradient(TW, TE, TS, TN, Δ, tBC, vBC)
    # West
    T_West = 0.; T_East = 0.
    if tBC.W==:Wc
        T_West = 2*vBC.W - TE
    elseif tBC.W==:Wv 
        T_West = vBC.W
    else
        T_West = TW
    end
    # East
    if tBC.E==:Ec
        T_East = 2*vBC.E - TW 
    elseif tBC.E==:Ev
        T_East = vBC.E 
    else
        T_East = TE
    end
    #--------------------------------
    # South
    T_South = 0.; T_North = 0.
    if tBC.S==:Sc
        T_South = 2*vBC.S - TN
    elseif tBC.S==:Sv
        T_South = vBC.S
    else
        T_South = TS
    end
    # North
    if tBC.N==:Nc
        T_North = 2*vBC.N - TS  
    elseif tBC.N==:Nv
        T_North = vBC.N  
    else
        T_North = TN
    end
    #--------------------------------
    # Gradients
    ∂T∂x = (T_East  - T_West )/Δ.ξ
    ∂T∂y = (T_North - T_South)/Δ.η
    return ∂T∂x, ∂T∂y
end

function Poisson2D( T, K, b, Δ, tBCW, tBCE, tBCS, tBCN, vBCW, vBCE, vBCS, vBCN, Frozen )
    k0 = 1.0
    kW, kE, kS, kN = 1.0, 1.0, 1.0, 1.0
    ∂T∂xW, ∂T∂xE = 0.0, 0.0
    ∂T∂yS, ∂T∂yN = 0.0, 0.0 
    ∂T∂yW, ∂T∂yE = 0.0, 0.0
    ∂T∂xS, ∂T∂xN = 0.0, 0.0
    ∂T∂xW, ∂T∂yW = Gradient(T[2], T[3], T[6], T[8], Δ, tBCW, vBCW)
    ∂T∂xE, ∂T∂yE = Gradient(T[3], T[4], T[7], T[9], Δ, tBCE, vBCE)
    ∂T∂xS, ∂T∂yS = Gradient(T[6], T[7], T[1], T[3], Δ, tBCS, vBCS)
    ∂T∂xN, ∂T∂yN = Gradient(T[8], T[9], T[3], T[5], Δ, tBCN, vBCN)
    # Non-linearity
    if Frozen
        kW = K[1]; kE = K[2]; kS = K[3]; kN = K[4]
    else
        kW = k0*sqrt(∂T∂xW.^2 + ∂T∂yW.^2)
        kE = k0*sqrt(∂T∂xE.^2 + ∂T∂yE.^2)
        kS = k0*sqrt(∂T∂xS.^2 + ∂T∂yS.^2)
        kN = k0*sqrt(∂T∂xN.^2 + ∂T∂yN.^2)
    end
    # Fluxes
    qxW = -kW*∂T∂xW
    qxE = -kE*∂T∂xE
    qyS = -kS*∂T∂yS
    qyN = -kN*∂T∂yN
    # Balance
    f  = b + (qxE - qxW)/Δ.ξ + (qyN - qyS)/Δ.ξ
    return f
end

function ResidualFSG!(K, f, T, b, k, Δ, num, val, typ, Assemble, Frozen)

    Tloc = zeros(9)
    ∂F∂T = zeros(9)
    kloc = zeros(4)
    nloc = zeros(Int64, 9)

    #-------------------------------------------#
    # Centroids
    for i in axes(f.c,1),  j in axes(f.c,2)
        if i>1 && i<size(f.c,1) && j>1 && j<size(f.c,2)

            tBCW  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCW  = (W=0., E=0., S=0., N=0.)
            tBCE  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCE  = (W=0., E=0., S=0., N=0.)
            tBCS  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCS  = (W=0., E=0., S=0., N=0.)
            tBCN  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCN  = (W=0., E=0., S=0., N=0.)
            Tloc .= [  T.c[i,j-1],   T.c[i-1,j],    T.c[i,j],   T.c[i+1,j],   T.c[i,j+1],   T.v[i-1,j-1],   T.v[i,j-1],   T.v[i-1,j],   T.v[i,j]]
            nloc .= [num.c[i,j-1], num.c[i-1,j],  num.c[i,j], num.c[i+1,j], num.c[i,j+1], num.v[i-1,j-1], num.v[i,j-1], num.v[i-1,j], num.v[i,j]]
            kloc .= [ k.ex[i-1,j],    k.ex[i,j], k.ey[i,j-1],    k.ey[i,j]]

            if typ.c[i-1,j]==2
                tBCW = (W=:Wc, E=:Internal, S=:Internal, N=:Internal)
                vBCW = (W=val.c[i-1,j], E=0., S=0., N=0.)
            elseif typ.c[i+1,j]==2
                tBCE = (W=:Internal, E=:Ec, S=:Internal, N=:Internal)
                vBCE = (W=0., E=val.c[i+1,j], S=0., N=0.)
            end
            if typ.c[i,j-1]==2
                tBCS = (W=:Internal, E=:Internal, S=:Sc, N=:Internal)
                vBCS = (W=0., E=0., S=val.c[i,j-1], N=0.)
            elseif typ.c[i,j+1]==2
                tBCN = (W=:Internal, E=:Internal, S=:Internal, N=:Nc)
                vBCN = (W=0., E=0., S=0., N=val.c[i,j+1])
            end

            f.c[i,j] = Poisson2D(Tloc, kloc, b.c[i,j], Δ, tBCW, tBCE, tBCS, tBCN, vBCW, vBCE, vBCS, vBCN, Frozen)

            if Assemble==true
                ∂F∂T .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(kloc), Const(b.c[i,j]), Const(Δ), Const(tBCW), Const(tBCE), Const(tBCS), Const(tBCN), Const(vBCW), Const(vBCE), Const(vBCS), Const(vBCN), Const(Frozen))
                for jeq in eachindex(nloc)
                    if abs(∂F∂T[jeq])>0.
                        K[nloc[3],nloc[jeq]] = ∂F∂T[jeq]
                    end
                end
            end
        elseif typ.c[i,j]==1
            K[num.c[i,j], num.c[i,j]] = 1.0
        end
    end

    #-------------------------------------------#
    # Vertices
    for j in axes(f.v,2), i in axes(f.v,1)
        if i>1 && i<size(f.v,1) && j>1 && j<size(f.v,2)
            
            tBCW  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCW  = (W=0., E=0., S=0., N=0.)
            tBCE  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCE  = (W=0., E=0., S=0., N=0.)
            tBCS  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCS  = (W=0., E=0., S=0., N=0.)
            tBCN  = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
            vBCN  = (W=0., E=0., S=0., N=0.)
            Tloc .= [  T.v[i,j-1],   T.v[i-1,j],   T.v[i,j],   T.v[i+1,j],   T.v[i,j+1],   T.c[i,j],   T.c[i+1,j],   T.c[i,j+1],   T.c[i+1,j+1]]
            nloc .= [num.v[i,j-1], num.v[i-1,j], num.v[i,j], num.v[i+1,j], num.v[i,j+1], num.c[i,j], num.c[i+1,j], num.c[i,j+1], num.c[i+1,j+1]]
            kloc .= [   k.ey[i,j],  k.ey[i+1,j],  k.ex[i,j],  k.ex[i,j+1]]

            if typ.v[i-1,j]==1
                BCx  = (W=:Wv, E=:Internal)
                TBC  = val.v[i-1,j]
                tBCW = (W=:Wv,          E=:Internal, S=:Internal, N=:Internal)
                vBCW = (W=val.v[i-1,j], E=0.,        S=0.,        N=0.)
            elseif typ.v[i+1,j]==1
                BCx  = (W=:Internal, E=:Ev) 
                TBC  = val.v[i+1,j]
                tBCE = (W=:Internal, E=:Ev,          S=:Internal, N=:Internal)
                vBCE = (W=0.,        E=val.v[i+1,j], S=0.,        N=0.)
            end

            if typ.v[i,j-1]==1
                BCy = (S=:Sv, N=:Internal)
                TBC = val.v[i,j-1]
                tBCS = (W=:Internal, E=:Internal, S=:Sv,          N=:Internal)
                vBCS = (W=0.,        E=0.,        S=val.v[i,j-1], N=0.)
            elseif typ.v[i,j+1]==1
                BCy = (S=:Internal, N=:Nv)
                TBC = val.v[i,j+1]
                tBCN = (W=:Internal, E=:Internal, S=:Internal, N=:Nv)
                vBCN = (W=0.,        E=0.,        S=0.,        N=val.v[i,j+1])
            end  

            f.v[i,j] = Poisson2D(Tloc, kloc, b.v[i,j], Δ, tBCW, tBCE, tBCS, tBCN, vBCW, vBCE, vBCS, vBCN, Frozen)

            if Assemble==true
                ∂F∂T .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(kloc), Const(b.c[i,j]), Const(Δ), Const(tBCW), Const(tBCE), Const(tBCS), Const(tBCN), Const(vBCW), Const(vBCE), Const(vBCS), Const(vBCN), Const(Frozen))
                for jeq in eachindex(nloc)
                    if abs(∂F∂T[jeq])>0.
                        K[nloc[3],nloc[jeq]] = ∂F∂T[jeq]
                    end
                end
            end
        elseif typ.v[i,j]==1
            K[num.v[i,j], num.v[i,j]] = 1.0
        end
    end

    #-------------------------------------------#
    # Finalise
    if Assemble==true flush!(K) end
end

function main()
    
    xmin =  0.
    xmax =  1.
    ymin =  0.
    ymax =  1.
    k0   =  1.
    nc   = (x=100, y=100)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    x    = (c=LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nc.x+2), v=LinRange(xmin, xmax, nc.x+1))
    y    = (c=LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nc.y+2), v=LinRange(ymin, ymax, nc.y+1))
    T    = (c=zeros(nce...), v=zeros(nv...))
    f    = (c=zeros(nce...), v=zeros(nv...))
    b    = (c=zeros(nce...), v=zeros(nv...))
    T_an = (c=zeros(nce...), v=zeros(nv...))
    num  = (c=zeros(Int64, nce...), v=zeros(Int64, nv...))
    typ  = (c=zeros(Int64, nce...), v=zeros(Int64, nv...))
    val  = (c=zeros(nce...), v=zeros(nv...))
    # Boundary conditions
    for j=1:nc.y+2
        typ.c[1,j]   = 2
        val.c[1,j]   = u_anal(x.v[1],   y.c[j])
        typ.c[end,j] = 2
        val.c[end,j] = u_anal(x.v[end], y.c[j])
    end
    for i=1:nc.x+2
        typ.c[i,1]   = 2
        val.c[i,1]   = u_anal(x.c[i], y.v[1])
        typ.c[i,end] = 2
        val.c[i,end] = u_anal(x.c[i], y.v[end])
    end
    for j=1:nv.y
        typ.v[1,j]   = 1
        val.v[1,j]   = u_anal(x.v[1],   y.v[j])
        typ.v[end,j] = 1
        val.v[end,j] = u_anal(x.v[end], y.v[j])
    end
    for i=1:nv.x
        typ.v[i,1]   = 1
        val.v[i,1]   = u_anal(x.v[i], y.v[1])
        typ.v[i,end] = 1
        val.v[i,end] = u_anal(x.v[i], y.v[end])
    end
    b.c                    .= b_anal.(x.c, y.c')
    b.v                    .= b_anal.(x.v, y.v')
    T_an.c                 .= u_anal.(x.c, y.c')
    T_an.v                 .= u_anal.(x.v, y.v')
    T.c                    .= T_an.c
    T.v                    .= T_an.v
    num.c[2:end-1,2:end-1] .= reshape(1:(nc.x*nc.y),nc.x,nc.y)
    num.v                  .= reshape(1:(nv.x*nv.y),nv.x,nv.y) .+ maximum(num.c)

    k = (ex=zeros(nv.x,nc.y+2), ey=zeros(nc.x+2, nv.y))

    ndof = nv.x*nv.y + nc.x*nc.y
    L    = ExtendableSparseMatrix(ndof, ndof)
    K    = ExtendableSparseMatrix(ndof, ndof)
    F    = zeros(ndof)
    δT   = zeros(ndof)
    Lc   = 0.

    for NLiter=1:20
        ComputeConductivity!( k, T, typ, val, k0, nc, Δ)
        ResidualFSG!(L, f, T, b, k, Δ, num, val, typ, true, true)
        NLerr = mean([norm(f.c)/sqrt(length(f.c)), norm(f.v)/sqrt(length(f.v))])
        @printf("Iter. %03d: %1.6e --- %1.6e\n", NLiter, abs(mean(f.c)), abs(mean(f.v)))
        ResidualFSG!(K, f, T, b, k, Δ, num, val, typ, true, false)
        if abs(mean(f.c))<1e-13 && abs(mean(f.v))<1e-13 break end
        F[1:nc.x*nc.y]     .= f.c[2:end-1,2:end-1][:]
        F[nc.x*nc.y+1:end] .= f.v[:]
        KJ  = dropzeros!(K.cscmatrix)
        LJ  = dropzeros!(L.cscmatrix)        
        if NLiter==1
            Lc  = cholesky(Hermitian(LJ))
        else
            cholesky!(Lc, LJ; check = false)
        end
        tol = NLerr/200
        KSP_GCR!( δT, KJ, F, tol, 1, Lc; restart=10 )
        T.c[2:end-1,2:end-1] .-= δT[num.c[2:end-1,2:end-1]]
        T.v                  .-= δT[num.v]
    end

    # ------------------------------------- #
    # Errors
    @show norm(T.c[2:end-1,2:end-1] .- T_an.c[2:end-1,2:end-1])/norm(T.c[2:end-1,2:end-1])
    @show norm(T.v[2:end-1,2:end-1] .- T_an.v[2:end-1,2:end-1])/norm(T.v[2:end-1,2:end-1])
    
    # ------------------------------------- #
    # Visu
    res = 800
    fig = Figure(resolution = (res, res), fontsize=25)
    ax  = Axis(fig[1, 1], title = L"$T$ - centroids", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    hm  = heatmap!(ax, x.c[2:end-1], y.c[2:end-1], T.c[2:end-1,2:end-1] .- 0.0.*T_an.c[2:end-1,2:end-1], colormap=:jet)
    Colorbar(fig[1, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    ax  = Axis(fig[2, 1], title = L"$T$ - vertices", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    hm  = heatmap!(ax, x.v, y.v, T.v .- 0.0.*T_an.v, colormap=:jet)
    Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    ComputeConductivity!( k, T, typ, val, k0, nc, Δ)
    keff = avWESN( k.ey[2:end-1,:], k.ex[:,2:end-1])
    ax  = Axis(fig[3, 1], title = L"$k$ - avg", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    hm  = heatmap!(ax, x.v, y.v, keff, colormap=:jet)
    Colorbar(fig[3, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    
    display(fig)
end 

@time main()