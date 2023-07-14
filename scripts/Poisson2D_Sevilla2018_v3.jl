using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays
import Statistics: mean

function u_anal(x, y)
    α = 0.1; β = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return exp(α*sin(a*x + b*y) + β*(cos(c*x + d*y)))
end

function b_anal(x, y)
    A = 0.1; B = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4
    return (-A .* a .^ 2 .* sin(a .* x + b .* y) - A .* b .^ 2 .* sin(a .* x + b .* y) - B .* c .^ 2 .* cos(c .* x + d .* y) - B .* d .^ 2 .* cos(c .* x + d .* y) + (A .* a .* cos(a .* x + b .* y) - B .* c .* sin(c .* x + d .* y)) .^ 2 + (A .* b .* cos(a .* x + b .* y) - B .* d .* sin(c .* x + d .* y)) .^ 2) .* exp(A .* sin(a .* x + b .* y) + B .* cos(c .* x + d .* y))
end

function Poisson2D( T, b, Δ, BCx, BCy, TBC )
    k0 = 1.0
    kW, kE, kS, kN = 1.0, 1.0, 1.0, 1.0 
    ∂T∂yW, ∂T∂yE = 0.0, 0.0
    ∂T∂xS, ∂T∂xN = 0.0, 0.0
    if BCx==:Wc
        ∂T∂xW = (T[3] - (2*TBC-T[3]))/Δ.ξ
        ∂T∂xE = (T[4] - T[3])/Δ.ξ
    elseif BCx==:Wv 
        ∂T∂xW = (T[3] - TBC )/Δ.ξ
        ∂T∂xE = (T[4] - T[3])/Δ.ξ
    elseif BCx==:Ec
        ∂T∂xW = (T[3] - T[2])/Δ.ξ
        ∂T∂xE = ((2*TBC-T[3]) - T[3])/Δ.ξ 
    elseif BCx==:Ev
        ∂T∂xW = (T[3] - T[2])/Δ.ξ
        ∂T∂xE = (TBC  - T[3])/Δ.ξ 
    else
        ∂T∂xW = (T[3] - T[2])/Δ.ξ
        ∂T∂xE = (T[4] - T[3])/Δ.ξ
        ∂T∂yW = (T[8] - T[6])/Δ.η
        ∂T∂yE = (T[9] - T[7])/Δ.η
    end
    if BCy==:Sc
        ∂T∂yS = (T[3] -(2*TBC-T[3]))/Δ.η
        ∂T∂yN = (T[5] - T[3])/Δ.η
    elseif BCy==:Sv
        ∂T∂yS = (T[3] - TBC )/Δ.η
        ∂T∂yN = (T[5] - T[3])/Δ.η
    elseif BCy==:Nc
        ∂T∂yS = (T[3] - T[1])/Δ.η
        ∂T∂yN = ((2*TBC-T[3]) - T[3])/Δ.η  
    elseif BCy==:Nv
        ∂T∂yS = (T[3] - T[1])/Δ.η
        ∂T∂yN = (TBC  - T[3])/Δ.η  
    else
        ∂T∂yS = (T[3] - T[1])/Δ.η
        ∂T∂yN = (T[5] - T[3])/Δ.η
        ∂T∂xS = (T[7] - T[6])/Δ.ξ
        ∂T∂xN = (T[9] - T[8])/Δ.ξ
    end
    # Non-linearity
    # kW = k0*sqrt(∂T∂xW.^2 + ∂T∂yW.^2)
    # kE = k0*sqrt(∂T∂xE.^2 + ∂T∂yE.^2)
    # kS = k0*sqrt(∂T∂xS.^2 + ∂T∂yS.^2)
    # kN = k0*sqrt(∂T∂xN.^2 + ∂T∂yN.^2)
    # Fluxes
    qxW = -kW*∂T∂xW
    qxE = -kE*∂T∂xE
    qyS = -kS*∂T∂yS
    qyN = -kN*∂T∂yN
    # Balance
    f  = b + (qxE - qxW)/Δ.ξ + (qyN - qyS)/Δ.ξ
    return f
end

function ResidualFSG!(K, f, T, b, x, y, Δ, num, typ, Assemble)

    Tloc = zeros(9)
    ∂F∂T = zeros(9)
    nloc = zeros(Int64, 9)
    TBC  = 0.

    #-------------------------------------------#
    # Centroids
    for j in axes(f.c,2), i in axes(f.c,1)
        if i>1 && i<size(f.c,1) && j>1 && j<size(f.c,2)
            BCx   = :Internal
            BCy   = :Internal
            Tloc .= [  T.c[i,j-1],   T.c[i-1,j],   T.c[i,j],   T.c[i+1,j],   T.c[i,j+1],   T.v[i-1,j-1],   T.v[i,j-1],   T.v[i-1,j],   T.v[i,j]]
            nloc .= [num.c[i,j-1], num.c[i-1,j], num.c[i,j], num.c[i+1,j], num.c[i,j+1], num.v[i-1,j-1], num.v[i,j-1], num.v[i-1,j], num.v[i,j]]
            
            if typ.c[i-1,j]==2
                BCx = :Wc
                x0  = 0.5*(x.c[i-1] + x.c[i])
                y0  = y.c[j]
                TBC = u_anal(x0, y0)
            elseif typ.c[i+1,j]==2
                BCx = :Ec
                x0  = 0.5*(x.c[i+1] + x.c[i])
                y0  = y.c[j]
                TBC = u_anal(x0, y0)
            end
            if typ.c[i,j-1]==2
                BCy = :Sc
                x0  = x.c[i]
                y0  = 0.5*(y.c[j-1] + y.c[j])
                TBC = u_anal(x0, y0)
            elseif typ.c[i,j+1]==2
                BCy = :Nc
                x0  = x.c[i]
                y0  = 0.5*(y.c[j+1] + y.c[j])
                TBC = u_anal(x0, y0)
            end

            f.c[i,j] = Poisson2D(Tloc, b.c[i,j], Δ, BCx, BCy, TBC)

            if Assemble==true
                ∂F∂T .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(b.c[i,j]), Const(Δ), Const(BCx), Const(BCy), Const(TBC))
                for jeq in eachindex(nloc)
                    if abs(∂F∂T[jeq])>0.
                        K[nloc[3],nloc[jeq]] = ∂F∂T[jeq]
                    end
                end
            end
        elseif typ.c[i,j]==1
            f[i,j] = 0.0
            K[num.c[i,j], num.c[i,j]] = 1.0
        end
    end

    #-------------------------------------------#
    # Vertices
    for j in axes(f.v,2), i in axes(f.v,1)
        if i>1 && i<size(f.v,1) && j>1 && j<size(f.v,2)
            BCx   = :Internal
            BCy   = :Internal
            Tloc .= [  T.v[i,j-1],   T.v[i-1,j],   T.v[i,j],   T.v[i+1,j],   T.v[i,j+1],   T.c[i,j],   T.c[i+1,j],   T.c[i,j+1],   T.c[i+1,j+1]]
            nloc .= [num.v[i,j-1], num.v[i-1,j], num.v[i,j], num.v[i+1,j], num.v[i,j+1], num.c[i,j], num.c[i+1,j], num.c[i,j+1], num.c[i+1,j+1]]
        
            if typ.v[i-1,j]==1
                BCx = :Wv
                x0  = x.v[i-1]
                y0  = y.v[j]
                TBC = u_anal(x0, y0)
            elseif typ.v[i+1,j]==1
                BCx = :Ev
                x0  = x.v[i+1]
                y0  = y.v[j]
                TBC = u_anal(x0, y0)
            end

            if typ.v[i,j-1]==1
                BCy = :Sv
                x0  = x.v[i]
                y0  = y.v[j-1]
                TBC = u_anal(x0, y0)
            elseif typ.v[i,j+1]==1
                BCy = :Nv
                x0  = x.v[i]
                y0  = y.v[j+1]
                TBC = u_anal(x0, y0)
            end

            f.v[i,j] = Poisson2D(Tloc, b.v[i,j], Δ, BCx, BCy, TBC)

            if Assemble==true
                ∂F∂T .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(b.v[i,j]), Const(Δ), Const(BCx), Const(BCy), Const(TBC))
                for jeq in eachindex(nloc)
                    if abs(∂F∂T[jeq])>0.
                        K[nloc[3],nloc[jeq]] = ∂F∂T[jeq]
                    end
                end
            end
        elseif typ.v[i,j]==1
            f.v[i,j] = 0.0
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
    nc   = (x=500, y=500)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    x    = (c=LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nc.x+2), v=LinRange(xmin, xmax, nc.x+1))
    y    = (c=LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nc.y+2), v=LinRange(ymin, ymax, nc.y+1))
    σ    = 0.1
    T    = (c=zeros(nce...), v=zeros(nv...))
    f    = (c=zeros(nce...), v=zeros(nv...))
    b    = (c=zeros(nce...), v=zeros(nv...))
    T_an = (c=zeros(nce...), v=zeros(nv...))
    num  = (c=zeros(Int64, nce...), v=zeros(Int64, nv...))
    typ  = (c=zeros(Int64, nce...), v=zeros(Int64, nv...))
    typ.c .= 2
    typ.c[2:end-1,2:end-1] .= 0
    typ.v .= 1
    typ.v[2:end-1,2:end-1] .= 0
    b.c                    .= b_anal.(x.c, y.c')
    b.v                    .= b_anal.(x.v, y.v')
    T_an.c                 .= u_anal.(x.c, y.c')
    T_an.v                 .= u_anal.(x.v, y.v')
    T.c                    .= T_an.c
    T.v                    .= T_an.v
    num.c[2:end-1,2:end-1] .= reshape(1:(nc.x*nc.y),nc.x,nc.y)
    num.v                  .= reshape(1:(nv.x*nv.y),nv.x,nv.y) .+ maximum(num.c)

    for iter=1:20
        # Centroid solve
        ndof = nv.x*nv.y + nc.x*nc.y
        K    = ExtendableSparseMatrix(ndof, ndof)
        F    = zeros(ndof)
        δT   = zeros(ndof)
        ResidualFSG!(K, f, T, b, x, y, Δ, num, typ, true)
        @printf("Iter. %03d: %1.6e --- %1.6e\n", iter, mean(f.c), mean(f.v))
        if abs(mean(f.c))<1e-13 && abs(mean(f.v))<1e-13 break end
        F[1:nc.x*nc.y]     .= f.c[2:end-1,2:end-1][:]
        F[nc.x*nc.y+1:end] .= f.v[:]
        KJ  = dropzeros!(K.cscmatrix)
        # δT .= cholesky(Hermitian(KJ))\F
        δT .= KJ\F
        T.c[2:end-1,2:end-1] .-= δT[num.c[2:end-1,2:end-1]]
        T.v                  .-= δT[num.v]
        ResidualFSG!(K, f, T, b, x, y, Δ, num, typ, false)
        # @printf("%1.6e\n", mean(f.c))
        # @printf("%1.6e\n", mean(f.v))
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
    hm  = heatmap!(ax, x.c[2:end-1], y.c[2:end-1], T.c[2:end-1,2:end-1], colormap=:jet)
    Colorbar(fig[1, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    ax  = Axis(fig[2, 1], title = L"$T$ - vertices", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    hm  = heatmap!(ax, x.v, y.v, T.v, colormap=:jet)
    Colorbar(fig[2, 2], hm, width = 20, labelsize = 25, ticklabelsize = 14 )
    display(fig)
end 

main()