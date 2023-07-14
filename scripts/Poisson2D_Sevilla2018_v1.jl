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

function Poisson2D( x, b, Δ, BCx, BCy, TBC )
    if BCx==:Wc
        qW = (x[3] - (2*TBC-x[3]))/Δ.ξ
        qE = (x[4] - x[3])/Δ.ξ
    elseif BCx==:Ec
        qW = (x[3] - x[2])/Δ.ξ
        qE = ((2*TBC-x[3]) - x[3])/Δ.ξ 
    else
        qW = (x[3] - x[2])/Δ.ξ
        qE = (x[4] - x[3])/Δ.ξ
    end
    if BCy==:Sc
        qS = (x[3] -(2*TBC-x[3]))/Δ.η
        qN = (x[5] - x[3])/Δ.η
    elseif BCy==:Nc
        qS = (x[3] - x[1])/Δ.η
        qN = ((2*TBC-x[3]) - x[3])/Δ.η  
    else
        qS = (x[3] - x[1])/Δ.η
        qN = (x[5] - x[3])/Δ.η
    end
    f  = b - (qE - qW)/Δ.ξ - (qN - qS)/Δ.ξ
    return f
end

function Residual!(K, f, T, b, xc, yc, Δ, num, Assemble)

    x   = zeros(5)
    dx  = zeros(5)
    neq = zeros(Int64,5)
    TBC = 0.

    for j in axes(f,2), i in axes(f,1)
        if i>1 && i<size(f,1) && j>1 && j<size(f,2)
            BCx = :Internal
            BCy = :Internal
            x   .= [  T[i,j-1],   T[i-1,j],   T[i,j],   T[i+1,j],   T[i,j+1]]
            neq .= [num[i,j-1], num[i-1,j], num[i,j], num[i+1,j], num[i,j+1]]
            if i==2
                BCx = :Wc
                x0  = 0.5*(xc[i-1] + xc[i])
                y0  = yc[j]
                TBC = u_anal(x0, y0)
            elseif i==size(f,1)-1
                BCx = :Ec
                x0  = 0.5*(xc[i+1] + xc[i])
                y0  = yc[j]
                TBC = u_anal(x0, y0)
            end
            if j==2
                BCy = :Sc
                x0  = xc[i]
                y0  = 0.5*(yc[j-1] + yc[j])
                TBC = u_anal(x0, y0)
            elseif j==size(f,2)-1
                BCy = :Nc
                x0  = xc[i]
                y0  = 0.5*(yc[j+1] + yc[j])
                TBC = u_anal(x0, y0)
            end
            f[i,j] = Poisson2D(x, b[i,j], Δ, BCx, BCy, TBC)

            if Assemble==true
                dx .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(x, dx), Const(b[i,j]), Const(Δ), Const(BCx), Const(BCy), Const(TBC))
                for jeq=1:5
                    if abs(dx[jeq])>0.
                        K[neq[3],neq[jeq]] = dx[jeq]
                    end
                end
            end
        end
    end
    if Assemble==true flush!(K) end
end

function main()
    
    xmin =  0.
    xmax =  1.
    ymin =  0.
    ymax =  1.
    nc   = (x=10, y=10)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    xv   = LinRange(xmin, xmax, nc.x+1)
    xc   = LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nc.x+2)
    yv   = LinRange(ymin, ymax, nc.y+1)
    yc   = LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nc.y+2)
    σ    = 0.1
    T    = (c=exp.(-xc.^2/σ^2) .+ exp.(-(yc').^2/σ^2), v=exp.(-xv.^2/σ^2) .+ exp.(-(yv').^2/σ^2))
    f    = (c=zeros(nce...), v=zeros(nv...))
    b    = (c=zeros(nce...), v=zeros(nv...))
    num  = (c=zeros(nce...), v=zeros(nv...))
    T_an = (c=zeros(nce...), v=zeros(nv...))
    b.c                    .= b_anal.(xc, yc')
    T_an.c                 .= u_anal.(xc, yc')
    num.c[2:end-1,2:end-1] .= reshape(1:(nc.x*nc.y),nc.x,nc.y)
    num.v                  .= reshape(1:(nv.x*nv.y),nv.x,nv.y) 

    # Centroid solve
    ndof = 0*nv.x*nv.y + nc.x*nc.y
    K    = ExtendableSparseMatrix(ndof, ndof)
    Residual!(K, f.c, T.c, b.c, xc, yc, Δ, num.c, true)
    @printf("%1.6e\n", mean(f.c))
    KJ = dropzeros!(K.cscmatrix)
    T.c[2:end-1,2:end-1] .-= reshape(KJ\f.c[2:end-1,2:end-1][:], nc...)
    Residual!(K, f.c, T.c, b.c, xc, yc, Δ, num.c, false)
    @printf("%1.6e\n", mean(f.c))

    # ------------------------------------- #
    res = 800
    fig = Figure(resolution = (res, res), fontsize=25)
    ax = Axis(fig[1, 1], title = L"$Vx$", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    heatmap!(ax, xc[2:end-1], yc[2:end-1], T.c[2:end-1,2:end-1], colormap=:jet)
    display(fig)

    @show norm(T.c[2:end-1,2:end-1] .- T_an.c[2:end-1,2:end-1])/norm(T.c[2:end-1,2:end-1])
end 

main()