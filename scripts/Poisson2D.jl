using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra, SparseArrays
import Statistics: mean

function Poisson2D( x, b, Δ, BCx, BCy, TBC )
    if BCx==:Wc
        qW = 0.0
        qE = (x[4] - x[3])/Δ.ξ
    elseif BCx==:Ec
        qW = (x[3] - x[2])/Δ.ξ
        qE = ((2*TBC-x[3]) - x[3])/Δ.ξ 
    else
        qW = (x[3] - x[2])/Δ.ξ
        qE = (x[4] - x[3])/Δ.ξ
    end
    if BCy==:Sc
        qS = 0.0
        qN = (x[5] - x[3])/Δ.η
    elseif BCy==:Nc
        qS = (x[3] - x[1])/Δ.η
        qN = ((2*TBC-x[3]) - x[3])/Δ.η  
    else
        qS = (x[3] - x[1])/Δ.η
        qN = (x[5] - x[3])/Δ.η
    end
    # @show qW, qS, BCx, BCy
    f  = b - (qE - qW)/Δ.ξ - (qN - qS)/Δ.ξ
    return f
end

function Residual!(K, f, T, b, Δ, num, Assemble)

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
            elseif i==size(f,1)-1
                BCx = :Ec
                TBC = 100.0
            end
            if j==2
                BCy = :Sc
            elseif j==size(f,2)-1
                BCy = :Nc
                TBC = 200.0
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
    
    xmin = -1.
    xmax =  1.
    ymin = -1.
    ymax =  1.
    nc   = (x=50, y=50)
    nv   = (x=nc.x+1, y=nc.y+1)
    nce  = (x=nc.x+2, y=nc.y+2)
    Δ    = (ξ=(xmax-xmin)/nc.x, η=(ymax-ymin)/nc.y)
    xv   = LinRange(xmin, xmax, nc.x+1)
    xc   = LinRange(xmin-Δ.ξ/2, xmax+Δ.ξ/2, nc.x+2)
    yv   = LinRange(ymin, ymax, nc.y+1)
    yc   = LinRange(ymin-Δ.η/2, ymax+Δ.η/2, nc.y+2)
    σ    = 0.1
    T    = (c=exp.(-xc.^2/σ^2) .+ exp.(-(yc').^2/σ^2), v=exp.(-xv.^2/σ^2) .+ exp.(-(yv').^2/σ^2))
    # k    = 1.0
    # q    = zeros(ncx+1, ncx+1)
    f    = (c=zeros(nce...), v=zeros(nv...))
    b    = (c=zeros(nce...), v=zeros(nv...))
    num  = (c=zeros(nce...), v=zeros(nv...))
    num.c[2:end-1,2:end-1] .= reshape(1:(nc.x*nc.y),nc.x,nc.y)
    num.v                  .= reshape(1:(nv.x*nv.y),nv.x,nv.y) 

    ndof = 0*nv.x*nv.y + nc.x*nc.y
    K    = ExtendableSparseMatrix(ndof, ndof)
    Residual!(K, f.c, T.c, b.c, Δ, num.c, true)
    @printf("%1.6e\n", mean(f.c))
    KJ = dropzeros!(K.cscmatrix)
    Kc = cholesky(Hermitian(KJ))
    @printf("%1.6e\n", mean(f.c))
    T.c[2:end-1,2:end-1] .-= reshape(Kc\f.c[2:end-1,2:end-1][:], nc...)

    Residual!(K, f.c, T.c, b.c, Δ, num.c, false)
    @printf("%1.6e\n", mean(f.c))

    res = 800
    fig = Figure(resolution = (res, res), fontsize=25)
    ax = Axis(fig[1, 1], title = L"$Vx$", xlabel = L"$x$ [km]", ylabel = L"$y$ [km]", aspect=1.0)
    heatmap!(ax, xc[2:end-1], yc[2:end-1], T.c[2:end-1,2:end-1])
    # spy!(ax, sparse(K))
    display(fig)

end 



main()