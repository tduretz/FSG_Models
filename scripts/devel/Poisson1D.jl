using Enzyme, CairoMakie, ExtendableSparse, Printf, LinearAlgebra
import Statistics: mean

function Poisson1D( x, b, Δx, BC, TBC )
    if BC==:West
        qW = 0.0
        qE = (x[3] - x[2])/Δx
    elseif BC==:East
        qW = (x[2] - x[1])/Δx
        qE = ((2*TBC-x[2]) - x[2])/Δx 
    else
        qW = (x[2] - x[1])/Δx
        qE = (x[3] - x[2])/Δx
    end
    f  = b - (qE - qW)/Δx
    return f
end

function Residual!(K, f, T, b, Δx, Assemble)

    x   = zeros(3)
    dx  = zeros(3)
    ncx = length(f)
    TBC = 100.0

    for i in eachindex(f)
        x .= [T[i], T[i+1], T[i+2]]
        if i==1
            BC = :West
        elseif i==ncx
            BC = :East
        else
            BC = :Internal
        end
        f[i] = Poisson1D(x, b[i], Δx, BC, TBC)
        if Assemble==true
            dx .= 0.
            autodiff(Enzyme.Reverse, Poisson1D, Duplicated(x, dx), Const(b[i]), Const(Δx), Const(BC), Const(TBC))
            for j=1:3
                if abs(dx[j])>0.
                    K[i,i+j-2] = dx[j]
                end
            end
        end
    end
    if Assemble==true flush!(K) end
end

function main()

    # Configure
    xmin = -1.
    xmax =  1.
    ncx  = 100
    Δx   = (xmax-xmin)/ncx
    xv   = LinRange(xmin, xmax, ncx+1)
    xc   = LinRange(xmin-Δx/2, xmax+Δx/2, ncx+2)
    f    = zeros(ncx)
    b    = zeros(ncx)
    σ    = 0.1
    T    = exp.(-xc.^2/σ^2)
    b    = 100*exp.(-(xc[2:end-1].-0.1).^2/σ^2)
    K    = ExtendableSparseMatrix(ncx, ncx)

    # Evaluate residual and assemble
    Residual!(K, f, T, b, Δx, true)
    @printf("Res. = %1.6e\n", mean(f))

    # Assemble
    T[2:end-1] .-= cholesky(K)\f

    # Evaluate residual 
    Residual!(K, f, T, b, Δx, false)
    @printf("Res. = %1.6e\n", mean(f))

    # Visualisation
    res = 800
    fig = Figure(resolution = (res, res), fontsize=25)
    ax = Axis(fig[1, 1], title = L"$T$", xlabel = L"$x$ [-]", ylabel = L"$y$ [-]", aspect=1.0)
    lines!(ax, xc[2:end-1], T[2:end-1])
    lines!(ax, xc[2:end-1], b)
    lines!(ax, xc[2:end-1], f)
    display(fig)

end 

main()