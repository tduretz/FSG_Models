    
@views function FreeSurfaceDiscretisation(ηy, ∂ξ, ∂η, hx )
    # Topography # See python notebook v5
    η_surf   = ηy[:,end]
    dkdx     = ∂ξ.∂x[1:2:end, end-1]
    dkdy     = ∂ξ.∂y[1:2:end, end-1]
    dedx     = ∂η.∂x[1:2:end, end-1]
    dedy     = ∂η.∂y[1:2:end, end-1]
    h_x      = hx[1:2:end]
    eta      = η_surf
    free_surface_params = (; # removed factor dz since we apply it directly to strain rates
        ∂Vx∂∂Vx∂x = (-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedy .* dkdx .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂∂Vy∂x = (dedx .* dkdy .* h_x .^ 2 .+ 2 * dedx .* dkdy .- dedy .* dkdx .* h_x .^ 2 .- 2 * dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vx∂P     = (3 // 2) .* (dedx .* h_x .^ 2 .- dedx .+ 2 .* dedy .* h_x) ./ (eta .* (2 .* dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 .* dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 .* dedy .^ 2)),
        ∂Vy∂∂Vx∂x = (.-2 * dedx .* dkdy .* h_x .^ 2 .- dedx .* dkdy .+ 2 * dedy .* dkdx .* h_x .^ 2 .+ dedy .* dkdx) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂∂Vy∂x = (.-2 * dedx .* dkdx .* h_x .^ 2 .- dedx .* dkdx .- 2 * dedx .* dkdy .* h_x .- dedy .* dkdy .* h_x .^ 2 .- 2 * dedy .* dkdy) ./ (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2),
        ∂Vy∂P     = (3 // 2) .* (2 * dedx .* h_x .- dedy .* h_x .^ 2 .+ dedy) ./ (eta .* (2 * dedx .^ 2 .* h_x .^ 2 .+ dedx .^ 2 .+ 2 * dedx .* dedy .* h_x .+ dedy .^ 2 .* h_x .^ 2 .+ 2 * dedy .^ 2)),
    )
    return free_surface_params
end

###############################


@views function CopyJacobian!(∂, ∂ξ, ∂η)
    ∂.ξ.∂x.ex .= ∂ξ.∂x[1:2:end-0,2:2:end-1]
    ∂.ξ.∂x.ey .= ∂ξ.∂x[2:2:end-1,1:2:end-0]
    ∂.ξ.∂x.c  .= ∂ξ.∂x[2:2:end-1,2:2:end-1]
    ∂.ξ.∂x.v  .= ∂ξ.∂x[1:2:end-0,1:2:end-0]
    ∂.ξ.∂y.ex .= ∂ξ.∂y[1:2:end-0,2:2:end-1]
    ∂.ξ.∂y.ey .= ∂ξ.∂y[2:2:end-1,1:2:end-0]
    ∂.ξ.∂y.c  .= ∂ξ.∂y[2:2:end-1,2:2:end-1]
    ∂.ξ.∂y.v  .= ∂ξ.∂y[1:2:end-0,1:2:end-0]
    ∂.η.∂x.ex .= ∂η.∂x[1:2:end-0,2:2:end-1]
    ∂.η.∂x.ey .= ∂η.∂x[2:2:end-1,1:2:end-0]
    ∂.η.∂x.c  .= ∂η.∂x[2:2:end-1,2:2:end-1]
    ∂.η.∂x.v  .= ∂η.∂x[1:2:end-0,1:2:end-0]
    ∂.η.∂y.ex .= ∂η.∂y[1:2:end-0,2:2:end-1]
    ∂.η.∂y.ey .= ∂η.∂y[2:2:end-1,1:2:end-0]
    ∂.η.∂y.c  .= ∂η.∂y[2:2:end-1,2:2:end-1]
    ∂.η.∂y.v  .= ∂η.∂y[1:2:end-0,1:2:end-0]
    return nothing
end

###############################

function Mesh_x( X, x0, m, xmin0, xmax0, σx, options; h=0.0 )
if options.swiss_x
        xmin1 = (sinh.( σx.*(xmin0.-x0) ))
        xmax1 = (sinh.( σx.*(xmax0.-x0) ))
        sx    = (xmax0-xmin0)/(xmax1-xmin1)
        x     = (sinh.( σx.*(X[1].-x0) )) .* sx  .+ x0
    else
        x = X[1]
    end
    return x
end

###############################

function Mesh_y( X, y0, m, ymin0, ymax0, σy, options; h=0.0 )
    # y0    = ymax0
    y     = X[2]
    if options.swiss_y
        ymin1 = (sinh.( σy.*(ymin0.-y0) ))
        ymax1 = (sinh.( σy.*(ymax0.-y0) ))
        sy    = (ymax0-ymin0)/(ymax1-ymin1)
        y     = (sinh.( σy.*(X[2].-y0) )) .* sy  .+ y0
    end
    if options.topo
        z0    = -h                     # topography height
        y     = (y/ymin0)*((z0+m))-z0  # shift grid vertically
    end   
    return y
end

###############################

@views function dhdx_num(xv4, yv4, Δ)
    #  ∂h∂ξ * ∂ξ∂x + ∂h∂η * ∂η∂x
    ∂h∂x = zero(xv4[:,end])
    ∂h∂x[2:2:end-1] .= (yv4[3:2:end-0,end-1] .- yv4[1:2:end-2,end-1])./Δ.ξ
    ∂h∂x[3:2:end-2] .= (yv4[4:2:end-1,end-1] .- yv4[2:2:end-3,end-1])./Δ.ξ
    ∂h∂x[[1 end]]   .= ∂h∂x[[2 end-1]]   
    return ∂h∂x
end

 ###############################

 function ComputeForwardTransformation!(∂x, ∂y, xv4, yv4, Δ)
    
    ∂x.∂ξ[2:2:end-1,:] .= (xv4[3:2:end-0,:] .- xv4[1:2:end-2,:])./Δ.ξ
    ∂x.∂ξ[3:2:end-2,:] .= (xv4[4:2:end-1,:] .- xv4[2:2:end-3,:])./Δ.ξ
    ∂x.∂ξ[[1 end],:]   .= ∂x.∂ξ[[2 end-1],:]
    # ----------------------
    ∂x.∂η[:,2:2:end-1] .= (xv4[:,3:2:end-0] .- xv4[:,1:2:end-2])./Δ.η
    ∂x.∂η[:,3:2:end-2] .= (xv4[:,4:2:end-1] .- xv4[:,2:2:end-3])./Δ.η
    ∂x.∂η[:,[1 end]]   .= ∂x.∂η[:,[2 end-1]]
    # ----------------------
    ∂y.∂ξ[2:2:end-1,:] .= (yv4[3:2:end-0,:] .- yv4[1:2:end-2,:])./Δ.ξ
    ∂y.∂ξ[3:2:end-2,:] .= (yv4[4:2:end-1,:] .- yv4[2:2:end-3,:])./Δ.ξ
    ∂y.∂ξ[[1 end],:]   .= ∂y.∂ξ[[2 end-1],:]  
    # ----------------------
    ∂y.∂η[:,2:2:end-1] .= (yv4[:,3:2:end-0] .- yv4[:,1:2:end-2])./Δ.η
    ∂y.∂η[:,3:2:end-2] .= (yv4[:,4:2:end-1] .- yv4[:,2:2:end-3])./Δ.η
    ∂y.∂η[:,[1 end]]   .= ∂y.∂η[:,[2 end-1]]    
    # ----------------------
    # @printf("min(∂x∂ξ) = %1.6f --- max(∂x∂ξ) = %1.6f\n", minimum(∂x.∂ξ), maximum(∂x.∂ξ))
    # @printf("min(∂x∂η) = %1.6f --- max(∂x∂η) = %1.6f\n", minimum(∂x.∂η), maximum(∂x.∂η))
    # @printf("min(∂y∂ξ) = %1.6f --- max(∂y∂ξ) = %1.6f\n", minimum(∂y.∂ξ), maximum(∂y.∂ξ))
    # @printf("min(∂y∂η) = %1.6f --- max(∂y∂η) = %1.6f\n", minimum(∂y.∂η), maximum(∂y.∂η))
    return nothing
end

##########################################
    
function InverseJacobian!(∂ξ,∂η,∂x,∂y)
    M = zeros(2,2)
    @time for i in eachindex(∂ξ.∂x)
        M[1,1]   = ∂x.∂ξ[i]
        M[1,2]   = ∂x.∂η[i]
        M[2,1]   = ∂y.∂ξ[i]
        M[2,2]   = ∂y.∂η[i]
        # display(M)
        invJ     = inv(M)
        ∂ξ.∂x[i] = invJ[1,1]
        ∂ξ.∂y[i] = invJ[1,2]
        ∂η.∂x[i] = invJ[2,1]
        ∂η.∂y[i] = invJ[2,2]
    end
    @printf("min(∂ξ∂x) = %1.6f --- max(∂ξ∂x) = %1.6f\n", minimum(∂ξ.∂x), maximum(∂ξ.∂x))
    @printf("min(∂ξ∂y) = %1.6f --- max(∂ξ∂y) = %1.6f\n", minimum(∂ξ.∂y), maximum(∂ξ.∂y))
    @printf("min(∂η∂x) = %1.6f --- max(∂η∂x) = %1.6f\n", minimum(∂η.∂x), maximum(∂η.∂x))
    @printf("min(∂η∂y) = %1.6f --- max(∂η∂y) = %1.6f\n", minimum(∂η.∂y), maximum(∂η.∂y))
    return nothing
end