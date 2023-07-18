function GradientLoc1(u, Δ, tBC, vBC; k=(;W=1.0, E=1.0, S=1.0, N=1.0), BC_order=2, ∂ξ=(∂x=1.0, ∂y=0.0), ∂η=(∂x=0.0, ∂y=1.0))
    # West
    u_West = 0.; u_East = 0.
    if tBC.W==2 
        u_West = 2*vBC.W - u.E
    elseif tBC.W==4
        u_West = Δ.ξ/k.W*vBC.W + u.E
    elseif tBC.W==1 
        u_West = vBC.W
    elseif tBC.W==3
        if BC_order==2
            qxE    = -k.E*(u.EE - u.E)/Δ.ξ # second order BC
            u_West = Δ.ξ/k.W*(2*vBC.W - qxE) + u.E
        elseif BC_order==1
            u_West = Δ.ξ/k.W*(vBC.W) + u.E # first order BC
        end
    else
        u_West = u.W
    end
    # East
    if tBC.E==2
        u_East = 2*vBC.E - u.W 
    elseif tBC.E==4
        u_East = -Δ.ξ*vBC.E/k.E + u.W
    elseif tBC.E==1
        u_East = vBC.E
    elseif tBC.E==3
        if BC_order==2
            qxW    = -k.W*(u.W-u.WW)/Δ.ξ # second order BC
            u_East = -Δ.ξ/k.E*(2*vBC.E - qxW) + u.W 
        elseif BC_order==1
            u_East = -Δ.ξ/k.E*(vBC.E) + u.W # first order BC
        end
    else
        u_East = u.E
    end
    #--------------------------------
    # South
    u_South = 0.; u_North = 0.
    if tBC.S==2
        u_South = 2*vBC.S - u.N
    elseif tBC.S==4
        u_South = Δ.η/k.S*vBC.S + u.N
    elseif tBC.S==1
        u_South = vBC.S
    elseif tBC.S==3
        if BC_order==2
            qyN     = -k.N*(u.NN - u.N)/Δ.η # second order BC
            u_South = Δ.η/k.S*(2*vBC.S - qyN) + u.N
        elseif BC_order==1
            u_South = Δ.η/k.S*vBC.S + u.N # first order BC
        end
    else
        u_South = u.S
    end
    # North
    if tBC.N==2
        u_North = 2*vBC.N - u.S
    elseif tBC.N==4
        u_North = -Δ.η/k.N*vBC.N + u.S 
    elseif tBC.N==1
        u_North = vBC.N  
    elseif tBC.N==3
        if BC_order==2
            qyS     = -k.S*(u.S - u.SS)/Δ.η # second order BC
            u_North = -Δ.η/k.N*(2*vBC.N - qyS) + u.S
        elseif BC_order==1
            u_North = -Δ.η/k.N*(vBC.N) + u.S # first order BC
        end
    else
        u_North = u.N
    end
    #--------------------------------
    # Gradients
    dudξ = (u_East  - u_West )/Δ.ξ
    dudη = (u_North - u_South)/Δ.η
    ∂u∂x = dudξ * ∂ξ.∂x + dudη * ∂η.∂x
    ∂u∂y = dudξ * ∂ξ.∂y + dudη * ∂η.∂y
    return ∂u∂x, ∂u∂y
end