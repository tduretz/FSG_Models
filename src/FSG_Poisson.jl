function Poisson2D( T, typ, val, k, b, ∂ξ∂x, ∂ξ∂y, ∂η∂x, ∂η∂y, Δ, params, Frozen, vertex, inds )
    kW, kE, kS, kN = 1.0, 1.0, 1.0, 1.0
    ∂T∂xW, ∂T∂xE = 0.0, 0.0
    ∂T∂yS, ∂T∂yN = 0.0, 0.0 
    ∂T∂yW, ∂T∂yE = 0.0, 0.0
    ∂T∂xS, ∂T∂xN = 0.0, 0.0
    kBC          = (W=params.kBC, E=params.kBC, S=params.kBC, N=params.kBC)
    if Frozen 
        BC_order = 1   # 1st order BC keeps preconditionner symmetric
        ∂ξ∂x    .= 1.0 # neglect mesh distrortion to keep preconditionner symmetric
        ∂ξ∂y    .= 0.0 # neglect mesh distrortion to keep preconditionner symmetric
        ∂η∂x    .= 0.0 # neglect mesh distrortion to keep preconditionner symmetric
        ∂η∂y    .= 1.0 # neglect mesh distrortion to keep preconditionner symmetric
    else 
        BC_order = 2 # 2nd order BC is what we aim for
    end

    # Gradient west
    TW = (W=  T[2], E=  T[3], S=  T[6], N=  T[8], EE=  T[4])
    tW = (W=typ[2], E=typ[3], S=typ[6], N=typ[8], EE=typ[4])
    vW = (W=val[2], E=val[3], S=val[6], N=val[8], EE=val[4])
    ∂ξ = (∂x=∂ξ∂x[2], ∂y=∂ξ∂y[2])
    ∂η = (∂x=∂η∂x[2], ∂y=∂η∂y[2])
    ∂T∂xW, ∂T∂yW = GradientLoc1(TW, Δ, tW, vW; k=kBC, BC_order=BC_order, ∂ξ=∂ξ, ∂η=∂η)
    isinW = tW.W==0 && tW.E==0 && tW.S==0  && tW.N==0 

    # Gradient east
    TE = (W=  T[3], E=  T[4], S=  T[7], N=  T[9], WW=  T[2])
    tE = (W=typ[3], E=typ[4], S=typ[7], N=typ[9], WW=typ[2])
    vE = (W=val[3], E=val[4], S=val[7], N=val[9], WW=val[2])
    ∂ξ = (∂x=∂ξ∂x[3], ∂y=∂ξ∂y[3])
    ∂η = (∂x=∂η∂x[3], ∂y=∂η∂y[3])
    ∂T∂xE, ∂T∂yE = GradientLoc1(TE, Δ, tE, vE; k=kBC, BC_order=BC_order, ∂ξ=∂ξ, ∂η=∂η)
    isinE = tE.W==0 && tE.E==0 && tE.S==0  && tE.N==0 
    
    # Gradient south
    TS = (W=  T[6], E=  T[7], S=  T[1], N=  T[3], NN=  T[5] )
    tS = (W=typ[6], E=typ[7], S=typ[1], N=typ[3], NN=typ[5] )
    vS = (W=val[6], E=val[7], S=val[1], N=val[3], NN=val[5] )
    ∂ξ = (∂x=∂ξ∂x[4], ∂y=∂ξ∂y[4])
    ∂η = (∂x=∂η∂x[4], ∂y=∂η∂y[4])
    ∂T∂xS, ∂T∂yS = GradientLoc1(TS, Δ, tS, vS; k=kBC, BC_order=BC_order, ∂ξ=∂ξ, ∂η=∂η)
    isinS = tS.W==0 && tS.E==0 && tS.S==0  && tS.N==0 

    # Gradient north
    TN = (W=  T[8], E=  T[9], S=  T[3], N=  T[5], SS=  T[1])
    tN = (W=typ[8], E=typ[9], S=typ[3], N=typ[5], SS=typ[1])
    vN = (W=val[8], E=val[9], S=val[3], N=val[5], SS=val[1])
    ∂ξ = (∂x=∂ξ∂x[5], ∂y=∂ξ∂y[5])
    ∂η = (∂x=∂η∂x[5], ∂y=∂η∂y[5])
    ∂T∂xN, ∂T∂yN = GradientLoc1(TN, Δ, tN, vN; k=kBC, BC_order=BC_order, ∂ξ=∂ξ, ∂η=∂η)
    isinN = tN.W==0 && tN.E==0 && tN.S==0  && tN.N==0 

    # Non-linearity
    if Frozen
        kW = k[1]; kE = k[2]; kS = k[3]; kN = k[4]
    else
        kW = EvaluateConductivity(params, ∂T∂xW, ∂T∂yW, isinW)
        kE = EvaluateConductivity(params, ∂T∂xE, ∂T∂yE, isinE)
        kS = EvaluateConductivity(params, ∂T∂xS, ∂T∂yS, isinS)
        kN = EvaluateConductivity(params, ∂T∂xN, ∂T∂yN, isinN)
    end
    # Fluxes
    qxW = -kW*∂T∂xW
    qxE = -kE*∂T∂xE
    qyS = -kS*∂T∂yS
    qyN = -kN*∂T∂yN
    # Balance
    f  = b + ((qxE - qxW)/Δ.ξ * (∂ξ∂x[1] + ∂ξ∂y[1]) + (qyN - qyS)/Δ.η * (∂η∂x[1] + ∂η∂y[1]))
    return f
end

function ResidualFSG!(K, f, T, b, k, q, ∂, Δ, num, val, typ, params, Assemble, Frozen)

    Tloc  = zeros(9)
    tloc  = zeros(Int64, 9)
    vloc  = zeros(9)
    ∂F∂T  = zeros(9)
    kloc  = zeros(4)
    nloc  = zeros(Int64, 9)
    ∂ξ∂x  = zeros(5)
    ∂ξ∂y  = zeros(5)
    ∂η∂x  = zeros(5)
    ∂η∂y  = zeros(5)
    #-------------------------------------------#
    # Centroids
    for i in axes(f.c,1),  j in axes(f.c,2)
        if i>1 && i<size(f.c,1) && j>1 && j<size(f.c,2)

            Tloc  .= [    T.c[i,j-1],   T.c[i-1,j],      T.c[i,j],   T.c[i+1,j],   T.c[i,j+1],   T.v[i,j],   T.v[i+1,j],   T.v[i,j+1],   T.v[i+1,j+1]]
            tloc  .= [  typ.c[i,j-1], typ.c[i-1,j],    typ.c[i,j], typ.c[i+1,j], typ.c[i,j+1], typ.v[i,j], typ.v[i+1,j], typ.v[i,j+1], typ.v[i+1,j+1]]
            vloc  .= [  val.c[i,j-1], val.c[i-1,j],    val.c[i,j], val.c[i+1,j], val.c[i,j+1], val.v[i,j], val.v[i+1,j], val.v[i,j+1], val.v[i+1,j+1]]
            nloc  .= [  num.c[i,j-1], num.c[i-1,j],    num.c[i,j], num.c[i+1,j], num.c[i,j+1], num.v[i,j], num.v[i+1,j], num.v[i,j+1], num.v[i+1,j+1]]
            kloc  .= [                     k.ex[i-1,j],      k.ex[i,j],      k.ey[i,j-1],      k.ey[i,j]]
            ∂ξ∂x  .= [ ∂.ξ.∂x.c[i,j], ∂.ξ.∂x.ex[i-1,j], ∂.ξ.∂x.ex[i,j], ∂.ξ.∂x.ey[i,j-1], ∂.ξ.∂x.ey[i,j]]
            ∂ξ∂y  .= [ ∂.ξ.∂y.c[i,j], ∂.ξ.∂y.ex[i-1,j], ∂.ξ.∂y.ex[i,j], ∂.ξ.∂y.ey[i,j-1], ∂.ξ.∂y.ey[i,j]]  
            ∂η∂x  .= [ ∂.η.∂x.c[i,j], ∂.η.∂x.ex[i-1,j], ∂.η.∂x.ex[i,j], ∂.η.∂x.ey[i,j-1], ∂.η.∂x.ey[i,j]]
            ∂η∂y  .= [ ∂.η.∂y.c[i,j], ∂.η.∂y.ex[i-1,j], ∂.η.∂y.ex[i,j], ∂.η.∂y.ey[i,j-1], ∂.η.∂y.ey[i,j]]

            f.c[i,j] = Poisson2D(Tloc, tloc, vloc, kloc, b.c[i,j], ∂ξ∂x, ∂ξ∂y, ∂η∂x, ∂η∂y, Δ, params, Frozen, false, (i=i,j=j))

            if Assemble==true
                ∂F∂T .= 0.
                autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(tloc), Const(vloc), Const(kloc), Const(b.c[i,j]), Const(∂ξ∂x), Const(∂ξ∂y), Const(∂η∂x), Const(∂η∂y), Const(Δ), Const(params), Const(Frozen), Const(false), Const( (i=i,j=j)))
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
            if typ.v[i,j]!=1

                Tloc  .= [  T.v[i,j-1],    T.v[i-1,j],    T.v[i,j],    T.v[i+1,j],   T.v[i,j+1],   T.c[i-1,j-1],   T.c[i,j-1],   T.c[i-1,j],   T.c[i,j]]
                tloc  .= [typ.v[i,j-1],  typ.v[i-1,j],  typ.v[i,j],  typ.v[i+1,j], typ.v[i,j+1], typ.c[i-1,j-1], typ.c[i,j-1], typ.c[i-1,j], typ.c[i,j]]
                vloc  .= [val.v[i,j-1],  val.v[i-1,j],  val.v[i,j],  val.v[i+1,j], val.v[i,j+1], val.c[i-1,j-1], val.c[i,j-1], val.c[i-1,j], val.c[i,j]]
                nloc  .= [num.v[i,j-1],  num.v[i-1,j],  num.v[i,j],  num.v[i+1,j], num.v[i,j+1], num.c[i-1,j-1], num.c[i,j-1], num.c[i-1,j], num.c[i,j]]
                kloc  .= [                     k.ey[i-1,j-1],      k.ey[i,j-1],      k.ex[i-1,j-1],      k.ex[i-1,j]]
                ∂ξ∂x  .= [ ∂.ξ.∂x.v[i,j], ∂.ξ.∂x.ey[i-1,j-1], ∂.ξ.∂x.ey[i,j-1], ∂.ξ.∂x.ex[i-1,j-1], ∂.ξ.∂x.ex[i-1,j]]
                ∂ξ∂y  .= [ ∂.ξ.∂y.v[i,j], ∂.ξ.∂y.ey[i-1,j-1], ∂.ξ.∂y.ey[i,j-1], ∂.ξ.∂y.ex[i-1,j-1], ∂.ξ.∂y.ex[i-1,j]]  
                ∂η∂x  .= [ ∂.η.∂x.v[i,j], ∂.η.∂x.ey[i-1,j-1], ∂.η.∂x.ey[i,j-1], ∂.η.∂x.ex[i-1,j-1], ∂.η.∂x.ex[i-1,j]]
                ∂η∂y  .= [ ∂.η.∂y.v[i,j], ∂.η.∂y.ey[i-1,j-1], ∂.η.∂y.ey[i,j-1], ∂.η.∂y.ex[i-1,j-1], ∂.η.∂y.ex[i-1,j]]
                
                f.v[i,j] = Poisson2D(Tloc, tloc, vloc, kloc, b.v[i,j], ∂ξ∂x, ∂ξ∂y, ∂η∂x, ∂η∂y, Δ, params, Frozen, true,  (i=i,j=j))

                if Assemble==true
                    ∂F∂T .= 0.
                    autodiff(Enzyme.Reverse, Poisson2D, Duplicated(Tloc, ∂F∂T), Const(tloc), Const(vloc), Const(kloc), Const(b.c[i,j]), Const(∂ξ∂x), Const(∂ξ∂y), Const(∂η∂x), Const(∂η∂y), Const(Δ), Const(params), Const(Frozen), Const(true), Const( (i=i,j=j)))
                    for jeq in eachindex(nloc)
                        if abs(∂F∂T[jeq])>0.
                            K[nloc[3],nloc[jeq]] = ∂F∂T[jeq]
                        end
                    end
                end
            else
                K[num.v[i,j], num.v[i,j]] = 1.0
            end
        end
    end

    #-------------------------------------------#
    # Finalise
    if Assemble==true flush!(K) end
end

function SetBoundaryConditions!(typ, val, BC, msh, nce, nve, u_anal, q_anal)
    # Boundary conditions
    for j=1:nce.y # Centroids W/E
        if BC.W == :Dirichlet   # West
            typ.c[1,j]     = 2
            val.c[1,j]     = u_anal(msh.ex.x[2,j], msh.c.y[2,j])
        elseif BC.W == :Neumann # West
            typ.c[1,j]     = 4
            qx, qy         = q_anal(msh.ex.x[2,j], msh.c.y[2,j])
            val.c[1,j]     = qx
        end
        if BC.E == :Dirichlet    # East
            typ.c[end,j]   = 2
            val.c[end,j]   = u_anal(msh.ex.x[end-1,j], msh.c.y[end-1,j])
        elseif BC.E == :Neumann  # East
            typ.c[end,j]   = 4
            qx, qy         = q_anal(msh.ex.x[end-1,j], msh.c.y[end-1,j])
            val.c[end,j]   = qx
        end
    end
    for i=1:nce.x # Centroids S/N
        if BC.S == :Dirichlet   # South
            typ.c[i,1]     = 2
            val.c[i,1]     = u_anal(msh.c.x[i,2], msh.ey.y[i,2])
        elseif BC.S == :Neumann  # South
            typ.c[i,1]     = 4
            qx, qy         = q_anal(msh.c.x[i,2], msh.ey.y[i,2])
            val.c[i,1]     = qy
        end
        if BC.N == :Dirichlet   # North
            typ.c[i,end]   = 2
            val.c[i,end]   = u_anal(msh.c.x[i,end-1], msh.ey.y[i,end-1])
        elseif BC.N == :Neumann  # North
            typ.c[i,end]   = 4
            qx, qy         = q_anal(msh.c.x[i,end-1], msh.ey.y[i,end-1])
            val.c[i,end]   = qy
        end
    end

    typ.v[[1,end], :]  .= -1;  typ.v[:,[1,end]]   .= -1
    for j=1:nve.y # Vertices W/E
        if BC.W == :Dirichlet   # West
            typ.v[2,j]     = 1
            val.v[2,j]     = u_anal(msh.v.x[2,j], msh.v.y[2,j])
        end
        if BC.W == :Neumann  # West
            typ.v[1,j]     = 3
            qx, qy         = q_anal(msh.v.x[2,j], msh.v.y[2,j])
            val.v[1,j]     = qx
        end
        if BC.E == :Dirichlet   # East
            typ.v[end-1,j] = 1
            val.v[end-1,j] = u_anal(msh.v.x[end-1,j], msh.v.y[end-1,j])
        elseif BC.E == :Neumann  # East
            typ.v[end,j]   = 3
            qx, qy         = q_anal(msh.v.x[end-1,j], msh.v.y[end-1,j])
            val.v[end,j]   = qx
        end
    end
    for i=1:nve.x # Vertices S/N
        if BC.S == :Dirichlet   # West
            typ.v[i,2]     = 1
            val.v[i,2]     = u_anal(msh.v.x[i,2], msh.v.y[i,2])
        elseif BC.S == :Neumann  # West
            typ.v[i,1]     = 3
            qx, qy         = q_anal(msh.v.x[i,2], msh.v.y[i,2])
            val.v[i,1]     = qy
        end
        if BC.N == :Dirichlet   # North
            typ.v[i,end-1] = 1
            val.v[i,end-1] = u_anal(msh.v.x[i,end-1], msh.v.y[i,end-1])
        elseif BC.N == :Neumann  # North
            typ.v[i,end]   = 3
            qx, qy         = q_anal(msh.v.x[i,end-1], msh.v.y[i,end-1])
            val.v[i,end]   = qy
        end
    end
    return nothing
end


function EvaluateConductivity(p, ∂T∂x, ∂T∂y, isin)
    if isin 
        return p.k0*sqrt(∂T∂x.^2 + ∂T∂y.^2)^(p.n)
    else
        return p.kBC
    end
end

function ComputeConductivity!(k, q, T, typ, val, params, nc, ∂, Δ)
    for j in axes(k.ex,2), i in axes(k.ex,1)
        if j>1 && j<nc.y+2
        tBC = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
        vBC = (W=0.,        E=0.,        S=0.,        N=0.)
        tBC  = (W=typ.c[i,j], E=typ.c[i+1,j], S=typ.v[i+1,j], N=typ.v[i+1,j+1])
        if i==1
            vBC  = (W=val.c[i,j], E=val.c[i+1,j], S=val.v[i+1,j], N=val.v[i+1,j+1], EE=val.c[i+2,j])
        elseif i==size(k.ex,1)
            vBC  = (W=val.c[i,j], E=val.c[i+1,j], S=val.v[i+1,j], N=val.v[i+1,j+1], WW=val.c[i-1,j])
        else
            vBC  = (W=val.c[i,j], E=val.c[i+1,j], S=val.v[i+1,j], N=val.v[i+1,j+1])
        end
        Tloc        = (W=  T.c[i,j], E=  T.c[i+1,j], S=  T.v[i+1,j], N=  T.v[i+1,j+1])
        ∂ξ          = (∂x=∂.ξ.∂x.ex[i,j], ∂y=∂.ξ.∂y.ex[i,j])
        ∂η          = (∂x=∂.η.∂x.ex[i,j], ∂y=∂.η.∂y.ex[i,j])
        ∂T∂x, ∂T∂y  = GradientLoc1(Tloc, Δ, tBC, vBC; ∂ξ=∂ξ, ∂η=∂η )
        isin        = tBC.W==0 && tBC.E==0 && tBC.S==0  && tBC.N==0 
        k.ex[i,j]   = EvaluateConductivity( params, ∂T∂x, ∂T∂y, isin)
        q.x.ex[i,j] = -k.ex[i,j]*∂T∂x
        q.y.ex[i,j] = -k.ex[i,j]*∂T∂y
    else
        k.ex[i,j]   = params.kBC
        end
    end
    
    for j in axes(k.ey,2), i in axes(k.ey,1)
        if i>1 && i<nc.x+2 
        tBC = (W=:Internal, E=:Internal, S=:Internal, N=:Internal)
        vBC = (W=0.,        E=0.,        S=0.,        N=0.)
        tBC         = (W=typ.v[i,j+1], E=typ.v[i+1,j+1], S=typ.c[i,j], N=typ.c[i,j+1])
        if j==1
            vBC         = (W=val.v[i,j+1], E=val.v[i+1,j+1], S=val.c[i,j], N=val.c[i,j+1], NN=val.c[i,j+2])
        elseif j==size(k.ey,2)
            vBC         = (W=val.v[i,j+1], E=val.v[i+1,j+1], S=val.c[i,j], N=val.c[i,j+1], SS=val.c[i,j-1])
        else
            vBC         = (W=val.v[i,j+1], E=val.v[i+1,j+1], S=val.c[i,j], N=val.c[i,j+1])
        end
        Tloc        = (W=T.v[i,j+1], E=T.v[i+1,j+1], S=T.c[i,j], N=T.c[i,j+1])
        ∂ξ          = (∂x=∂.ξ.∂x.ey[i,j], ∂y=∂.ξ.∂y.ey[i,j])
        ∂η          = (∂x=∂.η.∂x.ey[i,j], ∂y=∂.η.∂y.ey[i,j])
        ∂T∂x, ∂T∂y  = GradientLoc1(Tloc, Δ, tBC, vBC; ∂ξ=∂ξ, ∂η=∂η )
        isin        = tBC.W==0 && tBC.E==0 && tBC.S==0  && tBC.N==0 
        k.ey[i,j]   = EvaluateConductivity( params, ∂T∂x, ∂T∂y, isin)
        q.x.ey[i,j] = -k.ey[i,j]*∂T∂x
        q.y.ey[i,j] = -k.ey[i,j]*∂T∂y
    else
        k.ey[i,j]   = params.kBC
        end
    end
    return
end