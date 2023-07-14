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