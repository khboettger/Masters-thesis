## Compute the eigenpairs 

# Block version 
function eigenpairs(
    H::Matrix{Float64}, 
    X::ManifoldPoint, 
    parameter_manifold
    )

    m, n, M, r = parameter_manifold

    X_tilde = random_point(parameter_manifold)
    
    HX = H * X.X
    XHX = X.X' * HX 

    E_tilde, V = eigen(XHX)
    
    X_tilde.X .= X.X * V
    
    U_tilde, S_tilde, Vt_tilde = svd(reshape(X_tilde.X, m, n*M))
    U_tilde = U_tilde[:, 1:r]
    S_tilde = S_tilde[1:r]
    Vt_tilde = (Vt_tilde[:, 1:r])'
    
    X_tilde.U .= U_tilde
    X_tilde.S .= S_tilde
    X_tilde.Vt .= Vt_tilde

    return E_tilde, X_tilde
end

# Iterative version
function eigenpairs(
    H::Matrix{Float64}, 
    list_X::Vector{ManifoldPoint}, 
    parameter_manifold
    )

    m, n, M, r = parameter_manifold

    list_E_tilde = Float64[]
    list_X_tilde = ManifoldPoint[]
    
    for l=1:1:M
        X = list_X[l]

        X_tilde = random_point((m, n, 1, r))
        
        HX = H * X.X
        XHX = X.X' * HX 

        E_tilde, V = eigen(XHX)
        
        X_tilde.X .= X.X * V
        
        U_tilde, S_tilde, Vt_tilde = svd(reshape(X_tilde.X, m, n))
        U_tilde = U_tilde[:, 1:r]
        S_tilde = S_tilde[1:r]
        Vt_tilde = (Vt_tilde[:, 1:r])'
        
        X_tilde.U .= U_tilde
        X_tilde.S .= S_tilde
        X_tilde.Vt .= Vt_tilde

        append!(list_E_tilde, E_tilde)
        push!(list_X_tilde, X_tilde)
    end

    return list_E_tilde, list_X_tilde
end