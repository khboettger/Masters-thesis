## Define the backtracking

# Use the last alpha
function last_alpha(
    alpha::Float64, 
    alpha_max::Float64, 
    alpha_min::Float64,
    rho::Float64
    )

    return max(min(alpha/rho, alpha_max), alpha_min) 
end

# Compute an initial alpha - normal version 
function computed_alpha(
    Hzeta::Matrix{Float64},
    H::Matrix{Float64},
    zeta::ManifoldTangentVector,
    X::ManifoldPoint,
    divisor::Float64,
    alpha_max::Float64, 
    alpha_min::Float64,
    parameter_manifold
    )      

    mul!(Hzeta, H, zeta.p)
    project_full!(Hzeta, X, Hzeta, parameter_manifold)  
    divisor_Hzeta = dot(Hzeta, zeta.p)
    alpha_init = abs(divisor / divisor_Hzeta) 
    
    return max(min(alpha_init, alpha_max), alpha_min)  
end 

# Compute an initial alpha - normal version with Kronecker  
function computed_alpha(
    Hzeta::Matrix{Float64},
    H_kronecker::Vector{AbstractMatrix{Float64}},
    zeta::ManifoldTangentVector,
    X::ManifoldPoint,
    divisor::Float64,
    alpha_max::Float64, 
    alpha_min::Float64,
    parameter_manifold
    )      

    m, n, M, _ = parameter_manifold
    k = length(H_kronecker)

    H_BX_l = zeros(m, n)
    HX_reshaped_l_k = zeros(m, n)

    for l = 1:1:M
        HX_reshaped_l = zeros(m, n)

        X_l = @view zeta.p[:, l]
        X_reshaped_l = reshape(X_l, m, n)

        for i = 1:2:k
            A = H_kronecker[i]
            B = H_kronecker[i+1]

            mul!(H_BX_l, B, X_reshaped_l)
            mul!(HX_reshaped_l_k, H_BX_l, A)

            HX_reshaped_l .+= HX_reshaped_l_k       
        end

        Hzeta[:, l] .= vec(HX_reshaped_l)
    end

    project_full!(Hzeta, X, Hzeta, parameter_manifold)  
    divisor_Hzeta = dot(Hzeta, zeta.p)
    alpha_init = abs(divisor / divisor_Hzeta) 
    
    return max(min(alpha_init, alpha_max), alpha_min)  
end 

# Compute an initial alpha - normal version with Kronecker and higher eigenpairs
function computed_alpha(
    Hzeta::Matrix{Float64},
    H_kronecker::Vector{AbstractMatrix{Float64}},
    P,
    zeta::ManifoldTangentVector,
    X::ManifoldPoint,
    divisor::Float64,
    alpha_max::Float64, 
    alpha_min::Float64,
    parameter_manifold
    )      

    m, n, M, _ = parameter_manifold
    k = length(H_kronecker)

    H_BX_l = zeros(m, n)
    HX_reshaped_l_k = zeros(m, n)

    Pzeta = P*zeta.p

    for l = 1:1:M
        HX_reshaped_l = zeros(m, n)

        X_l = @view Pzeta[:, l]
        X_reshaped_l = reshape(X_l, m, n)

        for i = 1:2:k
            A = H_kronecker[i]
            B = H_kronecker[i+1]

            mul!(H_BX_l, B, X_reshaped_l)
            mul!(HX_reshaped_l_k, H_BX_l, A)

            HX_reshaped_l .+= HX_reshaped_l_k       
        end

        Hzeta[:, l] .= vec(HX_reshaped_l)
    end

    Hzeta .= P * Hzeta

    project_full!(Hzeta, X, Hzeta, parameter_manifold)  
    divisor_Hzeta = dot(Hzeta, zeta.p)
    alpha_init = abs(divisor / divisor_Hzeta) 
    
    return max(min(alpha_init, alpha_max), alpha_min)  
end 

# Compute an initial alpha - DMRG version with ProjMPO
function computed_alpha(
    Hzeta,
    H::ProjMPO,
    combiner_1, 
    zeta::ManifoldTangentVector,
    X::ManifoldPoint,
    divisor::Float64,
    alpha_max::Float64, 
    alpha_min::Float64,
    index_L,
    parameter_manifold
    )      

    zeta_tensor_combined = ITensor(zeta.p, inds(combiner_1)[1], index_L)
    zeta_tensor = combiner_1 * zeta_tensor_combined
    Hzeta .= array(combiner_1 * product(H, zeta_tensor))

    project_full!(Hzeta, X, Hzeta, direction, parameter_manifold)  
    divisor_Hzeta = dot(Hzeta, zeta.p)
    alpha_init = abs(divisor / divisor_Hzeta) 
    
    return max(min(alpha_init, alpha_max), alpha_min)  
end 

# Compute an initial alpha - DMRG version with ProjMPO_MPS
function computed_alpha(
    Hzeta,
    H::ProjMPO_MPS,
    combiner_1, 
    zeta::ManifoldTangentVector,
    X::ManifoldPoint,
    divisor::Float64,
    alpha_max::Float64, 
    alpha_min::Float64,
    index_L,
    parameter_manifold
    )      

    _, _, M, _ = parameter_manifold

    # Compute HX
    HX_array = []
    inds_X_tensor_1 = 0
    for l=1:1:M
        X_tensor_combined_l = ITensor(zeta.p[:, l], inds(combiner_1)[1])
        X_tensor_l = combiner_1 * X_tensor_combined_l

        HX_l = product(H, X_tensor_l)
        HX_array_l = array(HX_l)
        push!(HX_array, HX_array_l) 

        if l == 1
            inds_X_tensor_1 = inds(HX_l)
        end
    end 
    HX_array = stack(HX_array)
    HX_tensor = ITensor(HX_array, inds_X_tensor_1, index_L)
    Hzeta .= array(combiner_1 * HX_tensor)

    project_full!(Hzeta, X, Hzeta, direction, parameter_manifold)  
    divisor_Hzeta = dot(Hzeta, zeta.p)
    alpha_init = abs(divisor / divisor_Hzeta) 
    
    return max(min(alpha_init, alpha_max), alpha_min)  
end 

# Backtracking - normal version 
function backtracking_armijo!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, adaptive, epsilon, initial, _ = parameter_backtracking

    if initial == "last"
        alpha = last_alpha(alpha, alpha_max, alpha_min, rho)
    elseif initial == "computed"
        alpha = computed_alpha(Hzeta, H, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    cost_X = cost(H, Y.X, HX, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)
    
    if increase
        Yold = copy(Y.X)
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha

        while v >= 1.0 && alpha < alpha_max && increase
            Yold = copy(Y.X)
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            
            alpha /= rho
            
            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            cost_X = cost(H, Y.X, HX, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)
        end

        Y.X .= copy(Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
    end 

    while v < 1.0 && alpha > alpha_min 
        if adaptive 
            rho_hat = max(epsilon, rho*(1-c)/(1-c*v))
            alpha *= rho_hat
        else
            alpha *= rho
        end
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        cost_X = cost(H, Y.X, HX, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)
    end

    optimized_svd!(Y, parameter_manifold)

    return Y, alpha, cost_X
end

# Backtracking - normal version with Kronecker and higher eigenpairs
function backtracking_armijo!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    P,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )

    alpha0, c, rho, alpha_min, increase, alpha_max, adaptive, epsilon, initial, _ = parameter_backtracking

    if initial == "last"
        alpha = last_alpha(alpha, alpha_max, alpha_min, rho)
    elseif initial == "computed"
        alpha = computed_alpha(Hzeta, H, P, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end
    
    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    cost_X = cost(H, Y.X, HX, P, parameter_manifold) 
    v = (cost_X - cost_Xold) / (alpha*c*divisor)
    
    if increase
        Yold = copy(Y.X)
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha

        while v >= 1.0 && alpha < alpha_max && increase
            Yold = copy(Y.X)
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            cost_X = cost(H, Y.X, HX, P, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)
        end

        Y.X .= copy(Yold) 
        HX .= copy(HX)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
    end 

    while v < 1.0 && alpha > alpha_min 
        if adaptive 
            rho_hat = max(epsilon, rho*(1-c)/(1-c*v))
            alpha *= rho_hat
        else
            alpha *= rho
        end
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        cost_X = cost(H, Y.X, HX, P, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)
    end

    optimized_svd!(Y, parameter_manifold)

    return Y, alpha, cost_X
end

# Backtracking - DMRG version 
function backtracking_armijo!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    combiner_1,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta,
    HX,
    direction::String,
    index_L, 
    parameter_manifold,
    parameter_backtracking,
    )

    alpha0, c, rho, alpha_min, increase, alpha_max, adaptive, epsilon, initial, _ = parameter_backtracking

    if initial == "last"
        alpha = last_alpha(alpha, alpha_max, alpha_min, rho)
    elseif initial == "computed"
        alpha = computed_alpha(Hzeta, H, combiner_1, p, X, divisor, alpha_max, alpha_min, index_L, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end
    
    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
    cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)
    
    if increase
        Yold = copy(Y.X)
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha

        while v >= 1.0 && alpha < alpha_max && increase
            Yold = copy(Y.X)
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
            cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)
        end

        Y.X .= copy(Yold) 
        HX .= copy(HX)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
    end 

    while v < 1.0 && alpha > alpha_min 
        if adaptive 
            rho_hat = max(epsilon, rho*(1-c)/(1-c*v))
            alpha *= rho_hat
        else
            alpha *= rho
        end
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
        cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)
    end

    optimized_svd!(Y, direction, parameter_manifold)

    return Y, alpha, cost_X
end

# Wolfe Backtracking - normal version 
function backtracking_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    optimized_svd!(Y, parameter_manifold)

    cost_X = cost(H, Y.X, HX, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, Y, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && curvature >= 0.9*divisor) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            optimized_svd!(Y, parameter_manifold)

            cost_X = cost(H, Y.X, HX, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, Y, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || curvature < 0.9*divisor) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        optimized_svd!(Y, parameter_manifold)

        cost_X = cost(H, Y.X, HX, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, Y, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end

# Wolfe Backtracking - normal version with Kronecker and higher eigenpairs
function backtracking_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    P,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, P, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    optimized_svd!(Y, parameter_manifold)

    cost_X = cost(H, Y.X, HX, P, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, Y, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && curvature >= 0.9*divisor) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            optimized_svd!(Y, parameter_manifold)

            cost_X = cost(H, Y.X, HX, P, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, Y, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || curvature < 0.9*divisor) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        optimized_svd!(Y, parameter_manifold)

        cost_X = cost(H, Y.X, HX, P, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, Y, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end

# Wolfe Backtracking - DMRG version 
function backtracking_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    combiner_1,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta,
    HX,
    direction::String,
    index_L, 
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, combiner_1, p, X, divisor, alpha_max, alpha_min, index_L, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
    optimized_svd!(Y, direction, parameter_manifold)

    cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && curvature >= 0.9*divisor) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
            optimized_svd!(Y, direction, parameter_manifold)

            cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || curvature < 0.9*divisor) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
        optimized_svd!(Y, direction, parameter_manifold)

        cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end

# Strong Wolfe Backtracking - normal version 
function backtracking_strong_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    optimized_svd!(Y, parameter_manifold)

    cost_X = cost(H, Y.X, HX, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, Y, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && abs(curvature) <= 0.9*abs(divisor)) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            optimized_svd!(Y, parameter_manifold)

            cost_X = cost(H, Y.X, HX, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, Y, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || abs(curvature) > 0.9*abs(divisor)) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        optimized_svd!(Y, parameter_manifold)

        cost_X = cost(H, Y.X, HX, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, Y, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end

# Strong Wolfe Backtracking - normal version with Kronecker and higher eigenpairs
function backtracking_strong_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    P,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, P, p, X, divisor, alpha_max,  alpha_min, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
    optimized_svd!(Y, parameter_manifold)

    cost_X = cost(H, Y.X, HX, P, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, Y, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && abs(curvature) <= 0.9*abs(divisor)) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
            optimized_svd!(Y, parameter_manifold)

            cost_X = cost(H, Y.X, HX, P, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, Y, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || abs(curvature) > 0.9*abs(divisor)) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), parameter_manifold)
        optimized_svd!(Y, parameter_manifold)

        cost_X = cost(H, Y.X, HX, P, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, Y, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end

# Strong Wolfe Backtracking - DMRG version 
function backtracking_strong_wolfe!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    H,
    combiner_1,
    divisor::Float64,
    cost_Xold::Float64,
    alpha::Float64,
    Hzeta,
    HX,
    direction::String,
    index_L, 
    g::Matrix{Float64},
    tr_zeta::Matrix{Float64},
    parameter_manifold,
    parameter_backtracking,
    )
    
    alpha0, c, rho, alpha_min, increase, alpha_max, _, _, initial, _ = parameter_backtracking

    if initial == "computed"
        alpha = computed_alpha(Hzeta, H, combiner_1, p, X, divisor, alpha_max, alpha_min, index_L, parameter_manifold)      
    else
        alpha = max(min(alpha0, alpha_max), alpha_min) 
    end

    retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
    optimized_svd!(Y, direction, parameter_manifold)

    cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
    v = (cost_X - cost_Xold) / (alpha*c*divisor)

    grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
    
    copyto!(tr_zeta, p.p)
    transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

    curvature = dot(g, tr_zeta)

    if increase
        Yold = random_point(parameter_manifold)
        ManifoldCopy!(Yold, Y) 
        HYold = copy(HX)
        cost_Yold = cost_X
        vold = v
        alpha_old = alpha
        g_old = copy(g)
        tr_zeta_old = copy(tr_zeta)
        curvature_old = curvature

        while (v >= 1.0 && abs(curvature) <= 0.9*abs(divisor)) && alpha < alpha_max && increase
            ManifoldCopy!(Yold, Y) 
            HYold = copy(HX)
            cost_Yold = cost_X 
            vold = v
            alpha_old = alpha
            g_old = copy(g)
            tr_zeta_old = copy(tr_zeta)
            curvature_old = curvature
            
            alpha /= rho

            retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
            optimized_svd!(Y, direction, parameter_manifold)

            cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
            v = (cost_X - cost_Xold) / (alpha*c*divisor)

            grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
            
            copyto!(tr_zeta, p.p)
            transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

            curvature = dot(g, tr_zeta)
        end

        ManifoldCopy!(Y, Yold) 
        HX .= copy(HYold)
        cost_X = cost_Yold
        v = vold 
        alpha = alpha_old
        g .= copy(g_old)
        tr_zeta .= copy(tr_zeta_old)
    end 
    
    while (v < 1.0 || abs(curvature) > 0.9*abs(divisor)) && alpha > alpha_min 
        alpha *= rho
        
        retract_full!(Y.X, X, ManifoldScalarProduct(alpha, p), direction, parameter_manifold)
        optimized_svd!(Y, direction, parameter_manifold)

        cost_X = cost(H, combiner_1, Y.X, HX, index_L, parameter_manifold)
        v = (cost_X - cost_Xold) / (alpha*c*divisor)

        grad!(g, HX, combiner_1, Y, direction, parameter_manifold)
        
        copyto!(tr_zeta, p.p)
        transp!(tr_zeta, tr_zeta, Y, X, direction, parameter_manifold)

        curvature = dot(g, tr_zeta)
    end

    return Y, alpha, cost_X, g, tr_zeta
end
