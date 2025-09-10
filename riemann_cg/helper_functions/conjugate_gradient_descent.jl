## Define the conjugate gradient descent

# Normal version 
function conjugate_gradient_descent!(
    H,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_gold::Matrix{Float64}, 
    tr_zeta::Matrix{Float64}, 
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, _, type_beta = parameter_optimization 
    _, _, _, _, _, _, _, _, _, type_condition = parameter_backtracking  

    for it=1:1:iterations 

        # Test if gradnorm is smaller than tolerance 
        if gradnorm < tolerance_grad || rel_cost < tolerance_cost || cost_X < cost_max
            return X, alpha, cost_X, rel_cost, gradnorm, it-1, rel_X
        end

        # Compute alpha, cost_X and X, g, and tr_zeta
        if type_condition == "armijo"
            _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, parameter_manifold, parameter_backtracking) 

            copyto!(tr_gold, g)
            grad!(g, HX, X, parameter_manifold)

            copyto!(tr_zeta, zeta.p)
            transp!(tr_zeta, tr_zeta, X, Xold, parameter_manifold)
        elseif type_condition == "wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_wolfe!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)  
        elseif type_condition == "strong_wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_strong_wolfe!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)   
        end

        # Compute beta 
        if type_beta == "fr"
            beta = fletcher_reeves(g, tr_gold) 
        elseif type_beta == "dy"
            beta = dai_yuan(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        elseif type_beta == "cg"
            beta = conjugate_gradient(g, divisor) 
        elseif type_beta == "pr"
            beta = polak_ribiere(g, tr_gold, X, Xold, parameter_manifold) 
        elseif type_beta == "hs"
            beta = hestenes_stiefel(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        elseif type_beta == "ls"
            beta = liu_storey(g, tr_gold, X, Xold, divisor, parameter_manifold) 
        elseif type_beta == "hz"
            beta = hager_zhang(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        end 
        
        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, beta, parameter_manifold) 
        
        # Give outout
        if output && it%output_it==0
            println("# $it | F(X): $(cost_X) | Delta(F(X)): $(rel_cost) | |grad(F(X))|: $(gradnorm) | Delta(X): $(rel_X) | alpha: $(alpha)")
        end

        # Update Xold, cost_Xold, rel_cost, divisor and gradnorm  
        rel_X = norm(X.X - Xold.X)/norm(Xold.X)
        ManifoldCopy!(Xold, X)
        rel_cost = abs(cost_X - cost_Xold) / abs(cost_Xold)
        cost_Xold = cost_X
        divisor = dot(g, zeta.p)
        gradnorm = norm(g)
    end

    return X, alpha, cost_X, rel_cost, gradnorm, iterations, rel_X
end

# Normal version - Kronecker structure and higher eigenpairs
function conjugate_gradient_descent!(
    H::Vector{AbstractMatrix{Float64}},
    P,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_gold::Matrix{Float64}, 
    tr_zeta::Matrix{Float64}, 
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, _, type_beta = parameter_optimization
    _, _, _, _, _, _, _, _, _, type_condition = parameter_backtracking

    for it=1:1:iterations 

        # Test if gradnorm is smaller than tolerance 
        if gradnorm < tolerance_grad || rel_cost < tolerance_cost || cost_X < cost_max
            return X, alpha, cost_X, rel_cost, gradnorm, it-1, rel_X
        end

        # Compute alpha, cost_X and X, g, and tr_zeta
        if type_condition == "armijo"
            _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, parameter_manifold, parameter_backtracking)  

            copyto!(tr_gold, g)
            grad!(g, HX, X, parameter_manifold)

            copyto!(tr_zeta, zeta.p)
            transp!(tr_zeta, tr_zeta, X, Xold, parameter_manifold)
        elseif type_condition == "wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_wolfe!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)  
        elseif type_condition == "strong_wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_strong_wolfe!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)   
        end

        # Compute beta 
        if type_beta == "fr"
            beta = fletcher_reeves(g, tr_gold) 
        elseif type_beta == "dy"
            beta = dai_yuan(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        elseif type_beta == "cg"
            beta = conjugate_gradient(g, divisor) 
        elseif type_beta == "pr"
            beta = polak_ribiere(g, tr_gold, X, Xold, parameter_manifold) 
        elseif type_beta == "hs"
            beta = hestenes_stiefel(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        elseif type_beta == "ls"
            beta = liu_storey(g, tr_gold, X, Xold, divisor, parameter_manifold) 
        elseif type_beta == "hz"
            beta = hager_zhang(g, tr_gold, tr_zeta, X, Xold, parameter_manifold) 
        end 
        
        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, beta, parameter_manifold) 
        
        # Give outout
        if output && it%output_it==0
            println("# $it | F(X): $(cost_X) | Delta(F(X)): $(rel_cost) | |grad(F(X))|: $(gradnorm) | Delta(X): $(rel_X) | alpha: $(alpha)")
        end

        # Update Xold, cost_Xold, rel_cost, divisor and gradnorm  
        rel_X = norm(X.X - Xold.X)/norm(Xold.X)
        ManifoldCopy!(Xold, X)
        rel_cost = abs(cost_X - cost_Xold) / abs(cost_Xold)
        cost_Xold = cost_X
        divisor = dot(g, zeta.p)
        gradnorm = norm(g, Inf)
    end

    return X, alpha, cost_X, rel_cost, gradnorm, iterations, rel_X
end

# DMRG version 
function conjugate_gradient_descent!(
    H,
    combiner_1, 
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_gold::Matrix{Float64}, 
    tr_zeta::Matrix{Float64}, 
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta,
    HX,
    direction::String, 
    index_L, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, _, type_beta = parameter_optimization
    _, _, _, _, _, _, _, _, _, type_condition = parameter_backtracking

    max_line_search = 10
    it_line_search = 0
    for it=1:1:iterations 

        # Compute alpha, cost_X and X, g, and tr_zeta
        if type_condition == "armijo"
            _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, combiner_1, divisor, cost_Xold, alpha, Hzeta, HX, direction, index_L, parameter_manifold, parameter_backtracking)  

            copyto!(tr_gold, g)
            grad!(g, HX, combiner_1, X, direction, parameter_manifold)

            copyto!(tr_zeta, zeta.p)
            transp!(tr_zeta, tr_zeta, X, Xold, direction, parameter_manifold)
        elseif type_condition == "wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_wolfe!(X, Xold, zeta, H, combiner_1, divisor, cost_Xold, alpha, Hzeta, HX, direction, index_L, g, tr_zeta, parameter_manifold, parameter_backtracking)  
        elseif type_condition == "strong_wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_strong_wolfe!(X, Xold, zeta, H, combiner_1, divisor, cost_Xold, alpha, Hzeta, HX, direction, index_L, g, tr_zeta, parameter_manifold, parameter_backtracking)   
        end

        # Log if alpha was too small 
        _, _, _, alpha_min, _ = parameter_backtracking
        if alpha <= alpha_min 
            it_line_search += 1
        end

        # Compute beta 
        if type_beta == "fr"
            beta = fletcher_reeves(g, tr_gold) 
        elseif type_beta == "dy"
            beta = dai_yuan(g, tr_gold, tr_zeta, X, Xold, direction, parameter_manifold) 
        elseif type_beta == "cg"
            beta = conjugate_gradient(g, divisor) 
        elseif type_beta == "pr"
            beta = polak_ribiere(g, tr_gold, X, Xold, direction, parameter_manifold) 
        elseif type_beta == "hs"
            beta = hestenes_stiefel(g, tr_gold, tr_zeta, X, Xold, direction, parameter_manifold) 
        elseif type_beta == "ls"
            beta = liu_storey(g, tr_gold, X, Xold, divisor, direction, parameter_manifold) 
        elseif type_beta == "hz"
            beta = hager_zhang(g, tr_gold, tr_zeta, X, Xold, direction, parameter_manifold) 
        end 
        
        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, beta, direction, parameter_manifold) 
        
        # Give outout
        if output && it%output_it==0
            println("# $it | F(X): $(cost_X) | Delta(F(X)): $(rel_cost) | |grad(F(X))|: $(gradnorm) | Delta(X): $(rel_X) | alpha: $(alpha)")
        end

        # Update Xold, cost_Xold, rel_cost, divisor and gradnorm  
        rel_X = norm(X.X - Xold.X)/norm(Xold.X)
        ManifoldCopy!(Xold, X)
        rel_cost = abs(cost_X - cost_Xold) / abs(cost_Xold)
        cost_Xold = cost_X
        divisor = dot(g, zeta.p)
        gradnorm = norm(g)

        # Test if gradnorm is smaller than tolerance 
        if gradnorm < tolerance_grad || rel_cost < tolerance_cost || cost_X < cost_max
            return X, alpha, cost_X, rel_cost, gradnorm, it, rel_X, 0
        elseif it_line_search == max_line_search
            return X, alpha, cost_X, rel_cost, gradnorm, it, rel_X, 2
        end
    end

    return X, alpha, cost_X, rel_cost, gradnorm, iterations, rel_X, 1
end
