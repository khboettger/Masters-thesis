## Define the gradient descent

# Normal version
function gradient_descent!(
    H,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
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

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _ = parameter_optimization

    for it=1:1:iterations 
        # Test if gradnorm is smaller than tolerance 
        if gradnorm < tolerance_grad || rel_cost < tolerance_cost || cost_X < cost_max
            return X, alpha, cost_X, rel_cost, gradnorm, it-1, rel_X
        end

        # Compute alpha, cost_X and X
        _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, parameter_manifold, parameter_backtracking) 

        # Update g
        grad!(g, HX, X, parameter_manifold)

        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, 0.0, parameter_manifold)  
        
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

# Normal version - Kronecker structure and higher eigenpairs
function gradient_descent!(
    H::Vector{AbstractMatrix{Float64}},
    P,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
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

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _ = parameter_optimization

    for it=1:1:iterations 
        # Test if gradnorm is smaller than tolerance 
        if gradnorm < tolerance_grad || rel_cost < tolerance_cost || cost_X < cost_max
            return X, alpha, cost_X, rel_cost, gradnorm, it-1, rel_X
        end

        # Compute alpha, cost_X and X
        _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, parameter_manifold, parameter_backtracking) 

        # Update g
        grad!(g, HX, X, parameter_manifold)
        
        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, 0.0, parameter_manifold) 
        
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

# DMRG version
function gradient_descent!(
    H,
    combiner_1, 
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
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

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _ = parameter_optimization

    max_line_search = 10
    it_line_search = 0
    for it=1:1:iterations

        # Compute alpha, cost_X and X
        _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, combiner_1, divisor, cost_Xold, alpha, Hzeta, HX, direction, index_L, parameter_manifold, parameter_backtracking) 

        # Update g
        grad!(g, HX, combiner_1, X, direction, parameter_manifold)

        # Log if alpha was too small 
        _, _, _, alpha_min, _ = parameter_backtracking
        if alpha <= alpha_min 
            it_line_search += 1
        end

        # Update zeta 
        direction_update!(zeta, X, g, tr_zeta, 0.0, direction, parameter_manifold)  
        
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
