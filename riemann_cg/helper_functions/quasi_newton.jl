## Define the quasi newton descent

# Normal version
function quasi_newton!(
    H,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_zeta::Matrix{Float64}, 
    tr_gold::Matrix{Float64},  
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    list_s,
    list_y,
    list_r,
    p::Matrix{Float64},
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, L, _ = parameter_optimization
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
        elseif type_condition == "wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_wolfe!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)  
        elseif type_condition == "strong_wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_strong_wolfe!(X, Xold, zeta, H, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)   
        end

        # Update s, y and r
        s = transp(alpha*zeta.p, X, Xold, parameter_manifold)
        delta = norm(alpha*zeta.p) / norm(s)

        transp!(tr_gold, tr_gold, X, Xold, parameter_manifold)
        y = g/delta - tr_gold
        r = 1/dot(s, y)
        push!(list_s, s)
        push!(list_y, y)
        push!(list_r, r)
        
        # Update gamma 
        gamma = dot(s, y)/dot(y, y)

        # Delete old s, y and r 
        if length(list_s) > L 
            popfirst!(list_s)
            popfirst!(list_y)
            popfirst!(list_r)
        end

        # Transport all s and y to the new tangent space 
        for i=1:1:length(list_s)-1
            list_s[i] = transp(list_s[i], X, Xold, parameter_manifold)
            list_y[i] = transp(list_y[i], X, Xold, parameter_manifold)
        end

        # Update p 
        lbfgs_update!(p, gamma, g, list_s, list_y, list_r)
        project_full!(p, X, p, parameter_manifold)
        
        # Update zeta 
        direction_update!(zeta, X, p, tr_zeta, 0.0, parameter_manifold) 
        
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

# Version with Kronecker structure and higher eigenpairs
function quasi_newton!(
    H::Vector{AbstractMatrix{Float64}},
    P,
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_zeta::Matrix{Float64}, 
    tr_gold::Matrix{Float64},  
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta::Matrix{Float64},
    HX::Matrix{Float64},
    list_s,
    list_y,
    list_r,
    p::Matrix{Float64},
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, L, _ = parameter_optimization
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
        elseif type_condition == "wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_wolfe!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)  
        elseif type_condition == "strong_wolfe"
            copyto!(tr_gold, g)

            _, alpha, cost_X, g, tr_zeta = backtracking_strong_wolfe!(X, Xold, zeta, H, P, divisor, cost_Xold, alpha, Hzeta, HX, g, tr_zeta, parameter_manifold, parameter_backtracking)   
        end

        # Update s, y and r
        s = transp(alpha*zeta.p, X, Xold, parameter_manifold)
        delta = norm(alpha*zeta.p) / norm(s)

        transp!(tr_gold, tr_gold, X, Xold, parameter_manifold)
        y = g/delta - tr_gold
        r = 1/dot(s, y)
        push!(list_s, s)
        push!(list_y, y)
        push!(list_r, r)
        
        # Update gamma 
        gamma = dot(s, y)/dot(y, y)

        # Delete old s, y and r 
        if length(list_s) > L 
            popfirst!(list_s)
            popfirst!(list_y)
            popfirst!(list_r)
        end

        # Transport all s and y to the new tangent space 
        for i=1:1:length(list_s)-1
            list_s[i] = transp(list_s[i], X, Xold, parameter_manifold)
            list_y[i] = transp(list_y[i], X, Xold, parameter_manifold)
        end

        # Update p 
        lbfgs_update!(p, gamma, g, list_s, list_y, list_r)
        project_full!(p, X, p, parameter_manifold)
        
        # Update zeta 
        direction_update!(zeta, X, p, tr_zeta, 0.0, parameter_manifold) 
        
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
function quasi_newton!(
    H,
    combiner_1, 
    X::ManifoldPoint, 
    Xold::ManifoldPoint, 
    g::Matrix{Float64}, 
    zeta::ManifoldTangentVector, 
    tr_zeta::Matrix{Float64}, 
    tr_gold::Matrix{Float64},  
    gradnorm::Float64, 
    alpha::Float64, 
    cost_X::Float64, 
    cost_Xold::Float64, 
    rel_cost::Float64,
    rel_X::Float64, 
    divisor::Float64,
    Hzeta,
    HX,
    list_s,
    list_y,
    list_r,
    p::Matrix{Float64},
    direction::String, 
    index_L,
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization)

    tolerance_cost, tolerance_grad, iterations, cost_max, output, output_it, _, L, _ = parameter_optimization
    _, _, _, _, _, _, _, _, _, type_condition = parameter_backtracking

    max_line_search = 10
    it_line_search = 0
    for it=1:1:iterations 

        # Compute alpha, cost_X and X, g, and tr_zeta
        if type_condition == "armijo"
            _, alpha, cost_X = backtracking_armijo!(X, Xold, zeta, H, combiner_1, divisor, cost_Xold, alpha, Hzeta, HX, direction, index_L, parameter_manifold, parameter_backtracking) 

            copyto!(tr_gold, g)
            grad!(g, HX, combiner_1, X, direction, parameter_manifold)
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

        # Update s, y and r
        s = transp(alpha*zeta.p, X, Xold, direction, parameter_manifold)
        delta = norm(alpha*zeta.p) / norm(s)

        transp!(tr_gold, tr_gold, X, Xold, direction, parameter_manifold)
        y = g/delta - tr_gold
        r = 1/dot(s, y)
        push!(list_s, s)
        push!(list_y, y)
        push!(list_r, r)
        
        # Update gamma 
        gamma = dot(s, y)/dot(y, y)

        # Delete old s, y and r 
        if length(list_s) > L 
            popfirst!(list_s)
            popfirst!(list_y)
            popfirst!(list_r)
        end

        # Transport all s and y to the new tangent space 
        for i=1:1:length(list_s)-1
            list_s[i] = transp(list_s[i], X, Xold, direction, parameter_manifold)
            list_y[i] = transp(list_y[i], X, Xold, direction, parameter_manifold)
        end

        # Update p 
        lbfgs_update!(p, gamma, g, list_s, list_y, list_r)
        project_full!(p, X, p, direction, parameter_manifold)
        
        # Update zeta 
        direction_update!(zeta, X, p, tr_zeta, 0.0, direction, parameter_manifold) 
        
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