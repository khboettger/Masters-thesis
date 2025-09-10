### Function using the Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

using LinearAlgebra
using Arpack 
using ITensors 
using ITensorMPS

include("helper_functions/manifold.jl")
include("helper_functions/backtracking.jl")
include("helper_functions/beta.jl")
include("helper_functions/cost.jl")
include("helper_functions/direction_update.jl")
include("helper_functions/grad.jl")
include("helper_functions/initialize.jl")
include("helper_functions/low_rank_decomp.jl")
include("helper_functions/project.jl")
include("helper_functions/retract.jl")
include("helper_functions/transp.jl")
include("helper_functions/optimized_svd.jl")
include("helper_functions/lbfgs_update.jl")
include("helper_functions/gradient_descent.jl")
include("helper_functions/conjugate_gradient_descent.jl")
include("helper_functions/quasi_newton.jl")
include("helper_functions/reshape.jl") 

## Function

# Riemann method - normal version 
function riemann_eigs(
    H,
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold 
    if n*r<M 
        error("Dimensions are not correct!")
    end

    # Initialize all necessary functions 
    X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX = initialize(H, X0, parameter_manifold, parameter_backtracking, parameter_optimization)

    # Run the optimization algorithm 
    _, _, iterations, _, output, _, type, _ = parameter_optimization
    if type == "conjugate_gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = conjugate_gradient_descent!(H, X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = gradient_descent!(H, X, Xold, g, zeta, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "quasi_newton"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = quasi_newton!(H, X, Xold, g, zeta, tr_zeta, tr_gold, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, list_s, list_y, list_r, p ,parameter_manifold, parameter_backtracking, parameter_optimization)   
    end  

    # Give outputs
    if output
        if it == 1
            println("The algorithm converges without iteration with F(X): $(cost_X), |grad(F(X))|: $(gradnorm)")
        elseif it == iterations 
            println("Function breaks due to high iteration number with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        else
            println("The algorithm converges with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        end
    end

    return X, cost_X, rel_cost, gradnorm, it, rel_X
end

# Riemann method - normal version with higher eigenpairs
function riemann_eigs(
    H::Matrix{Float64},
    Z::Matrix{Float64},
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )
    
    # Construct new Hamiltonian 
    if length(Z) > 0
        P = (I - Z * Z')
        H_tilde = P * H * P 
    else 
        H_tilde = H 
    end

    # Solve using the standard Riemann eigs method 
    return riemann_eigs(H_tilde, X0, parameter_manifold, parameter_backtracking, 
    parameter_optimization
    )
end

# Riemann method - normal version with higher eigenpairs and shift
function riemann_eigs(
    H::Matrix{Float64},
    alpha::Float64,
    Z::Matrix{Float64},
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )
    
    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the standard Riemann eigs method 
    return riemann_eigs(H_tilde, Z, X0, parameter_manifold, parameter_backtracking, 
    parameter_optimization
    )
end

# Riemann method - normal version with Kronecker structure and higher eigenpairs
function riemann_eigs(
    H_kronecker::Vector{AbstractMatrix{Float64}},
    Z::Matrix{Float64},
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )
    
    # Construct the projection matrix
    if length(Z) > 0
        P = (I - Z * Z')
    else 
        P = I
    end

    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold 
    if n*r<M
        error("Dimensions are not correct!")
    end

    # Initialize all necessary functions 
    X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX = initialize(H_kronecker, P, X0, parameter_manifold, parameter_backtracking, parameter_optimization) 

    # Run the optimization algorithm 
    _, _, iterations, _, output, _, type, _ = parameter_optimization
    if type == "conjugate_gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = conjugate_gradient_descent!(H_kronecker, P, X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = gradient_descent!(H_kronecker, P, X, Xold, g, zeta, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "quasi_newton"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X = quasi_newton!(H_kronecker, P, X, Xold, g, zeta, tr_zeta, tr_gold, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, list_s, list_y, list_r, p ,parameter_manifold, parameter_backtracking, parameter_optimization)   
    end  

    # Give outputs
    if output
        if it == 1
            println("The algorithm converges without iteration with F(X): $(cost_X), |grad(F(X))|: $(gradnorm)")
        elseif it == iterations 
            println("Function breaks due to high iteration number with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        else
            println("The algorithm converges with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        end
    end

    return X, cost_X, rel_cost, gradnorm, it, rel_X
end

# Riemann method - normal version with Kronecker structure and higher eigenpairs and shift
function riemann_eigs(
    H_kronecker::Vector{AbstractMatrix{Float64}},
    alpha::Float64,
    Z::Matrix{Float64},
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )
    
    # Change H_kronecker 
    if alpha != 0.0
        H_kronecker_tilde = copy(H_kronecker)

        m, n, _ , _ = parameter_manifold
        Im = 1.0 * Matrix(I, m, m)
        In = 1.0 * Matrix(I, n, n)

        push!(H_kronecker_tilde, -alpha*Im)
        push!(H_kronecker_tilde, In)
    else
        H_kronecker_tilde = copy(H_kronecker)
    end

    return riemann_eigs(H_kronecker_tilde, Z, X0, parameter_manifold, parameter_backtracking, parameter_optimization)
end

# Riemann method - DMRG version 
function riemann_eigs(
    H,
    combiner_1, 
    X0::ManifoldPoint, 
    direction::String,
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold

    if direction=="left" && n*r<M 
        error("Dimensions are not correct!")
    end

    if direction=="right" && m*r<M 
        error("Dimensions are not correct!")
    end

    # Initialize all necessary functions 
    X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX, index_L = initialize(H, combiner_1, X0, direction, parameter_manifold, parameter_backtracking, parameter_optimization)   

    # Run the optimization algorithm 
    _, _, _, _, output, _, type, _ = parameter_optimization
    if type == "conjugate_gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X, breaks = conjugate_gradient_descent!(H, combiner_1, X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, direction, index_L, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "gradient_descent"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X, breaks = gradient_descent!(H, combiner_1, X, Xold, g, zeta, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, direction, index_L, parameter_manifold, parameter_backtracking, parameter_optimization)  
    elseif type == "quasi_newton"
        X, alpha, cost_X, rel_cost, gradnorm, it, rel_X, breaks = quasi_newton!(H, combiner_1, X, Xold, g, zeta, tr_zeta, tr_gold, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, HX, list_s, list_y, list_r, p, direction, index_L, parameter_manifold, parameter_backtracking, parameter_optimization)    
    end  

    # Give outputs
    if output
        if it == 1
            println("The algorithm converges without iteration with F(X): $(cost_X), |grad(F(X))|: $(gradnorm)")
        elseif breaks != 0
            println("Function breaks with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        else
            println("The algorithm converges with it: $it, F(X): $(cost_X), Delta(F(X)): $(rel_cost), |grad(F(X))|: $(gradnorm), Delta(X): $(rel_X), alpha: $(alpha)")
        end
    end

    return X, cost_X, rel_cost, gradnorm, it, rel_X, breaks
end