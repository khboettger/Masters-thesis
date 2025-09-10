### Function using the Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

using QuadGK 

include("../riemann_eigs.jl")
include("../helper_functions/polynomial_filter.jl") 
include("../helper_functions/eigenpairs.jl")
include("../helper_functions/eigenvalue_index.jl")

## Function

# Riemann method - normal version 
function riemann_eigs_middle(
    H,
    X0::ManifoldPoint, 
    parameter_middle,
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )

    sigma, lambda_max, lambda_min = parameter_middle

    H_tilde = jackson_chebyshev_filter_matrix(H, sigma, lambda_min, lambda_max) 

    return riemann_eigs(-H_tilde, X0, parameter_manifold, parameter_backtracking, parameter_optimization)    
end

# Riemann method - normal version with higher eigenpairs
function riemann_eigs_middle(
    H::Matrix{Float64},
    Z::Matrix{Float64},
    X0::ManifoldPoint,
    parameter_middle, 
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

    # Solve using the normal case
    return riemann_eigs_middle(H_tilde, X0, parameter_middle, parameter_manifold, parameter_backtracking, parameter_optimization)  
end

# Riemann method - normal version with higher eigenpairs and shift
function riemann_eigs_middle(
    H::Matrix{Float64},
    alpha::Float64,
    Z::Matrix{Float64},
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )
    
    # Shift H
    H_tilde = H - alpha * I 

    # Solve using the normal case
    return riemann_eigs_middle(H_tilde, Z, X0, parameter_middle, parameter_manifold, parameter_backtracking, parameter_optimization) 
end