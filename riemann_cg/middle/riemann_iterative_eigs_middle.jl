### Function using the iterative Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

include("../riemann_iterative_eigs.jl")
include("../helper_functions/polynomial_filter.jl") 
include("../helper_functions/eigenpairs.jl")
include("../helper_functions/eigenvalue_index.jl")

## Function

# Iterative Riemann method - normal version 
function riemann_iterative_eigs_middle(
    H,
    list_X0::Vector{ManifoldPoint}, 
    parameter_middle,
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    sigma, lambda_max, lambda_min = parameter_middle 

    H_tilde = jackson_chebyshev_filter_matrix(H, sigma, lambda_min, lambda_max)

    return riemann_iterative_eigs(-H_tilde, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)                
end

# Iterative Riemann method - normal version with shift
function riemann_iterative_eigs_middle(
    H,
    alpha::Float64, 
    list_X0::Vector{ManifoldPoint}, 
    parameter_middle,
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the normal case
    return riemann_iterative_eigs_middle(H_tilde, list_X0, parameter_middle, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end

# Iterative Riemann method - normal version with higher eigenpairs 
function riemann_iterative_eigs_middle(
    H::Matrix{Float64},
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_middle,
    parameter_manifold_overall, 
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
    return riemann_iterative_eigs_middle(H_tilde, list_X0, parameter_middle, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end

# Iterative Riemann method - normal version with higher eigenpairs and shift
function riemann_iterative_eigs_middle(
    H::Matrix{Float64},
    alpha::Float64,
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_middle,
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the normal case
    return riemann_iterative_eigs_middle(H_tilde, Z, list_X0, parameter_middle, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end
