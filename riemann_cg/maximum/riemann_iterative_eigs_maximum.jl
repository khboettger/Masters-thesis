### Function using the iterative Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

include("../riemann_iterative_eigs.jl")

## Function

# Iterative Riemann method - normal version 
function riemann_iterative_eigs_maximum(
    H,
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    return riemann_iterative_eigs(-H, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end

# Iterative Riemann method - normal version with shift
function riemann_iterative_eigs_maximum(
    H,
    alpha::Float64, 
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the normal case
    return riemann_iterative_eigs_maximum(H_tilde, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end

# Iterative Riemann method - normal version with higher eigenpairs 
function riemann_iterative_eigs_maximum(
    H::Matrix{Float64},
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
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
    return riemann_iterative_eigs_maximum(H_tilde, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end

# Iterative Riemann method - normal version with higher eigenpairs and shift
function riemann_iterative_eigs_maximum(
    H::Matrix{Float64},
    alpha::Float64,
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the normal case
    return riemann_iterative_eigs_maximum(H_tilde, Z, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)   
end