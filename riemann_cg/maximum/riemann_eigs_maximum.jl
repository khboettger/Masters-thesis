### Function using the Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

using LinearAlgebra 
using ITensors 
using ITensorMPS

include("../riemann_eigs.jl")

## Function

# Riemann method - normal version 
function riemann_eigs_maximum(
    H,
    X0::ManifoldPoint, 
    parameter_manifold, 
    parameter_backtracking, 
    parameter_optimization
    )

    return riemann_eigs(-H, X0, parameter_manifold, parameter_backtracking, parameter_optimization)
end

# Riemann method - normal version with higher eigenpairs
function riemann_eigs_maximum(
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

    # Solve using the normal case
    return riemann_eigs_maximum(H_tilde, X0, parameter_manifold, parameter_backtracking, parameter_optimization)
end

# Riemann method - normal version with higher eigenpairs and shift
function riemann_eigs_maximum(
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
    return riemann_eigs_maximum(H_tilde, Z, X0, parameter_manifold, parameter_backtracking, parameter_optimization)   
end