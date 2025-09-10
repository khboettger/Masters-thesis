### Iterative eigenspace method

## Pakets 

using LinearAlgebra
using Arpack
include("helper_functions/optimized_eigen.jl")
include("helper_functions/optimized_svd.jl")

## Functions 
function eigenspace_iterative(H, Z, m, n, M, rank)
    # Need to break, if dimensions are not right
    if rank>min(m,n)
        error("Dimensions are not correct!")
    end

    # Construct new Hamiltonian 
    if length(Z) > 0
        P = (I - Z * Z')
        H_tilde = P * H * P 
    else 
        H_tilde = H 
    end

    # Compute the full solution  
    X_full = optimized_eigen(H_tilde, M)

    # Initialize X, U, S, V
    X = zeros(m*n, M)
    U = zeros(m, rank, M)
    S = zeros(rank, M)
    Vt = zeros(rank, n, M)

    # Calculate low rank versions of the columns
    for l=1:1:M 
        U1_l, S1_l, V1_lt = optimized_svd(reshape(X_full[:, l], m, n), rank, "svds")

        X_l = U1_l * Diagonal(S1_l) * V1_lt
        X_l = reshape(X_l, m*n, 1)

        U2_l, _, V2_l = svd(X_l, full = false)
        X_l = U2_l * V2_l' 

        U3_l, S3_l, V3_lt = optimized_svd(reshape(X_l, m, n), rank, "svds")

        # Save X_l, U_l, S_l, V_l in X, U, S, V 
        X[:, l] = X_l
        U[:, :, l] = U3_l
        S[:, l] = S3_l
        Vt[:, :, l] = V3_lt
    end

    return X, U, S, Vt
end