### Eigenspace method

## Pakets 

using LinearAlgebra
using Arpack
include("../helper_functions/find_nearest_indices.jl")
include("../helper_functions/optimized_svd.jl")

## Functions 
function eigenspace_middle(H, Z, sigma, m, n, M, rank)
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
    Eeigen, Veigen = eigen(H_tilde)
    indices = find_nearest_indices(Eeigen, sigma, M)
    X_full = Veigen[:, indices[1]:indices[end]]

    # Compute low rank solution
    U, S, Vt = optimized_svd(reshape(X_full, m, n*M), rank, "svds")

    X_low = U * Diagonal(S) * Vt
    X_low = reshape(X_low, m*n, M)

    U, _, V = svd(X_low, full = false)
    X_low = U * V' 

    U_low, S_low, V_lowt = optimized_svd(reshape(X_low, m, n*M), rank, "svds")

    return X_low, U_low, S_low, V_lowt
end