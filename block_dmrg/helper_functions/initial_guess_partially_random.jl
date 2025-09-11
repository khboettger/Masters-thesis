## Helper function, which inflates the former solution with a random vector 

function initial_guess_partially_random(
    psies::Vector{MPS}, 
    j::Int64, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64, 
    combiners, 
    direction::String, 
    rank::Int64
    )

    # Compute the former solution
    Array_X = former_eigenvectors(psies, j, dims_left, dims_right, L, combiners)
    
    if direction=="left"
        # Reshape Array_X
        Array_X_reshaped = reshape(Array_X, dims_left, dims_right*L)
        
        # Compute the low rank decomposition
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank-1)
        
        # Build new components
        new_right, _ = qr(randn(dims_left, 1))
        new_right = new_right[:, 1:1]
        Array_S_initial_new = Matrix(1.0I, rank, rank)
        Array_S_initial_new[1:(rank-1), 1:(rank-1)] = Array_S_initial 
        new_left, _ = qr(randn(L * dims_right, 1))
        new_left = new_left[:, 1:1]
        
        Array_U_initial = hcat(Array_U_initial, new_right)
        Array_V_initial = hcat(Array_V_initial, new_left)
    else
        # Reshape Array_X
        Array_X_reshaped = reshape_mM_n(Array_X, dims_left, dims_right, L)
        
        # Compute the low rank decomposition
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank-1)
        
        # Build new components
        new_right, _ = qr(randn(dims_left * L, 1))
        new_right = new_right[:, 1:1]
        Array_S_initial_new = Matrix(1.0I, rank, rank)
        Array_S_initial_new[1:(rank-1), 1:(rank-1)] = Array_S_initial 
        new_left, _ = qr(randn(dims_right, 1))
        new_left = new_left[:, 1:1]

        Array_U_initial = hcat(Array_U_initial, new_right)
        Array_V_initial = hcat(Array_V_initial, new_left)
    end

    return Array_U_initial, Array_S_initial_new, Array_V_initial
end

