## Helper function used to compute the low-rank decomposition of Array_X 

# With Riemann 
function low_rank_decomposition_with_riemann(
    direction::String, 
    Array_X::Matrix{Float64}, 
    L::Int64, 
    dims_left::Int64, 
    dims_right::Int64, 
    rank::Int64
    )
    
    if direction=="left"
        # Reshape Array_X into column format
        Array_X_reshaped = reshape(Array_X, dims_left, dims_right*L)
        
        # Compute the SVD
        Array_U, Array_S, Array_V = optimized_svd(Array_X_reshaped, rank)
    else
        # Reshape Array_X into row format
        Array_X_reshaped = reshape_mM_n(Array_X, dims_left, dims_right, L)

        # Compute the SVD
        Array_U, Array_S, Array_V = optimized_svd(Array_X_reshaped, rank)  
    end

    return Array_U, Array_S, Array_V
end

# Without Riemann
function low_rank_decomposition_without_riemann(
    direction::String, 
    Array_X::Matrix{Float64}, 
    L::Int64, 
    dims_left::Int64, 
    dims_right::Int64, 
    adaptation::Bool, 
    maxdim::Int64, 
    mindim::Int64, 
    cutoff::Float64
    )

    if direction=="left"
        # Reshape Array_X into column format
        Array_X_reshaped = reshape(Array_X, dims_left, dims_right*L)

        # Compute the correct rank 
        maxrank = min(dims_left, dims_right*L) 
        if adaptation==true
            # Compute the SVD
            Array_U, Array_S, Array_V = svd(Array_X_reshaped, full = false)
            
            rank = rank_adaptation_without_riemann(Array_X_reshaped, Array_U, Array_S, Array_V, cutoff, maxdim, maxrank)

            # Find a low rank decomposition
            Array_U = Array_U[:, 1:rank]
            Array_S = Diagonal(Array_S[1:rank])
            Array_V = Array_V[:, 1:rank]
        else
            rank = min(mindim, maxrank)

            # Compute the SVD
            Array_U, Array_S, Array_V = optimized_svd(Array_X_reshaped, rank) 
        end
    else
        # Reshape Array_X into row format
        Array_X_reshaped = reshape_mM_n(Array_X, dims_left, dims_right, L)
        
        # Compute the correct rank 
        maxrank = min(dims_left*L, dims_right) 
        if adaptation==true
            # Compute the SVD
            Array_U, Array_S, Array_V = svd(Array_X_reshaped, full = false) 
            
            rank = rank_adaptation_without_riemann(Array_X_reshaped, Array_U, Array_S, Array_V, cutoff, maxdim, maxrank)

            # Find a low rank decomposition
            Array_U = Array_U[:, 1:rank]
            Array_S = Diagonal(Array_S[1:rank])
            Array_V = Array_V[:, 1:rank]
        else
            rank = min(mindim, maxrank)

            # Compute the SVD
            Array_U, Array_S, Array_V = optimized_svd(Array_X_reshaped, rank) 
        end
    end

    return Array_U, Array_S, Array_V, rank
end
