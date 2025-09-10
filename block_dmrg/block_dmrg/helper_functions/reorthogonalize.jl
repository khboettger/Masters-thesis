## Function used to reorthogonalize the eigenspace solution 

function reorthogonalization(
    direction::String,
    Array_U, 
    Array_S, 
    Array_V, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64 
    )

    if direction=="left" 
        Array_X_low_rank = reshape(Array_U*Array_S*Array_V', dims_left*dims_right, L)
        tmp_U, _, tmp_V = svd(Array_X_low_rank, full = false)
        Array_X_low_rank = tmp_U * tmp_V' 
    else 
        Array_X_low_rank = reshape_mn_M(Array_U*Array_S*Array_V', dims_left, dims_right, L)
        tmp_U, _, tmp_V = svd(Array_X_low_rank, full = false)
        Array_X_low_rank = tmp_U * tmp_V' 
    end

    return Array_X_low_rank
end