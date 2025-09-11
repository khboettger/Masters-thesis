## Helper function, which computes the initial guess using random matrices 

function initial_guess_random(
    direction::String, 
    dims_left::Int64, 
    dims_right::Int64, 
    rank::Int64, 
    L::Int64
    )

    if direction=="left"
        Array_U_initial, _ = qr(randn(dims_left, rank))
        Array_S_initial = Matrix(1.0I, rank, rank)
        Array_V_initial, _ = qr(randn(L * dims_right, rank))
    else
        Array_U_initial, _ = qr(randn(L * dims_left, rank))
        Array_S_initial = Matrix(1.0I, rank, rank)
        Array_V_initial, _ = qr(randn(dims_right, rank))
    end

    return Array_U_initial[:, 1:rank], Array_S_initial, Array_V_initial[:, 1:rank]
end
