## Helper Function used to implement the rank adaptation strategy 

# With Riemann - block version 
function rank_adaptation_with_riemann(
    ProjMPO_H::ProjMPO, 
    combiner_1, 
    Array_X::Matrix{Float64}, 
    cutoff::Float64
    )

    # Turn Array_X into ITensor_X 
    index_L = Index(L, "index_L")
    ITensor_X_combined = ITensor(Array_X, inds(combiner_1)[1], index_L)
    ITensor_X = combiner_1 * ITensor_X_combined

    # Compute the local residual
    xi = product(ProjMPO_H, ITensor_X)
    xi = array(combiner_1 * xi)
    Omega = 0.5 * (Array_X' * xi + xi' * Array_X)
    res_norm = norm(xi - Array_X * Omega)

    # Check whether the rank needs and can be increased
    #if res_norm>cutoff # Absolute error
    if res_norm/norm(xi)>cutoff # Relative error
        update = true
    else 
        update = false
    end

    return update
end

# With Riemann - with add-ons 
function rank_adaptation_with_riemann(
    ProjMPO_MPS_H::ProjMPO_MPS, 
    combiner_1, 
    Array_X::Matrix{Float64}, 
    cutoff::Float64, 
    L::Int64
    )

    # Compute Array_HX
    Array_HX = []
    inds_ITensor_X_1 = 0
    for l=1:1:L
        ITensor_X_combined_l = ITensor(Array_X[:, l], inds(combiner_1)[1])
        ITensor_X_l = combiner_1 * ITensor_X_combined_l

        if l == 1
            inds_ITensor_X_1 = inds(ITensor_X_l)
        end

        HX_l = product(ProjMPO_MPS_H, ITensor_X_l)
        Array_HX_l = array(HX_l)
        push!(Array_HX, Array_HX_l)
    end 
    Array_HX = stack(Array_HX)

    # Compute ITensor_HX
    index_L = Index(L, "index_L")
    ITensor_HX = ITensor(Array_HX, inds_ITensor_X_1, index_L)

    # Compute res_norm
    xi = array(combiner_1 * ITensor_HX)
    Omega = 0.5 * (Array_X' * xi + xi' * Array_X)
    res_norm = norm(xi - Array_X * Omega)

    # Check whether the rank needs and can be increased
    #if res_norm>cutoff # Absolute error
    if res_norm/norm(xi)>cutoff # Relative error
        update = true
    else 
        update = false
    end

    return update
end

# Without Riemann 
function rank_adaptation_without_riemann(
    Array_X_reshaped::Matrix{Float64}, 
    Array_U, 
    Array_S, 
    Array_V, 
    cutoff::Float64, 
    maxdim::Int64, 
    maxrank::Int64
    )

    rank = 2 
    stop = false

    while rank+1<=maxrank && rank+1<=maxdim && stop==false
        difference = Array_X_reshaped - Array_U[:, 1:rank] * Diagonal(Array_S[1:rank]) * Array_V[:, 1:rank]'
        #if norm(difference) > cutoff # Absolute error
        if norm(difference)/norm(Array_X_reshaped) > cutoff # Relative error
            rank += 1
        else 
            stop = true
        end
    end
    
    return rank
end
