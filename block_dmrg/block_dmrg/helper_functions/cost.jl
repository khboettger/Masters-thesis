## Cost function

# For block version 
function cost(
    X::Matrix{Float64}, 
    H::ProjMPO, 
    L::Int64, 
    combiner_1
    )

    # Turn X into an ITensor
    index_L = Index(L, "index_L")
    X_tensor_combined = ITensor(X, inds(combiner_1)[1], index_L)
    X_tensor = combiner_1 * X_tensor_combined

    # Compute cost function
    HX = product(H, X_tensor)

    return 0.5 * real(storage(X_tensor * HX)[1])
end

# For add-ons
function cost(
    Array_X::Matrix{Float64}, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    L::Int64, 
    combiner_1
    )
    
    # Compute Array_HX
    Array_HX = []
    inds_ITensor_X_1 = 0
    for l=1:1:L
        ITensor_X_combined_l = ITensor(Array_X[:, l], inds(combiner_1)[1])
        ITensor_X_l = combiner_1 * ITensor_X_combined_l

        HX_l = product(ProjMPO_MPS_H, ITensor_X_l)
        Array_HX_l = array(HX_l)
        push!(Array_HX, Array_HX_l)

        if l == 1
            inds_ITensor_X_1 = inds(HX_l)
        end
    end 
    Array_HX = stack(Array_HX)

    # Compute ITensor_HX
    index_L = Index(L, "index_L")
    ITensor_HX = ITensor(Array_HX, inds_ITensor_X_1, index_L)

    # Compute ITensor_X 
    ITensor_X_combined = ITensor(Array_X, inds(combiner_1)[1], index_L)
    ITensor_X = combiner_1 * ITensor_X_combined
    
    return 0.5 * real(storage(ITensor_X * ITensor_HX)[1])
end

