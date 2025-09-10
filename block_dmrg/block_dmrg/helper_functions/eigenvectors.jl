## Compute eigenvectors 

# For block version 
function eigenvectors(
    ProjMPO_H::ProjMPO, 
    Array_X::Matrix{Float64}, 
    L::Int64, 
    combiner_1
    )

    # Turn Array_X into ITensor_X 
    index_L = Index(L, "index_L")
    ITensor_X_combined = ITensor(Array_X, inds(combiner_1)[1], index_L)
    ITensor_X = combiner_1 * ITensor_X_combined
    
    # Compute the matrix A = ITensor_X'*ProjMPO_H*ITensor_X
    ITensor_HX = product(ProjMPO_H, ITensor_X)
    ITensor_XHX = prime(ITensor_X, index_L) * ITensor_HX

    # Calculate the spectral decomposition of ITensor_XHX
    _, ITensor_V = eigen(ITensor_XHX)

    # Calculate the eigenvectors X_tilde and eigenvalues D_tilde
    ITensor_X_tilde = ITensor_X_combined * ITensor_V

    # Turn into an array
    Array_X_tilde = array(ITensor_X_tilde)

    if typeof(Array_X_tilde)==Matrix{ComplexF64}
        return real(Array_X_tilde)
    else
        return Array_X_tilde 
    end
end

# For add-ons
function eigenvectors(
    ProjMPO_MPS_H::ProjMPO_MPS, 
    Array_X::Matrix{Float64}, 
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

    # Compute ITensor_XHX
    ITensor_XHX = prime(ITensor_X, index_L) * ITensor_HX

    # Calculate the spectral decomposition of ITensor_XHX
    _, ITensor_V = eigen(ITensor_XHX)

    # Calculate the eigenvectors X_tilde and eigenvalues D_tilde
    ITensor_X_tilde = ITensor_X_combined * ITensor_V

    return array(ITensor_X_tilde)
end