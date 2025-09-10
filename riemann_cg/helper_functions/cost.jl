## Define the cost function

# Normal version 
function cost(
    H::Matrix{Float64}, 
    X::Matrix{Float64}, 
    HX::Matrix{Float64},
    parameter_manifold
    )
    mul!(HX, H, X)

    return 0.5 * dot(X, HX)
end

# Normal version - Kronecker structure
function cost(
    H_kronecker::Vector{AbstractMatrix{Float64}}, 
    X::Matrix{Float64}, 
    HX::Matrix{Float64},
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold
    k = length(H_kronecker)

    H_BX_l = zeros(m, n)
    HX_reshaped_l_k = zeros(m, n)
    f = 0.0

    for l = 1:1:M
        HX_reshaped_l = zeros(m, n)

        X_l = @view X[:, l]
        X_reshaped_l = reshape(X_l, m, n)

        for i = 1:2:k
            A = H_kronecker[i]
            B = H_kronecker[i+1]

            mul!(H_BX_l, B, X_reshaped_l)
            mul!(HX_reshaped_l_k, H_BX_l, A)

            HX_reshaped_l .+= HX_reshaped_l_k       
        end

        HX[:, l] .= vec(HX_reshaped_l)
        HX_l = @view HX[:, l]

        f += 0.5 * dot(X_l, HX_l)
    end

    return f
end

# Normal version - Kronecker structure and higher eigenpairs
function cost(
    H_kronecker::Vector{AbstractMatrix{Float64}}, 
    X::Matrix{Float64}, 
    HX::Matrix{Float64},
    P,
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold
    k = length(H_kronecker)

    H_BX_l = zeros(m, n)
    HX_reshaped_l_k = zeros(m, n)
    f = 0.0
    PX = P*X

    for l = 1:1:M
        HX_reshaped_l = zeros(m, n)

        X_l = @view PX[:, l]
        X_reshaped_l = reshape(X_l, m, n)

        for i = 1:2:k
            A = H_kronecker[i]
            B = H_kronecker[i+1]

            mul!(H_BX_l, B, X_reshaped_l)
            mul!(HX_reshaped_l_k, H_BX_l, A)

            HX_reshaped_l .+= HX_reshaped_l_k       
        end

        HX[:, l] .= vec(HX_reshaped_l)
        HX_l = @view HX[:, l]

        f += 0.5 * dot(X_l, HX_l)
    end

    HX .= P * HX

    return f
end

# DMRG version - ProjMPO 
function cost(
    H::ProjMPO, 
    combiner_1,
    X::Matrix{Float64}, 
    HX,
    index_L,
    parameter_manifold
    )

    # Turn X into an ITensor
    X_tensor_combined = ITensor(X, inds(combiner_1)[1], index_L)
    X_tensor = combiner_1 * X_tensor_combined

    # Compute cost function
    HX .= product(H, X_tensor)

    return 0.5 * real(storage(X_tensor * HX)[1])
end

# DMRG version - ProjMPO_MPS
function cost(
    H::ProjMPO_MPS, 
    combiner_1,
    X::Matrix{Float64}, 
    HX_tensor,
    index_L,
    parameter_manifold
    )

    _, _, M, _ = parameter_manifold

    # Compute HX
    HX_array = []
    inds_X_tensor_1 = 0
    for l=1:1:M
        X_tensor_combined_l = ITensor(X[:, l], inds(combiner_1)[1])
        X_tensor_l = combiner_1 * X_tensor_combined_l

        HX_l = product(H, X_tensor_l)
        HX_array_l = array(HX_l)
        push!(HX_array, HX_array_l)

        if l == 1
            inds_X_tensor_1 = inds(HX_l)
        end
    end 
    HX_array = stack(HX_array)

    # Compute HX_tensor
    HX_tensor .= ITensor(HX_array, inds_X_tensor_1, index_L)

    # Compute X_tensor 
    X_tensor_combined = ITensor(X, inds(combiner_1)[1], index_L)
    X_tensor = combiner_1 * X_tensor_combined
    
    return 0.5 * real(storage(X_tensor * HX_tensor)[1])
end