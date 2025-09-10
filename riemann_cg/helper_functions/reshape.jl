## New reshape functions

# New reshape functions necessary for direction=="right"
function reshape_mM_n(
    X::Matrix{Float64}, 
    m::Int64, 
    n::Int64, 
    M::Int64
    )

    tmp = reshape(X, m, n*M)
    result = zeros(typeof(X[1, 1]), m*M, n)
    for l=0:M-1
        result[l*m+1:(l+1)*m, :] = tmp[:, l*n+1:(l+1)*n]
    end
    return result 
end

function reshape_mn_M(
    X::Matrix{Float64}, 
    m::Int64, 
    n::Int64, 
    M::Int64
    )

    result = zeros(typeof(X[1, 1]), m, n*M)
    for l=0:M-1
        result[:,l*n+1:(l+1)*n] = X[l*m+1:(l+1)*m, :]
    end
    
    return reshape(result, m*n, M) 
end


# Combine reshape functions for both directions
function reshape_low_rank(
    X::Matrix{Float64}, 
    direction::String,
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold

    if direction == "left"
        return reshape(X, m, n*M)
    else 
        return reshape_mM_n(X, m, n, M)
    end
end 

function reshape_Stiefel(
    X::Matrix{Float64}, 
    direction::String,
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold

    if direction == "left"
        return reshape(X, m*n, M)
    else 
        return reshape_mn_M(X, m, n, M)
    end
end 
