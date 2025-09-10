## Define the update of the search direction 

# Normal version 
function direction_update!(
    q::ManifoldTangentVector, 
    X::ManifoldPoint, 
    g::Matrix{Float64},
    p::Matrix{Float64},
    beta::Float64,
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold
    q.p .= -g + beta * p
    low_rank_decomp!(q, X, reshape(q.p, m, n*M))

    return q
end

# DMRG version 
function direction_update!(
    q::ManifoldTangentVector, 
    X::ManifoldPoint, 
    g::Matrix{Float64},
    p::Matrix{Float64},
    beta::Float64,
    direction::String,
    parameter_manifold
    )

    q.p .= -g + beta * p
    low_rank_decomp!(q, X, reshape_low_rank(q.p, direction, parameter_manifold))

    return q
end