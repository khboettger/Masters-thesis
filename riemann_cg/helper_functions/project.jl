
## Define the projection onto the tangent space of the manifold

# Define the Stiefel projection
function project_Stiefel!(
    q::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::Matrix{Float64}
    )

    Omega = X.X' * p
    T = eltype(q)
    copyto!(q, p)
    mul!(q, X.X, Omega + Omega', T(-0.5), true)

    return q
end

# Define the low rank projections - normal version
function project_low_rank!(
    q::Matrix{Float64},
    X::ManifoldPoint, 
    p::Matrix{Float64}
    )

    A = X.U' * p
    B = p * X.Vt'
    C = A * X.Vt'
    D = X.U * C

    q .= X.U * A .+ B * X.Vt .- D * X.Vt
    
    return q
end

# Define the low rank projections - DMRG version
function project_low_rank!(
    q::Matrix{Float64},
    X::ManifoldPoint, 
    p::Matrix{Float64}, 
    direction::String, 
    parameter_manifold
    )

    A = X.U' * p
    B = p * X.Vt'
    C = A * X.Vt'
    D = X.U * C

    q .= reshape_Stiefel(X.U * A .+ B * X.Vt .- D * X.Vt, direction, parameter_manifold)
    
    return q
end

# Define the full projection - normal version 
function project_full!(
    q::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::Matrix{Float64},
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold

    # Project X onto the Stiefel tangent space of p
    project_Stiefel!(q, X, p)

    # Project further onto the low rank tangent space of p
    project_low_rank!(reshape(q, m, n*M), X, reshape(q, m, n*M))

    return q 
end

# Define the full projection - DMRG version
function project_full!(
    q::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::Matrix{Float64},
    direction::String,
    parameter_manifold
    )

    # Project X onto the Stiefel tangent space of p
    project_Stiefel!(q, X, p)

    # Project further onto the low rank tangent space of p
    project_low_rank!(q, X, reshape_low_rank(q, direction, parameter_manifold), direction, parameter_manifold)

    return q 
end


