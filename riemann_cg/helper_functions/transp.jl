## Define the vector transport

# Normal version
function transp!( 
    q::Matrix{Float64},
    p::Matrix{Float64}, 
    X::ManifoldPoint, 
    Y::ManifoldPoint, 
    parameter_manifold 
    )

    project_full!(q, X, p, parameter_manifold)
    A = Y.X' * q 
    q .-= Y.X * A 

    return q
end

function transp( 
    p::Matrix{Float64}, 
    X::ManifoldPoint, 
    Y::ManifoldPoint, 
    parameter_manifold 
    )

    m, n, M, _ = parameter_manifold

    q = zeros(m*n, M)
    transp!(q, p, X, Y, parameter_manifold )

    return q
end

# DMRG version
function transp!( 
    q::Matrix{Float64},
    p::Matrix{Float64}, 
    X::ManifoldPoint, 
    Y::ManifoldPoint, 
    direction::String, 
    parameter_manifold 
    )

    project_full!(q, X, p, direction, parameter_manifold)
    A = Y.X' * q 
    q .-= Y.X * A 

    return q
end

function transp( 
    p::Matrix{Float64}, 
    X::ManifoldPoint, 
    Y::ManifoldPoint, 
    direction::String, 
    parameter_manifold 
    )

    m, n, M, _ = parameter_manifold

    q = zeros(m*n, M)
    transp!(q, p, X, Y, direction, parameter_manifold )

    return q
end