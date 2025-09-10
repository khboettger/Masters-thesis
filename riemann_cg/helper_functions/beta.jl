## Define different coefficients for the conjugate gradient descent

# Define the Liu-Storey coefficient - normal version 
function liu_storey(
    q::Matrix{Float64}, 
    p::Matrix{Float64}, 
    X::ManifoldPoint,
    Y::ManifoldPoint,
    divisor::Float64,
    parameter_manifold
    )

    transp!(p, p, X, Y, parameter_manifold)

    A = q - p
    B = dot(q, A)

    beta = (-1.0) * B / divisor
    
    return beta 
end 

# Define the Fletcher-Reeves coefficient - normal and DMRG version 
function fletcher_reeves(
    q::Matrix{Float64}, # q = g
    p::Matrix{Float64}, # p = gold
    )

    A = dot(q, q) 
    B = dot(p, p)

    beta = A / B 

    return beta 
end 

# Define the Dai-Yuan coefficient - normal version 
function dai_yuan(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    parameter_manifold
    )

    A = dot(q, q)

    transp!(p, p, X, Y, parameter_manifold)

    B = q - p
    C = dot(g, B)

    beta = A / C
    
    return beta 
end 

# Define the conjugate-gradient coefficient - normal and DMRG version 
function conjugate_gradient(
    q::Matrix{Float64}, # q = g 
    divisor::Float64,
    )

    A = dot(q, q)

    beta = (-1.0) * A / divisor
    
    return beta 
end 

# Define the Polak-Ribiere coefficient - normal version 
function polak_ribiere(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    X::ManifoldPoint,
    Y::ManifoldPoint,
    parameter_manifold
    )

    A = dot(p, p)
    transp!(p, p, X, Y, parameter_manifold)

    B = q - p
    C = dot(q, B)
    

    beta = C / A
    
    return beta 
end 

# Define the Hestenes-Stiefel coefficient - normal version 
function hestenes_stiefel(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    parameter_manifold
    )

    transp!(p, p, X, Y, parameter_manifold)

    A = q - p
    B = dot(q, A)

    C = dot(g, A)

    beta = B / C
    
    return beta 
end 

# Define the Liu-Storey coefficient - normal version 
function liu_storey(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    X::ManifoldPoint,
    Y::ManifoldPoint,
    divisor::Float64,
    parameter_manifold
    )

    transp!(p, p, X, Y, parameter_manifold)

    A = q - p
    B = dot(q, A)

    beta = (-1.0) * B / divisor
    
    return beta 
end 

# Define the Hager-Zhang coefficient - normal version 
function hager_zhang(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    parameter_manifold
    )

    transp!(p, p, X, Y, parameter_manifold)

    A = q - p
    B = dot(g, A)
    C = dot(A, A)
    D = dot(A - 2.0*g*C/B, q)

    beta = D / B
    
    return beta 
end 

# Define the Liu-Storey coefficient - DMRG version 
function liu_storey(
    q::Matrix{Float64}, 
    p::Matrix{Float64}, 
    X::ManifoldPoint,
    Y::ManifoldPoint,
    divisor::Float64,
    direction::String, 
    parameter_manifold
    )

    transp!(p, p, X, Y, direction, parameter_manifold)

    A = q - p
    B = dot(q, A)

    beta = (-1.0) * B / divisor
    
    return beta 
end 

# Define the Dai-Yuan coefficient - DMRG version 
function dai_yuan(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    direction::String, 
    parameter_manifold
    )

    A = dot(q, q)

    transp!(p, p, X, Y, direction, parameter_manifold)

    B = q - p
    C = dot(g, B)

    beta = A / C
    
    return beta 
end 

# Define the Polak-Ribiere coefficient - DMRG version 
function polak_ribiere(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    X::ManifoldPoint,
    Y::ManifoldPoint,
    direction::String, 
    parameter_manifold
    )

    A = dot(p, p)
    transp!(p, p, X, Y, direction, parameter_manifold)

    B = q - p
    C = dot(q, B)
    

    beta = C / A
    
    return beta 
end 

# Define the Hestenes-Stiefel coefficient - DMRG version 
function hestenes_stiefel(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    direction::String, 
    parameter_manifold
    )

    transp!(p, p, X, Y, direction, parameter_manifold)

    A = q - p
    B = dot(q, A)

    C = dot(g, A)

    beta = B / C
    
    return beta 
end 

# Define the Liu-Storey coefficient - DMRG version 
function liu_storey(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    X::ManifoldPoint,
    Y::ManifoldPoint,
    divisor::Float64,
    direction::String, 
    parameter_manifold
    )

    transp!(p, p, X, Y, direction, parameter_manifold)

    A = q - p
    B = dot(q, A)

    beta = (-1.0) * B / divisor
    
    return beta 
end 

# Define the Hager-Zhang coefficient - DMRG version 
function hager_zhang(
    q::Matrix{Float64}, # q = g 
    p::Matrix{Float64}, # p = gold
    g::Matrix{Float64}, # g = tr_zeta
    X::ManifoldPoint,
    Y::ManifoldPoint,
    direction::String, 
    parameter_manifold
    )

    transp!(p, p, X, Y, direction, parameter_manifold)

    A = q - p
    B = dot(g, A)
    C = dot(A, A)
    D = dot(A - 2.0*g*C/B, q)

    beta = D / B
    
    return beta 
end 