## Define everything related to the manifold

# Define a point on the manifold
struct ManifoldPoint 
    X::Matrix{Float64}  
    U::Matrix{Float64}   
    S::Vector{Float64}  
    Vt::Matrix{Float64}  
end 

# Define a tangential vector on the manifold
struct ManifoldTangentVector
    p::Matrix{Float64}  
    Up::Matrix{Float64}
    Sdot::Matrix{Float64}
    Vpt::Matrix{Float64}
end 

# Define a multiplication on the manifold 
function ManifoldScalarProduct(
    alpha::Float64,
    p::ManifoldTangentVector,
    )

    return ManifoldTangentVector(alpha * p.p, alpha * p.Up, alpha * p.Sdot, alpha * p.Vpt)
end 

# Define a inplace-copy function on the manifold 
function ManifoldCopy!(
    Y::ManifoldPoint,
    X::ManifoldPoint,
    )

    copyto!(Y.X, X.X)
    copyto!(Y.U, X.U)
    copyto!(Y.S, X.S)
    copyto!(Y.Vt, X.Vt)

    return Y
end 

# Define a copy function on the manifold - normal version
function ManifoldCopy(
    X::ManifoldPoint,
    parameter_manifold
    )

    Y = zero_point(parameter_manifold)
    ManifoldCopy!(Y, X)

    return Y
end 

# Define a copy function on the manifold - DMRG version 
function ManifoldCopy(
    X::ManifoldPoint,
    direction::String, 
    parameter_manifold
    )

    Y = zero_point(direction, parameter_manifold)
    ManifoldCopy!(Y, X)

    return Y
end 

# Define the zero point - normal version 
function zero_point(
    parameter_manifold
    )
    m, n, M, r = parameter_manifold
    
    return ManifoldPoint(zeros(m*n, M), zeros(m, r), zeros(r), zeros(r, n*M))
end

# Define the zero point - DMRG version
function zero_point(
    direction::String, 
    parameter_manifold
    )
    m, n, M, r = parameter_manifold
    
    if direction=="left"
        return ManifoldPoint(zeros(m*n, M), zeros(m, r), zeros(r), zeros(r, n*M))
    else
        return ManifoldPoint(zeros(m*n, M), zeros(m*M, r), zeros(r), zeros(r, n))
    end
end

# Define a random matrix that lies on the manifold - normal version 
function random_point(
    parameter_manifold
    )
    m, n, M, r = parameter_manifold
    
    U, S, V = svd(randn(m, n*M))
    U = U[:, 1:r]
    S = S[1:r]
    Vt = (V[:, 1:r])'

    Q, _ = qr(reshape(U*Diagonal(S)*Vt, m*n, M))
    X = Matrix(Q[:, 1:M])

    U, S, V = svd(reshape(X, m, n*M))
    U = U[:, 1:r]
    S = S[1:r]
    Vt = (V[:, 1:r])'    

    return ManifoldPoint(X, U, S, Vt)
end

# Define a random matrix that lies on the manifold - DMRG version 
function random_point(
    direction::String, 
    parameter_manifold
    )
    m, n, M, r = parameter_manifold
    
    U, S, V = svd(randn(m, n*M))
    U = U[:, 1:r]
    S = S[1:r]
    Vt = (V[:, 1:r])'

    Q, _ = qr(reshape_Stiefel(U*Diagonal(S)*Vt, direction, parameter_manifold))
    X = Matrix(Q[:, 1:M])

    U, S, V = svd(reshape_low_rank(X, direction, parameter_manifold))
    U = U[:, 1:r]
    S = S[1:r]
    Vt = (V[:, 1:r])'    

    return ManifoldPoint(X, U, S, Vt)
end

# Define the zero vector - normal version 
function zero_vector( 
    parameter_manifold
    ) 

    m, n, M, r = parameter_manifold

    return ManifoldTangentVector(zeros(m*n, M), zeros(m, r), zeros(r, r), zeros(r, n*M))
end

# Define the zero vector - DMRG version 
function zero_vector( 
    direction::String,
    parameter_manifold
    ) 

    m, n, M, r = parameter_manifold

    if direction=="left"
        return ManifoldTangentVector(zeros(m*n, M), zeros(m, r), zeros(r, r), zeros(r, n*M))
    else
        return ManifoldTangentVector(zeros(m*n, M), zeros(m*M, r), zeros(r, r), zeros(r, n))
    end
end

# Define the check for points 
function check_point(
    X::ManifoldPoint,
    parameter_manifold, 
    atol::Float64 
    )

    c = X.X' * X.X
    if norm(c - I) > atol
        return error(
            norm(c - I), " The point $(p) does not lie on $(N), because X.X'*X.X is not the unit matrix.",
        )
    end
    
    _, _, _, r = parameter_manifold
    k = length(X.S)
    s = " The point $(X) does not lie on the manifold, "
    if k > r
        return error(k, string(s, "since its rank is too large ($(k))."))
    end

    return nothing
end

# Define the check for vectors - normal version 
function check_vector(
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    parameter_manifold,  
    atol::Float64
    )
    
    m, n, M, r = parameter_manifold

    if norm(X.X' * p.p + conj(p.p' * X.X)) > atol
        return error(
            norm(X.X' * p.p + conj(p.p' * X.X)),
            " The vector $(p) is does not lie in the tangent space of $(X) on the Stiefel manifold of dimension ($(m*n),$(M)), since X.X' * p.p + conj(p.p' * X.X) is not the zero matrix.",
        )
    end

    if norm(X.U' * p.Up) > atol
        return error(
            norm(X.U' * p.Up),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold since X.U' * p.Up is not zero.",
        )
    end

    if norm(X.Vt * p.Vpt') > atol
        return error(
            norm(X.Vt * p.Vpt'),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold  since X.Vt * p.Vpt' is not zero.",
        )
    end

    U, _, V = svd(reshape(X.X, m, n * M), full=true)
    Omega = U' * reshape(p.p, m, n * M) * V
    if norm(Omega[r+1:m, r+1:n*M]) > atol
        return error(
            norm(Omega[r+1:m, r+1:n*M]),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold  since Omega[r+1:m, r+1:n*M] is not zero.",
        )
    end
    
    return nothing
end

# Define the check for vectors - DMRG version 
function check_vector(
    X::ManifoldPoint,
    p::ManifoldTangentVector,
    direction::String, 
    parameter_manifold,  
    atol::Float64
    )
    
    m, n, M, r = parameter_manifold

    if norm(X.X' * p.p + conj(p.p' * X.X)) > atol
        return error(
            norm(X.X' * p.p + conj(p.p' * X.X)),
            " The vector $(p) is does not lie in the tangent space of $(X) on the Stiefel manifold of dimension ($(m*n),$(M)), since X.X' * p.p + conj(p.p' * X.X) is not the zero matrix.",
        )
    end

    if norm(X.U' * p.Up) > atol
        return error(
            norm(X.U' * p.Up),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold since X.U' * p.Up is not zero.",
        )
    end

    if norm(X.Vt * p.Vpt') > atol
        return error(
            norm(X.Vt * p.Vpt'),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold  since X.Vt * p.Vpt' is not zero.",
        )
    end

    U, _, V = svd(reshape_low_rank(X.X, direction, parameter_manifold), full=true)
    Omega = U' * reshape_low_rank(p.p, direction, parameter_manifold) * V
    if norm(Omega[r+1:m, r+1:n*M]) > atol
        return error(
            norm(Omega[r+1:m, r+1:n*M]),
            " The tangent vector $(p) is not a tangent vector to $(X) on the low rank manifold  since Omega[r+1:m, r+1:n*M] is not zero.",
        )
    end
    
    return nothing
end