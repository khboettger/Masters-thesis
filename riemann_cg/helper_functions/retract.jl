## Define the retraction onto the manifold

# Define the low rank retraction 
function retract_low_rank!(
    Z::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::ManifoldTangentVector
    )

    A = Diagonal(X.S) + p.Sdot
    B = inv(A)
    C = X.U + p.Up * B
    D = A * X.Vt + p.Vpt 
    
    mul!(Z, C, D)

    return Z
end

# Define the polar decomposition 
function polar_decomp!(
    Z::Matrix{Float64}
    )

    M = size(Z)[2]

    if M == 1
        Q, _ = qr(Z)
        Z .= Matrix(Q)
    else
        A = Z' * Z
        try
            R = cholesky(A).U
            Z .= Z * inv(R)
        catch
            Q, _ = qr(Z)
            Z .= Matrix(Q)
        end
    end

    return Z
end


# Define the full retraction - normal version
function retract_full!(
    Z::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::ManifoldTangentVector, 
    parameter_manifold
    )

    m, n, M, _ = parameter_manifold  

    # Compute the low rank retraction of p
    retract_low_rank!(reshape(Z, m, n*M), X, p)

    # Compute the polar decomposition 
    polar_decomp!(Z)  

    return Z
end

# Define the full retraction - DMRG version
function retract_full!(
    Z::Matrix{Float64}, 
    X::ManifoldPoint, 
    p::ManifoldTangentVector, 
    direction::String,
    parameter_manifold
    )

    # Compute the low rank retraction of p
    Z .= reshape_Stiefel(retract_low_rank!(reshape_low_rank(Z, direction, parameter_manifold), X, p), direction, parameter_manifold)

    # Compute the polar decomposition 
    polar_decomp!(Z)  

    return Z
end