## My own eigen decomposition optimized for runtime

# Pakets

using LinearAlgebra
using KrylovKit

# Function 
function optimized_eigen(
    A::Matrix{Float64}, 
    L::Int64,
    type=:SR
    )
    
    n = size(A)[1] 

    if n < 100 && (typeof(A)==Matrix{Float64} || typeof(A)==Matrix{ComplexF64})
        X = eigvecs(A)

        if type==:SR 
            X = X[:, 1:L]
        elseif type==:LR 
            X = X[:, end-L+1:end]
            X[:, end:-1:1]
        end
    else
        if 2*L>30
            _, X = eigsolve(A, L, type; ishermitian=true, krylovdim=2*L, tol=1E-16)
        else
            _, X = eigsolve(A, L, type; ishermitian=true, tol=1E-16)
        end
        X = stack(X)
        
        if size(X)[2] > L
            X = X[:, 1:L]
        end
    end 

    return X
end