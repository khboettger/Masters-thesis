## My own svd, here optimized for runtime and not memory

function optimized_svd(
    A::Matrix{Float64}, 
    rank::Int64
    )
    m, n = size(A)

    U = 0
    S = 0
    V = 0

    try
        if rank < min(m, n)
            tmp = svds(A; nsv=rank)[1]
            
            U = tmp.U 
            S = Diagonal(tmp.S)
            V = tmp.V
        else
            U, S, V = svd(A, full=false)

            U = U[:, 1:rank]
            S = Diagonal(S[1:rank])
            V = V[:, 1:rank]
        end
    catch 
        try 
            println("svds failed!")
            U, S, V = svd(A, full=false)
            U = U[:,1:rank]
            S = Diagonal(S[1:rank])
            V = V[:, 1:rank]
        catch 
            println("svd failed with standard algorithm!")
            U, S, V = svd(A, full=false, alg=LinearAlgebra.QRIteration())
            U = U[:,1:rank]
            S = Diagonal(S[1:rank])
            V = V[:, 1:rank]
        end
    end 

    return U, S, V
end