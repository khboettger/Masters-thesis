## Define the optimized svd 

# Normal version 
function optimized_svd!(
    Y::ManifoldPoint, 
    parameter_manifold
    )

    m, n, M, r = parameter_manifold

    if M == 1
        if r <= 0.2 * min(m, n*M)
            U, S, Vt = optimized_svd(reshape(Y.X, m, n*M), r, "svds") 
            Y.U .= U
            Y.S .= S
            Y.Vt .= Vt
        else 
            U, S, Vt = optimized_svd(reshape(Y.X, m, n*M), r, "svd")
            Y.U .= U
            Y.S .= S
            Y.Vt .= Vt
        end
    else 
        if min(m, n*M) <= 64
            U, S, Vt = optimized_svd(reshape(Y.X, m, n*M), r, "svds")  
            Y.U .= U
            Y.S .= S
            Y.Vt .= Vt
        else 
            U, S, Vt = optimized_svd(reshape(Y.X, m, n*M), r, "svd")
            Y.U .= U
            Y.S .= S
            Y.Vt .= Vt
        end
    end

    return Y
end

# DMRG version
function optimized_svd!(
    Y::ManifoldPoint, 
    direction::String,
    parameter_manifold
    )

    m, n, M, r = parameter_manifold
    
    if direction == "left"
        if M == 1
            if r <= 0.2 * min(m, n*M)
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svds")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            else 
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svd")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            end
        else 
            if min(m, n*M) <= 64
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svds")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            else 
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svd")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            end
        end
    else 
        if M == 1
            if r <= 0.2 * min(m*M, n)
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svds")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            else 
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svd")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            end
        else 
            if min(m*M, n) <= 64
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svds")
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            else 
                U, S, Vt = optimized_svd(reshape_low_rank(Y.X, direction, parameter_manifold), r, "svd") 
                Y.U .= U
                Y.S .= S
                Y.Vt .= Vt
            end
        end
    end

    return Y
end

# Helper full version - old 
#=
function optimized_svd(
    X::Matrix{Float64}, 
    r::Int64, 
    type::String
    )
    
    if type == "svd"
        U = 0
        S = 0
        Vt = 0

        try 
            U, S, V = svd(X, full=false)

            U = U[:, 1:r]
            S = S[1:r]
            Vt = (V[:, 1:r])'
        catch
            println("svd failed with standard algorithm!")

            U, S, V = svd(X, full=false, alg=LinearAlgebra.QRIteration())

            U = U[:, 1:r]
            S = S[1:r]
            Vt = (V[:, 1:r])'
        end

        return U, S, Vt
    elseif type == "svds"
        if r < minimum(size(X))
            U = 0
            S = 0
            Vt = 0

            try
                tmp = svds(X; nsv=r)[1]
                        
                U = tmp.U 
                S = tmp.S
                Vt = tmp.Vt
            catch 
                try 
                    println("svds failed!")
                    U, S, V = svd(X, full=false)

                    U = U[:, 1:r]
                    S = S[1:r]
                    Vt = (V[:, 1:r])'
                catch 
                    println("svd failed with standard algorithm!")

                    U, S, V = svd(X, full=false, alg=LinearAlgebra.QRIteration())

                    U = U[:, 1:r]
                    S = S[1:r]
                    Vt = (V[:, 1:r])'
                end
            end 
        else
            U = 0
            S = 0
            Vt = 0

            try 
                U, S, V = svd(X, full=false)

                U = U[:, 1:r]
                S = S[1:r]
                Vt = (V[:, 1:r])'
            catch
                println("svd failed with standard algorithm!")

                U, S, V = svd(X, full=false, alg=LinearAlgebra.QRIteration())

                U = U[:, 1:r]
                S = S[1:r]
                Vt = (V[:, 1:r])'
            end
        end

        return U, S, Vt
    end 
end
=#

function optimized_svd(
    X::Matrix{Float64}, 
    r::Int64, 
    type::String
    )
    
    # Pre-check for invalid values
    if any(isnan, X) || any(isinf, X)
        error("Matrix contains NaN or Inf values, cannot compute SVD.")
    end

    # Helper to extract top r components
    function top_r(U, S, V, r)
        return U[:, 1:r], S[1:r], (V[:, 1:r])'
    end

    # Safe fallback SVD function
    function safe_svd(X::Matrix{Float64}, r::Int64)
        try
            U, S, V = svd(X, full=false)
            return top_r(U, S, V, r)
        catch
            println("Standard SVD failed. Trying QRIteration.")
            try
                U, S, V = svd(X, full=false, alg=LinearAlgebra.QRIteration())
                return top_r(U, S, V, r)
            catch e
                error("SVD failed with both algorithms. Details: ", e)
            end
        end
    end

    if type == "svd"
        return safe_svd(X, r)

    elseif type == "svds"
        if r < minimum(size(X))
            try
                res = svds(X; nsv=r)[1]
                return res.U, res.S, res.Vt
            catch
                println("`svds` failed. Falling back to full SVD.")
                return safe_svd(X, r)
            end
        else
            return safe_svd(X, r)
        end
    else
        error("Unknown type: $type. Use \"svd\" or \"svds\".")
    end
end

