### Function using the iterative Riemannian low rank optimization algorithm to compute eigenpairs of a symmetric matrix 

## Pakets 

include("riemann_eigs.jl")

## Function

# Iterative Riemann method - normal version 
function riemann_iterative_eigs(
    H,
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold_overall 
    if m*r<M || min(m, n)<r
        error("Dimensions are not correct!")
    end

    # Initialize the overall lists 
    list_X = ManifoldPoint[]
    list_cost_X = Float64[]
    list_rel_cost = Float64[]
    list_gradnorm = Float64[]
    list_it = Int64[]
    list_rel_X = Float64[]

    # Initialize parameter_manifold
    parameter_manifold = m, n, 1, r

    # Initialize Z 
    Z = zeros(m*n, M)

    tolerance_cost, tolerance_grad, iterations, list_cost_max, output, output_it, type, L, type_beta = parameter_optimization

    for l=1:1:M
        println("Eigenvalue l = ", l)

        if length(list_cost_max) == M
            parameter_optimization = tolerance_cost, tolerance_grad, iterations, list_cost_max[l], output, output_it, type, L, type_beta
        end 

        # Run the optimization algorithm 
        X, cost_X, rel_cost, gradnorm, it, rel_X = riemann_eigs(H, Z[:, 1:l-1], list_X0[l], parameter_manifold, parameter_backtracking, parameter_optimization)

        # Update the lists 
        push!(list_X, X)
        push!(list_cost_X, cost_X)
        push!(list_rel_cost, rel_cost)
        push!(list_gradnorm, gradnorm)
        push!(list_it, it)
        push!(list_rel_X, rel_X)

        # Update Z 
        Z[:, l] = list_X[l].X
    end

    return list_X, list_cost_X, list_rel_cost, list_gradnorm, list_it, list_rel_X
end

# Iterative Riemann method - normal version with shift
function riemann_iterative_eigs(
    H,
    alpha::Float64, 
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    #=
    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold_overall 
    if m*r<M || min(m, n)<r
        error("Dimensions are not correct!")
    end

    # Initialize the overall lists 
    list_X = []
    list_cost_X = []
    list_rel_cost = []
    list_gradnorm = []
    list_it = []
    list_rel_X = []

    # Initialize parameter_manifold
    parameter_manifold = m, n, 1, r

    # Initialize Z 
    Z = zeros(m*n, M)

    for l=1:1:M
        println("Eigenvalue l = ", l)

        # Run the optimization algorithm 
        X, cost_X, rel_cost, gradnorm, it, rel_X = riemann_eigs(H, alpha, Z[:, 1:l-1], list_X0[l], parameter_manifold, parameter_backtracking, parameter_optimization)

        # Update the lists 
        push!(list_X, X)
        push!(list_cost_X, cost_X)
        push!(list_rel_cost, rel_cost)
        push!(list_gradnorm, gradnorm)
        push!(list_it, it)
        push!(list_rel_X, rel_X)

        # Update Z 
        Z[:, l] = list_X[l].X
    end

    return list_X, list_cost_X, list_rel_cost, list_gradnorm, list_it, list_rel_X
    =#

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the standard Riemann eigs method 
    return riemann_iterative_eigs(H_tilde, list_X0::Vector{ManifoldPoint}, parameter_manifold_overall, parameter_backtracking, parameter_optimization)
end

# Iterative Riemann method - normal version with higher eigenpairs 
function riemann_iterative_eigs(
    H::Matrix{Float64},
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Construct new Hamiltonian 
    if length(Z) > 0
        P = (I - Z * Z')
        H_tilde = P * H * P 
    else 
        H_tilde = H 
    end

    # Solve using the standard Riemann eigs method 
    return riemann_iterative_eigs(H_tilde, list_X0::Vector{ManifoldPoint}, parameter_manifold_overall, parameter_backtracking, parameter_optimization)
end

# Iterative Riemann method - normal version with higher eigenpairs and shift
function riemann_iterative_eigs(
    H::Matrix{Float64},
    alpha::Float64,
    Z::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Shift the eigenvalues of the Hamiltonian
    H_tilde = H - alpha * I

    # Solve using the standard Riemann eigs method 
    return riemann_iterative_eigs(H_tilde, Z, list_X0::Vector{ManifoldPoint}, parameter_manifold_overall, parameter_backtracking, parameter_optimization)
end

# Iterative Riemann method - normal version with Kronecker and higher eigenpairs 
function riemann_iterative_eigs(
    H_kronecker::Vector{AbstractMatrix{Float64}},
    A::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Need to break, if dimensions are not right
    m, n, M, r = parameter_manifold_overall 
    if m*r<M || min(m, n)<r
        error("Dimensions are not correct!")
    end

    # Initialize the overall lists 
    list_X = ManifoldPoint[]
    list_cost_X = Float64[]
    list_rel_cost = Float64[]
    list_gradnorm = Float64[]
    list_it = Int64[]
    list_rel_X = Float64[]

    # Initialize parameter_manifold
    parameter_manifold = m, n, 1, r

    # Initialize Z 
    Z = zeros(m*n, M)

    tolerance_cost, tolerance_grad, iterations, list_cost_max, output, output_it, type, L, type_beta = parameter_optimization

    for l=1:1:M
        println("Eigenvalue l = ", l)

        if length(list_cost_max) == M
            parameter_optimization = tolerance_cost, tolerance_grad, iterations, list_cost_max[l], output, output_it, type, L, type_beta
        end 

        # Run the optimization algorithm 
        X, cost_X, rel_cost, gradnorm, it, rel_X = riemann_eigs(H_kronecker, hcat(A, Z[:, 1:l-1]), list_X0[l], parameter_manifold, parameter_backtracking, parameter_optimization)

        # Update the lists 
        push!(list_X, X)
        push!(list_cost_X, cost_X)
        push!(list_rel_cost, rel_cost)
        push!(list_gradnorm, gradnorm)
        push!(list_it, it)
        push!(list_rel_X, rel_X)

        # Update Z 
        Z[:, l] = list_X[l].X
    end

    return list_X, list_cost_X, list_rel_cost, list_gradnorm, list_it, list_rel_X
end

# Iterative Riemann method - normal version with Kronecker and higher eigenpairs and shift
function riemann_iterative_eigs(
    H_kronecker::Vector{AbstractMatrix{Float64}},
    alpha::Float64,
    A::Matrix{Float64},
    list_X0::Vector{ManifoldPoint}, 
    parameter_manifold_overall, 
    parameter_backtracking, 
    parameter_optimization
    )

    # Change H_kronecker 
    if alpha != 0.0
        H_kronecker_tilde = copy(H_kronecker)

        m, n, _ , _ = parameter_manifold_overall
        Im = 1.0 * Matrix(I, m, m)
        In = 1.0 * Matrix(I, n, n)

        push!(H_kronecker_tilde, -alpha*Im)
        push!(H_kronecker_tilde, In)
    else
        H_kronecker_tilde = copy(H_kronecker)
    end

    return riemann_iterative_eigs(
    H_kronecker_tilde, A, list_X0, parameter_manifold_overall, parameter_backtracking, parameter_optimization)
end