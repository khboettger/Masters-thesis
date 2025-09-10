## Initialize necessary parameter

# Normal version 
function initialize(
    H,
    X0::ManifoldPoint,   
    parameter_manifold, 
    parameter_backtracking,
    parameter_optimization
    )

    m, n, M, _ = parameter_manifold 
    _, _, _, cost_max, _ = parameter_optimization
    alpha0, _ = parameter_backtracking

    # Initialize X
    X = ManifoldCopy(X0, parameter_manifold)

    # Initialize Xold 
    Xold = ManifoldCopy(X, parameter_manifold)

    # Initialize HX 
    HX = zeros(m*n, M)

    # Initialize cost_Xold, cost_X, rel_cost
    cost_X = cost_max
    cost_Xold = cost(H, X.X, HX, parameter_manifold)
    rel_cost = 1.0

    # Initialize g 
    g = zeros(m*n, M)
    grad!(g, HX, X, parameter_manifold)

    # Initialize zeta 
    zeta = zero_vector(parameter_manifold)
    zeta.p .= -g
    low_rank_decomp!(zeta, X, reshape(zeta.p, m, n*M))

    # Initialize tr_gold 
    tr_gold = copy(g) 

    # Initialize tr_gold 
    tr_zeta = copy(zeta.p) 

    # Initialize gradnorm 
    gradnorm = norm(g, Inf)

    # Initialize alpha 
    alpha = alpha0

    # Initialize rel_X 
    rel_X = 1.0

    # Initialize divisor 
    divisor = dot(g, zeta.p)

    # Initialize Hzeta
    Hzeta = zeros(m*n, M)

    # Initialize list_s, list_y, list_r 
    list_s = []
    list_y = [] 
    list_r = []

    #  Initialize p 
    p = zeros(m*n, M)

    return X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX
end

# Normal version - Kronecker structure and higher eigenpairs
function initialize(
    H::Vector{AbstractMatrix{Float64}},
    P,
    X0::ManifoldPoint,   
    parameter_manifold,  
    parameter_backtracking,
    parameter_optimization
    )

    m, n, M, _ = parameter_manifold 
    _, _, _, cost_max, _ = parameter_optimization
    alpha0, _ = parameter_backtracking

    # Initialize X
    X = ManifoldCopy(X0, parameter_manifold)

    # Initialize Xold 
    Xold = ManifoldCopy(X, parameter_manifold)

    # Initialize HX 
    HX = zeros(m*n, M)

    # Initialize cost_Xold, cost_X, rel_cost
    cost_X = cost_max
    cost_Xold = cost(H, X.X, HX, P, parameter_manifold)
    rel_cost = 1.0

    # Initialize g 
    g = zeros(m*n, M)
    grad!(g, HX, X, parameter_manifold)

    # Initialize zeta 
    zeta = zero_vector(parameter_manifold)
    zeta.p .= -g
    low_rank_decomp!(zeta, X, reshape(zeta.p, m, n*M))

    # Initialize tr_gold 
    tr_gold = copy(g) 

    # Initialize tr_gold 
    tr_zeta = copy(zeta.p) 

    # Initialize gradnorm 
    gradnorm = norm(g)

    # Initialize alpha 
    alpha = alpha0

    # Initialize rel_X 
    rel_X = 1.0

    # Initialize divisor 
    divisor = dot(g, zeta.p)

    # Initialize Hzeta
    Hzeta = zeros(m*n, M)

    # Initialize list_s, list_y, list_r 
    list_s = []
    list_y = [] 
    list_r = []

    #  Initialize p 
    p = zeros(m*n, M)

    return X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX
end

# DMRG version - ProjMPO
function initialize(
    H::ProjMPO,
    combiner_1, 
    X0::ManifoldPoint,   
    direction::String,
    parameter_manifold,  
    parameter_backtracking,
    parameter_optimization
    )

    m, n, M, _ = parameter_manifold 
    _, _, _, cost_max, _ = parameter_optimization
    alpha0, _ = parameter_backtracking

    # Initialize X
    X = ManifoldCopy(X0, direction, parameter_manifold)

    # Initialize Xold 
    Xold = ManifoldCopy(X, direction, parameter_manifold)

    # Initialize HX 
    index_L = Index(M, "index_L")
    X_tensor_combined = ITensor(X.X, inds(combiner_1)[1], index_L)
    X_tensor = combiner_1 * X_tensor_combined
    HX = product(H, X_tensor) 

    # Initialize cost_Xold, cost_X, rel_cost
    cost_X = cost_max
    cost_Xold = cost(H, combiner_1, X.X, HX, index_L, parameter_manifold)
    rel_cost = 1.0

    # Initialize g 
    g = zeros(m*n, M)
    grad!(g, HX, combiner_1, X, direction, parameter_manifold)

    # Initialize zeta 
    zeta = zero_vector(direction, parameter_manifold)
    zeta.p .= -g
    low_rank_decomp!(zeta, X, reshape_low_rank(zeta.p, direction, parameter_manifold))

    # Initialize tr_gold 
    tr_gold = copy(g) 

    # Initialize tr_gold 
    tr_zeta = copy(zeta.p) 

    # Initialize gradnorm 
    gradnorm = norm(g)

    # Initialize alpha 
    alpha = alpha0

    # Initialize rel_X 
    rel_X = 1.0

    # Initialize divisor 
    divisor = dot(g, zeta.p)

    # Initialize Hzeta
    Hzeta = zeros(m*n, M)

    # Initialize list_s, list_y, list_r 
    list_s = []
    list_y = [] 
    list_r = []

    #  Initialize p 
    p = zeros(m*n, M)

    return X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX, index_L
end

# DMRG version - ProjMPO_MPS
function initialize(
    H::ProjMPO_MPS,
    combiner_1, 
    X0::ManifoldPoint,   
    direction::String,
    parameter_manifold,  
    parameter_backtracking,
    parameter_optimization
    )

    m, n, M, _ = parameter_manifold 
    _, _, _, cost_max, _ = parameter_optimization
    alpha0, _ = parameter_backtracking

    # Initialize X
    X = ManifoldCopy(X0, direction, parameter_manifold)

    # Initialize Xold 
    Xold = ManifoldCopy(X, direction, parameter_manifold)

    # Initialize HX 
    index_L = Index(M, "index_L")
    HX_array = []
    inds_X_tensor_1 = 0
    for l=1:1:M
        X_tensor_combined_l = ITensor(X.X[:, l], inds(combiner_1)[1])
        X_tensor_l = combiner_1 * X_tensor_combined_l

        HX_l = product(H, X_tensor_l)
        HX_array_l = array(HX_l)
        push!(HX_array, HX_array_l)

        if l == 1
            inds_X_tensor_1 = inds(HX_l)
        end
    end 
    HX_array = stack(HX_array)

    # Compute HX_tensor
    HX = ITensor(HX_array, inds_X_tensor_1, index_L)

    # Initialize cost_Xold, cost_X, rel_cost
    cost_X = cost_max
    cost_Xold = cost(H, combiner_1, X.X, HX, index_L, parameter_manifold)
    rel_cost = 1.0

    # Initialize g 
    g = zeros(m*n, M)
    grad!(g, HX, combiner_1, X, direction, parameter_manifold)

    # Initialize zeta 
    zeta = zero_vector(direction, parameter_manifold)
    zeta.p .= -g
    low_rank_decomp!(zeta, X, reshape_low_rank(zeta.p, direction, parameter_manifold))

    # Initialize tr_gold 
    tr_gold = copy(g) 

    # Initialize tr_gold 
    tr_zeta = copy(zeta.p) 

    # Initialize gradnorm 
    gradnorm = norm(g)

    # Initialize alpha 
    alpha = alpha0

    # Initialize rel_X 
    rel_X = 1.0

    # Initialize divisor 
    divisor = dot(g, zeta.p)

    # Initialize Hzeta
    Hzeta = zeros(m*n, M)

    # Initialize list_s, list_y, list_r 
    list_s = []
    list_y = [] 
    list_r = []

    #  Initialize p 
    p = zeros(m*n, M)

    return X, Xold, g, zeta, tr_gold, tr_zeta, gradnorm, alpha, cost_X, cost_Xold, rel_cost, rel_X, divisor, Hzeta, list_s, list_y, list_r, p, HX, index_L
end
