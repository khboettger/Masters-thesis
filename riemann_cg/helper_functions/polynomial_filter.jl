## Define the polynomial filter

# Coefficients for a Jackson-Chebychev filter 
function jackson_coefficients(
    d::Int64
    )

    g = zeros(d+1)
    for j in 0:d
        g[j+1] = ((d - j + 1) * cos(pi * j / (d + 1)) + sin(pi * j / (d + 1)) / tan(pi / (d + 1))) / (d + 1)
    end
    
    return g
end

# Jackson-Chebychev filter 
function jackson_chebyshev_filter_matrix(
    H::Matrix{Float64}, 
    σ::Float64, 
    λmin::Float64, 
    λmax::Float64; 
    degree=100
    )

    # Rescale H
    n = size(H, 1)
    α = (λmax + λmin) / 2
    β = (λmax - λmin) / 2

    H̃ = (H - α * I) / β
    σ̃ = (σ - α) / β

    # Compute Chebyshev coefficients (a_j) for filter centered at σ̃
    a = [cos(j * acos(σ̃)) for j in 0:degree]

    # Jackson damping coefficients g_j
    g = jackson_coefficients(degree)

    # Apply damping: b_j = g_j * a_j
    b = g .* a

    # Initialize T_0(H̃) = I, T_1(H̃) = H̃
    T0 = Matrix(I, n, n)
    T1 = copy(H̃)

    # Compute P recursively
    P = b[1] * T0 + b[2] * T1
    for j in 2:degree
        T2 = 2 * H̃ * T1 - T0
        P += b[j+1] * T2
        T0, T1 = T1, T2
    end

    return P
end
