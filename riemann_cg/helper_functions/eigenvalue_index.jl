## Define the estimation of the index of the eigenvalue 

# Stochastic trace estimation
function estimate_eigenvalue_count(
    H::Matrix{Float64}, 
    θ::Float64, 
    λmin::Float64,
    λmax::Float64; 
    m=50, 
    s=20
    )

    n = size(H, 1)

    # Step 1: Affine transformation: map spectrum to [-1,1]
    a = (λmax - λmin) / 2
    b = (λmax + λmin) / 2

    # Rescaled step function to approximate: 1 if λ < θ, else 0
    function step_approx(x)
        return 0.5 * (1 - tanh(10 * (x - (θ - b) / a)))
    end

    # Step 2: Compute Chebyshev coefficients
    coeffs = zeros(m+1)
    for k in 0:m
        integrand(x) = step_approx(x) * cos(k * acos(x)) / sqrt(1 - x^2)
        coeffs[k+1] = (2 - (k == 0)) / π * quadgk(integrand, -1, 1, rtol=1e-6)[1]
    end

    # Step 3: Stochastic trace estimation
    trace_est = 0.0
    for j in 1:s
        z = randn(n)
        T0 = z
        T1 = ((H - b*I)/a) * z
        y = coeffs[1] * T0 + coeffs[2] * T1

        for k in 2:m
            T2 = 2 * ((H - b*I)/a) * T1 - T0
            y += coeffs[k+1] * T2
            T0, T1 = T1, T2
        end

        trace_est += dot(z, y)
    end

    return trace_est / s
end

# Estimate all eigenvalues
function estimate_eigenvalue_count_all(
    H::Matrix{Float64}, 
    list_θ::Vector{Float64},
    M::Int64, 
    λmin::Float64,
    λmax::Float64
    )

    list_indices = Float64[]
    for l=1:1:M
        θ = list_θ[l]
        index = estimate_eigenvalue_count(H, θ, λmin, λmax)
        push!(list_indices, index)
    end

    return list_indices
end