## Helper function, which computes the initial guess using a truncated eigenspace solution of the right rank 

# For block version 
function initial_guess_with_truncated_eigenspace_solution(
    Spin::String, 
    direction::String, 
    ProjMPO_H::ProjMPO, 
    phi_1::ITensor, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64, 
    j::Int64, 
    rank::Int64, 
    N::Int64, 
    parameter_eigsolve
    ) 

    # Check, which way to tackle the problem works
    Array_X_full = 0
    try 
        # Calculate the local solutions
        vecs = optimized_eigen(ProjMPO_H, phi_1, L, parameter_eigsolve)
        phies = vecs[1:L]
        
        # Calculate the combiners of the initial guess and dimensions
        combiner_1, _, _ = combiner_state_1(inds(phi_1), j, Spin)

        # Turn the solutions into an array
        Array_X_full = collect_eigenstates(phies, combiner_1, L, dims_left, dims_right)
    catch 
        # Warning, that the old strategy is needed 
        println("Not enough eigenvectors where calculated, need the old strategy!")
        
        # Calculate the Tensor-form of the local Hamiltonian
        ITensor_H = local_tensor_form(ProjMPO_H, j, N)

        # Calculate the Array-form of the local Hamiltonian
        Array_H = local_array_form(ITensor_H)

        # The L Eigenvectors of the matrix Array_H are calculated using an iterative solver
        Array_X_full = optimized_eigen(Array_H, L)
    end

    # Reshape Array_X 
    if direction=="left"
        Array_X_reshaped = reshape(Array_X_full, dims_left, dims_right*L) 

        # Compute the full SVD
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank)

        # To compare Riemann cg method and eigenspace method
        Array_X_truncated = Array_U_initial * Array_S_initial * Array_V_initial'
        Array_X_truncated = reshape(Array_X_truncated, dims_left*dims_right, L)
    else
        Array_X_reshaped = reshape_mM_n(Array_X_full, dims_left, dims_right, L)

        # Compute the full SVD
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank)

        # To compare Riemann cg method and eigenspace method
        Array_X_truncated = Array_U_initial * Array_S_initial * Array_V_initial'
        Array_X_truncated = reshape_mn_M(Array_X_truncated, dims_left, dims_right, L)
    end

    return Array_U_initial, Array_S_initial, Array_V_initial, Array_X_truncated, Array_X_full
end

# For add-ons
function initial_guess_with_truncated_eigenspace_solution(
    Spin::String, 
    direction::String, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    phi_1::ITensor, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64, 
    j::Int64, 
    rank::Int64, 
    N::Int64, 
    parameter_eigsolve
    )

    # Calculate the local solutions
    vecs = optimized_eigen(ProjMPO_MPS_H, phi_1, L, parameter_eigsolve)
    phies = vecs[1:L]
        
    # Calculate the combiners of the initial guess and dimensions
    combiner_1, _, _ = combiner_state_1(inds(phi_1), j, Spin)

    # Turn the solutions into an array
    Array_X_full = collect_eigenstates(phies, combiner_1, L, dims_left, dims_right)

    # Reshape Array_X 
    if direction=="left"
        Array_X_reshaped = reshape(Array_X_full, dims_left, dims_right*L) 

        # Compute the full SVD
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank)

        # To compare Riemann cg method and eigenspace method
        Array_X_truncated = Array_U_initial * Array_S_initial * Array_V_initial'
        Array_X_truncated = reshape(Array_X_truncated, dims_left*dims_right, L)
    else
        Array_X_reshaped = reshape_mM_n(Array_X_full, dims_left, dims_right, L)

        # Compute the full SVD
        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank)

        # To compare Riemann cg method and eigenspace method
        Array_X_truncated = Array_U_initial * Array_S_initial * Array_V_initial'
        Array_X_truncated = reshape_mn_M(Array_X_truncated, dims_left, dims_right, L)
    end

    return Array_U_initial, Array_S_initial, Array_V_initial, Array_X_truncated, Array_X_full
end