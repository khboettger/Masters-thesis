## Helper function used to solve the local problem using the eigenspace method 

# For block version
function eigenspace_solution(
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_H::ProjMPO, 
    psies::Vector{MPS}, 
    psi_1::MPS, 
    direction::String,
    parameter_dmrg,
    parameter_eigsolve
    )
    
    # Initialize necessary parameter
    _, maxdim, mindim, adaptation, cutoff = parameter_dmrg
    
    # All objects that are necessary
    Array_X = 0
    dims_left = 0
    dims_right = 0
    combiner_1 = 0

    # Check, which way to tackle the problem works
    try 
        # Calculate an initial guess
        phi_1 = psi_1[j]*psi_1[j+1]

        # Calculate the local solutions
        vecs = optimized_eigen(ProjMPO_H, phi_1, L, parameter_eigsolve)
        phies = vecs[1:L]

        # Calculate the combiners of the initial guess and dimensions
        combiner_1, dims_left, dims_right = combiner_state_1(inds(phi_1), j, Spin)

        # Turn the solutions into an array
        Array_X = collect_eigenstates(phies, combiner_1, L, dims_left, dims_right)
    catch 
        # Warning, that the old strategy is needed 
        println("Not enough eigenvectors where calculated, need the old strategy!")
        
        # Calculate the Tensor-form of the local Hamiltonian
        ITensor_H = local_tensor_form(ProjMPO_H, j, N)

        # Calculate the Array-form of the local Hamiltonian
        Array_H = local_array_form(ITensor_H)

        # Compute the correct dimensions of Array_H
        dims_left, dims_right = dimension_of_local_hamiltonian(ITensor_H, j, Spin) 

        # The L Eigenvectors of the matrix Array_H are calculated using an iterative solver
        Array_X = optimized_eigen(Array_H, L)
    end

    # Calculate the necessary combiners of all states
    _, combiners_left, combiners_right = combiners_states(psies, L, j, Spin)

    # Compute the low rank decomposition of Array_X 
    Array_U, Array_S, Array_V, rank = low_rank_decomposition_without_riemann(direction, Array_X, L, dims_left, dims_right, adaptation, maxdim, mindim, cutoff) 

    # Re-Orthogonalize before computing the eigenvalues
    Array_X_low_rank = reorthogonalization(direction, Array_U, Array_S, Array_V, dims_left, dims_right, L)
    
    # Compute the eigenvalues 
    energies = eigenvalues(ProjMPO_H, Array_X_low_rank, L, combiner_1)

    return energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank
end

# For add-ons
function eigenspace_solution(
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    psies::Vector{MPS}, 
    psi_1::MPS,  
    direction::String,
    parameter_dmrg, 
    parameter_eigsolve
    )
    
    # Initialize necessary parameter
    _, maxdim, mindim, adaptation, cutoff = parameter_dmrg

    # Calculate an initial guess
    phi_1 = psi_1[j]*psi_1[j+1]

    # Calculate the local solutions
    vecs = optimized_eigen(ProjMPO_MPS_H, phi_1, L, parameter_eigsolve)
    phies = vecs[1:L]

    # Calculate the combiners of the initial guess and dimensions
    combiner_1, dims_left, dims_right = combiner_state_1(inds(phi_1), j, Spin)

    # Turn the solutions into an array
    Array_X = collect_eigenstates(phies, combiner_1, L, dims_left, dims_right)

    # Calculate the necessary combiners of all states
    _, combiners_left, combiners_right = combiners_states(psies, L, j, Spin)

    # Compute the low rank decomposition of Array_X 
    Array_U, Array_S, Array_V, rank = low_rank_decomposition_without_riemann(direction, Array_X, L, dims_left, dims_right, adaptation, maxdim, mindim, cutoff) 

    # Re-Orthogonalize before computing the eigenvalues
    Array_X_low_rank = reorthogonalization(direction, Array_U, Array_S, Array_V, dims_left, dims_right, L)
    
    # Compute the eigenvalues 
    energies = eigenvalues(ProjMPO_MPS_H, Array_X_low_rank, L, combiner_1)

    return energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank
end
