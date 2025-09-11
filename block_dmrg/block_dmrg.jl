### Code to calculate the L lowest eigenpairs for a given lattice model using the optimized (Riemannian) block DMRG algorithm

## Pakets 

using ITensors
using ITensorMPS 
using LinearAlgebra
using KrylovKit
using Arpack 

ITensors.disable_warn_order()

include("../models/lattice_models.jl")
include("../models/site_models.jl")
include("../riemann_cg/riemann_eigs.jl")
include("../riemann_cg/helper_functions/optimized_eigen.jl")
include("helper_functions/block_dmrg_sweep.jl")
include("helper_functions/collect_eigenstates.jl")
include("helper_functions/combiner_state_1.jl")
include("helper_functions/combiners_states.jl")
include("helper_functions/cost.jl")
include("helper_functions/create_manifold_point.jl")
include("helper_functions/dimension_of_local_hamiltonian.jl")
include("helper_functions/eigenspace_solution.jl")
include("helper_functions/eigenvalues.jl")
include("helper_functions/eigenvectors.jl")
include("helper_functions/former_eigenvectors.jl")
include("helper_functions/initial_guess_former.jl")
include("helper_functions/initial_guess_partially_random.jl")
include("helper_functions/initial_guess_random.jl")
include("helper_functions/initial_guess_with_truncated_eigenspace_solution.jl")
include("helper_functions/local_array_form.jl")
include("helper_functions/local_tensor_form.jl")
include("helper_functions/low_rank_decomposition.jl")
include("helper_functions/optimized_eigen.jl")
include("helper_functions/optimized_svd.jl")
include("helper_functions/position_update.jl")
include("helper_functions/rank_adaptation.jl")
include("helper_functions/reorthogonalize.jl")
include("helper_functions/riemann_solution.jl")

## Different functions used to calculate the L lowest eigenpairs 

# To compute the lowest lying eigenstates and eigenvalues - with Riemann
function block_dmrg(
    N::Int64, 
    L::Int64, 
    Spin::String,
    H::MPO, 
    psies_initial::Vector{MPS}, 
    parameter_dmrg,
    parameter_extra,
    parameter_backtracking,
    parameter_optimization;
    parameter_eigsolve=(1E-16, 4*L, 100, 0)
    )

    # Initialize necessary parameter
    nsweeps, _, mindim, _, _ = parameter_dmrg

    # Need to check if the chosen parameter are allowed
    if Spin=="S=1/2" 
        true_L = 2.0^N
    elseif Spin=="New_S=1/2" || Spin=="S=1" 
        true_L = 3.0^N
    elseif Spin=="New_S=1"
        true_L = 4.0^N
    else
        error("Wrong Spin!")
    end

    if L>true_L 
        error("Number of eigenvalues is too large!")
    end

    if 2*mindim*mindim<L
        error("Dimensions are too small in the beginning for the truncated eigenspace method!")
    end

    # Creation of the initial states 
    psies = deepcopy(psies_initial)
    for l=1:1:L
        psi_l = deepcopy(psies[l])
        
        #orthogonalize!(psi_l, 1) # Shift orthogonality center to positions 1 
        set_ortho_lims!(psi_l, 1:2) # Shift orthogonality center to positions 1 and 2 
        
        psies[l] = psi_l
    end

    # Creation of the output energies 
    energies = 0

    # Creation of the Hamiltonian
    ProjMPO_H = ProjMPO(H)
    psi_1 = deepcopy(psies[1])
    position!(ProjMPO_H, psi_1, 1)

    # Initialize the starting ranks and an update list for the Riemann cg method
    update = Bool[false for j=1:1:N-1]
    ranks = Int64[mindim for j=1:1:N-1] 

    # Initialize comparison matrices 
    compare_riemann = zeros(nsweeps, 2*(N-1))
    compare_eigen = zeros(nsweeps, 2*(N-1))

    # Sweeps of the Block DMRG algorithm 
    for sw in 1:nsweeps
        println("Starting sweep number $sw")

        # Run through the left-to-right or the right-to-left half-sweep
        for (j, ha) in sweepnext(N)
            direction = ha == 1 ? "left" : "right"       
            println("Direction $direction and position $j")

            # Do the half-sweep 
            energies, psies, ProjMPO_H, compare_riemann_sw_j, compare_eigen_sw_j = block_dmrg_sweep(N, L, j, Spin, ProjMPO_H, psies, direction, update, ranks, sw, parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization, parameter_eigsolve)
            
            # Update comparison matrices
            if direction=="left"
                compare_riemann[sw, j] = compare_riemann_sw_j
                compare_eigen[sw, j] = compare_eigen_sw_j
            else 
                compare_riemann[sw, 2*N-j-1] = compare_riemann_sw_j
                compare_eigen[sw, 2*N-j-1] = compare_eigen_sw_j
            end
        end
    end

    # Check normalization of the eigenstates in the end
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        psies[l] = psi_l
    end

    # Old way to compute the eigenenergies connected with the eigenstates
    #=
    energies = Float64[]
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        # Check if normalization is necessary 
        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        energy_l = inner(psi_l', H, psi_l)/inner(psi_l, psi_l)
        push!(energies, real(energy_l))

        psies[l] = psi_l
    end
    =#
    
    return energies, psies, compare_riemann, compare_eigen
end

# To compute the lowest lying eigenstates and eigenvalues - without Riemann
function block_dmrg(
    N::Int64, 
    L::Int64, 
    Spin::String,
    H::MPO, 
    psies_initial::Vector{MPS}, 
    parameter_dmrg;
    parameter_eigsolve=(1E-16, 4*L, 100, 0)
    )

    # Initialize necessary parameter
    nsweeps, _, mindim, _, _ = parameter_dmrg

    # Need to check if the chosen parameter are allowed
    if Spin=="S=1/2" 
        true_L = 2.0^N
    elseif Spin=="New_S=1/2" || Spin=="S=1" 
        true_L = 3.0^N
    elseif Spin=="New_S=1"
        true_L = 4.0^N
    else
        error("Wrong Spin!")
    end

    if L>true_L 
        error("Number of eigenvalues is too large!")
    end

    if 2*mindim*mindim<L
        error("Dimensions are too small in the beginning for the truncated eigenspace method!")
    end

    # Creation of the initial states 
    psies = deepcopy(psies_initial)
    for l=1:1:L
        psi_l = deepcopy(psies[l])
        
        #orthogonalize!(psi_l, 1) # Shift orthogonality center to positions 1 
        set_ortho_lims!(psi_l, 1:2) # Shift orthogonality center to positions 1 and 2 
        
        psies[l] = psi_l
    end

    # Creation of the output energies 
    energies = 0

    # Creation of the Hamiltonian
    ProjMPO_H = ProjMPO(H)
    psi_1 = deepcopy(psies[1])
    position!(ProjMPO_H, psi_1, 1)

    # Sweeps of the Block DMRG algorithm 
    for sw in 1:nsweeps
        println("Starting sweep number $sw")

        # Run through the left-to-right or the right-to-left half-sweep
        for (j, ha) in sweepnext(N)
            direction = ha == 1 ? "left" : "right"       
            println("Direction $direction and position $j")

            # Do the half-sweep 
            energies, psies, ProjMPO_H = block_dmrg_sweep(N, L, j, Spin, ProjMPO_H, psies, direction, parameter_dmrg, parameter_eigsolve)
        end
    end

    # Check normalization of the eigenstates in the end
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        psies[l] = psi_l
    end

    # Old way to compute the eigenenergies connected with the eigenstates
    #=
    energies = Float64[]
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        # Check if normalization is necessary 
        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        energy_l = inner(psi_l', H, psi_l)/inner(psi_l, psi_l)
        push!(energies, real(energy_l))

        psies[l] = psi_l
    end
    =#
    
    return energies, psies
end

# To compute the lowest lying eigenstates and eigenvalues orthogonal to some given states - with Riemann 
function block_dmrg(
    N::Int64, 
    L::Int64, 
    Spin::String, 
    H::MPO, 
    psies_given::Vector{MPS}, 
    weight::Float64, 
    psies_initial::Vector{MPS}, 
    parameter_dmrg,
    parameter_extra,
    parameter_backtracking,
    parameter_optimization;
    parameter_eigsolve=(1E-16, 4*L, 100, 0)
    )

    # Initialize necessary parameter
    nsweeps, _, mindim, _, _ = parameter_dmrg

    # Need to check if the chosen parameter are allowed
    if Spin=="S=1/2" 
        true_L = 2.0^N
    elseif Spin=="New_S=1/2" || Spin=="S=1" 
        true_L = 3.0^N
    elseif Spin=="New_S=1"
        true_L = 4.0^N
    else
        error("Wrong Spin!")
    end

    if L>true_L 
        error("Number of eigenvalues is too large!")
    end

    if 2*mindim*mindim<L
        error("Dimensions are too small in the beginning for the truncated eigenspace method!")
    end

    if weight <= 0
        error("Weight parameter should be > 0.0!")
    end

    M = length(psies_given) 
    for m=1:1:M
        if abs(norm(psies_given[m]) - 1.0) > 1E-6
            error("Not all given states are normalized!")
        end
    end

    #=
    for m=1:1:M 
        for n=1:1:M 
            if m!=n && ITensors.inner(psies_given[m], psies_given[n]) > 1E-6
                error("Not all given eigenstates are orthogonal!")
            end
        end
    end
    =# 
    
    # Creation of the initial states 
    psies = deepcopy(psies_initial)
    for l=1:1:L
        psi_l = deepcopy(psies[l])
        
        #orthogonalize!(psi_l, 1) # Shift orthogonality center to positions 1 
        set_ortho_lims!(psi_l, 1:2) # Shift orthogonality center to positions 1 and 2 
        
        psies[l] = psi_l
    end

    # Creation of the output energies 
    energies = 0

    # Creation of the Hamiltonian 
    ProjMPO_MPS_H = ProjMPO_MPS(H, psies_given; weight)
    psi_1 = deepcopy(psies[1])
    position!(ProjMPO_MPS_H, psi_1, 1)

    # Initialize the starting ranks and an update list for the Riemann cg method
    update = Bool[false for j=1:1:N-1]
    ranks = Int64[mindim for j=1:1:N-1] 

    # Initialize comparison matrices 
    compare_riemann = zeros(nsweeps, 2*(N-1))
    compare_eigen = zeros(nsweeps, 2*(N-1))

    # Sweeps of the Block DMRG algorithm 
    for sw in 1:nsweeps
        #println("Starting sweep number $sw")

        # Run through the left-to-right or the right-to-left half-sweep
        for (j, ha) in sweepnext(N)
            direction = ha == 1 ? "left" : "right"       
            #println("Direction $direction and position $j")

            # Do the half-sweep 
            energies, psies, ProjMPO_MPS_H, compare_riemann_sw_j, compare_eigen_sw_j = block_dmrg_sweep(N, L, j, Spin, ProjMPO_MPS_H, psies, direction, update, ranks, sw, parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization, parameter_eigsolve)

            # Update comparisson matrices
            if direction=="left"
                compare_riemann[sw, j] = compare_riemann_sw_j
                compare_eigen[sw, j] = compare_eigen_sw_j
            else 
                compare_riemann[sw, 2*N-j-1] = compare_riemann_sw_j
                compare_eigen[sw, 2*N-j-1] = compare_eigen_sw_j
            end
        end
    end

    # Check normalization of the eigenstates in the end
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        psies[l] = psi_l
    end

    # Old way to compute the eigenenergies connected with the eigenstates
    #=
    energies = Float64[]
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        # Check if normalization is necessary 
        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        energy_l = ITensors.inner(psi_l', H, psi_l)/ITensors.inner(psi_l, psi_l)
        push!(energies, real(energy_l))

        psies[l] = psi_l
    end
    =#
    
    return energies, psies, compare_riemann, compare_eigen
end

# To compute the lowest lying eigenstates and eigenvalues orthogonal to some given states - without Riemann 
function block_dmrg(
    N::Int64, 
    L::Int64, 
    Spin::String, 
    H::MPO, 
    psies_given::Vector{MPS}, 
    weight::Float64, 
    psies_initial::Vector{MPS}, 
    parameter_dmrg;
    parameter_eigsolve=(1E-16, 4*L, 100, 0)
    )

    # Initialize necessary parameter
    nsweeps, _, mindim, _, _ = parameter_dmrg

    # Need to check if the chosen parameter are allowed
    if Spin=="S=1/2" 
        true_L = 2.0^N
    elseif Spin=="New_S=1/2" || Spin=="S=1" 
        true_L = 3.0^N
    elseif Spin=="New_S=1"
        true_L = 4.0^N
    else
        error("Wrong Spin!")
    end

    if L>true_L 
        error("Number of eigenvalues is too large!")
    end

    if 2*mindim*mindim<L
        error("Dimensions are too small in the beginning for the truncated eigenspace method!")
    end

    if weight <= 0
        error("Weight parameter should be > 0.0!")
    end

    M = length(psies_given) 
    for m=1:1:M
        if abs(norm(psies_given[m]) - 1.0) > 1E-6
            error("Not all given states are normalized!")
        end
    end

    #=
    for m=1:1:M 
        for n=1:1:M 
            if m!=n && ITensors.inner(psies_given[m], psies_given[n]) > 1E-6
                error("Not all given eigenstates are orthogonal!")
            end
        end
    end
    =# 
    
    # Creation of the initial states 
    psies = deepcopy(psies_initial)
    for l=1:1:L
        psi_l = deepcopy(psies[l])
        
        #orthogonalize!(psi_l, 1) # Shift orthogonality center to positions 1 
        set_ortho_lims!(psi_l, 1:2) # Shift orthogonality center to positions 1 and 2 
        
        psies[l] = psi_l
    end

    # Creation of the output energies 
    energies = 0

    # Creation of the Hamiltonian 
    ProjMPO_MPS_H = ProjMPO_MPS(H, psies_given; weight)
    psi_1 = deepcopy(psies[1])
    position!(ProjMPO_MPS_H, psi_1, 1)

    # Sweeps of the Block DMRG algorithm 
    for sw in 1:nsweeps
        #println("Starting sweep number $sw")

        # Run through the left-to-right or the right-to-left half-sweep
        for (j, ha) in sweepnext(N)
            direction = ha == 1 ? "left" : "right"       
            #println("Direction $direction and position $j")

            # Do the half-sweep 
            energies, psies, ProjMPO_MPS_H = block_dmrg_sweep(N, L, j, Spin, ProjMPO_MPS_H, psies, direction, parameter_dmrg, parameter_eigsolve)
        end
    end

    # Check normalization of the eigenstates in the end
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        psies[l] = psi_l
    end

    # Old way to compute the eigenenergies connected with the eigenstates
    #=
    energies = Float64[]
    for l=1:1:L
        psi_l = deepcopy(psies[l])

        # Check if normalization is necessary 
        if abs(norm(psi_l) - 1.0) > 1E-6
            println("ATTENTION: Eigenvector l = $(l) needs to be normalized with norm(psi_l) = $(norm(psi_l))!")
            normalize!(psi_l)
        end

        energy_l = ITensors.inner(psi_l', H, psi_l)/ITensors.inner(psi_l, psi_l)
        push!(energies, real(energy_l))

        psies[l] = psi_l
    end
    =#
    
    return energies, psies
end

