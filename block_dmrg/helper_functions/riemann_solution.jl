## Helper function used to solve the local problem using the Riemannian cg method

# For block version 
function riemann_solution(
    N::Int64,
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_H::ProjMPO, 
    psies::Vector{MPS}, 
    psi_1::MPS,
    direction::String, 
    update::Vector{Bool}, 
    ranks::Vector{Int64}, 
    sw::Int64,
    parameter_dmrg, 
    parameter_extra, 
    parameter_backtracking, 
    parameter_optimization, 
    parameter_eigsolve
    )
    
    # Initialize all necessary parameter
    _, maxdim, mindim, adaptation, cutoff = parameter_dmrg
    initialization, increase, compare, repeat, check = parameter_extra
    tolerance_cost, tolerance_grad, _, _, _, _, _, _ = parameter_optimization

    # Initialize the initial guess 
    phi_1 = psi_1[j] * psi_1[j+1]

    # Calculate the combiners of the initial guess and dimensions
    combiner_1, dims_left, dims_right = combiner_state_1(inds(phi_1), j, Spin)

    # Calculate the necessary combiners of all states
    combiners, combiners_left, combiners_right = combiners_states(psies, L, j, Spin)

    # Increase the needed rank by one if needed
    update_j = update[j]
    if update_j==true
        ranks[j] += increase
    end

    # Decrease the rank if it surpasses maxdim 
    decreased = false
    if ranks[j]>maxdim
        ranks[j] = maxdim
        decreased = true
    end

    # Update the current rank
    rank = deepcopy(ranks[j])    

    # Calculate the maximal rank for this iteration
    if direction=="left"
        maxrank = min(dims_left, dims_right*L)
    else
        maxrank = min(dims_left*L, dims_right)
    end

    # Decrease the rank if it surpasses maxrank 
    if ranks[j]>maxrank
        rank = maxrank
        decreased = true
    end

    # Initialize parameter_manifold 
    parameter_manifold = dims_left, dims_right, L, rank

    # Calculate the initial matrix initial_Array_X
    if initialization=="random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_random(direction, dims_left, dims_right, rank, L)
    elseif initialization=="eigen" 
        Array_U_initial, Array_S_initial, Array_V_initial, Array_X_truncated, Array_X_full = initial_guess_with_truncated_eigenspace_solution(Spin, direction, ProjMPO_H, phi_1, dims_left, dims_right, L, j, rank, N, parameter_eigsolve) 
    elseif initialization=="former_eigen" || initialization=="former_random" || initialization=="former_partially_random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_former(Spin, sw, direction, initialization, psies, phi_1, j, dims_left, dims_right, L, combiners, rank, ProjMPO_H, update_j, N, parameter_eigsolve)
    end 

    if initialization=="random" || initialization=="former_random" || initialization=="former_partially_random"
        # The Riemann cg method is used to calculate L Eigenvectors
        ManifoldPoint_X_initial = create_manifold_point(Array_U_initial, Array_S_initial, Array_V_initial, direction, parameter_manifold) 
        ManifoldPoint_X, cost_X, cost_X, gradnorm, _, _, breaks = riemann_eigs(ProjMPO_H, combiner_1, ManifoldPoint_X_initial, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
        Array_X = ManifoldPoint_X.X
        Array_U = ManifoldPoint_X.U
        Array_S = Diagonal(ManifoldPoint_X.S)
        Array_V = (ManifoldPoint_X.Vt)'

        # Re-do if Riemann cg method breaks
        if breaks != 0 && repeat==true
            if (gradnorm<10*tolerance_grad || cost_X<10*tolerance_cost) && breaks==1
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = Array_U, Array_S, Array_V
            else
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = initial_guess_random(direction, dims_left, dims_right, rank, L)
            end

            ManifoldPoint_X_initial_new = create_manifold_point(Array_U_initial_new, Array_S_initial_new, Array_V_initial_new, direction, parameter_manifold) 
            ManifoldPoint_X_new, cost_X_new, _, _, _, _, _ = riemann_eigs(ProjMPO_H, combiner_1, ManifoldPoint_X_initial_new, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
            Array_X_new = ManifoldPoint_X_new.X

            if cost_X_new < cost_X
                println("New solution is better than the old one with $(cost_X_new) and $(cost_X)!")
                Array_X = Array_X_new
            end
        end

        # Calculate the eigenvectors from Array_X 
        Array_X = eigenvectors(ProjMPO_H, Array_X, L, combiner_1)

        # Compute the low rank decomposition of Array_X 
        Array_U, Array_S, Array_V = low_rank_decomposition_with_riemann(direction, Array_X, L, dims_left, dims_right, rank)
    elseif initialization=="eigen" || initialization=="former_eigen"
        # The Riemann cg method is used to calculate L Eigenvectors 
        ManifoldPoint_X_initial = create_manifold_point(Array_U_initial, Array_S_initial, Array_V_initial, direction, parameter_manifold)  
        ManifoldPoint_X, cost_X, _, gradnorm, it, _, breaks = riemann_eigs(ProjMPO_H, combiner_1, ManifoldPoint_X_initial, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
        Array_X = ManifoldPoint_X.X
        Array_U = ManifoldPoint_X.U
        Array_S = Diagonal(ManifoldPoint_X.S)
        Array_V = (ManifoldPoint_X.Vt)'

        # Re-do if Riemann cg method breaks
        if breaks != 0 && repeat==true
            if (gradnorm<10*tolerance_grad || cost_X<10*tolerance_cost) && breaks==1
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = Array_U, Array_S, Array_V
            else
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = initial_guess_random(direction, dims_left, dims_right, rank, L)
            end

            ManifoldPoint_X_initial_new = create_manifold_point(Array_U_initial_new, Array_S_initial_new, Array_V_initial_new, direction, parameter_manifold) 
            ManifoldPoint_X_new, cost_X_new, _, _, _, _, _ = riemann_eigs(ProjMPO_H, combiner_1, ManifoldPoint_X_initial_new, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
            Array_X_new = ManifoldPoint_X_new.X

            if cost_X_new < cost_X
                println("New solution is better than the old one with $(cost_X_new) and $(cost_X)!")
                Array_X = Array_X_new
                Array_U_initial = Array_U_initial_new
                Array_S_initial = Array_S_initial_new 
                Array_V_initial = Array_V_initial_new
                Array_U = ManifoldPoint_X_new.U
                Array_S = Diagonal(ManifoldPoint_X_new.S)
                Array_V = (ManifoldPoint_X_new.Vt)'
            end
        end
        
        # Check, whether Riemann solution is actually better
        if check==true 
            # Re-Orthogonalize before computing the eigenvalues
            Array_X_initial_low_rank = reorthogonalization(direction, Array_U_initial, Array_S_initial, Array_V_initial, dims_left, dims_right, L)
    
            if cost(Array_X_initial_low_rank, ProjMPO_H, L, combiner_1)<cost_X && it>1
                println("Riemann solution is worse with $cost(Array_X_initial_low_rank, ProjMPO_H, L, combiner_1) and $(cost_X)!")
                Array_X = Array_X_initial_low_rank
                Array_U = Array_U_initial
                Array_S = Array_S_initial
                Array_V = Array_V_initial
            end
        end
    end

    # Do a rank update or not
    if decreased==true || adaptation==false 
        # Do not update, if the rank surpassed maxrank or maxdim
        update[j] = false
    else
        # Check whether the rank needs to be raised in the next iteration
        update[j] = rank_adaptation_with_riemann(ProjMPO_H, combiner_1, Array_X, cutoff)  
    end

    # Compare Riemann CG method and eigenspace method 
    if compare==true && initialization=="eigen"
        cost_X_full = cost(Array_X_full, ProjMPO_H, L, combiner_1)
        compare_riemann_sw_j = norm(cost_X_full - cost_X)/norm(cost_X_full)
        compare_eigen_sw_j = norm(cost_X_full - cost(Array_X_truncated, ProjMPO_H, L, combiner_1))/norm(cost_X_full)
    else
        compare_riemann_sw_j = 0
        compare_eigen_sw_j = 0
    end

    # Compute the eigenvalues 
    energies = eigenvalues(ProjMPO_H, Array_X, L, combiner_1)

    return energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, compare_riemann_sw_j, compare_eigen_sw_j
end

# For add-ons
function riemann_solution(
    N::Int64,
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    psies::Vector{MPS}, 
    psi_1::MPS,
    direction::String, 
    update::Vector{Bool}, 
    ranks::Vector{Int64}, 
    sw::Int64,
    parameter_dmrg, 
    parameter_extra, 
    parameter_backtracking, 
    parameter_optimization, 
    parameter_eigsolve
    )
    
    # Initialize all necessary parameter
    _, maxdim, mindim, adaptation, cutoff = parameter_dmrg
    initialization, increase, compare, repeat, check = parameter_extra
    tolerance_cost, tolerance_grad, _, _, _, _, _, _ = parameter_optimization

    # Initialize the initial guess 
    phi_1 = psi_1[j] * psi_1[j+1]

    # Calculate the combiners of the initial guess and dimensions
    combiner_1, dims_left, dims_right = combiner_state_1(inds(phi_1), j, Spin)

    # Calculate the necessary combiners of all states
    combiners, combiners_left, combiners_right = combiners_states(psies, L, j, Spin)

    # Increase the needed rank by one if needed
    update_j = update[j]
    if update_j==true
        ranks[j] += increase
    end

    # Decrease the rank if it surpasses maxdim 
    decreased = false
    if ranks[j]>maxdim
        ranks[j] = maxdim
        decreased = true
    end

    # Update the current rank
    rank = deepcopy(ranks[j])    

    # Calculate the maximal rank for this iteration
    if direction=="left"
        maxrank = min(dims_left, dims_right*L)
    else
        maxrank = min(dims_left*L, dims_right)
    end

    # Decrease the rank if it surpasses maxrank 
    if ranks[j]>maxrank
        rank = maxrank
        decreased = true
    end

    # Initialize parameter_manifold 
    parameter_manifold = dims_left, dims_right, L, rank
    
    # Calculate the initial matrix initial_Array_X
    if initialization=="random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_random(direction, dims_left, dims_right, rank, L)
    elseif initialization=="eigen" 
        Array_U_initial, Array_S_initial, Array_V_initial, Array_X_truncated, Array_X_full = initial_guess_with_truncated_eigenspace_solution(Spin, direction, ProjMPO_MPS_H, phi_1, dims_left, dims_right, L, j, rank, N, parameter_eigsolve)  
    elseif initialization=="former_eigen" || initialization=="former_random" || initialization=="former_partially_random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_former(Spin, sw, direction, initialization, psies, phi_1, j, dims_left, dims_right, L, combiners, rank, ProjMPO_MPS_H, update_j, N, parameter_eigsolve) 
    end 

    if initialization=="random" || initialization=="former_random" || initialization=="former_partially_random"
        # The Riemann cg method is used to calculate L Eigenvectors
        ManifoldPoint_X_initial = create_manifold_point(Array_U_initial, Array_S_initial, Array_V_initial, direction, parameter_manifold) 
        ManifoldPoint_X, cost_X, cost_X, gradnorm, _, _, breaks = riemann_eigs(ProjMPO_MPS_H, combiner_1, ManifoldPoint_X_initial, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
        Array_X = ManifoldPoint_X.X
        Array_U = ManifoldPoint_X.U
        Array_S = Diagonal(ManifoldPoint_X.S)
        Array_V = (ManifoldPoint_X.Vt)'

        # Re-do if Riemann cg method breaks
        if breaks != 0 && repeat==true
            if (gradnorm<10*tolerance_grad || cost_X<10*tolerance_cost) && breaks==1
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = Array_U, Array_S, Array_V
            else
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = initial_guess_random(direction, dims_left, dims_right, rank, L)
            end

            ManifoldPoint_X_initial_new = create_manifold_point(Array_U_initial_new, Array_S_initial_new, Array_V_initial_new, direction, parameter_manifold) 
            ManifoldPoint_X_new, cost_X_new, _, _, _, _, _ = riemann_eigs(ProjMPO_MPS_H, combiner_1, ManifoldPoint_X_initial_new, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
            Array_X_new = ManifoldPoint_X_new.X

            if cost_X_new < cost_X
                println("New solution is better than the old one with $(cost_X_new) and $(cost_X)!")
                Array_X = Array_X_new
            end
        end

        # Calculate the eigenvectors from Array_X 
        Array_X = eigenvectors(ProjMPO_MPS_H, Array_X, L, combiner_1)

        # Compute the low rank decomposition of Array_X 
        Array_U, Array_S, Array_V = low_rank_decomposition_with_riemann(direction, Array_X, L, dims_left, dims_right, rank)
    elseif initialization=="eigen" || initialization=="former_eigen"
        # The Riemann cg method is used to calculate L Eigenvectors 
        ManifoldPoint_X_initial = create_manifold_point(Array_U_initial, Array_S_initial, Array_V_initial, direction, parameter_manifold)  
        ManifoldPoint_X, cost_X, _, gradnorm, it, _, breaks = riemann_eigs(ProjMPO_MPS_H, combiner_1, ManifoldPoint_X_initial, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
        Array_X = ManifoldPoint_X.X
        Array_U = ManifoldPoint_X.U
        Array_S = Diagonal(ManifoldPoint_X.S)
        Array_V = (ManifoldPoint_X.Vt)'

        # Re-do if Riemann cg method breaks
        if breaks != 0 && repeat==true
            if (gradnorm<10*tolerance_grad || cost_X<10*tolerance_cost) && breaks==1
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = Array_U, Array_S, Array_V
            else
                Array_U_initial_new, Array_S_initial_new, Array_V_initial_new = initial_guess_random(direction, dims_left, dims_right, rank, L)
            end

            ManifoldPoint_X_initial_new = create_manifold_point(Array_U_initial_new, Array_S_initial_new, Array_V_initial_new, direction, parameter_manifold) 
            ManifoldPoint_X_new, cost_X_new, _, _, _, _, _ = riemann_eigs(ProjMPO_MPS_H, combiner_1, ManifoldPoint_X_initial_new, direction, parameter_manifold, parameter_backtracking, parameter_optimization)
            Array_X_new = ManifoldPoint_X_new.X

            if cost_X_new < cost_X
                println("New solution is better than the old one with $(cost_X_new) and $(cost_X)!")
                Array_X = Array_X_new
                Array_U_initial = Array_U_initial_new
                Array_S_initial = Array_S_initial_new 
                Array_V_initial = Array_V_initial_new
                Array_U = ManifoldPoint_X_new.U
                Array_S = Diagonal(ManifoldPoint_X_new.S)
                Array_V = (ManifoldPoint_X_new.Vt)'
            end
        end
        
        # Check, whether Riemann solution is actually better
        if check==true 
            # Re-Orthogonalize before computing the eigenvalues
            Array_X_initial_low_rank = reorthogonalization(direction, Array_U_initial, Array_S_initial, Array_V_initial, dims_left, dims_right, L)
    
            if cost(Array_X_initial_low_rank, ProjMPO_H, L, combiner_1)<cost_X && it>1
                println("Riemann solution is worse with $cost(Array_X_initial_low_rank, ProjMPO_H, L, combiner_1) and $(cost_X)!")
                Array_X = Array_X_initial_low_rank
                Array_U = Array_U_initial
                Array_S = Array_S_initial
                Array_V = Array_V_initial
            end
        end
    end

    # Do a rank update or not
    if decreased==true || adaptation==false 
        # Do not update, if the rank surpassed maxrank or maxdim
        update[j] = false
    else
        # Check whether the rank needs to be raised in the next iteration
        update[j] = rank_adaptation_with_riemann(ProjMPO_MPS_H, combiner_1, Array_X, cutoff, L)  
    end

    # Compare Riemann CG method and eigenspace method 
    if compare==true && initialization=="eigen"
        cost_X_full = cost(Array_X_full, ProjMPO_MPS_H, L, combiner_1)
        compare_riemann_sw_j = norm(cost_X_full - cost_X)/norm(cost_X_full)
        compare_eigen_sw_j = norm(cost_X_full - cost(Array_X_truncated, ProjMPO_MPS_H, L, combiner_1))/norm(cost_X_full)
    else
        compare_riemann_sw_j = 0
        compare_eigen_sw_j = 0
    end

    # Compute the eigenvalues 
    energies = eigenvalues(ProjMPO_MPS_H, Array_X, L, combiner_1)

    return energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, compare_riemann_sw_j, compare_eigen_sw_j
end

