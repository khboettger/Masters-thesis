## Function which runs a single sweep of the block DMRG algorithm with or without the Riemann CG method

# For block version - with Riemann
function block_dmrg_sweep(
    N::Int64, 
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_H::ProjMPO, 
    psies::Vector{MPS}, 
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
    
    # Move the fixed sites to j and j+1 by contracting H with psi_1 and psi_1'
    psi_1 = deepcopy(psies[1])
    ProjMPO_H = position!(ProjMPO_H, psi_1, j)

    # Compute the solution either using the Riemannian cg method
    energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, compare_riemann_sw_j, compare_eigen_sw_j = riemann_solution(N, L, j, Spin, ProjMPO_H, psies, psi_1, direction, update, ranks, sw, parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization, parameter_eigsolve)

    # The states need to be updated at the positions j and j+1 and orthogonality and normalization need to be taken care of
    psies = position_update(psies, L, j, direction, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, N)

    return energies, psies, ProjMPO_H, compare_riemann_sw_j, compare_eigen_sw_j
end

# For block version - without Riemann
function block_dmrg_sweep(
    N::Int64, 
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_H::ProjMPO, 
    psies::Vector{MPS}, 
    direction::String, 
    parameter_dmrg,  
    parameter_eigsolve
    )
    
    # Move the fixed sites to j and j+1 by contracting H with psi_1 and psi_1'
    psi_1 = deepcopy(psies[1])
    ProjMPO_H = position!(ProjMPO_H, psi_1, j)

    # Compute the solution using the eigenspace method 
    energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank = eigenspace_solution(L, j, Spin, ProjMPO_H, psies, psi_1, direction, parameter_dmrg, parameter_eigsolve)

    # The states need to be updated at the positions j and j+1 and orthogonality and normalization need to be taken care of
    psies = position_update(psies, L, j, direction, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, N)

    return energies, psies, ProjMPO_H
end

# For add-ons - with Riemann
function block_dmrg_sweep(
    N::Int64, 
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    psies::Vector{MPS}, 
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

    # Move the fixed sites to j and j+1 by contracting H with psi_1 and psi_1'
    psi_1 = deepcopy(psies[1])
    ProjMPO_MPS_H = position!(ProjMPO_MPS_H, psi_1, j)

    # Compute the solution using the Riemannian cg method
    energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, compare_riemann_sw_j, compare_eigen_sw_j = riemann_solution(N, L, j, Spin, ProjMPO_MPS_H, psies, psi_1, direction, update, ranks, sw, parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization, parameter_eigsolve)

    # The states need to be updated at the positions j and j+1 and orthogonality and normalization need to be taken care of
    psies = position_update(psies, L, j, direction, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, N)

    return energies, psies, ProjMPO_MPS_H, compare_riemann_sw_j, compare_eigen_sw_j
end

# For add-ons - without Riemann
function block_dmrg_sweep(
    N::Int64, 
    L::Int64, 
    j::Int64, 
    Spin::String, 
    ProjMPO_MPS_H::ProjMPO_MPS, 
    psies::Vector{MPS}, 
    direction::String, 
    parameter_dmrg, 
    parameter_eigsolve
    )

    # Move the fixed sites to j and j+1 by contracting H with psi_1 and psi_1'
    psi_1 = deepcopy(psies[1])
    ProjMPO_MPS_H = position!(ProjMPO_MPS_H, psi_1, j)

    # Compute the solution using the eigenspace method
    energies, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank = eigenspace_solution(L, j, Spin, ProjMPO_MPS_H, psies, psi_1, direction, parameter_dmrg, parameter_eigsolve)

    # The states need to be updated at the positions j and j+1 and orthogonality and normalization need to be taken care of
    psies = position_update(psies, L, j, direction, combiners_left, combiners_right, Array_U, Array_S, Array_V, dims_left, dims_right, rank, N)

    return energies, psies, ProjMPO_MPS_H
end