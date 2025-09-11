### Full code used to calculate the energies and states for the pyrochlore example 

## Pakets

using LinearAlgebra
using ITensorMPS
using ITensors

include("crystal_structure.jl")
include("paths.jl")
include("heisenberg.jl")
include("../dmrg_for_comparison/dmrg_for_comparison.jl")
include("../block_dmrg/block_dmrg.jl")

## Function used to calculate the energies and states with periodic boundary conditions

# DMRG for comparison
function full_code_pyrochlore_for_comparison(
    N1::Int64, 
    N2::Int64, 
    N3::Int64,
    L::Int64, 
    J::Float64, 
    Delta::Float64, 
    h::Vector{Float64}, 
    Spin::String, 
    site, 
    psies_initial::Vector{MPS}, 
    which_path::String, 
    parameter_dmrg
    )

    # Calculate the 4xN1xN2xN3-cluster (also with periodic boundary)
    cluster = crystal_lattice_open(N1, N2, N3)
    cluster_with_boundary = crystal_lattice_periodic(cluster, N1, N2, N3)

    # Calculate the node list (also with periodic boundary)
    nodes = list_nodes_open(N1, N2, N3)
    nodes_with_boundary = list_nodes_periodic(nodes)

    # Calculate the nearest neighbors with periodic boundary 
    idxs, _, degrees = all_nearest_neighbors_periodic(cluster_with_boundary, cluster, nodes_with_boundary)
    
    # Calculate the connections between neaest neighbors with periodic boundary
    connections = all_connections(idxs, degrees)

    # Calculate a mapping of the cluster onto a 1D chain
    N = N1*N2*N3*4
    if which_path=="identity"
        path, connectivity_matrix, _, _ = identity_path(nodes, N, connections)    
    elseif which_path=="random"
        path, connectivity_matrix, _, _ = random_path(nodes, N, connections) 
    elseif which_path=="random_fixed"
        path, connectivity_matrix, _, _ = random_fixed_path(nodes, N, connections, Random.seed!(1))
    elseif which_path=="cuthill_mckee"
        path, connectivity_matrix, _, _ = cuthill_mckee_path(N, nodes, idxs, degrees, connections, true, "all")
    elseif which_path=="sloane"
        path, connectivity_matrix, _, _ = sloane_path(N, nodes, idxs, degrees, connections, 1, 2, "pseudo_peripheral_all")
    end 

    # Calculate the Hamiltonian 
    os = heisenberg_xxz_pyrochlore(J, Delta, h, nodes, path, connections)
    H = MPO(os, site)

    # Calculate the energies and states using a chosen dmrg method 
    energies, psies = dmrg_for_comparison(N, L, Spin, H, psies_initial, parameter_dmrg)

    # Returns energies and states (the order of the states is the one chosen by the path)
    return energies, psies, connectivity_matrix
end

# Block DMRG - with Riemann
function full_code_pyrochlore(
    N1::Int64, 
    N2::Int64, 
    N3::Int64,
    L::Int64, 
    J::Float64, 
    Delta::Float64, 
    h::Vector{Float64}, 
    Spin::String, 
    site, 
    psies_initial::Vector{MPS}, 
    which_path::String, 
    parameter_dmrg, 
    parameter_extra, 
    parameter_backtracking, 
    parameter_optimization; 
    parameter_eigsolve=(1E-16, 2*L, 100, 0)
    )

    # Calculate the 4xN1xN2xN3-cluster (also with periodic boundary)
    cluster = crystal_lattice_open(N1, N2, N3)
    cluster_with_boundary = crystal_lattice_periodic(cluster, N1, N2, N3)

    # Calculate the node list (also with periodic boundary)
    nodes = list_nodes_open(N1, N2, N3)
    nodes_with_boundary = list_nodes_periodic(nodes)

    # Calculate the nearest neighbors with periodic boundary 
    idxs, _, degrees = all_nearest_neighbors_periodic(cluster_with_boundary, cluster, nodes_with_boundary)
    
    # Calculate the connections between neaest neighbors with periodic boundary
    connections = all_connections(idxs, degrees)

    # Calculate a mapping of the cluster onto a 1D chain
    N = N1*N2*N3*4
    if which_path=="identity"
        path, connectivity_matrix, _, _ = identity_path(nodes, N, connections)    
    elseif which_path=="random"
        path, connectivity_matrix, _, _ = random_path(nodes, N, connections) 
    elseif which_path=="random_fixed"
        path, connectivity_matrix, _, _ = random_fixed_path(nodes, N, connections, Random.seed!(1))
    elseif which_path=="cuthill_mckee"
        path, connectivity_matrix, _, _ = cuthill_mckee_path(N, nodes, idxs, degrees, connections, true, "all")
    elseif which_path=="sloane"
        path, connectivity_matrix, _, _ = sloane_path(N, nodes, idxs, degrees, connections, 1, 2, "pseudo_peripheral_all")
    end 

    # Calculate the Hamiltonian 
    os = heisenberg_xxz_pyrochlore(J, Delta, h, nodes, path, connections)
    H = MPO(os, site)

    # Calculate the energies and states using a chosen dmrg method 
    energies, psies = block_dmrg(N, L, Spin, H, psies_initial, parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization; parameter_eigsolve)

    # Returns energies and states (the order of the states is the one chosen by the path)
    return energies, psies, connectivity_matrix
end

# Block DMRG - without Riemann
function full_code_pyrochlore(
    N1::Int64, 
    N2::Int64, 
    N3::Int64,
    L::Int64, 
    J::Float64, 
    Delta::Float64, 
    h::Vector{Float64}, 
    Spin::String, 
    site, 
    psies_initial::Vector{MPS}, 
    which_path::String, 
    parameter_dmrg; 
    parameter_eigsolve=(1E-16, 2*L, 100, 0)
    )

    # Calculate the 4xN1xN2xN3-cluster (also with periodic boundary)
    cluster = crystal_lattice_open(N1, N2, N3)
    cluster_with_boundary = crystal_lattice_periodic(cluster, N1, N2, N3)

    # Calculate the node list (also with periodic boundary)
    nodes = list_nodes_open(N1, N2, N3)
    nodes_with_boundary = list_nodes_periodic(nodes)

    # Calculate the nearest neighbors with periodic boundary 
    idxs, _, degrees = all_nearest_neighbors_periodic(cluster_with_boundary, cluster, nodes_with_boundary)
    
    # Calculate the connections between neaest neighbors with periodic boundary
    connections = all_connections(idxs, degrees)

    # Calculate a mapping of the cluster onto a 1D chain
    N = N1*N2*N3*4
    if which_path=="identity"
        path, connectivity_matrix, _, _ = identity_path(nodes, N, connections)    
    elseif which_path=="random"
        path, connectivity_matrix, _, _ = random_path(nodes, N, connections) 
    elseif which_path=="random_fixed"
        path, connectivity_matrix, _, _ = random_fixed_path(nodes, N, connections, Random.seed!(1))
    elseif which_path=="cuthill_mckee"
        path, connectivity_matrix, _, _ = cuthill_mckee_path(N, nodes, idxs, degrees, connections, true, "all")
    elseif which_path=="sloane"
        path, connectivity_matrix, _, _ = sloane_path(N, nodes, idxs, degrees, connections, 1, 2, "pseudo_peripheral_all")
    end 

    # Calculate the Hamiltonian 
    os = heisenberg_xxz_pyrochlore(J, Delta, h, nodes, path, connections)
    H = MPO(os, site)

    # Calculate the energies and states using a chosen dmrg method 
    energies, psies = block_dmrg(N, L, Spin, H, psies_initial, parameter_dmrg; parameter_eigsolve)

    # Returns energies and states (the order of the states is the one chosen by the path)
    return energies, psies, connectivity_matrix

end
