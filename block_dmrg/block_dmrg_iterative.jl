### Code to calculate L eigenpairs for a given lattice model using the optimized iterative (Riemannian) block DMRG algorithm

## Pakets 

include("block_dmrg.jl")

## Different functions used to calculate the L lowest eigenpairs 

# With Riemann
function block_dmrg_iterative(
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
    parameter_eigsolve=(1E-16, 3, 100, 0)
    )

    # Create the states to which the solution has to be orthogonal to 
    if length(psies_given) == 0 
        psies_constraint = MPS[]
    else 
        psies_constraint = deepcopy(psies_given) 
    end 

    # Create the initial ground state
    psi_0_init = deepcopy(psies_initial[1])

    # Create a list of energies and states
    energies = Float64[]
    psies = MPS[]

    # Create a list of comparissons 
    compare_riemann = Matrix{Float64}[]
    compare_eigen = Matrix{Float64}[]

    # Compute the energy0, psi0, compare_riemann and compare_eigen
    energy_0, psi_0, compare_riemann_0, compare_eigen_0 = block_dmrg(N, 1, Spin, H, psies_constraint, weight, [psi_0_init], parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization; parameter_eigsolve) 

    # Update energies and psies
    push!(energies, energy_0[1])
    push!(psies, psi_0[1])

    # Update compare_riemann and compare_eigen
    push!(compare_riemann, compare_riemann_0)
    push!(compare_eigen, compare_eigen_0)

    # Update psies_constraint
    push!(psies_constraint, psi_0[1])

    println("l = 1, energy = ", energy_0[1])

    for l=2:1:L    
        # Create the initial excited state
        psi_l_init = deepcopy(psies_initial[l])

        # Compute the energy0, psi0, compare_riemann and compare_eigen
        energy_l, psi_l, compare_riemann_l, compare_eigen_l = block_dmrg(N, 1, Spin, H, psies_constraint, weight, [psi_l_init], parameter_dmrg, parameter_extra, parameter_backtracking, parameter_optimization; parameter_eigsolve) 

        # Update energies and psies 
        push!(energies, energy_l[1])
        push!(psies, psi_l[1])

        # Update compare_riemann and compare_eigen
        push!(compare_riemann, compare_riemann_l)
        push!(compare_eigen, compare_eigen_l)

        # Update psies_constraint
        push!(psies_constraint, psi_l[1])

        println("l = $(l), energy = ", energy_l[1])
    end

    return energies, psies, compare_riemann, compare_eigen
end

# Without Riemann
function block_dmrg_iterative(
    N::Int64, 
    L::Int64, 
    Spin::String,
    H::MPO, 
    psies_given::Vector{MPS}, 
    weight::Float64, 
    psies_initial::Vector{MPS}, 
    parameter_dmrg;
    parameter_eigsolve=(1E-16, 3, 100, 0)
    )

    # Create the states to which the solution has to be orthogonal to 
    if length(psies_given) == 0 
        psies_constraint = MPS[]
    else 
        psies_constraint = deepcopy(psies_given) 
    end 

    # Create the initial ground state
    psi_0_init = deepcopy(psies_initial[1])

    # Create a list of energies and states
    energies = Float64[]
    psies = MPS[]

    # Compute the energy0, psi0, compare_riemann and compare_eigen
    energy_0, psi_0 = block_dmrg(N, 1, Spin, H, psies_constraint, weight, [psi_0_init], parameter_dmrg; parameter_eigsolve) 

    # Update energies and psies
    push!(energies, energy_0[1])
    push!(psies, psi_0[1])

    # Update psies_constraint
    push!(psies_constraint, psi_0[1])

    println("l = 1, energy = ", energy_0[1])

    for l=2:1:L    
        # Create the initial excited state
        psi_l_init = deepcopy(psies_initial[l])

        # Compute the energy0, psi0, compare_riemann and compare_eigen
        energy_l, psi_l = block_dmrg(N, 1, Spin, H, psies_constraint, weight, [psi_l_init], parameter_dmrg; parameter_eigsolve) 

        # Update energies and psies 
        push!(energies, energy_l[1])
        push!(psies, psi_l[1])

        # Update psies_constraint
        push!(psies_constraint, psi_l[1])

        println("l = $(l), energy = ", energy_l[1])
    end

    return energies, psies
end
