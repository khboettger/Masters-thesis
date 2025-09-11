### Code to calculate the L lowest eigenpairs for a given lattice model using the DMRG algorithm

## Pakets 
using ITensors
using ITensorMPS 

ITensors.disable_warn_order()

include("../models/lattice_models.jl")
include("../models/site_models.jl")

## Observer function calculating the memory used in the DMRG algorithm
mutable struct SizeObserver <: AbstractObserver
end

function ITensors.measure!(o::SizeObserver; bond, sweep, half_sweep, psi, projected_operator, kwargs...)
    if bond==1 && half_sweep==2
        psi_size = Base.format_bytes(Base.summarysize(psi))
        PH_size = Base.format_bytes(Base.summarysize(projected_operator))
        println("After sweep $sweep, |psi| = $psi_size, |PH| = $PH_size")
    end
end

## Function to calculate the L lowest eigenpairs for a given Hamiltonian
function dmrg_for_comparison(
    N::Int64, 
    L::Int64,  
    Spin::String,
    H::MPO,  
    psies_initial::Vector{MPS}, 
    parameter_dmrg
    )

    # Initialize parameter  
    weight, nsweeps, maxdim, mindim, cutoff, noise = parameter_dmrg

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

    # Create the initial ground state
    psi_0_init = deepcopy(psies_initial[1])

    # Create an observer
    obs = SizeObserver() 

    # Create a list of energies and states
    energies = Float64[]
    psies = MPS[]

    # Compute the ground state psi0
    #_, psi_0 = dmrg(H, psi_0_init; nsweeps, maxdim, mindim, cutoff, noise, observer=obs, outputlevel=1)
    _, psi_0 = dmrg(H, psi_0_init; nsweeps, maxdim, mindim, cutoff, noise, outputlevel=0)

    energy_0 = ITensors.inner(psi_0', H, psi_0)/ITensors.inner(psi_0, psi_0)
    push!(energies, real(energy_0))
    push!(psies, psi_0)

    println("l = 1, energy = ", real(energy_0))

    for l=2:1:L
        # Create the initial excited state
        psi_l_init = deepcopy(psies_initial[l])

        # Compute the excited state 
        #_, psi_l = dmrg(H, psies, psi_l_init; nsweeps, maxdim, mindim, cutoff, noise, weight, observer=obs, outputlevel=1)
        _, psi_l = dmrg(H, psies, psi_l_init; nsweeps, maxdim, mindim, cutoff, noise, weight, outputlevel=0)

        energy_l = ITensors.inner(psi_l', H, psi_l)/ITensors.inner(psi_l, psi_l)
        push!(energies, real(energy_l))
        push!(psies, psi_l)

        println("l = $(l), energy = ", real(energy_l))
    end

    return energies, psies
end