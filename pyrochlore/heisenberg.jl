### Functions used to calculate the Heisenberg model for pyrochlore  

## Pakets

using LinearAlgebra
using ITensors
using ITensorMPS

include("crystal_structure.jl")
include("paths.jl")

## Functions

# Function used to calculate the Heisenberg-Hamiltonian
function heisenberg_pyrochlore(J, h, list_nodes, path, connections)
    os = OpSum()

    for v in connections
        os -= J[1],"Sx",path[v[1]],"Sx",path[v[2]] 
        os -= J[2],"Sy",path[v[1]],"Sy",path[v[2]]
        os -= J[3],"Sz",path[v[1]],"Sz",path[v[2]]
    end

    for j in list_nodes
        os -= h[1],"Sx",path[j];
        os -= h[2],"Sy",path[j];
        os -= h[3],"Sz",path[j];
    end

    return os
end

# Function used to calculate the Heisenberg-XXZ-Hamiltonian (only if necessary)
function heisenberg_xxz_pyrochlore(J, Delta, h, list_nodes, path, connections)
    os = OpSum()

    for v in connections
        os -= J/2,"S+",path[v[1]],"S-",path[v[2]] 
        os -= J/2,"S-",path[v[1]],"S+",path[v[2]]
        os -= Delta,"Sz",path[v[1]],"Sz",path[v[2]]
    end

    for j in list_nodes
        os -= h[1],"Sx",path[j];
        os -= h[2],"Sy",path[j];
        os -= h[3],"Sz",path[j];
    end

    return os
end