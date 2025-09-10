### Code to implement different kinds of lattice models all dependent on N, J, h and pbc

## Pakets

using ITensors

include("site_models.jl")

## Functions to create the Hamiltonian for a chosen model

# Ising model 
function ising(N, J, h, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J,"Sz",j,"Sz",j+1
    end
    for j=1:N
        os -= h[1],"Sx",j;
        os -= h[2],"Sy",j;
        os -= h[3],"Sz",j;
    end
    if pbc == true
        os -= J,"Sz",N,"Sz",1 # Periodic boundary conditions?
    end

    return os
end

# Ising model 
function ising_with_sym(N, J, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J,"Sz",j,"Sz",j+1
    end
    if pbc == true
        os -= J,"Sz",N,"Sz",1 # Periodic boundary conditions?
    end

    return os
end

# ANNNI model 
function annni(N, J, h, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J[1],"Sz",j,"Sz",j+1
    end
    for j=1:N-2
        os += J[2],"Sz",j,"Sz",j+2
    end
    for j=1:N
        os -= h[1],"Sx",j;
        os -= h[2],"Sy",j;
        os -= h[3],"Sz",j;
    end
    if pbc == true
        os -= J[1],"Sz",N,"Sz",1 # Periodic boundary conditions?
        os += J[2],"Sz",N-1,"Sz",1
        os += J[2],"Sz",N,"Sz",2
    end

    return os
end

# ANNNI model 
function annni_with_sym(N, J, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J[1],"Sz",j,"Sz",j+1
    end
    for j=1:N-2
        os += J[2],"Sz",j,"Sz",j+2
    end
    if pbc == true
        os -= J[1],"Sz",N,"Sz",1 # Periodic boundary conditions?
        os += J[2],"Sz",N-1,"Sz",1
        os += J[2],"Sz",N,"Sz",2
    end

    return os
end

# XY model 
function xy(N, J, h, pbc)
    os = OpSum()

    for j=1:N-1
        os -= 0.5*J,"S+",j,"S-",j+1 
        os -= 0.5*J,"S-",j,"S+",j+1
    end

    for j=1:N
        os -= h[1],"Sx",j;
        os -= h[2],"Sy",j;
        os -= h[3],"Sz",j;
    end

    if pbc == true
        os -= 0.5*J,"S+",N,"S-",1 
        os -= 0.5*J,"S-",N,"S+",1
    end

    return os
end

function xy_with_sym(N, J, pbc)
    os = OpSum()

    for j=1:N-1
        os -= 0.5*J,"S+",j,"S-",j+1 
        os -= 0.5*J,"S-",j,"S+",j+1
    end

    if pbc == true
        os -= 0.5*J,"S+",N,"S-",1 
        os -= 0.5*J,"S-",N,"S+",1
    end

    return os
end

# Heisenberg-XXZ model 
function heisenberg_xxz(N, J, Delta, h, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J/2,"S+",j,"S-",j+1 
        os -= J/2,"S-",j,"S+",j+1
        os -= Delta,"Sz",j,"Sz",j+1
    end

    for j=1:N
        os -= h[1],"Sx",j;
        os -= h[2],"Sy",j;
        os -= h[3],"Sz",j;
    end
    
    if pbc == true
        os -= J/2,"S+",N,"S-",1 # Periodic boundary conditions?
        os -= J/2,"S-",N,"S+",1
        os -= Delta,"Sz",N,"Sz",1
    end

    return os
end

function heisenberg_xxz_with_sym(N, J, Delta, pbc)
    os = OpSum()

    for j=1:N-1
        os -= J/2,"S+",j,"S-",j+1 
        os -= J/2,"S-",j,"S+",j+1
        os -= Delta,"Sz",j,"Sz",j+1
    end
    
    if pbc == true
        os -= J/2,"S+",N,"S-",1 # Periodic boundary conditions?
        os -= J/2,"S-",N,"S+",1
        os -= Delta,"Sz",N,"Sz",1
    end

    return os
end

## Function to create the necessary sites for a chosen model

function sites(Spin, N, conserve)
    if conserve==true
        sites = siteinds(Spin, N; conserve_qns=conserve)
    else
        sites = siteinds(Spin, N)
    end

    return sites
end