### Functions used to calculate the crystal structure of pyrochlore  

## Pakets

using LinearAlgebra
using Plots
using NearestNeighbors

## Functions 

# Function used to calculate a lattice point 
function lattice_point(n1, n2, n3, alpha, a1, a2, a3, b1, b2, b3)

    # Lattice point 
    result = n1 .* a1 + n2 .* a2 + n3 .* a3

    if alpha == 1 
        result += b1
    elseif alpha == 2 
        result += b2
    elseif alpha == 3 
        result += b3
    end
    
    return result 
end

# Function used to calculate the lattice points (open bc)
function crystal_lattice_open(N1, N2, N3)
    # Lattice vectors
    a1 = 0.5 .* [1.0, 1.0, 0.0]
    a2 = 0.5 .* [1.0, 0.0, 1.0]
    a3 = 0.5 .* [0.0, 1.0, 1.0]
    
    # Basis vectors 
    b1 = 0.5 .* a1
    b2 = 0.5 .* a2
    b3 = 0.5 .* a3

    # Lattice points
    result = []
    for n3=0:1:N3-1
        for n2=0:1:N2-1
            for n1=0:1:N1-1
                for alpha=0:1:3
                    push!(result, lattice_point(n1, n2, n3, alpha, a1, a2, a3, b1, b2, b3))
                end
            end
        end
    end 

    return result
end

# Function used to calculate the lattice points (periodic bc)
function crystal_lattice_periodic(crystal_lattice_open, N1, N2, N3)
    # Lattice vectors
    a1 = 0.5 .* [1.0, 1.0, 0.0]
    a2 = 0.5 .* [1.0, 0.0, 1.0]
    a3 = 0.5 .* [0.0, 1.0, 1.0]
    
    # New lattice points
    result = []
    prefactors1 = [-N1, 0.0, N1]
    prefactors2 = [-N2, 0.0, N2]
    prefactors3 = [-N3, 0.0, N3]
    
    for i in prefactors1
        for j in prefactors2
            for k in prefactors3
                translation = i .* a1 + j .* a2 + k .* a3
                for lattice_point in crystal_lattice_open
                    push!(result, lattice_point .+ translation)
                end
            end
        end
    end 
    
    return result
end

# Plot the lattice 
function lattice_plot(crystal_lattice)
    p = plot()
    crystal_lattice = stack(crystal_lattice)

    for i=0:1:Int64(size(crystal_lattice)[2]/4-1)
        x = crystal_lattice[1, 4*i+1:4*i+4] 
        y = crystal_lattice[2, 4*i+1:4*i+4]
        z = crystal_lattice[3, 4*i+1:4*i+4]
        plot!(x, y, z, seriestype=:scatter, label="tetra $(i+1)")
        xlabel!("x")
        ylabel!("y")
        ylabel!("z")
    end

    return p 
end

# Calculate the node list (open bc)
function list_nodes_open(N1, N2, N3)
    result = [i for i=1:1:N1*N2*N3*4]

    return result
end

# Calculate the node list (periodic bc)
function list_nodes_periodic(list_nodes_open)
    result = []

    for i = 1:1:3^3
        append!(result, list_nodes_open)
    end

    return result
end

# Function used to calculate the nearest neighbor of a particular point
function nearest_neighbors(lattice_point, crystal_lattice, list_nodes)
    k = length(crystal_lattice)[1]
    kdtree = KDTree(stack(crystal_lattice))
    all_idxs, all_dists = knn(kdtree, lattice_point, k, true)
    popfirst!(all_idxs)
    popfirst!(all_dists)

    idxs = []
    dists = []
    for i=1:1:length(all_idxs)
        min, arg = findmin(all_dists)

        if i>1 && min>dists[i-1]
            break
        else
            push!(idxs, list_nodes[all_idxs[arg]])
            push!(dists, min)
            popfirst!(all_idxs)
            popfirst!(all_dists)
        end
    end

    degrees = length(idxs)

    return idxs, dists, degrees 
end

# Function used to calculate all nearest neighbors
function all_nearest_neighbors_open(crystal_lattice_open, list_nodes_open)
    list_idxs = []
    list_dists = []
    list_degrees = []
    
    for lattice_point in crystal_lattice_open
        idxs, dists, degrees = nearest_neighbors(lattice_point, crystal_lattice_open, list_nodes_open)
        push!(list_idxs, idxs)
        #push!(list_idxs, stack(idxs))
        push!(list_dists, dists)
        #push!(list_dists, stack(dists))
        push!(list_degrees, degrees)
    end

    return list_idxs, list_dists, list_degrees
end

# Function used to calculate all nearest neighbors
function all_nearest_neighbors_periodic(crystal_lattice_periodic, crystal_lattice_open, list_nodes_periodic)
    list_idxs = []
    list_dists = []
    list_degrees = []
    
    for lattice_point in crystal_lattice_open
        idxs, dists, degrees = nearest_neighbors(lattice_point, crystal_lattice_periodic, list_nodes_periodic)
        push!(list_idxs, idxs)
        #push!(list_idxs, stack(idxs))
        push!(list_dists, dists)
        #push!(list_dists, stack(dists))
        push!(list_degrees, degrees)
    end

    return list_idxs, list_dists, list_degrees
end

# Function used to calculate the connections to the nearest neighbors of one point
function connections(position, idxs, degrees) 
    connections = []

    for i=1:1:degrees 
        if position<idxs[i]
            connection = [position, idxs[i]]
        else
        connection = [idxs[i], position]
        end
        push!(connections, connection)
    end

    return connections
end

# Function used to calculate all connections in the crystal
function all_connections(list_idxs, list_degrees)
    list_connections = []

    for position=1:1:length(list_degrees) 
        idxs = list_idxs[position]
        degrees = list_degrees[position] 
        current_connections = connections(position, idxs, degrees)
        
        for connection in current_connections
            if isempty(findall(x->x==connection, list_connections)) == true
                push!(list_connections, connection)
            end
        end
    end

    return list_connections
end