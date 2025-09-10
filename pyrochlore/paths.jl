### Functions used to calculate the paths chosen through the pyrochlore crystal  

## Pakets

using LinearAlgebra
using Random
using SparseArrays
using BandedMatrices

include("crystal_structure.jl")

## Functions 

# Function used to compute the interaction matrix and calculating the bandwidth and enveloping size
function connection_matrix(N, connections, path)
    result = spzeros(N,N)
    
    for v in connections
        result[path[v[1]],path[v[2]]] = 1
        result[path[v[2]],path[v[1]]] = 1 
    end

    envelope_size = 0
    for i = 1:1:N 
        tmp = spzeros(N,N)
        tmp[i, :] = result[i, :]
        envelope_size += max(bandwidths(tmp)[1], bandwidths(tmp)[2])
    end

    return result, max(bandwidths(result)[1], bandwidths(result)[2]), envelope_size
end

# Function used to calculate a truly random path through the crystal
function random_path(list_nodes, N, connections)
    random_path = shuffle(list_nodes)
    random_connection_matrix, random_bandwidth, random_envelope_size = connection_matrix(N, connections, random_path)

    return random_path, random_connection_matrix, random_bandwidth, random_envelope_size
end

# Function used to calculate a fixed random path through the crystal
function random_fixed_path(list_nodes, N, connections, seed)
    random_path = shuffle(seed, list_nodes)
    random_connection_matrix, random_bandwidth, random_envelope_size = connection_matrix(N, connections, random_path)

    return random_path, random_connection_matrix, random_bandwidth, random_envelope_size
end

# Function wich gives the identity as a path map through the crystal
function identity_path(list_nodes, N, connections)
    identity_path = list_nodes
    identity_connection_matrix, identity_bandwidth, identity_envelope_size = connection_matrix(N, connections, identity_path)

    return identity_path, identity_connection_matrix, identity_bandwidth, identity_envelope_size
end

# Function used to generate the level structure rooted at a given node
function level_structure(N, start_node, list_idxs)
    # Initialize level list and check list
    list_levels = []
    already_numbered = [0 for i=1:1:N]

    # Consider the zeroth level
    level_0 = []
    append!(level_0, start_node) 
    already_numbered[start_node] = 1
    push!(list_levels, level_0)

    # Consider the first level
    level_1 = []
    for current_node in list_idxs[start_node]
        if already_numbered[current_node]==0
            append!(level_1, current_node)
            already_numbered[current_node] = 1
        end
    end
    push!(list_levels, level_1)

    # Set first leve as current level 
    level_current = level_1

    # Break if counter is too large
    counter = 1
    
    # Consider all other levels
    while prod(already_numbered)==0 && counter < N

        # Initializing the new level
        level_new = []

        # Add nodes to level
        for current_node in level_current   
            for current_current_node in list_idxs[current_node]
                if already_numbered[current_current_node]==0
                    append!(level_new, current_current_node)
                    already_numbered[current_current_node] = 1
                end
            end
        end

        # Add level to other levels
        push!(list_levels, level_new)

        # Set new level as current one 
        level_current = level_new

        # Raise counter by one
        counter += 1
    end

    return list_levels
end

# Function used to calculate the height of the level structure
function height_level_structure(list_levels)
    return length(list_levels)
end

# Function used to calculate the width of the level structure
function width_level_structure(list_levels)
    first_level = list_levels[1]
    maximal_width = length(first_level)

    for current_level in list_levels
        current_width = length(current_level)

        if current_width > maximal_width 
            maximal_width = current_width
        end
    end

    return maximal_width
end

# Function used to select pseudo-peripheral nodes for a given starting node
function pseudo_peripheral_nodes_without_start(N, list_idxs, list_degrees, start_node)
    # Generate level structure 
    list_levels = level_structure(N, start_node, list_idxs)
    start_height = height_level_structure(list_levels) 
    end_node = list_levels[end][end] 
    
    # Break criterion
    counter = 0

    while counter<1000
        # Increase counter 
        counter += 1

        # Sort the last level 
        list_nodes_last_level = list_levels[end]
        list_degrees_last_level = list_degrees[list_nodes_last_level]
        sorting_last_level = sortperm(list_degrees_last_level)
        list_nodes_last_level = list_nodes_last_level[sorting_last_level]
        list_degrees_last_level = list_degrees_last_level[sorting_last_level]

        # Shrink the last level 
        list_nodes_q = []
        list_degrees_q = []
        append!(list_nodes_q, list_nodes_last_level[1])
        append!(list_degrees_q, list_degrees_last_level[1])
        for current_node in list_nodes_last_level
            if list_degrees[current_node] ∉ list_degrees_q && current_node != list_nodes_last_level[1]
                append!(list_nodes_q, current_node)
                append!(list_degrees_q, list_degrees[current_node])
            end
        end

        # Sort the shrinked last level
        sorting_q = sortperm(list_degrees_q)
        list_nodes_q = list_nodes_q[sorting_q]
        list_degrees_q = list_degrees_q[sorting_q]

        # Initialize parameter
        end_node = list_nodes_q[end]
        end_width = N

        # Test for termination 
        for current_node in list_nodes_q 
            current_list_levels = level_structure(N, current_node, list_idxs)
            current_width = width_level_structure(current_list_levels)
            current_height = height_level_structure(current_list_levels)

            if current_height > start_height && current_width < end_width 
                start_node = current_node
                list_levels = current_list_levels
                start_height = current_height
                end_width = current_width
                break
            end
            
            if current_width < end_width 
                end_node = current_node 
                end_width = current_width
            end 
            
            if current_node == list_nodes_q[end]
                return start_node, end_node, counter
            end
        end
    end

    return start_node, end_node, counter
end

# Function wich a path map through the crystal according to the Cuthill-McKee algorithm, for a given starting node 
function cuthill_mckee_path_without_start(N, list_idxs, list_degrees, connections, start_node, reverse)
    # Initialize new ordering and check list
    new_list_nodes = []
    already_numbered = [0 for i=1:1:N]

    # Add the first node to the new list 
    append!(new_list_nodes, start_node)
    already_numbered[start_node] = true

    # Initializing the first level
    list_nodes_level_1 = []
    list_degrees_level_1 = []

    # Add nodes to level if not already numbered and mark them as numbered
    for current_node in list_idxs[start_node]
        if already_numbered[current_node]==0
            append!(list_nodes_level_1, current_node)
            append!(list_degrees_level_1, list_degrees[current_node])
            already_numbered[current_node] = 1
        end
    end
    
    # Sort the first level
    sorting_level_1 = sortperm(list_degrees_level_1)
    list_nodes_level_1 = list_nodes_level_1[sorting_level_1]
    list_degrees_level_1 = list_degrees_level_1[sorting_level_1]

    # Add nodes to list 
    for current_node in list_nodes_level_1
        append!(new_list_nodes, current_node)
    end

    # Set first leve as current level 
    list_nodes_level_current = list_nodes_level_1

    # Break if counter is too large
    counter = 1
    
    # Number all remaining nodes
    while prod(already_numbered)==0 && counter < N
        
        # Initializing the new level
        list_nodes_level_new = []
        list_degrees_level_new = []

        # Add nodes to level if not already numbered and mark them as numbered
        for current_node in list_nodes_level_current   
            for current_current_node in list_idxs[current_node]
                if already_numbered[current_current_node]==0
                    append!(list_nodes_level_new, current_current_node)
                    append!(list_degrees_level_new, list_degrees[current_current_node])
                    already_numbered[current_current_node] = 1
                end
            end
        end

        # Sort the new level
        sorting_level_new = sortperm(list_degrees_level_new)
        list_nodes_level_new = list_nodes_level_new[sorting_level_new]
        list_degrees_level_new = list_degrees_level_new[sorting_level_new]

        # Add nodes to list
        for current_node in list_nodes_level_new    
            push!(new_list_nodes, current_node)
        end

        # Set new level as current one 
        list_nodes_level_current = list_nodes_level_new

        # Raise counter by one
        counter += 1
    end

    # Reverse the order or not
    if reverse==true
        reverse!(new_list_nodes)
    end

    # Calculate the path related to the new numbering
    cuthill_mckee_path = zeros(Int64, N)
    for current_node in new_list_nodes 
        cuthill_mckee_path[current_node] = findfirst(==(current_node), new_list_nodes)
    end

    # Calculate the bandwidth of the path
    cuthill_mckee_connection_matrix, cuthill_mckee_bandwidth, cuthill_mckee_envelope_size = connection_matrix(N, connections, cuthill_mckee_path)

    return cuthill_mckee_path, cuthill_mckee_connection_matrix, cuthill_mckee_bandwidth, cuthill_mckee_envelope_size
end

# Function used to extract the nodes of minimal degree
function nodes_of_minimal_degree(N, list_nodes, list_degrees)
    sorting_degrees = sortperm(list_degrees)
    list_nodes_sorted = list_nodes[sorting_degrees]
    list_degrees_sorted = list_degrees[sorting_degrees]

    list_minimal_nodes = []
    list_minimal_degrees = []

    for i=1:1:N
        min, arg = findmin(list_degrees_sorted)

        if i>1 && min>list_minimal_degrees[i-1]
            break
        else
            push!(list_minimal_nodes, list_nodes_sorted[arg])
            push!(list_minimal_degrees, min)
            popfirst!(list_nodes_sorted)
            popfirst!(list_degrees_sorted)
        end
    end

    return list_minimal_nodes
end

# Function wich a path map through the crystal according to the Cuthill-McKee algorithm
function cuthill_mckee_path(N, list_nodes, list_idxs, list_degrees, connections, reverse, start)
    # Initialize minimal parameter
    minimal_bandwidth = N
    minimal_envelope_size = N*N 
    minimal_path = 0
    minimal_connection_matrix = zeros(N, N)

    # Check for the minimal path
    if start=="smallest_degree"
        list_nodes_of_minimal_degree = nodes_of_minimal_degree(N, list_nodes, list_degrees)
        
        for current_node in list_nodes_of_minimal_degree
            current_start_node = current_node
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = cuthill_mckee_path_without_start(N, list_idxs, list_degrees, connections, current_start_node, reverse)

            if current_bandwidth < minimal_bandwidth 
            #if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="all"
        for current_node in list_nodes
            current_start_node = current_node
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = cuthill_mckee_path_without_start(N, list_idxs, list_degrees, connections, current_start_node, reverse)

            if current_bandwidth < minimal_bandwidth 
            #if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="pseudo_peripheral_smallest_degree"
        list_nodes_of_minimal_degree = nodes_of_minimal_degree(N, list_nodes, list_degrees)
        
        for current_node in list_nodes_of_minimal_degree
            current_start_node, _, _ = pseudo_peripheral_nodes_without_start(N, list_idxs, list_degrees, current_node)
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = cuthill_mckee_path_without_start(N, list_idxs, list_degrees, connections, current_start_node, reverse)
        
            if current_bandwidth < minimal_bandwidth 
            #if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="pseudo_peripheral_all"
        for current_node in list_nodes
            current_start_node, _, _ = pseudo_peripheral_nodes_without_start(N, list_idxs, list_degrees, current_node)
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = cuthill_mckee_path_without_start(N, list_idxs, list_degrees, connections, current_start_node, reverse)
            
            if current_bandwidth < minimal_bandwidth 
            #if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    end

    return minimal_path, minimal_connection_matrix, minimal_bandwidth, minimal_envelope_size
end

# Function wich a path map through the crystal according to the Sloane algorithm with given start and endpoint
function sloane_path_without_start_and_end(N, list_nodes, list_idxs, list_degrees, connections, start_node, end_node, W1, W2)
    # Initialize new node list
    new_list_nodes = []
    
    # Generate level structure and compute distances 
    list_levels_end = level_structure(N, end_node, list_idxs)
    height_end = height_level_structure(list_levels_end)
    list_distances = [0 for i=1:1:N]
    for j=1:1:height_end
        level_j = list_levels_end[j]
        for current_node in level_j
            list_distances[current_node] = j-1
        end 
    end

    # Assign initial status and priority
    list_modes = [0 for i=1:1:N] # inactive=0, preactive=1, active=2, postactive=3
    list_priorities = [0 for i=1:1:N]
    for current_node in list_nodes 
        list_priorities[current_node] = W1*list_distances[current_node]-W2*(list_degrees[current_node]+1)
    end

    # Initialize node count and priority queue
    l = 0 
    n = 1
    list_modes[start_node] = 1 # preactive
    priority_queue = [start_node]

    # Test for termination
    while n>0
        # Select node to be labelled
        _, current_index = findmax(list_priorities[priority_queue])
        current_node = priority_queue[current_index]

        # Update queue and priorities 
        deleteat!(priority_queue, current_index)
        n -= 1 
        if list_modes[current_node] == 1 # preactive
            for current_current_node in list_idxs[current_node]
                list_priorities[current_current_node] += W2
                if list_modes[current_current_node] == 0 # inactive
                    list_modes[current_current_node] = 1 # preactive
                    n += 1 
                    push!(priority_queue, current_current_node)
                end
            end
        end

        # Label the next node
        push!(new_list_nodes, current_node)
        l += 1
        list_modes[current_node] = 3 # postactive

        # Update priorities and queue 
        for current_current_node in list_idxs[current_node]
            if list_modes[current_current_node] == 1 # inactive eigentlich aber preactive
                list_modes[current_current_node] = 2 # active
                list_priorities[current_current_node] += W2
                for current_current_current_node in list_idxs[current_current_node]
                    if list_modes[current_current_current_node] != 3 # postactive 
                        list_priorities[current_current_current_node] += W2
                        if list_modes[current_current_current_node] == 0 # inactive 
                            list_modes[current_current_current_node] = 1 # preactive
                            n += 1 
                            push!(priority_queue, current_current_current_node)
                        end
                    end
                end
            end
        end
    end

    # Calculate the path related to the new numbering
    sloane_path = zeros(Int64, N)
    for current_node in new_list_nodes 
        sloane_path[current_node] = findfirst(==(current_node), new_list_nodes)
    end

    # Calculate the bandwidth of the path
    sloane_connection_matrix, sloane_bandwidth , sloane_envelope_size = connection_matrix(N, connections, sloane_path)

    return sloane_path, sloane_connection_matrix, sloane_bandwidth, sloane_envelope_size
end

# Function wich a path map through the crystal according to the Sloane algorithm
function sloane_path(N, list_nodes, list_idxs, list_degrees, connections, W1, W2, start)
    # Initialize minimal parameter
    minimal_bandwidth = N
    minimal_envelope_size = N*N 
    minimal_path = 0
    minimal_connection_matrix = zeros(N, N)
    
    # Check for the minimal path
    if start=="smallest_degree"
        # Compte all nodes of minimal degree
        list_nodes_of_minimal_degree = nodes_of_minimal_degree(N, list_nodes, list_degrees)
        
        for current_node in list_nodes_of_minimal_degree
            current_start_node = current_node
            current_list_levels = level_structure(N, current_start_node, list_idxs)
            current_end_node = current_list_levels[end][end]
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = sloane_path_without_start_and_end(N, list_nodes, list_idxs, list_degrees, connections, current_start_node, current_end_node, W1, W2)

            #if current_bandwidth < minimal_bandwidth 
            if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="all"
        for current_node in list_nodes
            current_start_node = current_node
            current_list_levels = level_structure(N, current_start_node, list_idxs)
            current_end_node = current_list_levels[end][end]
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = sloane_path_without_start_and_end(N, list_nodes, list_idxs, list_degrees, connections, current_start_node, current_end_node, W1, W2)

            #if current_bandwidth < minimal_bandwidth 
            if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="pseudo_peripheral_smallest_degree"
        # Compte all nodes of minimal degree
        list_nodes_of_minimal_degree = nodes_of_minimal_degree(N, list_nodes, list_degrees)
        
        for current_node in list_nodes_of_minimal_degree
            current_start_node, current_end_node, _ = pseudo_peripheral_nodes_without_start(N, list_idxs, list_degrees, current_node)
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = sloane_path_without_start_and_end(N, list_nodes, list_idxs, list_degrees, connections, current_start_node, current_end_node, W1, W2)
        
            #if current_bandwidth < minimal_bandwidth 
            if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    elseif start=="pseudo_peripheral_all"
        for current_node in list_nodes
            current_start_node, current_end_node, _ = pseudo_peripheral_nodes_without_start(N, list_idxs, list_degrees, current_node)
    
            current_path, current_connection_matrix, current_bandwidth, current_envelope_size = sloane_path_without_start_and_end(N, list_nodes, list_idxs, list_degrees, connections, current_start_node, current_end_node, W1, W2)
            
            #if current_bandwidth < minimal_bandwidth 
            if current_envelope_size < minimal_envelope_size
                minimal_bandwidth = current_bandwidth
                minimal_envelope_size = current_envelope_size
                minimal_path = current_path
                minimal_connection_matrix = current_connection_matrix
            end
        end
    end

    return minimal_path, minimal_connection_matrix, minimal_bandwidth, minimal_envelope_size
end