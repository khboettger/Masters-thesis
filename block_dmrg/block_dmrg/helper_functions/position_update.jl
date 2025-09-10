## Helper function used to update the states at the positions j and j+1 and normalize the states

function position_update(
    psies::Vector{MPS}, 
    L::Int64, 
    j::Int64, 
    direction::String, 
    combiners_left, 
    combiners_right, 
    Array_U, 
    Array_S, 
    Array_V, 
    dims_left::Int64, 
    dims_right::Int64, 
    rank::Int64, 
    N::Int64
    )

    for l=1:1:L
        # Access the current state
        psi_l = deepcopy(psies[l])

        # Combiners of the current state are called
        combiners_l_left = deepcopy(combiners_left[l])
        combiners_l_right = deepcopy(combiners_right[l])

        # Create new index
        new_index = Index(rank, "Link,l=$j")

        # Update the positions j and j+1 of state psi_l dependent on the direction of the half-sweep
        if direction=="left"
            # Combine Array_V and Array_S into the new Array_C
            Array_C = Array_S * Array_V'

            # Access the l-th entry of Array_C
            Array_C_l = Array_C[:, (l-1)*dims_right+1:l*dims_right]

            # Turn Array_U and Array_C into Tensors
            indices_U = (combinedind(combiners_l_left), new_index)
            ITensor_U = ITensor(Array_U, indices_U)
            indices_C_l = (new_index, combinedind(combiners_l_right))
            ITensor_C_l = ITensor(Array_C_l, indices_C_l)
            
            # Update the position j and j+1
            psi_l[j] = dag(combiners_l_left) * ITensor_U
            psi_l[j+1] = ITensor_C_l * dag(combiners_l_right) 

            # Normalize and move the orthogonalization center
            #set_ortho_lims!(psi_l, j+1:j+1)
            if j<N-1
                set_ortho_lims!(psi_l, j+1:j+2)
            end
            psi_l[j+1] ./= norm(psi_l[j+1])
        else
            # Combine Array_U and Array_S into the new Array_C
            Array_C = Array_U * Array_S
            
            # Access the l-th entry of Array_C
            Array_C_l = Array_C[(l-1)*dims_left+1:l*dims_left, :]

            # Turn Array_U and Array_C into Tensors
            indices_C_l = (combinedind(combiners_l_left), new_index)
            ITensor_C_l = ITensor(Array_C_l, indices_C_l)
            indices_V = (new_index, combinedind(combiners_l_right))
            ITensor_V = ITensor(Array_V', indices_V)
            
            # Update the position j and j+1
            psi_l[j] = dag(combiners_l_left) * ITensor_C_l
            psi_l[j+1] = ITensor_V * dag(combiners_l_right) 

            # Normalize and move the orthogonalization center
            #set_ortho_lims!(psi_l, j:j)
            if j>1
                set_ortho_lims!(psi_l, j-1:j)
            end
            psi_l[j] ./= norm(psi_l[j])
        end

        psies[l] = psi_l
    end

    return psies
end
