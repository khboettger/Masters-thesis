## Helper function which calculates the current phies

function former_eigenvectors(
    psies::Vector{MPS}, 
    j::Int64, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64, 
    combiners
    )

    Array_X = zeros((dims_left*dims_right, L))

    for l=1:1:L
        psi_l = deepcopy(psies[l])
        phi_l = psi_l[j]*psi_l[j+1]

        combiners_l = combiners[l]

        ITensor_phi_l = combiners_l * phi_l
        Array_phi_l = array(ITensor_phi_l, inds(ITensor_phi_l))
        
        Array_X[:,l] = Array_phi_l
    end

    return Array_X
end