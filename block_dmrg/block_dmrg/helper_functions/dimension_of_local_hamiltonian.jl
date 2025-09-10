## Helper function used to compute the correct dimensions of the Tensor-form and Array-form of the local Hamiltonian

function dimension_of_local_hamiltonian(
    ITensor_H::ITensor, 
    j::Int64, 
    Spin::String
    )

    r_l, n_j, n_j_1, r_r = 1, 1, 1, 1

    link_l_tag = "\"Link,l=$(j-1)\""
    site_j_tag = "\"$Spin,Site,n=$j\"" 
    site_j_1_tag = "\"$Spin,Site,n=$(j+1)\"" 

    for ind in inds(ITensor_H)
        if plev(ind) == 0    
            tags_ind = tags(ind)
            string_tags_ind = "$tags_ind"

            if string_tags_ind == "$link_l_tag" 
                r_l *= dim(ind)
            elseif string_tags_ind == "$site_j_tag"
                n_j *= dim(ind)
            elseif string_tags_ind == "$site_j_1_tag"
                n_j_1 *= dim(ind) 
            else
                r_r *= dim(ind)
            end
        end
    end
    dims_left = r_l*n_j
    dims_right = n_j_1*r_r
    
    return dims_left, dims_right
end



