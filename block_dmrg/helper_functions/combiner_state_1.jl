## Helper function used to compute the combiners of the initial guess phi_1 as well as the necessary dimensions

function combiner_state_1(
    inds_phi_1, 
    j::Int64, 
    Spin::String 
    )
    
    link_l_tag = "\"Link,l=$(j-1)\""
    site_j_tag = "\"$Spin,Site,n=$j\""  

    dims_left = 1
    dims_right = 1

    inds = []

    for ind in inds_phi_1
        tags_ind = tags(ind)
        string_tags_ind = "$tags_ind"
        if string_tags_ind == "$link_l_tag" || string_tags_ind == "$site_j_tag"
            dims_left *= dim(ind)
        else
            dims_right *= dim(ind)
        end
        push!(inds, ind)
    end

    combiners_1 = combiner(inds; tags="combiners_1")

    return combiners_1, dims_left, dims_right
end
