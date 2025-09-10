## Helper function used to compute the combiners of each state 

function combiners_states(
    psies::Vector{MPS}, 
    L::Int64, 
    j::Int64, 
    Spin::String
    )
    
    combiners = []
    combiners_left = []
    combiners_right = []

    link_l_tag = "\"Link,l=$(j-1)\""
    site_j_tag = "\"$Spin,Site,n=$j\""  

    for l=1:1:L
        psi_l = deepcopy(psies[l]) 
        phi_l = psi_l[j]*psi_l[j+1]

        inds_l = []
        inds_l_left = []
        inds_l_right = []

        for ind in inds(phi_l)
            tags_ind = tags(ind)
            string_tags_ind = "$tags_ind"
            if string_tags_ind == "$link_l_tag" || string_tags_ind == "$site_j_tag"
                push!(inds_l_left, ind)
            else
                push!(inds_l_right, ind)
            end

            push!(inds_l, ind)
        end

        combiners_l = combiner(inds_l; tags="combiners_l")
        combiners_l_left = combiner(inds_l_left; tags="combiners_l_left")
        combiners_l_right = combiner(inds_l_right; tags="combiners_l_right")

        push!(combiners, combiners_l)
        push!(combiners_left, combiners_l_left)
        push!(combiners_right, combiners_l_right)
    end

    return combiners, combiners_left, combiners_right
end
