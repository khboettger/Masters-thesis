## Helper function used to calculate the Array-form of the local Hamiltonian

function local_array_form(
    ITensor_H::ITensor
    )
    # Collect the primed and unprimed indices of the local Hamiltonian
    inds_prime = []
    inds_noprime = []

    for ind in inds(ITensor_H)
        if plev(ind) == 1
            push!(inds_prime, ind)
        elseif plev(ind) == 0
            push!(inds_noprime, ind)
        end
    end

    # Compute the combiners
    combiner_prime = combiner(inds_prime; tags="combiner_prime")
    combiner_noprime = combiner(inds_noprime; tags="combiner_noprime")

    # Turn the local Hamiltonian into a combined Tensor 
    ITensor_H_combined = combiner_prime * (ITensor_H * combiner_noprime)

    # Turn the local Hamiltonian into a sparse or dense Array
    Array_H = matrix(ITensor_H_combined, inds(ITensor_H_combined)) # dense
    #Array_H = sparse(matrix(ITensor_H_combined, inds(ITensor_H_combined))) # sparse

    return Array_H
end
