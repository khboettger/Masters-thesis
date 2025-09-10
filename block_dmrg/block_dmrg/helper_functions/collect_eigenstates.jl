## Helper function used to turn the eigenvectors phies into an array Array_X 

function collect_eigenstates(
    phies::Vector{ITensor},
    combiner_1, 
    L::Int64, 
    dims_left::Int64, 
    dims_right::Int64
    )
    
    phi_1 = phies[1]
    Array_phi_1 = array(combiner_1*phi_1)
    Array_X = zeros(typeof(Array_phi_1[1]), dims_left*dims_right, L)
    Array_X[:, 1] = Array_phi_1

    for l=2:1:L
        phi_l = phies[l]
        Array_phi_l = array(combiner_1*phi_l)
        Array_X[:, l] = Array_phi_l
    end

    return Array_X
end
