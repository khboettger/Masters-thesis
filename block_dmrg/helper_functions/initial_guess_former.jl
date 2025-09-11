## Helper function, which computes the initial guess using the current phies

function initial_guess_former(
    Spin::String, 
    sw::Int64, 
    direction::String, 
    initialization::String, 
    psies::Vector{MPS}, 
    phi_1::ITensor, 
    j::Int64, 
    dims_left::Int64, 
    dims_right::Int64, 
    L::Int64, 
    combiners, 
    rank::Int64, 
    ProjMPO_H, 
    update_j::Bool, 
    N::Int64, 
    parameter_eigsolve
    )

    if sw==1 && direction=="left" 
        if initialization=="former_eigen"
            Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_with_truncated_eigenspace_solution(Spin, direction, ProjMPO_H, phi_1, dims_left, dims_right, L, j, rank, N, parameter_eigsolve)  
        elseif initialization=="former_random" || initialization=="former_partially_random"
            Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_random(direction, dims_left, dims_right, rank, L)
        end
    elseif update_j==false
        Array_X = former_eigenvectors(psies, j, dims_left, dims_right, L, combiners)
        
        # Reshape Array_X
        if direction=="left"
            Array_X_reshaped = reshape(Array_X, dims_left, dims_right*L)
        else
            Array_X_reshaped = reshape_mM_n(Array_X, dims_left, dims_right, L)
        end

        Array_U_initial, Array_S_initial, Array_V_initial = optimized_svd(Array_X_reshaped, rank)
    elseif update_j==true && initialization=="former_eigen"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_with_truncated_eigenspace_solution(Spin, direction, ProjMPO_H, phi_1, dims_left, dims_right, L, j, rank, N, parameter_eigsolve)  
    elseif update_j==true && initialization=="former_random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_random(direction, dims_left, dims_right, rank, L)
    elseif update_j==true && initialization=="former_partially_random"
        Array_U_initial, Array_S_initial, Array_V_initial = initial_guess_partially_random(psies, j, dims_left, dims_right, L, combiners, direction, rank)
    end

    return Array_U_initial, Array_S_initial, Array_V_initial
end
