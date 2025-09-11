## Function used to create a manifold point from given Arrays 

function create_manifold_point(
    Array_U_initial, 
    Array_S_initial, 
    Array_V_initial, 
    direction::String, 
    parameter_manifold
    ) 

    dims_left, dims_right, L, _ = parameter_manifold

    Array_Vt_initial = (Array_V_initial)'

    if direction=="left"    
        Array_X_initial = reshape(Array_U_initial*Array_S_initial*Array_Vt_initial, dims_left*dims_right, L) 
    else 
        Array_X_initial = reshape_mn_M(Array_U_initial*Array_S_initial*Array_Vt_initial, dims_left, dims_right, L)
    end

    return ManifoldPoint(Array_X_initial, Array_U_initial, Vector{Float64}(diag(Array_S_initial)), Array_Vt_initial)
end