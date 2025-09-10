## Define the projected gradient

# Normal version 
function grad!(
    g::Matrix{Float64},
    HX::Matrix{Float64}, 
    X::ManifoldPoint, 
    parameter_manifold 
    )

    return project_full!(g, X, HX, parameter_manifold)
end

# DMRG version 
function grad!(
    g::Matrix{Float64},
    HX, 
    combiner_1,
    X::ManifoldPoint, 
    direction::String, 
    parameter_manifold 
    )

    return project_full!(g, X, array(combiner_1 * HX), direction, parameter_manifold)
end