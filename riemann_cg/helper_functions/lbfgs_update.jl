## Define the lbfgs update 

function lbfgs_update!(
    p::Matrix{Float64},
    gamma::Float64, 
    g::Matrix{Float64}, 
    list_s, 
    list_y, 
    list_r
    )

    B = gamma .* I
    q = g
    list_xi = []

    for i = length(list_s):-1:1
        if i <= 0
            break 
        end
        xi = list_r[i] .* dot(list_s[i], q)
        q -= xi .* list_y[i] 
        push!(list_xi, xi)
    end

    p .= B * q

    for i = 1:1:length(list_s)
        if i <= 0
            break 
        end
        w = list_r[i] .* dot(list_y[i], p)
        p .+= list_s[i] .* (list_xi[length(list_s)-i+1] - w) 
    end

    return p
end