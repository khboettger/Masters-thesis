## Define the low rank decomposition 

function low_rank_decomp!(
    q::ManifoldTangentVector, 
    X::ManifoldPoint,
    p::Matrix{Float64}
    )

    A = p * (X.Vt') 
    B = X.U' * A
    C = p' * X.U

    q.Up .= A - X.U * B
    q.Sdot .= B 
    q.Vpt .= (C - X.Vt' * B')'

    return q
end