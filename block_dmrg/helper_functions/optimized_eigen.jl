## Compute L eigenvectors 

function optimized_eigen(
    A, 
    phi::ITensor, 
    L::Int64,
    parameter_eigsolve
    )

    eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter, eigsolve_verbosity = parameter_eigsolve

    if 2*L>30
        _, X = eigsolve(A, phi, L, :SR; ishermitian=true,  tol=eigsolve_tol, krylovdim=2*L, maxiter=eigsolve_maxiter, verbosity=eigsolve_verbosity)
    else
        _, X = eigsolve(A, phi, L, :SR; ishermitian=true, tol=eigsolve_tol, krylovdim=eigsolve_krylovdim, maxiter=eigsolve_maxiter, verbosity=eigsolve_verbosity)
    end

    return X
end