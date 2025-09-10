## Helper function used to calculate the Tensor-form of the local Hamiltonian 

function local_tensor_form(
    ProjMPO_H::ProjMPO, 
    j::Int64, 
    N::Int64
    )

    lpos = ProjMPO_H.lpos
    rpos = ProjMPO_H.rpos

    # Multiply with the left side
    if (lpos > 0) 
        ITensor_H = ProjMPO_H.LR[lpos]
        ITensor_H *= ProjMPO_H.H[j]
        ITensor_H *= ProjMPO_H.H[j+1]
    else
        ITensor_H = ProjMPO_H.H[j]
        ITensor_H *= ProjMPO_H.H[j+1]
    end

    # Multiply with the right side
    if (rpos < N+1) 
        ITensor_H *= ProjMPO_H.LR[rpos]
    end

    return ITensor_H
end
