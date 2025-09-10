### Code to implement different kinds of site models 

## Pakets

using ITensors
using ITensorMPS

## Definitions regarding the existing sitetype "Electrons"

# Definition of the Sy-operator for the sitetype "Electron"
ITensors.op(::OpName"Sy",::SiteType"Electron") = 
    [0.0 0.0 0.0 0.0
    0.0 0.0 -0.5im 0.0
    0.0 0.5im 0.0 0.0
    0.0 0.0 0.0 0.0]

## Definions regarding a new sidetype "New_Spin=1/2"

# Definition of the space and its quantum numbers
function ITensors.space(
    ::SiteType"New_S=1/2";
    conserve_qns=false,
    conserve_number=conserve_qns,
    conserve_sz=false,
    qnname_number="N",
    qnname_sz="Sz")

    if conserve_number && conserve_sz
        return [
            QN((qnname_number, 0), (qnname_sz, 0))  => 1,
            QN((qnname_number, 1), (qnname_sz, +1)) => 1,
            QN((qnname_number, 1), (qnname_sz, -1)) => 1
        ]
    elseif conserve_number
        return [
            QN(qnname_number, 0) => 1,
            QN(qnname_number, 1) => 1,
            QN(qnname_number, 1) => 1
        ]
    elseif conserve_sz
        return [
            QN(qnname_sz, 0) => 1,
            QN(qnname_sz, +1) => 1,
            QN(qnname_sz, -1) => 1
        ]
    end

    return 3
end

# Definition of the states
ITensors.val(::ValName"Emp", ::SiteType"New_S=1/2") = 1
ITensors.val(::ValName"Up", ::SiteType"New_S=1/2") = 2
ITensors.val(::ValName"Down", ::SiteType"New_S=1/2") = 3

ITensors.state(::StateName"Emp", ::SiteType"New_S=1/2") = [1, 0, 0]
ITensors.state(::StateName"Up", ::SiteType"New_S=1/2") = [0, 1, 0]
ITensors.state(::StateName"Dn", ::SiteType"New_S=1/2") = [0, 0, 1]

# Definition of the operators
function ITensors.op(::OpName"Sx", ::SiteType"New_S=1/2")
    return [
        0.0 0.0 0.0 
        0.0 0.0 0.5 
        0.0 0.5 0.0 
    ]
end

function ITensors.op(::OpName"Sy", ::SiteType"New_S=1/2")
    return [
        0.0 0.0 0.0 
        0.0 0.0 -0.5im 
        0.0 0.5im 0.0 
    ]
end

function ITensors.op(::OpName"Sz", ::SiteType"New_S=1/2")
    return [
        0.0 0.0 0.0 
        0.0 0.5 0.0 
        0.0 0.0 -0.5 
    ]
end

function ITensors.op(::OpName"S+", ::SiteType"New_S=1/2")
    return [
        0.0 0.0 0.0
        0.0 0.0 1.0 
        0.0 0.0 0.0
    ]
end

function ITensors.op(::OpName"S-", ::SiteType"New_S=1/2")
    return [
        0.0 0.0 0.0
        0.0 0.0 0.0 
        0.0 1.0 0.0
    ]
end

## Definions regarding a new sidetype "New-Spin-1"

# Definition of the space and its quantum numbers
function ITensors.space(
    ::SiteType"New_S=1";
    conserve_qns=false,
    conserve_number=conserve_qns,
    conserve_sz=false,
    qnname_number="N",
    qnname_sz="Sz")

    if conserve_number && conserve_sz
        return [
            QN((qnname_number, 0), (qnname_sz, 0))  => 1,
            QN((qnname_number, 1), (qnname_sz, +2)) => 1,
            QN((qnname_number, 1), (qnname_sz, 0)) => 1,
            QN((qnname_number, 1), (qnname_sz, -2)) => 1
        ]
    elseif conserve_number
        return [
            QN(qnname_number, 0) => 1,
            QN(qnname_number, 1) => 1,
            QN(qnname_number, 1) => 1,
            QN(qnname_number, 1) => 1
        ]
    elseif conserve_sz
        return [
            QN(qnname_sz, 0) => 1,
            QN(qnname_sz, +2) => 1,
            QN(qnname_sz, 0) => 1,
            QN(qnname_sz, -2) => 1
        ]
    end

    return 4
end

# Definition of the states
ITensors.val(::ValName"Emp", ::SiteType"New_S=1") = 1
ITensors.val(::ValName"Up", ::SiteType"New_S=1") = 2
ITensors.val(::ValName"Z0", ::SiteType"New_S=1") = 3
ITensors.val(::ValName"Down", ::SiteType"New_S=1") = 4

ITensors.state(::StateName"Emp", ::SiteType"New_S=1") = [1, 0, 0, 0]
ITensors.state(::StateName"Up", ::SiteType"New_S=1") = [0, 1, 0, 0]
ITensors.state(::StateName"Z0", ::SiteType"New_S=1") = [0, 0, 1, 0]
ITensors.state(::StateName"Dn", ::SiteType"New_S=1") = [0, 0, 0, 1]

# Definition of the operators
function ITensors.op(::OpName"Sx", ::SiteType"New_S=1")
    return [
        0.0 0.0 0.0 0.0  
        0.0 0.0 1/√2 0.0
        0.0 1/√2 0.0 1/√2
        0.0 0.0 1/√2 0.0
    ]
end

function ITensors.op(::OpName"Sy", ::SiteType"New_S=1")
    return [
        0.0 0.0 0.0 0.0  
        0.0 0.0 -im/√2 0.0
        0.0 im/√2 0.0 -im/√2
        0.0 0.0 im/√2 0.0
    ]
end

function ITensors.op(::OpName"Sz", ::SiteType"New_S=1")
    return [
        0.0 0.0 0.0 0.0  
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
    ]
end

function ITensors.op(::OpName"S+", ::SiteType"New_S=1")
    return [
        0.0 0.0 0.0 0.0  
        0.0 0.0 √2 0.0
        0.0 0.0 0.0 √2
        0.0 0.0 0.0 0.0
    ]
end

function ITensors.op(::OpName"S-", ::SiteType"New_S=1")
    return [
        0.0 0.0 0.0 0.0  
        0.0 0.0 0.0 0.0
        0.0 √2 0.0 0.0
        0.0 0.0 √2 0.0
    ]
end