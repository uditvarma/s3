using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV


function von_neumann_entropy(ψ::Vector{<:Complex}, cut::Int, L::Int)
    dimA = 3^cut
    dimB = 3^(L - cut)
    ψ_matrix = reshape(ψ, (dimA, dimB))
    svals = svdvals(ψ_matrix)
    S = 0.0
    for s in svals
        if s > 1e-15
            p = abs2(s)
            S -= p * log(p)
        end
    end
    return S
end

function random_product_state(L::Int)
    product_state = nothing

    for i in 1:L
        θ1 = rand() * π
        θ2 = rand() * π
        ϕ1 = rand() * 2π
        ϕ2 = rand() * 2π

        c1 = cos(θ1 / 2)
        c2 = exp(im * ϕ1) * sin(θ1 / 2) * sin(θ2 / 2)
        c3 = exp(im * ϕ2) * sin(θ1 / 2) * cos(θ2 / 2)

        temp_state = [c1, c2, c3]

        if i == 1
            product_state = temp_state
        else
            product_state = kron(product_state, temp_state)
        end
    end

    # Normalize the state
    product_state /= norm(product_state)

    return product_state
end

"""
function time_evolution(ψ::Vector{ComplexF64}, dt::Float64, L)
    ψ /= norm(ψ)
    H, _, _ = Hamiltonian(L)
    a = 2π * rand()
    # Apply exp(-im * H * dt) directly to ψ
    ψ_new = expmv(-im * a * dt, H, ψ)

    # Normalize the state
    ψ_new /= norm(ψ_new)
    
    return ψ_new
end
"""

function I3(psi::Vector{ComplexF64})
    log3(x) = log(x) / log(3)

    L = Int(round(log3(length(psi))))
    qL = L ÷ 4

    SA   = von_neumann_entropy(psi, qL, L)
    SB   = von_neumann_entropy(psi, 2qL, L)
    SC   = von_neumann_entropy(psi, 3qL, L)
    SABC = von_neumann_entropy(psi, L, L) ### seems quite sus
    SAB  = von_neumann_entropy(psi, 2qL, L)
    SAC  = von_neumann_entropy(psi, 3qL, L)
    SBC  = von_neumann_entropy(psi, 3qL, L)
    
    return SA + SB + SC + SABC - SAB - SAC - SBC
end

### This function is for Z measurements only

function Entropy_t_z(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    Random.seed!(shot)  # Set random seed
    s_t = random_product_state(L)
    S_list = Float64[]

    # Define Hamiltonian and local observables
    H, sz2l, szl = Hamiltonian(L)
    sm_list = szl
    steps = Int(floor(T / dt))

    for _ in 1:steps
        #push!(S_list, von_neumann_entropy(s_t, L÷2, L))
        push!(S_list, I3(s_t))

        # Time evolution
        s_t = time_evolution(s_t, dt, L)
        s_t /= norm(s_t)

        # Measurements
        if p != 0
            for l in 1:L
                x = rand()
                if x < p
                    p_m_mone = 0.5 * real(s_t' * (sz2l[l] - sm_list[l]) * s_t)
                    p_m_one = 0.5 * real(s_t' * (sz2l[l] + sm_list[l]) * s_t)
                    x1 = rand()
                    if x1 < p_m_mone
                        s_t = 0.5 * ((sz2l[l]-sm_list[l]) * s_t) / sqrt(p_m_mone)
                    elseif p_m_mone < x1 < p_m_one + p_m_mone
                        s_t = 0.5 * ((sz2l[l]+sm_list[l]) * s_t) / sqrt(p_m_one)
                    else
                        s_t = (s_t - sz2l[l] * s_t) / sqrt(1 - p_m_mone - p_m_one)
                    end
                end
            end
        end
    end

    # Save result to disk
    filename = "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot).npy"
    npzwrite(filename, S_list)

    return S_list
end

"""
function Hamiltonian(L)
    # Define Pauli matrices as complex sparse matrices
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    sx = 1/sqrt(2) * sparse(ComplexF64[0 1 0; 1 0 1; 0 1 0])
    sy = 1/sqrt(2) * sparse(ComplexF64[0 -im 0; im 0 -im; 0 im 0])
    sz = sparse(ComplexF64[1 0 0; 0 0 0; 0 0 -1])

    sp = 1/sqrt(2) * (sx + im * sy)
    sm = 1/sqrt(2) * (sx - im * sy)

    # Preallocate vectors of operators with correct type
    #sx_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    #sy_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    szl = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sz operators
    sz2l = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sz² operators
    spl = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sp operators
    sml = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sm operators

    for i_site in 1:L
        p_ops = fill(id, L)
        m_ops = fill(id, L)
        z_ops = fill(id, L)
        z2_ops = fill(id, L)
        p_ops[i_site] = sp
        m_ops[i_site] = sm
        z_ops[i_site] = sz
        z2_ops[i_site] = sz^2

        # Build the full operator by tensoring
        P = p_ops[1]
        M = m_ops[1]
        Z = z_ops[1]
        Z2 = z2_ops[1]
        for j in 2:L
            P = kron(P, p_ops[j])
            M = kron(M, m_ops[j])
            Z = kron(Z, z_ops[j])
            Z2 = kron(Z2, z2_ops[j])
        end

        spl[i_site] = P
        sml[i_site] = M
        szl[i_site] = Z
        sz2l[i_site] = Z2
    end

    dim = 3^L
    H = spzeros(ComplexF64, dim, dim)

    for i in 1:L
        ip = (i % L) + 1  # Periodic boundary
        H += ((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2) + adjoint((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)))
    end

    return H, sz2l, szl 
end
"""

function tmi(ψ::Vector{ComplexF64})
    log3(x) = log(x) / log(3)

    L = Int(round(log3(length(ψ))))
    qL = L ÷ 4

    SA   = entropy_vn(ψ, L, 1:qL)
    SB   = entropy_vn(ψ, L, (qL+1):(2qL))
    SC   = entropy_vn(ψ, L, (2qL+1):(3qL))
    SABC = entropy_vn(ψ, L, 1:3qL)
    SAB  = entropy_vn(ψ, L, 1:(2qL))
    SAC  = entropy_vn(ψ, L, (1:(qL)) ∪ ((2qL+1):3qL))
    SBC  = entropy_vn(ψ, L, (2qL+1):L)

    return SA + SB + SC + SABC - SAB - SAC - SBC

end

function entropy_vn(ψ::Vector{<:Complex}, L::Int, subsystem::AbstractArray{Int})
    cut = length(subsystem)
    dimA = 3^cut
    dimB = 3^(L - cut)
    ψ_matrix = reshape(ψ, (dimA, dimB))
    svals = svdvals(ψ_matrix)
    S = 0.0
    for s in svals
        if s > 1e-15
            p = abs2(s)
            S -= p * log(p)
        end
    end
    return S
end

function Hamiltonian(L)
    # Define Pauli matrices as complex sparse matrices
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    sx = 1/sqrt(2) * sparse(ComplexF64[0 1 0; 1 0 1; 0 1 0])
    sy = 1/sqrt(2) * sparse(ComplexF64[0 -im 0; im 0 -im; 0 im 0])
    sz = sparse(ComplexF64[1 0 0; 0 0 0; 0 0 -1])

    sp = 1/sqrt(2) * (sx + im * sy)
    sm = 1/sqrt(2) * (sx - im * sy)

    # Preallocate vectors of operators with correct type
    #sx_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    #sy_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    szl = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sz operators
    sz2l = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sz² operators
    spl = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sp operators
    sml = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L) # List of sm operators

    for i_site in 1:L
        p_ops = fill(id, L)
        m_ops = fill(id, L)
        z_ops = fill(id, L)
        z2_ops = fill(id, L)
        p_ops[i_site] = sp
        m_ops[i_site] = sm
        z_ops[i_site] = sz
        z2_ops[i_site] = sz^2

        # Build the full operator by tensoring
        P = p_ops[1]
        M = m_ops[1]
        Z = z_ops[1]
        Z2 = z2_ops[1]
        for j in 2:L
            P = kron(P, p_ops[j])
            M = kron(M, m_ops[j])
            Z = kron(Z, z_ops[j])
            Z2 = kron(Z2, z2_ops[j])
        end

        spl[i_site] = P
        sml[i_site] = M
        szl[i_site] = Z
        sz2l[i_site] = Z2
    end

    dim = 3^L
    Ha = spzeros(ComplexF64, dim, dim)

    for i in 1:L
        ip = (i % L) + 1  # Periodic boundary
        Ha += ((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2) + adjoint((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)))
    end

    Hb = spzeros(ComplexF64, dim, dim)

    for i in 1:L
        ip = (i % L) + 1  # Periodic boundary
        Hb += (spl[i] * szl[i] * sml[ip] * szl[ip] + sml[i] * szl[i] * spl[ip] * szl[ip] + adjoint(spl[i] * szl[i] * sml[ip] * szl[ip]) + adjoint(sml[i] * szl[i] * spl[ip] * szl[ip]))
    end

    Hc = spzeros(ComplexF64, dim, dim)

    for i in 1:L
        ip = (i % L) + 1  # Periodic boundary
        Hc += (spl[i]^2 * spl[ip] * szl[ip] + sml[i]^2 * sml[ip] * szl[ip] + adjoint(spl[i]^2 * spl[ip] * szl[ip]) + adjoint(sml[i]^2 * sml[ip] * szl[ip]))
    end

    return Ha, Hb, Hc, sz2l, szl 
end

function time_evolution(ψ::Vector{ComplexF64}, dt::Float64, L)
    ψ /= norm(ψ)
    Ha, Hb, Hc, _, _ = Hamiltonian(L)
    a = 2π * rand()
    b = 2π * rand()
    c = 2π * rand()
    # Apply exp(-im * H * dt) directly to ψ
    H = a * Ha + b * Hb + c * Hc
    ψ_new = expmv(-im * dt, H, ψ)

    # Normalize the state
    ψ_new /= norm(ψ_new)
    
    return ψ_new
end