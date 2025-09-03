using SparseArrays
using LinearAlgebra
using Random
using NPZ
using ExpmV

# ----------------------------
# Single-site / product states
# ----------------------------
function random_product_state(L::Int)
    ψ = nothing
    for i in 1:L
        θ1, θ2 = rand()*π, rand()*π
        ϕ1, ϕ2 = rand()*2π, rand()*2π

        c1 = cos(θ1/2)
        c2 = exp(im*ϕ1) * sin(θ1/2) * sin(θ2/2)
        c3 = exp(im*ϕ2) * sin(θ1/2) * cos(θ2/2)

        temp_state = ComplexF64.([c1, c2, c3])

        ψ = i == 1 ? temp_state : kron(ψ, temp_state)
    end
    return ψ ./ norm(ψ)
end

function spin1_state(L::Int)
    site = ComplexF64.(fill(1.0, 3)) ./ sqrt(3)  # (|1> + |0> + |-1>)/√3
    ψ = site
    for _ in 2:L
        ψ = kron(ψ, site)
    end
    return ψ ./ norm(ψ)
end

# ----------------------------
# Entropy (von Neumann on pure-state bipartition)
# ----------------------------
function entropy_vn(ψ::Vector{<:Complex}, L::Int, subsystem::AbstractArray{Int})
    cut = length(subsystem)
    dimA, dimB = 3^cut, 3^(L - cut)
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

# ----------------------------
# Hamiltonian builder (returns Ha, Hd, He and local operators)
# ----------------------------
function Hamiltonian(L::Int)
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    sx = 1/sqrt(2) * sparse(ComplexF64[0 1 0; 1 0 1; 0 1 0])
    sy = 1/sqrt(2) * sparse(ComplexF64[0 -im 0; im 0 -im; 0 im 0])
    sz = sparse(ComplexF64[1 0 0; 0 0 0; 0 0 -1])

    sp = 1/sqrt(2) * (sx + im*sy)
    sm = 1/sqrt(2) * (sx - im*sy)

    # Preallocate operator lists
    szl  = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, L)
    sz2l = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, L)
    spl  = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, L)
    sml  = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, L)

    # Build local operators as full-system sparse matrices
    for i_site in 1:L
        ops = fill(id, L)
        p_ops, m_ops, z_ops, z2_ops = copy(ops), copy(ops), copy(ops), copy(ops)
        p_ops[i_site] = sp
        m_ops[i_site] = sm
        z_ops[i_site] = sz
        z2_ops[i_site] = sz^2

        P, M, Z, Z2 = p_ops[1], m_ops[1], z_ops[1], z2_ops[1]
        for j in 2:L
            P  = kron(P,  p_ops[j])
            M  = kron(M,  m_ops[j])
            Z  = kron(Z,  z_ops[j])
            Z2 = kron(Z2, z2_ops[j])
        end

        spl[i_site]  = P
        sml[i_site]  = M
        szl[i_site]  = Z
        sz2l[i_site] = Z2
    end

    dim = 3^L
    Ha = spzeros(ComplexF64, dim, dim)
    Hd = spzeros(ComplexF64, dim, dim)
    He = spzeros(ComplexF64, dim, dim)

    # Ha: nearest-neighbor pair terms (like in original)
    for i in 1:L
        ip = (i % L) + 1
        term = (spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)
        Ha += term + adjoint(term)
    end

    # Hd: next-nearest (i to i+2) terms
    if L >= 3
        for i in 1:L-2
            ip = (i % L) + 2
            term = (spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)
            Hd += term + adjoint(term)
        end
        # wrap-around pairs for periodic boundary
        Hd += (spl[L-1]^2 * sml[1]^2 + sml[L-1]^2 * spl[1]^2)
        Hd += (spl[L]^2   * sml[2]^2 + sml[L]^2   * spl[2]^2)
    end

    # He: Sz_i * Sz_{i+1}
    for i in 1:L
        ip = (i % L) + 1
        He += szl[i] * szl[ip]
    end

    return Ha, Hd, He, sz2l, szl
end

"""
# ----------------------------
# In-place Hamiltonian update (Option A)
# ----------------------------
# In-place Hamiltonian update (safe version)
function update_H!(H::SparseMatrixCSC{ComplexF64,Int},
                   a::Float64, Ha::SparseMatrixCSC{ComplexF64,Int},
                   b::Float64, Hd::SparseMatrixCSC{ComplexF64,Int},
                   c::Float64, He::SparseMatrixCSC{ComplexF64,Int})
    H .= a .* Ha .+ b .* Hd .+ c .* He
    return H
end

# ----------------------------
# Time evolution (uses preallocated H and updates in-place)
# ----------------------------
function time_evolution!(ψ::Vector{ComplexF64},
                         H::SparseMatrixCSC{ComplexF64,Int},
                         Ha::SparseMatrixCSC{ComplexF64,Int},
                         Hd::SparseMatrixCSC{ComplexF64,Int},
                         He::SparseMatrixCSC{ComplexF64,Int},
                         dt::Float64)
    a, b, c = 2π*rand(), 2π*rand(), 2π*rand()
    update_H!(H, a, Ha, b, Hd, c, He)
    ψ_new = expmv(-im * dt, H, ψ)
    return ψ_new ./ norm(ψ_new)
end
"""

# ----------------------------
# Entropy_t_z2 (Z^2 measurements) using same in-place H
# ----------------------------
function Entropy_t_z2(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    Random.seed!(shot)
    s_t = spin1_state(L) 
    S_list = Float64[]
    Q_qnv_list = Float64[]

    # Build Hamiltonians once
    _, _, _, sz2l, _ = Hamiltonian(L)
    #H = copy(Ha)  # preallocate H

    # Build global Q operators: Q = sum_i Szi^2
    Q_op = spzeros(ComplexF64, 3^L, 3^L)
    for i in 1:L
        Q_op += sz2l[i]
    end
    Q2_op = Q_op * Q_op  # sparse times sparse -> sparse/dense depending on structure

    steps = Int(floor(T/dt))
    for _ in 1:steps
        push!(S_list, entropy_vn(s_t, L, 1:(L ÷ 2)))
        #push!(Q_qnv_list, real(s_t' * (Q2_op * s_t)) - real(s_t' * (Q_op * s_t))^2)

        s_t = time_evolution(s_t, dt, L, shot)

        # local Z^2 measurements with probability p per site
        if p != 0
            for l in 1:L
                if rand() < p
                    p_m_zero = real(s_t' * (sz2l[l] * s_t))
                    x1 = rand()
                    if x1 < p_m_zero 
                        s_t = (sz2l[l] * s_t) / sqrt(p_m_zero)
                    else
                        s_t = (s_t - sz2l[l] * s_t) / sqrt(1 - p_m_zero)
                    end
                end
            end
        end
    end

    """
    # Save result to disk
    filename = "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_hc.npy" ## half-chain entropy
    npzwrite(filename, S_list)
    filenameq = "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_qnv.npy" ## QNV variance
    npzwrite(filenameq, Q_qnv_list)

    """


    folder_hc = "/Users/uditvarma/Documents/s3_data/data_hc_p"
    #folder_qnv = "/Users/uditvarma/Documents/s3_data/data_qnv_p"
    filename_hc = joinpath(folder_hc, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_hc.npy")
    #filename_qnv = joinpath(folder_qnv, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_qnv.npy")
    npzwrite(filename_hc, S_list)
    #npzwrite(filename_qnv, Q_qnv_list)
    #"""

    return Q_qnv_list
end

function time_evolution(ψ::Vector{ComplexF64}, dt::Float64, L, shot::Int)
    Random.seed!(shot)  # Set random seed for reproducibility
    ψ /= norm(ψ)
    Ha, Hd, He, _, _ = Hamiltonian(L)
    a = 2π * rand()
    b = 2π * rand()
    c = 2π * rand()
    # Apply exp(-im * H * dt) directly to ψ
    #H = a * Ha + b * Hb + c * Hc
    H = spzeros(ComplexF64, length(ψ), length(ψ))
    H .= a .* Ha .+ b .* Hd .+ c .* He ## Sz^2 conserving part
    #H = Hd # Only Heisenberg part
    ψ_new = expmv(-im * dt, H, ψ)

    # Normalize the state
    ψ_new /= norm(ψ_new)
    
    return ψ_new
end