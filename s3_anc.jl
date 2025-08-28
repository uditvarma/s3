using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV

### Add Lth spin to the Hamiltonian with the coupling as zeros

function Hamiltonian_an(L)
    # Define Pauli matrices as complex sparse matrices
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    sx = 1/sqrt(2) * sparse(ComplexF64[0 1 0; 1 0 1; 0 1 0])
    sy = 1/sqrt(2) * sparse(ComplexF64[0 -im 0; im 0 -im; 0 im 0])
    sz = sparse(ComplexF64[1 0 0; 0 0 0; 0 0 -1])

    sp = 1/sqrt(2) * (sx + im * sy)
    sm = 1/sqrt(2) * (sx - im * sy)

    J_list = [i < L-1 ? 1.0 : 0.0 for i in 1:L-1]
    J_list_nnn = [i < L-2 ? 1.0 : 0.0 for i in 1:L-2]

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

    dim = 3^(L)
    Ha = spzeros(ComplexF64, dim, dim)

    for i in 1:L-1
        ip = (i  + 1)  # Open boundary
        Ha += J_list[i] * ((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2) + adjoint((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)))
    end

    Hb = spzeros(ComplexF64, dim, dim)

    for i in 1:L-1
        ip = (i + 1) # Open boundary
        Hb += J_list[i] * (spl[i] * szl[i] * sml[ip] * szl[ip] + sml[i] * szl[i] * spl[ip] * szl[ip] + adjoint(spl[i] * szl[i] * sml[ip] * szl[ip]) + adjoint(sml[i] * szl[i] * spl[ip] * szl[ip]))
    end

    Hc = spzeros(ComplexF64, dim, dim)

    for i in 1:L-1
        ip = (i + 1)  # Open boundary
        Hc += J_list[i] * (spl[i]^2 * spl[ip] * szl[ip] + sml[i]^2 * sml[ip] * szl[ip] + adjoint(spl[i]^2 * spl[ip] * szl[ip]) + adjoint(sml[i]^2 * sml[ip] * szl[ip]))
    end

    Hd = spzeros(ComplexF64, dim, dim)

    for i in 1:L-2
        ip = (i + 2)  # Open boundary
        Hd += J_list_nnn[i] * ((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2) + adjoint((spl[i]^2 * sml[ip]^2 + sml[i]^2 * spl[ip]^2)))
    end
    #Hd += ((spl[L-1]^2 * sml[1]^2 + sml[L-1]^2 * spl[1]^2) + adjoint((spl[L-1]^2 * sml[1]^2 + sml[L-1]^2 * spl[1]^2)))
    #Hd += ((spl[L]^2 * sml[2]^2 + sml[L]^2 * spl[2]^2) + adjoint((spl[L]^2 * sml[2]^2 + sml[L]^2 * spl[2]^2)))
    
    
    He = spzeros(ComplexF64, dim, dim)

    for i in 1:L-1
        ip = (i + 1)  # Open boundary
        He += J_list[i] * (szl[i] * szl[ip])
    end

    return Ha, Hb, Hc, Hd, He, sz2l, szl 
end

function Entropy_z_an(L::Int, dt::Float64, p::Float64, shot::Int)
    ## Change s_t to the Bell state as required
    #s_t = spin1_bell(L)
    s_t = superposition_state(L)
    T   = 10 * L ## PRX paper
    _, _, _, _, _, sz2l, szl = Hamiltonian_an(L)
    sm_list = szl
    Sanc_list = Float64[]
    

    steps = Int(floor(T / dt))

    for n in 1:steps
        push!(Sanc_list, entropy_vn(s_t, L, 1:L-1)) ## Entropy of the ancilla spin
    

        # Time evolution
        s_t = time_evolution(s_t, dt, L, shot)

        # Effective measurement probability:
        #   = 0.0  for n ≤ steps/2
        #   = p    for n > steps/2
        p_eff = (n <= steps ÷ 5 ? 0.0 : p)

        # Measurements
        if p_eff != 0
            for l in 1:L-1 ## Not measuring the ancilla spin
                x = rand()
                if x < p_eff
                    p_m_mone = 0.5 * real(s_t' * (sz2l[l] - sm_list[l]) * s_t)
                    p_m_one  = 0.5 * real(s_t' * (sz2l[l] + sm_list[l]) * s_t)
                    x1 = rand()
                    if x1 < p_m_mone
                        s_t = 0.5 * ((sz2l[l] - sm_list[l]) * s_t) / sqrt(p_m_mone)
                    elseif p_m_mone < x1 < (p_m_one + p_m_mone)
                        s_t = 0.5 * ((sz2l[l] + sm_list[l]) * s_t) / sqrt(p_m_one)
                    else
                        s_t = (s_t - sz2l[l] * s_t) / sqrt(1 - p_m_mone - p_m_one)
                    end
                end
            end
        end
    end

    
    # Save result to disk
    """
    filename = "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_anc.npy"
    npzwrite(filename, Sanc_list)
    """
    folder = "/Users/uditvarma/Documents/s3_data/data_anc"
    filename = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_anc.npy")
    
    npzwrite(filename, Sanc_list)
    #"""
    return Sanc_list
end

### This creates a Bell pair state |Up, Down, ...> \kron |Up>_anc + |Down, Up, ...> \kron |Down>_anc

function spin1_bell(L::Int)
    dim = 3^L
    psi = zeros(ComplexF64, dim)

    # Helper: convert spin pattern to basis index
    # mapping: -1 -> 0, 0 -> 1, +1 -> 2
    function pattern_to_index(pattern)
        idx = 1
        for i in 1:L
            idx += (pattern[i]+1)*3^(L-i)
        end
        return idx
    end

    # Build the two patterns
    pattern1 = [i%2==1 ? 1 : -1 for i in 1:L-1]  # alternating pattern except last
    push!(pattern1, 1)  # enforce last spin = up

    pattern2 = [i%2==1 ? -1 : 1 for i in 1:L-1] # alternating pattern except last
    push!(pattern2, -1) # enforce last spin = down

    # Get computational basis indices
    idx1 = pattern_to_index(pattern1)
    idx2 = pattern_to_index(pattern2)

    # Set amplitudes
    psi[idx1] = 1/sqrt(2)
    psi[idx2] = 1/sqrt(2)

    return psi
end

### This creates a GHZ S=1 state

function spin1_ghz(L::Int)
    dim = 3^L
    psi = zeros(ComplexF64, dim)

    # Helper: convert spin pattern to basis index
    # mapping: -1 -> 0, 0 -> 1, +1 -> 2
    function pattern_to_index(pattern)
        idx = 1
        for i in 1:L
            idx += (pattern[i]+1)*3^(L-i)
        end
        return idx
    end

    # Pattern 1: Up,Down,Up,... and last spin = +1 (Up)
    pattern1 = [(i % 2 == 1 ? 1 : -1) for i in 1:L-1]
    push!(pattern1, 1)

    # Pattern 2: all zeros and last spin = 0
    pattern2 = fill(0, L-1)
    push!(pattern2, 0)

    # Pattern 3: Down,Up,Down,... and last spin = -1 (Down)
    pattern3 = [(i % 2 == 1 ? -1 : 1) for i in 1:L-1]
    push!(pattern3, -1)

    # Get computational basis indices
    idx1 = pattern_to_index(pattern1)
    idx2 = pattern_to_index(pattern2)
    idx3 = pattern_to_index(pattern3)

    # Set amplitudes (use 1 for an unnormalized sum, or divide by √3 to normalize)
    psi[idx1] = 1
    psi[idx2] = 1
    psi[idx3] = 1

    # Normalization:
    psi ./= sqrt(3)

    return psi
end

function total_Sz2(L)
    _, _, _, _, _, sz2l, _ = Hamiltonian_an(L)  # get sz² list from your Hamiltonian builder
    dim = 3^L
    Sz2_tot = spzeros(ComplexF64, dim, dim)

    for i in 1:L
        Sz2_tot += sz2l[i]
    end

    return Sz2_tot
end

"""
    superposition_state(N::Int)

Construct the normalized superposition state (|000...0⟩ + |111...1⟩)/√2 
for N spin-1 sites. 
Here |0⟩ is the m=0 eigenstate of Sz and |1⟩ is the m=+1 eigenstate.
Returns a vector of length 3^N.
"""
function superposition_state(N::Int)
    # Local spin-1 basis states
    m_pos1 = [1.0, 0.0, 0.0]  # |m=-1>
    m_0    = [0.0, 1.0, 0.0]  # |m=0>  (our "0" state)
    m_neg1 = [0.0, 0.0, 1.0]  # |m=+1> (our "1" state)
    
    # Build |000...0⟩
    state0 = m_0
    for i in 2:N
        state0 = kron(state0, m_0)
    end
    
    # Build |111...1⟩
    state1 = m_pos1
    for i in 2:N
        state1 = kron(state1, m_pos1)
    end

    # Build |-1-1-1...-1⟩
    state2 = m_neg1
    for i in 2:N
        state2 = kron(state2, m_neg1)
    end
    
    # Superposition (normalized)
    state = (state0 .+ state1 .+ state2) ./ sqrt(3)
    
    return ComplexF64.(state)
end

function Entropy_z2_an(L::Int, dt::Float64, p::Float64, shot::Int)
    ## Change s_t to the Bell state as required
    #s_t = spin1_bell(L)
    #s_t = spin1_ghz(L)
    s_t = superposition_state(L)
    T   = 10 * L ## PRX paper
    _, _, _, _, _, sz2l, _ = Hamiltonian_an(L)
    Sanc_list = Float64[]
    
    steps = Int(floor(T / dt))

    for n in 1:steps
        push!(Sanc_list, entropy_vn(s_t, L, 1:L-1)) ## Entropy of the ancilla spin

        # Time evolution
        s_t = time_evolution(s_t, dt, L, shot)

        # Effective measurement probability:
        #   = 0.0  for n ≤ steps/2
        #   = p    for n > steps/2
        p_eff = (n <= steps ÷ 5 ? 0.0 : p)

        # Measurements
        if p_eff != 0
            for l in 1:L-1 ## Not measuring the ancilla spin
                x = rand()
                if x < p_eff
                    p_m_zero = real(s_t' * (sz2l[l]) * s_t)
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

    
    # Save result to disk
    """
    filename = "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_anc.npy"
    npzwrite(filename, Sanc_list)
    """
    folder = "/Users/uditvarma/Documents/s3_data/data_anc"
    filename = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_anc.npy")
    npzwrite(filename, Sanc_list)
    
    #"""
    return Sanc_list
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



function time_evolution(ψ::Vector{ComplexF64}, dt::Float64, L, shot::Int)
    Random.seed!(shot)  # Set random seed for reproducibility
    ψ /= norm(ψ)
    Ha, Hb, Hc, Hd, He, _, _ = Hamiltonian_an(L)
    a = 2π * rand()
    b = 2π * rand()
    c = 2π * rand()
    # Apply exp(-im * H * dt) directly to ψ
    #H = a * Ha + b * Hb + c * Hc
    H = a * Ha + b * Hd + c * He
    ψ_new = expmv(-im * dt, H, ψ)

    # Normalize the state
    ψ_new /= norm(ψ_new)
    
    return ψ_new
end

############################################################################


# Function to generate spin-1 Néel state vector of length N as complex numbers
function neel_spin1_complex(N::Int)
    # Construct the Néel pattern: Up, Dn, Up, Dn...
    neel_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    
    # Start with first spin
    psi = spin1_vector(neel_state[1])
    
    # Build the full tensor product
    for j in 2:N
        psi = kron(psi, spin1_vector(neel_state[j]))
    end
    
    # Convert to complex vector
    psi_complex = ComplexF64.(psi)  # broadcast conversion
    
    return psi_complex
end

