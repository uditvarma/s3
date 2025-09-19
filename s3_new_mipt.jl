using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV
using Dates

Random.seed!(Dates.now().instant.periods.value)

# --- Utility Functions ---

"""
    random_product_state(L::Int)

Generates a random product state for L qutrits (spin-1 particles).
Each site state is a random superposition of the three basis states.
"""
function random_product_state(L::Int)
    dim_site = 3
    state_vec = ComplexF64[]
    
    for _ in 1:L
        # Generate a random state vector for a single qutrit
        c1 = rand(ComplexF64)
        c2 = rand(ComplexF64)
        c3 = rand(ComplexF64)
        site_state = normalize!([c1, c2, c3])

        if isempty(state_vec)
            state_vec = site_state
        else
            state_vec = kron(state_vec, site_state)
        end
    end
    
    return normalize!(state_vec)
end

"""
    spin1_state(L::Int)

Generates a specific initial state for L qutrits, a uniform superposition
of the three basis states at each site.
"""
function spin1_state(L::Int)
    site = normalize!([1.0 + 0.0im, 1.0 + 0.0im, 1.0 + .0im])
    ψ = site
    for _ in 2:L
        ψ = kron(ψ, site)
    end
    return normalize!(ψ)
end

"""
    entropy_vn(ψ::Vector{<:Complex}, L::Int, subsystem::AbstractArray{Int})

Calculates the von Neumann entanglement entropy for a given subsystem.
"""
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

# --- Operator and Hamiltonian Construction ---

"""
    get_spin1_operators()

Returns the local spin-1 operators (Pauli matrices) as sparse matrices.
"""
function get_spin1_operators()
    id = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    sx = 1/sqrt(2) * sparse(ComplexF64[0 1 0; 1 0 1; 0 1 0])
    sy = 1/sqrt(2) * sparse(ComplexF64[0 -im 0; im 0 -im; 0 im 0])
    sz = sparse(ComplexF64[1 0 0; 0 0 0; 0 0 -1])
    sp = 1/sqrt(2) * (sx + im * sy)
    sm = 1/sqrt(2) * (sx - im * sy)
    sz2 = sz * sz
    
    return id, sx, sy, sz, sp, sm, sz2
end

"""
    build_term(operators::Vector{<:SparseMatrixCSC})

Helper function to tensor a list of local operators together.
"""
function build_term(operators::Vector{<:SparseMatrixCSC})
    term = operators[1]
    for j in 2:length(operators)
        term = kron(term, operators[j])
    end
    return term
end

"""
    create_local_sz2_operator(L::Int, site::Int)

Creates a full-space operator with sz^2 at a given site and
identity matrices on all other sites.
"""
function create_local_sz2_operator(L::Int, site::Int)
    id, _, _, _, _, _, sz2 = get_spin1_operators()
    ops = fill(id, L)
    ops[site] = sz2
    return build_term(ops)
end

"""
    create_local_sz_operator(L::Int, site::Int)

Creates a full-space operator with sz at a given site and
identity matrices on all other sites.
"""
function create_local_sz_operator(L::Int, site::Int)
    id, _, _, sz, _, _, _ = get_spin1_operators()
    ops = fill(id, L)
    ops[site] = sz
    return build_term(ops)
end

"""
    total_Sz(L::Int)

Constructs the total Sz operator for L sites.
"""
function total_Sz2(L::Int)
    dim = 3^L
    Sz2_tot = spzeros(ComplexF64, dim, dim)
    for i in 1:L
        Sz2_tot += create_local_sz2_operator(L, i)
    end
    return Sz2_tot
end


"""
    Hamiltonian(L::Int)

Constructs the total Hamiltonian for L sites, consisting of several
components (Ha, Hb, Hc, Hd, He).
"""
function Hamiltonian(L::Int)
    id, _, _, sz, sp, sm, _ = get_spin1_operators()
    dim = 3^L
    
    # Initialize Hamiltonians
    Ha = spzeros(ComplexF64, dim, dim)
    Hb = spzeros(ComplexF64, dim, dim)
    Hc = spzeros(ComplexF64, dim, dim)
    Hd = spzeros(ComplexF64, dim, dim)
    He = spzeros(ComplexF64, dim, dim)
    
    for i in 1:L
        # Periodic boundary condition
        ip = mod1(i + 1, L)
        
        # Ha term: (sp^2_i * sm^2_ip)
        ops_Ha_1 = fill(id, L)
        ops_Ha_1[i] = sp^2
        ops_Ha_1[ip] = sm^2
        Ha += build_term(ops_Ha_1)
        
        # Ha term: (sm^2_i * sp^2_ip)
        ops_Ha_2 = fill(id, L)
        ops_Ha_2[i] = sm^2
        ops_Ha_2[ip] = sp^2
        Ha += build_term(ops_Ha_2)
        
        # Hb term: (sp_i * sz_i * sm_ip * sz_ip)
        ops_Hb_1 = fill(id, L)
        ops_Hb_1[i] = sp * sz
        ops_Hb_1[ip] = sm * sz
        Hb += build_term(ops_Hb_1)
        
        # Hb term: (sm_i * sz_i * sp_ip * sz_ip)
        ops_Hb_2 = fill(id, L)
        ops_Hb_2[i] = sm * sz
        ops_Hb_2[ip] = sp * sz
        Hb += build_term(ops_Hb_2)

        # Hc term: (sp^2_i * sp_ip * sz_ip)
        ops_Hc_1 = fill(id, L)
        ops_Hc_1[i] = sp^2
        ops_Hc_1[ip] = sp * sz
        Hc += build_term(ops_Hc_1)
        
        # Hc term: (sm^2_i * sm_ip * sz_ip)
        ops_Hc_2 = fill(id, L)
        ops_Hc_2[i] = sm^2
        ops_Hc_2[ip] = sm * sz
        Hc += build_term(ops_Hc_2)

        # He term: (sz_i * sz_ip)
        ops_He = fill(id, L)
        ops_He[i] = sz
        ops_He[ip] = sz
        He += build_term(ops_He)
    end
    
    # Hd term: (long-range)
    for i in 1:L
        ip = mod1(i + 2, L)
        ops_Hd_1 = fill(id, L)
        ops_Hd_1[i] = sp^2
        ops_Hd_1[ip] = sm^2
        Hd += build_term(ops_Hd_1)

        ops_Hd_2 = fill(id, L)
        ops_Hd_2[i] = sm^2
        ops_Hd_2[ip] = sp^2
        Hd += build_term(ops_Hd_2)
    end
    
    # Ha, Hb, Hc, Hd must be Hermitian
    Ha += adjoint(Ha)
    Hb += adjoint(Hb)
    Hc += adjoint(Hc)
    Hd += adjoint(Hd)

    return Ha, Hb, Hc, Hd, He
end

"""
    time_evolution(ψ::Vector{ComplexF64}, H::SparseMatrixCSC, dt::Float64)

Performs a single step of time evolution using the ExpmV library.
"""
function time_evolution(ψ::Vector{ComplexF64}, Ha, Hd, He, dt::Float64)
    #Ha, _, _, Hd, He = Hamiltonian(L)
    
    # Generate random Hamiltonian
    a = 2π * rand()
    b = 2π * rand()
    c = 2π * rand()

    H = a * Ha + b * Hd + c * He
    #H = a * Ha

    ψ_new = expmv(-im * dt, H, ψ)
    return normalize!(ψ_new)
end

# --- Main Simulation Function ---

"""
    Entropy_t_z2(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)

Simulates the time evolution of a quantum state with random Hamiltonians and Z² measurements.
Records and saves the half-chain entanglement entropy and QNV at each time step.
"""
function Entropy_t_z(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    
    # Initialize state
    #s_t = spin1_state(L)
    s_t = random_product_state(L)
    
    
    # Build Hamiltonians once per shot
    Ha, _, _, Hd, He = Hamiltonian(L)
    
    """
    # Generate random Hamiltonian
    a = 2π * rand()
    b = 2π * rand()
    c = 2π * rand()
    #H = a * Ha + b * Hd + c * He
    """
    
    # Build a list of single-site sz^2 operators for measurement
    sz2l = [create_local_sz2_operator(L, i) for i in 1:L]
    szl = [create_local_sz_operator(L, i) for i in 1:L]
        
    # Build total Sz and Sz^2 operators for QNV calculation
    #Q_op = total_Sz2(L) ## conserved charge
    #Q2_op = Q_op * Q_op
    
    # Initialize lists to store results
    S_list = Float64[]
    #Q_qnv_list = Float64[]
    steps = Int(floor(T / dt))

    for _ in 1:steps
        # Record half-chain entropy
        push!(S_list, entropy_vn(s_t, L, 1:L÷2))
        
        # Record QNV
        #exp_Q2 = dot(s_t, Q2_op * s_t)
        #exp_Q = dot(s_t, Q_op * s_t)
        #qnv = real(exp_Q2)  - real(exp_Q^2)
        #qnv = real(exp_Q)

        # Time evolution
        s_t = time_evolution(s_t, Ha, Hd, He, dt)

        # Measurements
        if p != 0
            for l in 1:L
                x = rand()
                if x < p
                    P_mone = 0.5 * (sz2l[l] - szl[l]) ## projection operators
                    P_one = 0.5 * (sz2l[l] + szl[l])
                    psi_mone = P_mone * s_t # non-normalized states
                    psi_one = P_one * s_t
                    p_m_mone = real(psi_mone' * psi_mone) # probabilities
                    p_m_one = real(psi_one' * psi_one)
                    x1 = rand()
                    if x1 < p_m_mone
                        s_t = psi_mone / sqrt(p_m_mone)
                    elseif p_m_mone ≤ x1 < p_m_one + p_m_mone
                        s_t = psi_one / sqrt(p_m_one)
                    else
                        s_t = (s_t - sz2l[l] * s_t) / sqrt(1 - p_m_mone - p_m_one)
                    end
                    #p_total = p_m_mone + p_m_one + p_m_zero
                    #push!(P_list, p_total)
                end
            end
        end
    end

    # Save results to a file
    folder = "/Users/uditvarma/Documents/s3_data/data_hcn"
    #folderq = "/Users/uditvarma/Documents/s3_data/data_qnvn"
    mkpath(folder) # Create the folder if it doesn't exist
    #mkpath(folderq)
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_hc.npy")
    npzwrite(filename_entropy, S_list)
    return S_list
end