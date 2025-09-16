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
    site = normalize!([1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im])
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
        if s > 1e-12
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
    time_evolution(ψ::Vector{ComplexF64}, H::SparseMatrixCSC, dt::Float64)

Performs a single step of time evolution using the ExpmV library.
"""
function time_evolution(ψ::Vector{ComplexF64}, L)

    U_odd  = odd_layer(L)
    U_even = even_layer(L)

    # One full brick-wall step:
    U_step = U_even * U_odd
    ψ_new = U_step * ψ

    return normalize!(ψ_new)
end

# --- Main Simulation Function ---

"""
    Entropy_t_z2(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)

Simulates the time evolution of a quantum state with random Hamiltonians and Z² measurements.
Records and saves the half-chain entanglement entropy and QNV at each time step.
"""
function Entropy_t_z2(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    
    # Initialize state
    #s_t = spin1_state(L)
    s_t = random_product_state(L)
    
    
    
    # Build a list of single-site sz^2 operators for measurement
    sz2l = [create_local_sz2_operator(L, i) for i in 1:L]
    szl = [create_local_sz_operator(L, i) for i in 1:L]
    
    # Build total Sz and Sz^2 operators for QNV calculation
    Q_op = total_Sz2(L)
    Q2_op = Q_op * Q_op
    
    # Initialize lists to store results
    S_list = Float64[]
    Q_qnv_list = Float64[]
    P_list = Float64[]
    steps = Int(floor(T / dt))

    for _ in 1:steps
        # Record half-chain entropy
        push!(S_list, entropy_vn(s_t, L, 1:L÷2))
        
        # Record QNV
        exp_Q2 = dot(s_t, Q2_op * s_t)
        exp_Q = dot(s_t, Q_op * s_t)
        qnv = real(exp_Q2)  - real(exp_Q^2)
        #qnv = real(exp_Q)
        push!(Q_qnv_list, qnv)

        # Time evolution
        s_t = time_evolution(s_t, L)
        
        #"""
        # Measurements S^z
        if p != 0
            for l in 1:L
                x = rand()
                if x < p
                    P_mone = 0.5 * (sz2l[l] - szl[l]) ## projection operators
                    P_one = 0.5 * (sz2l[l] + szl[l])
                    psi_mone = P_mone * s_t # non-normalized states
                    psi_one = P_one * s_t 
                    psi_zero = (s_t - sz2l[l] * s_t)
                    p_m_mone = real(psi_mone' * psi_mone) # probabilities
                    p_m_one = real(psi_one' * psi_one)
                    p_m_zero = real(psi_zero' * psi_zero)
                    x1 = rand()
                    if x1 < p_m_mone
                        s_t = psi_mone / sqrt(p_m_mone)
                    elseif p_m_mone ≤ x1 < p_m_one + p_m_mone
                        s_t = psi_one / sqrt(p_m_one)
                    else
                        s_t = (s_t - sz2l[l] * s_t) / sqrt(p_m_zero)
                    end
                    p_total = p_m_mone + p_m_one + p_m_zero
                    push!(P_list, p_total)
                end
            end
        end
    end
        """
        # Measurements S^z2
        if p != 0
            for l in 1:L
                x = rand()
                if x < p
                    # Measurement on sz^2 operator
                    op = sz2l[l]
                    s_t_proj = op * s_t
                    p_m_zero = real(dot(s_t, s_t_proj))
                    
                    x1 = rand()
                    if x1 < p_m_zero
                        s_t = s_t_proj / sqrt(p_m_zero)
                    else
                        s_t = (s_t - s_t_proj) / sqrt(1 - p_m_zero)
                    end
                end
            end
        end
    end
    """

    # Save results to a file
    folder = "/Users/uditvarma/Documents/s3_data/data_hcrn"
    folderq = "/Users/uditvarma/Documents/s3_data/data_qnvrn"
    mkpath(folder) # Create the folder if it doesn't exist
    mkpath(folderq)
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_hc.npy")
    npzwrite(filename_entropy, S_list)
    filename_qnv = joinpath(folderq, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_qnv.npy")
    npzwrite(filename_qnv, Q_qnv_list)
    return P_list
end


function odd_layer(L::Int)
    id, sx, sy, sz, sp, sm, sz2 = get_spin1_operators()
    U = two_qubit_unitary()
    for i in 3:2:L-1
        U = kron(U, two_qubit_unitary())
    end
    if isodd(L)
        U = kron(U, id)   # pad last site if odd number of spins
    end
    return U
end

function even_layer(L::Int)
    id, sx, sy, sz, sp, sm, sz2 = get_spin1_operators()
    U = id                          # start with a free spin at site 1
    for _ in 2:2:L-1
        U = kron(U, two_qubit_unitary())
    end
    if isodd(L)
        # if L is even, we’ve already used up all spins
        return U
    else
        # if L is odd, pad last site
        return kron(U, id)
    end
end

using ExponentialUtilities

function two_qubit_unitary()
    
    id, sx, sy, sz, sp, sm, sz2 = get_spin1_operators()
    dim = 3^2
    
    Ha = spzeros(ComplexF64, dim, dim)
    He = spzeros(ComplexF64, dim, dim)
    
    term1 = kron(sp^2, sm^2) + kron(sm^2, sp^2)
    Ha += 2π * rand() * (term1 + adjoint(term1))
    #Ha += (term1 + adjoint(term1))
    
    
    H = Matrix(Ha)               
    U = exp(-im * H)            
    return sparse(U)
end
