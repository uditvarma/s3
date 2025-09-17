using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using ExponentialUtilities
using NPZ
using ExpmV
using Dates

Random.seed!(Dates.now().instant.periods.value)

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

### Ancilla time evolution

function time_evolution_an(ψ::Vector{ComplexF64}, L)
    id, sx, sy, sz, sp, sm, sz2 = get_spin1_operators()

    U_odd  = odd_layer(L)
    U_even = even_layer(L)

    U_odd_anc = kron(U_odd, id) ## To couple with the ancilla
    U_even_anc = kron(U_even, id)

    # One full brick-wall step:
    U_step = U_even_anc * U_odd_anc
    ψ_new = U_step * ψ

    return normalize!(ψ_new)
end

function build_state(L::Int; normalize::Bool=true)
    up, zero, down = [1.0,0,0], [0,1.0,0], [0,0,1.0]

    kron_list(vs) = reduce(kron, vs)

    # alternating sequences
    alt_up   = [ isodd(i) ? up   : down for i in 1:L ]
    alt_down = [ isodd(i) ? down : up   for i in 1:L ]

    # if L odd, replace last with |0>
    if isodd(L)
        alt_up[end]   = zero
        alt_down[end] = zero
    end

    state1 = kron_list(alt_up)   |> x -> kron(x, up)
    state2 = kron_list(alt_down) |> x -> kron(x, down)
    state3 = kron_list(fill(zero, L)) |> x -> kron(x, zero)

    ψ = state1 + state2 + state3
    ψ = ComplexF64.(ψ)
    return normalize ? ψ / norm(ψ) : ψ
end


#### Main simulation function

function Entropy_anc_z(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    
    # Initialize state
    #s_t = spin1_state(L)
    #s_t = random_product_state(L)
    s_t = build_state(L) ## Same sector state but maximally entangled
    
    
    # Build a list of single-site sz^2 operators for measurement
    sz2l = [create_local_sz2_operator(L+1, i) for i in 1:L]
    szl = [create_local_sz_operator(L+1, i) for i in 1:L]
    
    
    
    # Initialize lists to store results
    S_list = Float64[]
    #P_list = Float64[]
    steps = Int(floor(T / dt))

    for n in 1:steps
        # Record half-chain entropy
        push!(S_list, entropy_vn(s_t, L+1, 1:L))

        # Time evolution
        s_t = time_evolution_an(s_t, L)

        p_eff = (n <= steps ÷ 5 ? 0.0 : p)
        
        #"""
        # Measurements S^z
        if p_eff != 0
            for l in 1:L
                x = rand()
                if x < p
                    P_mone = 0.5 * (sz2l[l] - szl[l]) ## projection operators
                    P_one = 0.5 * (sz2l[l] + szl[l])
                    psi_mone = P_mone * s_t # non-normalized states
                    psi_one = P_one * s_t 
                    #psi_zero = (s_t - sz2l[l] * s_t)
                    p_m_mone = real(psi_mone' * psi_mone) # probabilities
                    p_m_one = real(psi_one' * psi_one)
                    #p_m_zero = real(psi_zero' * psi_zero)
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
    folder = "/Users/uditvarma/Documents/s3_data/data_anc_c"
    mkpath(folder) # Create the folder if it doesn't exist
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_anc.npy")
    npzwrite(filename_entropy, S_list)
    return S_list
end

function bell_state_3(N)
    # local basis vectors: |+1>, |0>, |-1>
    up   = [1.0, 0.0, 0.0]   # |+1>
    zero = [0.0, 1.0, 0.0]   # |0>
    down = [0.0, 0.0, 1.0]   # |-1>

    # helper: kron repeated N times
    function kron_chain(v, N)
        result = v
        for _ in 2:N
            result = kron(result, v)
        end
        return result
    end

    # product states
    psi_up   = kron_chain(up, N)
    psi_zero = kron_chain(zero, N)
    psi_down = kron_chain(down, N)

    # build the superposition (they are orthogonal computational basis states)
    psi = psi_up .+ psi_zero .+ psi_down

    # normalize
    normpsi = norm(psi)
    psi ./= normpsi

    return ComplexF64.(psi)
end

function Entropy_anc_z2(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    
    # Initialize state
    #s_t = spin1_state(L)
    #s_t = random_product_state(L)
    s_t = bell_state_3(L+1)
    
    
    # Build a list of single-site sz^2 operators for measurement
    sz2l = [create_local_sz2_operator(L+1, i) for i in 1:L]
    #szl = [create_local_sz_operator(L, i) for i in 1:L]
    """
    # Build total Sz and Sz^2 operators for QNV calculation
    Q_op = total_Sz2(L)
    Q2_op = Q_op * Q_op
    """
    
    # Initialize lists to store results
    S_list = Float64[]
    #Q_qnv_list = Float64[]
    #P_list = Float64[]
    steps = Int(floor(T / dt))

    for n in 1:steps
        # Record half-chain entropy
        push!(S_list, entropy_vn(s_t, L+1, 1:L))
        
        # Record QNV
        #exp_Q2 = dot(s_t, Q2_op * s_t)
        #exp_Q = dot(s_t, Q_op * s_t)
        #qnv = real(exp_Q2)  - real(exp_Q^2)
        #qnv = real(exp_Q)
        #push!(Q_qnv_list, qnv)

        # Time evolution
        s_t = time_evolution_an(s_t, L)

        p_eff = (n <= steps ÷ 5 ? 0.0 : p)
        
        # Measurements S^z2
        if p_eff != 0
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

    # Save results to a file
    folder = "/Users/uditvarma/Documents/s3_data/data_anc_cc"
    #folderq = "/Users/uditvarma/Documents/s3_data/data_qnvrn"
    mkpath(folder) # Create the folder if it doesn't exist
    #mkpath(folderq)
    filename_entropy = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_anc.npy")
    npzwrite(filename_entropy, S_list)
    #filename_qnv = joinpath(folderq, "L$(L),T$(T),dt$(dt),p$(p),dirZ2,s$(shot)_qnv.npy")
    #npzwrite(filename_qnv, Q_qnv_list)
    return S_list
end