# ----------------------------
# Entropy_t_z (Z measurements) using in-place H
# ----------------------------
function Entropy_t_z(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    Random.seed!(shot)
    s_t = random_product_state(L)
    S_list = Float64[]

    # Build Hamiltonians once
    Ha, Hd, He, sz2l, szl = Hamiltonian(L)
    H = copy(Ha)  # preallocate H with same sparsity

    steps = Int(floor(T/dt))
    for _ in 1:steps
        push!(S_list, entropy_vn(s_t, L, 1:(L ÷ 2)))
        s_t = time_evolution!(s_t, H, Ha, Hd, He, dt)

        # local Z-basis measurements with probability p per site
        if p != 0
            for l in 1:L
                if rand() < p
                    # probabilities for m = -1 and +1 outcomes (projectors built from sz2 & sz)
                    p_m_mone = 0.5 * real(s_t' * (sz2l[l] - szl[l]) * s_t)
                    p_m_one  = 0.5 * real(s_t' * (sz2l[l] + szl[l]) * s_t)
                    x1 = rand()
                    if x1 < p_m_mone && p_m_mone > 0
                        s_t = 0.5 * ((sz2l[l] - szl[l]) * s_t) / sqrt(p_m_mone)
                    elseif x1 < p_m_one + p_m_mone && p_m_one > 0
                        s_t = 0.5 * ((sz2l[l] + szl[l]) * s_t) / sqrt(p_m_one)
                    else
                        # the |0> outcome (complement)
                        normp = 1 - p_m_mone - p_m_one
                        if normp > 0
                            s_t = (s_t - sz2l[l] * s_t) / sqrt(normp)
                        end
                    end
                end
            end
        end
    end

    # Save results
    folder = "/Users/uditvarma/Documents/s3_data/data_hc"
    filename = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_hc.npy")
    npzwrite(filename, S_list)

    return S_list
end




### This function is for Z measurements only

function Entropy_t_z(L::Int, T::Float64, dt::Float64, p::Float64, shot::Int)
    Random.seed!(shot)  # Set random seed
    s_t = random_product_state(L)
    #s_t = neel_spin1_complex(L)  # Use the Néel state as the initial state 
    S_list = Float64[]

    # Define Hamiltonian and local observables
    _, _, _, _, _, sz2l, szl = Hamiltonian(L)
    sm_list = szl
    steps = Int(floor(T / dt))

    for _ in 1:steps
        push!(S_list, entropy_vn(s_t, L, 1:L÷2)) ## Half-chain entropy
        #push!(S_list, tmi(s_t))

        # Time evolution
        s_t = time_evolution(s_t, dt, L, shot)
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

    """
    # Save result to disk
    filename = "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot).npy"
    npzwrite(filename, S_list)
    

    """
    folder = "/Users/uditvarma/Documents/s3_data/data_hc"
    filename = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dirZ,s$(shot)_hc.npy")
    npzwrite(filename, S_list)
    #"""
    
    return S_list
end