using LinearAlgebra, CSV, DataFrames, DelimitedFiles, StatsBase, Random

HD = (1/sqrt(2))*[1 1; 1 -1]
ID2 = [1 0; 0 1]
HD = kron(HD, ID2)
CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
EG = CNOT * HD

function f(x)
    if x == 1; return [0,1]; end
    if x == 0; return [1,0]; end
end

function tob10(vecs)
    join(vecs)
    parse(Int, join(vecs); base = 2)
end

function splitleft(state, dimstate)
    statenew = zeros(dimstate)
    for i in 1:dimstate
        statenew[i] = Int64(state[i])
    end
    return round.(Int,statenew)
end

function splitright(state,dimstate)
    statenew = zeros(length(state)-dimstate)
    for i in dimstate+1:length(state)
        statenew[i-dimstate] = Int64(state[i])
    end
    return round.(Int,statenew)
end

function splicee(state,left,right)
    statevo = zeros(2)
    statevo[1] = state[left]
    statevo[2] = state[right]
    return round.(Int,statevo)
end

function splicer(state,left,right)
    statevo = zeros(length(state)-2)
    jp = 0
    for i=1:left-1
        jp += 1
        statevo[jp] = state[i]
    end
    for i=left+1:right-1
        jp += 1
        statevo[jp] = state[i]
    end
    for i=right+1:length(state)
        jp += 1
        statevo[jp] = state[i]
    end
    return round.(Int,statevo)
end

px = [0 1;1 0]
py = [0 -1im; 1im 0]
pz = [1 0; 0 -1]

function Haargate()
    a = randn(ComplexF64,(4,4))
    Q,R = qr(a)
    λ = Diagonal(R)/(abs.(Diagonal(R)))
    U = Matrix(Q)*λ
    return U
end

function binaryvec(n)
    ket0 = string(n, base=2, pad = nspin)
    for k in 1:length(ket0)
        kettemp[k] = parse(Int64,ket0[k:k])
    end
    return kettemp
end

function reyni(state,nth,cut)
    n = nth
    diml = cut
    dimr = nspin - cut
    A = zeros(ComplexF64,2^dimr,2^diml)
    srn = 0
    for j in 0:dim-1
        l = tob10(splitright(ket[j+1],cut))
        m = tob10(splitleft(ket[j+1],cut))
        A[l+1,m+1] = state[j+1]
    end
    evls = svdvals(A)
    for i in 1:length(evls)
        if 0 < ((norm(evls[i]))^2n)
            srn += ((norm(evls[i]))^2n)
        end
    end
    if n==1
        srni = -sum(((norm(evls[i]))^2)*log2(norm(evls[i])^2) for i in 1:length(evls) if 0 < (norm(evls[i]))^2)
    else
        srni = (1/(1-n))*log2(srn)
    end
    return srni
end

function engate(state, enspin)
    nl = enspin
    nr = nspin
    state = state / norm(state)
    A = zeros(ComplexF64,4,2^(nspin-2))
    f = zeros(Int64,4,2^(nspin-2))
    for j in 1:dim
        l = tob10(splicee(ket[j],nl,nr))
        m = tob10(splicer(ket[j],nl,nr))
        A[l+1,m+1] = state[j]
        f[l+1,m+1] = j
    end
    A = EG*A
    for i = 1:size(A)[1], j = 1:size(A)[2]
        state[f[i,j]] = A[i,j]
    end
    return state
end

function gatean(statein,i)
    L = nspin
    U = Haargate()
    psi_temp = zeros(ComplexF64, 4)
    if i != nspin
        for nL = 0:2^(L-1-i)-1, nR = 0:2^(i-1)-1
            ind1 = nR + nL*2^(i+1) + 1
            ind2 = nR + nL*2^(i+1) + 2^(i-1) + 1
            ind3 = nR + nL*2^(i+1) + 2^(i) + 1
            ind4 = nR + nL*2^(i+1) + 2^(i) + 2^(i-1) + 1
            psi_temp[1] = statein[ind1]
            psi_temp[2] = statein[ind2]
            psi_temp[3] = statein[ind3]
            psi_temp[4] = statein[ind4]
            psi_temp = U * psi_temp
            statein[ind1] = psi_temp[1]
            statein[ind2] = psi_temp[2]
            statein[ind3] = psi_temp[3]
            statein[ind4] = psi_temp[4]
        end
    end
    return statein
end

function projmsman(statein)
    L = nspin
    for i = 2:L
        p1 = rand()
        pup = 0.0
        if p1 < prob
            for nL = 0:(2^(L-i)-1), nR = 0:(2^(i-1)-1)
                n = nR + 2^(i)*nL + 2^(i-1) + 1
                pup += statein[n]*conj(statein[n])
            end
            p2 = rand()
            if p2 < Real(pup)
                for nL = 0:(2^(L-i)-1), nR = 0:(2^(i-1)-1)
                    n = nR + 2^(i)*nL + 1
                    statein[n] = 0
                end
            else
                for nL = 0:(2^(L-i)-1), nR = 0:(2^(i-1)-1)
                    n = nR + 2^(i)*nL + 2^(i-1) + 1
                    statein[n] = 0
                end
            end
        end
        statein = statein / norm(statein)
    end
    return statein
end

function runanc(nspins, iteration, probability, filename)
    global prob = probability
    global nspin = nspins + 1
    global dim = 2^nspin
    global ket = [zeros(Int64,nspin)]
    global kettemp = zeros(Int64,nspin)
    timestep = 2 * nspin
    iterations = iteration
    x = zeros(iterations)
    vals1 = zeros(iterations)
    vals2 = zeros(iterations)
    vals3 = zeros(iterations)
    n1th, n2th, n3th = 1, 2, 3
    cut = nspin - 1
    for j in 1:dim-1
        temp = Vector(binaryvec(j))
        push!(ket, temp)
    end
    for k = 1:iterations
        stateun = rand(Complex{Float64}, dim)
        state = stateun / norm(stateun)
        state = engate(state, nspin-1)
        for i = 1:Int64(timestep/2)
            for j = 1:(nspin-1)÷2
                gatean(state, 2*j)
            end
            for j = 2:(nspin-1)÷2+1
                j == 2 ? gatean(state, nspin) : gatean(state, 2*(j-3)+3)
            end
        end
        for i = Int64(timestep/2+1):timestep
            for j = 1:(nspin-1)÷2
                gatean(state, 2*j)
            end
            state = projmsman(state)
            if i == timestep
                vals1[k] += reyni(state, n1th, cut)/2
                vals2[k] += reyni(state, n2th, cut)/2
                vals3[k] += reyni(state, n3th, cut)/2
            end
            for j = 2:(nspin-1)÷2+1
                j == 2 ? gatean(state, nspin) : gatean(state, 2*(j-3)+3)
            end
            state = projmsman(state)
            if i == timestep
                vals1[k] += reyni(state, n1th, cut)/2
                vals2[k] += reyni(state, n2th, cut)/2
                vals3[k] += reyni(state, n3th, cut)/2
            end
        end
    end
    writedlm(string(filename, "it_reyni1"), vals1, ", ")
    writedlm(string(filename, "it_reyni2"), vals2, ", ")
    writedlm(string(filename, "it_reyni3"), vals3, ", ")
    writedlm(string(filename, "reyni1"), mean(vals1), ", ")
    writedlm(string(filename, "reyni2"), mean(vals2), ", ")
    writedlm(string(filename, "reyni3"), mean(vals3), ", ")
end