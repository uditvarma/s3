using MPI
using Printf
using Statistics: mean 
using PyCall

function kicked_rotor_map(theta, I, K)
    I_next = mod(I + K*sin(theta), 2*pi)
    theta_next = mod(theta + I + K*sin(theta), 2*pi)
    return theta_next, I_next
end

function control_map(theta, I, a, theta_fp, I_fp)
    I_next = I
    if I-I_fp > pi
        I_next = I - 2*pi
    elseif I-I_fp < - pi
        I_next = I + 2*pi
    end
    I_next = mod(a*I_next + (1-a)*I_fp, 2*pi)
    theta_next = theta
    if theta-theta_fp > pi
        theta_next = theta - 2*pi
    elseif theta-theta_fp < - pi
        theta_next = theta + 2*pi
    end
    theta_next = mod(a*theta_next + (1-a)*theta_fp, 2*pi)
    return theta_next, I_next
end

function random_map(theta, I, K, a, theta_fp, I_fp, probability)
    p = rand()
    if p > probability
        theta, I = kicked_rotor_map(theta, I, K)
    else
        theta, I = control_map(theta, I, a, theta_fp, I_fp)
    end
    return theta, I
end

function distance(theta, I, theta_fp, I_fp)
    if abs(I-I_fp) < pi
        r1 = (I-I_fp)^2
    else
        r1 = (2*pi - abs(I-I_fp))^2
    end
    if abs(theta-theta_fp) < pi
        r2 = (theta-theta_fp)^2
    else
        r2 = (2*pi-abs(theta-theta_fp))^2
    end
    return sqrt(r1+r2)
end

function O(theta, I, theta_fp, I_fp)
    O = distance(theta, I, theta_fp, I_fp)
    O = O / (sqrt(2) * pi)
    return O
end


function main()
    # MPI initialization
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Parameters
    number_of_trajectories = 128000
    theta_fp = 0.0
    I_fp = 0.0
    number_of_prob_points = 100
    a = 0.5

    number_diff_steps = 5
    number_of_K = 1
    number_of_prob_iterv = 8

    u = 1

    if rank == 0
        # 1 initial interval from plots
        in_intervals = [[4.0, 0.7084, 0.7224]] #[[4.0, 0.69, 0.79]]

        steps_values = [4000, 6000, 8000, 10000, 12000] #[200, 300, 400, 500, 600, 700, 800, 900, 1100] 

        #distribution of initial intervals among 40 (number_diff_steps*number_of_K*number_of_prob_iterv) processes (number_of_prob_points probabilities per process)
        
        range_arr = zeros(Float64, (4, size))
        
        for i_st in 1:number_diff_steps
            for k in 1:number_of_K
                for i_int in 0:(number_of_prob_iterv-1)
                    diff = (in_intervals[k][3] - in_intervals[k][2])/number_of_prob_iterv
                    range_arr[1, u] = in_intervals[k][1]
                    range_arr[2, u] = in_intervals[k][2] + i_int * diff
                    range_arr[3, u] = in_intervals[k][2] + (i_int + 1) * diff
                    range_arr[4, u] = steps_values[i_st]
                    u = u + 1
                end
            end
        end
    else
        range_arr = zeros(Float64, 4)
    end

    values = zeros(Float64, 4)

    values = MPI.Scatter(range_arr, 4, 0, comm)

    K = values[1]
    p_in = values[2]
    p_f = values[3]
    number_of_steps = values[4]

    @printf("rank = %d | K = %.1f | p_in = %.4f | p_f = %.4f | number_of_steps = %d\n", rank, K, p_in, p_f, number_of_steps)

    data = zeros(Float64, 4 , number_of_prob_points*number_of_trajectories)

    for i_prob in 1:number_of_prob_points
        p  = p_in + (p_f - p_in) * (i_prob - 1) / number_of_prob_points
        for i_traj in 1:number_of_trajectories
            theta = 2 * pi * rand()
            I = 2 * pi * rand()
            for i in 1:number_of_steps
                theta, I = random_map(theta, I, K, a, theta_fp, I_fp, p)
            end
            data[1, number_of_trajectories*(i_prob-1) + i_traj] = K
            data[2, number_of_trajectories*(i_prob-1) + i_traj] = p
            data[3, number_of_trajectories*(i_prob-1) + i_traj] = number_of_steps
            data[4, number_of_trajectories*(i_prob-1) + i_traj] = O(theta, I, theta_fp, I_fp)
        end
    end

    data_recv = MPI.Gather(data, 0, comm)

    if rank == 0
        data_f = reshape(data_recv, (4, size*number_of_prob_points*number_of_trajectories))  
        data_f = transpose(data_f)

        pickle = pyimport("pickle")
        pd = pyimport("pandas")
        np = pyimport("numpy")

        py"""
        import pandas as pd, numpy as np

        df = pd.DataFrame($data_f, columns = ['K', 'p', 'L', "observations"])
        df = df.groupby(['K', 'p', 'L'])["observations"].apply(lambda x: np.array(x)).reset_index(name="observations")
        df = df.sort_values(['L', 'p'])
        """

        df = py"df" 

        pyopen  = pybuiltin("open")
        filename = "Data_for_scaling_K_$(K)_lo_cl_2.pkl"
        io = pyopen(filename, "wb")
        pickle.dump(df, io, pickle.HIGHEST_PROTOCOL)
        io.close()
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
