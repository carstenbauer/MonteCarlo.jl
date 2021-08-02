# This is meant to be run with something like
# mpiexec -np 4 julia mpi.jl

using Test, MonteCarlo
import MPI

let
    function workload(x)
        sleep(x)
        return (MPI.Comm_rank(MPI.COMM_WORLD), x)
    end

    MPI.Init()
    N_workers = MPI.Comm_size(MPI.COMM_WORLD) - 1

    # Test if the scheduler generates this grouping:
    # first group:   [4, 7, ..., 7]
    # second group:  [11, 4, ..., 4]
    # third group:   [_, 4, ..., 4]
    # where each number corresponds to the sleep time of the relevant processes
    # More specifically checks if the "8" gets pairs with the initial "3" by checking
    # the total runtime

    inputs = fill(4, 3N_workers - 1)
    inputs[N_workers + 1] = 11
    inputs[2:N_workers] .= 7

    output = mpi_queue(workload, inputs, verbose = true)
    
    MPI.Barrier(MPI.COMM_WORLD)
    
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        results = zeros(Int64, N_workers)
        for (id, val) in output
            results[id] += val
        end
        println("Time slept: $results")
        @test all(results .== 15)
    end
end