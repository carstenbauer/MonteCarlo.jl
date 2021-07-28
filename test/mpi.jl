# This is meant to be run with something like
# mpiexec -np 4 julia mpi.jl

using Tests, MonteCarlo
import MPI

@testset "mpi_queue" begin
    function workload(x)
        sleep(x)
        nothing
    end

    MPI.Init()
    N_workers = MPI.Comm_size() - 1

    # Test if the scheduler generates this grouping:
    # first group:   [3, 5, ..., 5]
    # second group:  [8, 3, ..., 3]
    # third group:   [_, 3, ..., 3]
    # where each number corresponds to the sleep time of the relevant processes
    # More specifically checks if the "8" gets pairs with the initial "3" by checking
    # the total runtime

    inputs = fill(3, 3N_workers - 1)
    inputs[N_workers + 1] = 8
    inputs[2:N_wprkers] .= 5

    t0 = time()
    mpi_queue(workload, inputs)
    dt = time() - t0
    @info dt
    @test 10.9 < dt < 12
end