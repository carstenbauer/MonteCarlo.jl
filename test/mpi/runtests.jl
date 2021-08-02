using MPI

testdir = @__DIR__

mpiexec() do cmd
    run(`$cmd -n 3 $(Base.julia_cmd()) $(joinpath(testdir, "mpi_queue.jl"))`)
end
