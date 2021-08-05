using MPI

testdir = @__DIR__

mpiexec() do cmd
    run(`$cmd -n 3 $(Base.julia_cmd()) $(joinpath(testdir, "mpi_queue.jl"))`)
end

for file in readdir()
    if endswith(file, "log")
        rm(file)
    end
end

mpiexec() do cmd
    run(`$cmd -n 4 $(Base.julia_cmd()) $(joinpath(testdir, "mpi_updates.jl"))`)
end