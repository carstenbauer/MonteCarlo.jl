import MPI

# Modified from the examples in the MPI.jl docs
# https://juliaparallel.github.io/MPI.jl/latest/examples/05-job_schedule/
"""
    mpi_queue(function, data::Vector; kwargs...)

Distributes `data` to all available mpi workers (i.e. not rank = 0). Each worker 
calls `function` with the received data and sends the result back to master 
(rank = 0). When all work is done, master (rank = 0) return the collected results.

This is somewhat similar to `pmap`.

You can set the `result_type = Nothing` as a keyword argument. You can also set 
an `end_time = now() + Year(100)` this way.
"""
function mpi_queue(
        f::Function, data::Vector{T};
        end_time = now() + Year(100), result_type = Nothing
    ) where T

    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    N = length(data)
    new_data = Array{result_type}(undef, N)

    if rank == root # I am root

        idx_recv = 0
        idx_sent = 1

        # Array of workers requests
        sreqs_workers = Array{MPI.Request}(undef,nworkers)
        # -1 = start, 0 = channel not available, 1 = channel available
        status_workers = ones(nworkers).*-1
        
        # Distribute initial set of tasks to workers
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
            sreq = MPI.Isend(idx_sent, dst, dst+32, comm)
            idx_sent += 1
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            print("Root: Sent index $idx_sent/$N to worker $dst/$world_size\n")
        end

        # recieve finalization messages from worker and send new work
        while idx_recv != N
            now() > end_time && return

            # Check to see if there is an available message to receive
            for dst in 1:nworkers
                if status_workers[dst] == 0
                    (flag, status) = MPI.Test!(sreqs_workers[dst])
                    if flag
                        status_workers[dst] = 1
                    end
                end
            end

            for dst in 1:nworkers
                if status_workers[dst] == 1
                    ismessage, status = MPI.Iprobe(dst,dst+32, comm)
                    if ismessage
                        # Receives message
                        result = MPI.Recv(result_type, dst, dst+32, comm)
                        idx_recv += 1
                        new_data[idx_recv] = result
                        print("Root: Received result $idx_recv/$N from worker $dst/$world_size\n")
                        if idx_sent <= N
                            # Sends new message
                            sreq = MPI.Isend(idx_sent, dst, dst+32, comm)
                            idx_sent += 1
                            sreqs_workers[dst] = sreq
                            status_workers[dst] = 1
                            print("Root: Sent index $idx_sent/$N to worker $dst/$world_size\n")
                        end
                    end
                end
            end
        end
        
        # Send termination messages to workers
        for dst in 1:nworkers
            sreq = MPI.Isend(-1, dst, dst+32, comm)
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            print("Root: Finish worker $dst\n")
        end
        
        MPI.Waitall!(sreqs_workers)

    else # If rank == worker

        # -1 = start, 0 = channel not available, 1 = channel available
        status_worker = -1
        while true
            now() > end_time && return

            sreqs_workers = Array{MPI.Request}(undef,1)
            ismessage, status = MPI.Iprobe(root, rank+32, comm)
            
            if ismessage
                # Receives message
                idx = MPI.Recv(Int64, root, rank+32, comm)
                # Termination message from root
                print("Worker $rank: Received index $idx from root\n")
                if idx == -1
                    print("Worker $rank: Finish\n")
                    break
                end
                # Apply function (add number 100) to array
                send_mesg = f(data[idx])
                @assert typeof(send_msg) == result_type
                sreq = MPI.Isend(send_mesg, root, rank+32, comm)
                sreqs_workers[1] = sreq
                status_worker = 0
            end

            # Check to see if there is an available message to receive
            if status_worker == 0
                (flag, status) = MPI.Test!(sreqs_workers[1])
                if flag
                    status_worker = 1
                end
            end
        end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return new_data
end