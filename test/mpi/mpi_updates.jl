using MonteCarlo, Test
import MPI

#=

t1 = MPI.Initialized()
m = HubbardModelAttractive(2, 2)
scheduler = SimpleScheduler(LocalSweep(), MPIReplicaPull())
mc = DQMC(m, beta = 1.0, scheduler = scheduler)
t2 = MPI.Initialized()

io = open("worker$(MPI.Comm_rank(MPI.COMM_WORLD)).log", "w")
old_stdout = stdout
old_stderr = stderr
redirect_stdout(io)
redirect_stderr(io)

append!(MonteCarlo.connected_ids, filter(i -> i != MPI.Comm_rank(MPI.COMM_WORLD), 0:3))
run!(mc)

=#


MPI.Initialized() || MPI.Init()

io = open("worker$(MPI.Comm_rank(MPI.COMM_WORLD)).log", "w")
old_stdout = stdout
old_stderr = stderr
redirect_stdout(io)
redirect_stderr(io)

# maps 0 1 2 3 ...
#       X   X
# to   0 1 2 3 ...
partner1(rank = MPI.Comm_rank(MPI.COMM_WORLD)) = rank - isodd(rank) + iseven(rank)
# maps N 0 1 2 ...
#       X   X  
# to   N 0 1 2 ...
function partner2(rank = MPI.Comm_rank(MPI.COMM_WORLD), max = MPI.Comm_size(MPI.COMM_WORLD) - 1)
    mod1(rank + isodd(rank) - iseven(rank) + 1, max + 1) - 1
end

m = HubbardModelAttractive(2, 2)
scheduler = SimpleScheduler(
    LocalSweep(), MPIReplicaExchange(partner1()), MPIReplicaExchange(partner2())
)
println(MPI.Comm_rank(MPI.COMM_WORLD), " < - > ", partner1(), ", ", partner2())
show(scheduler)
mc = DQMC(m, beta = 1.0, scheduler = scheduler)

run!(mc)

println("Simulations finished, good.")
# @testset "MPI Updates" begin
#     # Check if init is working as expected
#     @test t1
#     @test t2
# end

redirect_stdout(old_stdout)
redirect_stderr(old_stderr)

MPI.Barrier(MPI.COMM_WORLD)

exit()