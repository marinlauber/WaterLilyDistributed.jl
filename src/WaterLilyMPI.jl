using MPI
using StaticArrays
using WaterLilyDistributed

const NDIMS_MPI = 3           # Internally, we set the number of dimensions always to 3 for calls to MPI. This ensures a fixed size for MPI coords, neigbors, etc and in general a simple, easy to read code.
const NNEIGHBORS_PER_DIM = 2

#@TODO these can be cleaned up
function send_flat(I,N,n,d)
    J = LinearIndices(buff(N,d)); Is=buff(N,d)[1]
    J[I-Is+oneunit(I)] + n*length(J)
end
function fill_send!(a::MPIArray,d,::Val{:Scalar}) # fill scalar field
    N=size(a)
    @loop a.send[1][send_flat(I,N,0,-d)] = a[I] over I ∈ buff(N,-d)
    @loop a.send[2][send_flat(I,N,0,+d)] = a[I] over I ∈ buff(N,+d)
    # @loop a.send[1][I] = a[buff(N,-d)[I]] over I ∈ CartesianIndices(a.send[1])
    # @loop a.send[2][I] = a[buff(N,+d)[I]] over I ∈ CartesianIndices(a.send[2])
end
function fill_send!(a::MPIArray,d,::Val{:Vector})# fill vector field
    N,n = WaterLily.size_u(a)
    for i ∈ 1:n # copy every component in that direction
        @loop a.send[1][send_flat(I,N,i-1,-d)] = a[I,i] over I ∈ buff(N,-d)
        @loop a.send[2][send_flat(I,N,i-1,+d)] = a[I,i] over I ∈ buff(N,+d)
    end
end
function recv_flat(I,N,n,d)
    J = LinearIndices(halos(N,d)); Is=halos(N,d)[1]
    J[I-Is+oneunit(I)] + n*length(J)
end
function copyto!(a::MPIArray,d,::Val{:Scalar}) # copy scalar field back from rcv buffer
    N=size(a); i = d<0 ? 1 : 2
    # @loop a[halos(N,d)[I]] = a.recv[i][I]  over I ∈ CartesianIndices(a.recv[i])
    @loop a[I] = a.recv[i][recv_flat(I,N,0,d)] over I ∈ halos(N,d)
end
function copyto!(a::MPIArray,d,::Val{:Vector}) # copy scalar field back from rcv buffer
    N,n = WaterLily.size_u(a); i = d<0 ? 1 : 2
    for j ∈ 1:n # copy every component in that direction
        @loop a[I,j] = a.recv[i][recv_flat(I,N,j-1,d)] over I ∈ halos(N,d)
    end
end


"""
    mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)

This function swaps the data between two MPI processes. The data is sent from `send1` to `neighbor[1]` and received in `recv1`.
The data is sent from `send2` to `neighbor[2]` and received in `recv2`. The function is non-blocking and returns when all data 
has been sent and received. 
"""
function mpi_swap!(send1,recv1,send2,recv2,neighbor,comm)
    reqs=MPI.Request[]
    # Send to / receive from neighbor 1 in dimension d
    push!(reqs,MPI.Isend(send1,  neighbor[1], 0, comm))
    push!(reqs,MPI.Irecv!(recv1, neighbor[1], 1, comm))
    # Send to / receive from neighbor 2 in dimension d
    push!(reqs,MPI.Irecv!(recv2, neighbor[2], 0, comm))
    push!(reqs,MPI.Isend(send2,  neighbor[2], 1, comm))
    # wair for all transfer to be done
    MPI.Waitall!(reqs)
end
function mpi_swap!(a::MPIArray,neighbor,comm)
    # prepare the transfer
    reqs=MPI.Request[]
    # Send to / receive from neighbor 1 in dimension d
    push!(reqs,MPI.Isend(a.send[1],  neighbor[1], 0, comm))
    push!(reqs,MPI.Irecv!(a.recv[1], neighbor[1], 1, comm))
    # Send to / receive from neighbor 2 in dimension d
    push!(reqs,MPI.Irecv!(a.recv[2], neighbor[2], 0, comm))
    push!(reqs,MPI.Isend(a.send[2],  neighbor[2], 1, comm))
    # wair for all transfer to be done
    MPI.Waitall!(reqs)
end

struct MPIGrid #{I,C<:MPI.Comm,N<:AbstractVector,M<:AbstractArray,G<:AbstractVector}
    me::Int                    # rank
    comm::MPI.Comm             # communicator
    coords::AbstractVector     # coordinates
    neighbors::AbstractArray   # neighbors
    global_loc::AbstractVector # the location of the lower left corner in global index space
end
const MPI_GRID_NULL = MPIGrid(-1,MPI.COMM_NULL,[-1,-1,-1],[-1 -1 -1; -1 -1 -1],[0,0,0])

let
    global MPIGrid, set_mpi_grid, mpi_grid, mpi_initialized, check_mpi

    # allows to access the global mpi grid
    _mpi_grid::MPIGrid          = MPI_GRID_NULL
    mpi_grid()::MPIGrid         = (check_mpi(); _mpi_grid::MPIGrid)
    set_mpi_grid(grid::MPIGrid) = (_mpi_grid = grid;)
    mpi_initialized()           = (_mpi_grid.comm != MPI.COMM_NULL)
    check_mpi()                 = !mpi_initialized() && error("MPI not initialized")
end

function init_mpi(Dims::NTuple{D};dims=[0, 0, 0],periods=[0, 0, 0],comm::MPI.Comm=MPI.COMM_WORLD,
                  disp::Integer=1,reorder::Bool=true) where D
    # MPI
    MPI.Init()
    nprocs = MPI.Comm_size(comm)
    # create cartesian communicator
    MPI.Dims_create!(nprocs, dims)
    comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
    me     = MPI.Comm_rank(comm_cart)
    coords = MPI.Cart_coords(comm_cart)
    # make the cart comm
    neighbors = fill(MPI.PROC_NULL, NNEIGHBORS_PER_DIM, NDIMS_MPI);
    for i = 1:NDIMS_MPI
        neighbors[:,i] .= MPI.Cart_shift(comm_cart, i-1, disp);
    end
    # global index coordinate in grid space
    global_loc = SVector([coords[i]*Dims[i] for i in 1:D]...)
    set_mpi_grid(MPIGrid(me,comm_cart,coords,neighbors,global_loc))
    return me; # this is the most usefull MPI vriable to have in the local space
end
finalize_mpi() = MPI.Finalize()

# helper functions
me() = mpi_grid().me
master() = me()==0
grid_loc() = mpi_grid().global_loc
neighbors(dim) = mpi_grid().neighbors[:,dim]
mpi_wall(dim,i) = mpi_grid().neighbors[i,dim]==MPI.PROC_NULL
mpi_dims() = MPI.Cart_get(mpi_grid().comm)[1]
