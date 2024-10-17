## util.jl

import WaterLily: splitn,size_u,slice,inside_u,loc

"""
    halos(dims,d)

Return the CartesianIndices of the halos in dimension `±d` of an array of size `dims`.
"""
function halos(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (1:2) : (dims[i]-1:dims[i]) : (1:dims[i]), N))
end
"""
    buff(dims,d)

Return the CartesianIndices of the buffer in dimension `±d` of an array of size `dims`.
"""
function buff(dims::NTuple{N},j) where N
    CartesianIndices(ntuple( i-> i==abs(j) ? j<0 ? (3:4) : (dims[i]-3:dims[i]-2) : (1:dims[i]), N))
end

"""
    inside(a)

Return CartesianIndices range excluding the double layer of cells on all boundaries.
"""
@inline WaterLily.inside(a::MPIArray;buff=2) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function WaterLily.inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (4:dims[i]-2) : (3:dims[i]-1), N))
end
@inline WaterLily.inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(3:i-2),dims)...,1:N))
@inline WaterLily.inside_u(u::MPIArray) = CartesianIndices(map(i->(3:i-2),Base.front(size(u))))

using StaticArrays
"""
loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`.
"""
@inline WaterLily.loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(grid_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
@inline WaterLily.loc(Ii::CartesianIndex,T=Float32) = WaterLily.loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))

"""
    L₂/∞(a::MPIArray)

Return the L₂/∞ norm of a distributed memory array `a<:MPIArray`.
"""
WaterLily.L∞(a::MPIArray) = MPI.Allreduce(maximum(abs.(a)),Base.max,mpi_grid().comm)
WaterLily.L₂(a::MPIArray{T}) where T = MPI.Allreduce(sum(T,abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)

# hepler function for vtk writer
function get_extents(a::MPIArray)
    xs = Tuple(ifelse(x==0,1,x+1):ifelse(x==0,n+4,n+x+4) for (n,x) in zip(size(inside(a)),mpi_grid().global_loc))
    MPI.Allgather(xs, mpi_grid().comm)
end

using EllipsisNotation
"""
    BC!(a::MPIArray,A)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`. A zero Neumann condition
is applied to the tangential components.
"""
function WaterLily.BC!(a::MPIArray,A,saveexit=false,perdir=())
    N,n = size_u(a)
    for d ∈ 1:n # transfer full halos in each direction
        # get data to transfer @TODO use @views
        send1 = a[buff(N,-d),:]; send2 = a[buff(N,+d),:]
        recv1 = zero(send1);     recv2 = zero(send2)
        # swap 
        mpi_swap!(send1,recv1,send2,recv2,neighbors(d),mpi_grid().comm)

        # mpi boundary swap
        !mpi_wall(d,1) && (a[halos(N,-d),:] .= recv1) # halo swap
        !mpi_wall(d,2) && (a[halos(N,+d),:] .= recv2) # halo swap
        
        for i ∈ 1:n # this sets the BCs on the physical boundary
            if mpi_wall(d,1) # left wall
                if i==d # set flux
                    a[halos(N,-d),i] .= A[i]
                    a[slice(N,3,d),i] .= A[i]
                else # zero gradient
                    a[halos(N,-d),i] .= reverse(send1[..,i]; dims=d)
                end
            end
            if mpi_wall(d,2) # right wall
                if i==d && (!saveexit || i>1) # convection exit
                    a[halos(N,+d),i] .= A[i]
                else # zero gradient
                    a[halos(N,+d),i] .= reverse(send2[..,i]; dims=d)
                end
            end
        end
    end
end

"""
    exitBC!(u,u⁰,U,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
Because we have a distributed memory, we need to correct the flux imbalance on the MPI
grid as well.
"""
function WaterLily.exitBC!(u::MPIArray{T},u⁰,U,Δt) where T
    N,_ = size_u(u)
    exitR = slice(N.-2,N[1]-1,1,3) # exit slice excluding ghosts
    ∮udA = zero(T)
    if mpi_wall(1,2) # right wall
        @loop u[I,1] = u⁰[I,1]-U[1]*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
        @loop u[I+δ(1,I),1] = u[I,1] over I ∈ exitR
        ∮udA = sum(@view(u[exitR,1]))/length(exitR)-U[1]   # mass flux imbalance
    end
    ∮u = MPI.Allreduce(∮udA,+,mpi_grid().comm)           # domain imbalance
    N = prod(mpi_dims()[2:end]) # for now we only have 1D exit
    mpi_wall(1,2) && (@loop u[I,1] -= ∮u/N over I ∈ exitR;
                      @loop u[I+δ(1,I),1] -= ∮u/N over I ∈ exitR) # correct flux only on right wall
end


"""
    perBC!(a::MPIArray,perdir)
Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
WaterLily.perBC!(a::MPIArray,::Tuple{})            = _perBC!(a, size(a), true)
WaterLily.perBC!(a::MPIArray, perdir, N = size(a)) = _perBC!(a, N, true)
_perBC!(a, N, mpi::Bool) = for d ∈ eachindex(N)
    # fill with data to transfer
    fill_send!(a,d,Val(:Scalar))

    # swap
    mpi_swap!(a,neighbors(d),mpi_grid().comm)

    # this sets the BCs
    !mpi_wall(d,1) && copyto!(a,-d,Val(:Scalar)) # halo swap
    !mpi_wall(d,2) && copyto!(a,+d,Val(:Scalar)) # halo swap
end

"""
    interp(x::SVector, arr::AbstractArray)

    Linear interpolation from array `arr` at index-coordinate `x`.
    Note: This routine works for any number of dimensions.
"""
# function interp(x::SVector{D}, arr::MPIArray{T,D}) where {D,T}
#     # Index below the interpolation coordinate and the difference
#     i = floor.(Int,x); y = x.-i

#     # CartesianIndices around x
#     I = CartesianIndex(i...); R = I:I+oneunit(I)

#     # Linearly weighted sum over arr[R] (in serial)
#     s = zero(T)
#     @fastmath @inbounds @simd for J in R
#         weight = prod(@. ifelse(J.I==I.I,1-y,y))
#         s += arr[J]*weight
#     end
#     return s
# end
# function interp(x::SVector{D}, varr::MPIArray) where {D}
#     # Shift to align with each staggered grid component and interpolate
#     @inline shift(i) = SVector{D}(ifelse(i==j,0.5,0.0) for j in 1:D)
#     return SVector{D}(interp(x+shift(i),@view(varr[..,i])) for i in 1:D)
# end