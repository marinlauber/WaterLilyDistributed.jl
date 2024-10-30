using KernelAbstractions: get_backend, @index, @kernel

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

@inline CI(a...) = CartesianIndex(a...)
"""
    CIj(j,I,jj)
Replace jᵗʰ component of CartesianIndex with k
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a)

Return CartesianIndices range excluding the double layer of cells on all boundaries.
"""
@inline inside(a::AbstractArray;buff=2) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (4:dims[i]-2) : (3:dims[i]-1), N))
end
@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(3:i-2),dims)...,1:N))
@inline inside_u(u::AbstractArray) = CartesianIndices(map(i->(3:i-2),Base.front(size(u))))
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = sum(abs2,@inbounds(a[I]) for I ∈ inside(a))
L₂(a::MPIArray{T}) where T = MPI.Allreduce(sum(T,abs2,@inbounds(a[I]) for I ∈ inside(a)),+,mpi_grid().comm)
L∞(a::MPIArray) = MPI.Allreduce(maximum(abs.(a)),Base.max,mpi_grid().comm)

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example,

    @inside p[I] = sum(loc(0,I))

becomes

    @loop p[I] = sum(loc(0,I)) over I ∈ inside(p)

See [`@loop`](@ref).
"""
macro inside(ex)
    # Make sure it's a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLilyDistributed.@loop $ex over $I ∈ inside($a)
    end |> esc
end

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

For example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution, or

    @kernel function kern(a,i,@Const(I0))
        I ∈ @index(Global,Cartesian)+I0
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end
    kern(get_backend(a),64)(a,i,R[1]-oneunit(R[1]),ndrange=size(R))

when multi-threading on CPU or using CuArrays.
Note that `get_backend` is used on the _first_ variable in `expr` (`a` in this example).
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym(kern, kern_) # generate unique kernel function names for serial and KA execution
    return quote
        function $kern($(rep.(sym)...),::Val{1})
            @simd for $I ∈ $R
                @fastmath @inbounds $ex
            end
        end
        @kernel function $kern_($(rep.(sym)...),@Const(I0)) # replace composite arguments
            $I = @index(Global,Cartesian)
            $I += I0
            @fastmath @inbounds $ex
        end
        function $kern($(rep.(sym)...),_)
            $kern_(get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
        end
        $kern($(sym...),Val{Threads.nthreads()}()) # dispatch to SIMD for -t 1, or KA otherwise
    end |> esc
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])          # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

using StaticArrays
"""
loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`.
"""
grid_loc(arg) = 0 # no offset in serial
global_loc() = grid_loc(Val(:WaterLilyDistributed_MPIExt))
master(arg) = true # always on master in serial
master() = master(Val(:WaterLilyDistributed_MPIExt))
@inline loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(global_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
@inline loc(Ii::CartesianIndex,T=Float32) = loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    apply!(f, c)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c` or
a function `f(x)` to the center of a uniform array `c`.
"""
apply!(f,c) = hasmethod(f,Tuple{Int,CartesianIndex}) ? applyV!(f,c) : applyS!(f,c)
applyV!(f,c) = @loop c[Ii] = f(last(Ii),loc(Ii,eltype(c))) over Ii ∈ CartesianIndices(c)
applyS!(f,c) = @loop c[I] = f(loc(0,I,eltype(c))) over I ∈ CartesianIndices(c)
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end

"""
    BC!(a,A)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`. A zero Neumann condition
is applied to the tangential components.
"""
function BC!(a,A,saveexit=false,perdir=())
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        if j in perdir
            #@TODO make sure this is correct
            @loop a[I,i] = a[CIj(j,I,N[j]-3),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,N[j]-2),i] over I ∈ slice(N,2,j)
            @loop a[I,i] = a[CIj(j,I,3),i] over I ∈ slice(N,N[j]-1,j)
            @loop a[I,i] = a[CIj(j,I,4),i] over I ∈ slice(N,N[j],j)
        else
            if i==j # Normal direction, Dirichlet
                for s ∈ (1,2,3)
                    @loop a[I,i] = A[i] over I ∈ slice(N,s,j)
                end
                (!saveexit || i>1) && (@loop a[I,i] = A[i] over I ∈ slice(N,N[j]-1,j);
                                       @loop a[I,i] = A[i] over I ∈ slice(N,N[j],j)) # overwrite exit
            else    # Tangential directions, Neumann
                @loop a[I,i] = a[I+δ(j,I),i] over I ∈ slice(N,2,j)
                @loop a[I,i] = a[I+2δ(j,I),i] over I ∈ slice(N,1,j)
                @loop a[I,i] = a[I-δ(j,I),i] over I ∈ slice(N,N[j]-1,j)
                @loop a[I,i] = a[I-2δ(j,I),i] over I ∈ slice(N,N[j],j)
            end
        end
    end
end
function BC!(a::MPIArray,A,saveexit=false,perdir=())
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
                    a[WaterLilyDistributed.slice(N,3,d),i] .= A[i]
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
"""
function exitBC!(u,u⁰,U,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-2,N[1]-1,1,3)            # exit slice excluding ghosts
    @loop u[I,1] = u⁰[I,1]-U[1]*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U[1]   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ exitR         # correct flux
end
function exitBC!(u::MPIArray{T},u⁰,U,Δt) where T
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
    perBC!(a,perdir)
Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-3)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,N[j]-2)] over I ∈ slice(N,2,j)
    @loop a[I] = a[CIj(j,I,3)] over I ∈ slice(N,N[j]-1,j)
    @loop a[I] = a[CIj(j,I,4)] over I ∈ slice(N,N[j],j)
end
perBC!(a::MPIArray,::Tuple{})            = _perBC!(a, size(a), true)
perBC!(a::MPIArray, perdir, N = size(a)) = _perBC!(a, N, true)
_perBC!(a, N, mpi::Bool) = for d ∈ eachindex(N)
    # fill with data to transfer
    fill_send!(a,d,Val(:Scalar))

    # swap
    mpi_swap!(a,neighbors(d),mpi_grid().comm)

    # this sets the BCs
    !mpi_wall(d,1) && copyto!(a,-d,Val(:Scalar)) # halo swap
    !mpi_wall(d,2) && copyto!(a,+d,Val(:Scalar)) # halo swap
end

# hepler function for vtk writer
function get_extents(a::MPIArray)
    xs = Tuple(ifelse(x==0,1,x+1):ifelse(x==0,n+4,n+x+4) for (n,x) in zip(size(inside(a)),mpi_grid().global_loc))
    MPI.Allgather(xs, mpi_grid().comm)
end

# """
#     interp(x::SVector, arr::AbstractArray)

#     Linear interpolation from array `arr` at index-coordinate `x`.
#     Note: This routine works for any number of dimensions.
# """
# function interp(x::SVector{D}, arr::AbstractArray{T,D}) where {D,T}
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
# function interp(x::SVector{D}, varr::AbstractArray) where {D}
#     # Shift to align with each staggered grid component and interpolate
#     @inline shift(i) = SVector{D}(ifelse(i==j,0.5,0.0) for j in 1:D)
#     return SVector{D}(interp(x+shift(i),@view(varr[..,i])) for i in 1:D)
# end