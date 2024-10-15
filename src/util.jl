using KernelAbstractions: get_backend, @index, @kernel

struct MPIArray{T,N,V<:AbstractArray{T,N},W<:AbstractVector{T}} <: AbstractArray{T,N}
    A :: V
    send :: Vector{W}
    recv :: Vector{W}
    function MPIArray(::Type{T}, dims::NTuple{N, Integer}) where {T,N}
        A = Array{T,N}(undef, dims); M,n = last(dims)==N-1 ? (N-1,dims[1:end-1]) : (1,dims)
        send = Array{T}(undef,M*2max(prod(n).÷n...))
        recv = Array{T}(undef,M*2max(prod(n).÷n...))
        new{T,N,typeof(A),typeof(send)}(A,[send,copy(send)],[recv,copy(recv)])
    end
    MPIArray(A::AbstractArray{T}) where T = (B=MPIArray(T,size(A)); B.A.=A; B)
end
export MPIArray
for fname in (:size, :length, :ndims, :eltype) # how to write 4 lines of code in 5...
    @eval begin
        Base.$fname(A::MPIArray) = Base.$fname(A.A)
    end
end
Base.getindex(A::MPIArray, i...)       = Base.getindex(A.A, i...)
Base.setindex!(A::MPIArray, v, i...)   = Base.setindex!(A.A, v, i...)
Base.copy(A::MPIArray)                 = MPIArray(A)
Base.similar(A::MPIArray)              = MPIArray(eltype(A),size(A))
Base.similar(A::MPIArray, dims::Tuple) = MPIArray(eltype(A),dims)
# required for the @loop function
using KernelAbstractions
KernelAbstractions.get_backend(A::MPIArray) = KernelAbstractions.get_backend(A.A)

import WaterLily: inside, inside_u

"""
    inside(a)

Return CartesianIndices range excluding the double layer of cells on all boundaries.
"""
@inline inside(a::MPIArray;buff=2) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (4:dims[i]-2) : (3:dims[i]-1), N))
end

@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(3:i-2),dims)...,1:N))

@inline inside_u(u::MPIArray) = CartesianIndices(map(i->(3:i-2),Base.front(size(u))))

splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

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
@inline WaterLily.loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(global_loc() .+ I.I .- 2.5 .- 0.5 .* δ(i,I).I)
@inline WaterLily.loc(Ii::CartesianIndex,T=Float32) = WaterLily.loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
