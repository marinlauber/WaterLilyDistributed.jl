module WaterLilyDistributed

using WaterLily
using WaterLily: @loop,CI,slice
import WaterLily: sim_step!,sim_time

"""
    MPIArray{T,N,V<:AbstractArray{T,N},W<:AbstractVector{T}}

A distributed array that contains the array `A` and send and receive buffers `send` and `recv`.
"""
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
export MPIArray

include("util.jl")
export get_extents

include("Poisson.jl")

include("MultiLevelPoisson.jl")

include("Flow.jl")

include("WaterLilyMPI.jl")
export init_mpi,me,global_loc,mpi_grid,mpi_dims,finalize_mpi,master

function WaterLily.Simulation(dims::NTuple{N}, u_BC, L::Number;
                    Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                    uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                    T=Float32, mem=Array) where N
    @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
    @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
    isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,zero(T)),N)).==T) "`u_BC` is not type stable"
    uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
    U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
    r = init_mpi(dims) # initialize the MPI grid
    dims = dims.+2 # make the halos 2 cells wide
    mpi_mem = MPIArray # at some point there should be a MPICuArray here
    flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mpi_mem,perdir,exitBC)
    measure!(flow,body;ϵ)
    WaterLily.Simulation(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
end

function WaterLily.sim_step!(sim::Simulation{D,T,S},t_end;remeasure=true,
                   max_steps=typemax(Int),verbose=false) where {D,T,S<:MPIArray{T}}
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        (verbose && me()==0) && println("tU/L=",round(sim_time(sim),digits=4),
                                        ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

using WriteVTK
using Printf: @sprintf

# import utils fuctions
components_first = Base.get_extension(WaterLily, :WaterLilyWriteVTKExt).components_first
VTKWriter = Base.get_extension(WaterLily, :WaterLilyWriteVTKExt).VTKWriter
pvd_collection = Base.get_extension(WaterLily, :WaterLilyWriteVTKExt).pvd_collection

function WaterLily.vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32)
    (master() && !isdir(dir)) && mkdir(dir)
    VTKWriter(fname,dir,pvd_collection(fname),attrib,[0])
end
function WaterLily.write!(w::VTKWriter,a::Simulation{D,T,S};N=size(inside(a.flow.p))) where {D,T,S<:MPIArray{T}}
    k,part = w.count[1], Int(me()+1)
    extents = get_extents(a.flow.p)
    pvtk = pvtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), extents[part];
                     part=part, extents=extents, ghost_level=2)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        pvtk[name] = size(func(a))==size(a.flow.p) ? func(a) : components_first(func(a))
    end
    vtk_save(pvtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=pvtk
end

end # module
