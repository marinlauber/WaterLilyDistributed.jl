module WaterLilyDistributed

using WaterLily
import WaterLily: @loop,CI,slice,measure!,sim_time,time,sim_step!,update!

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
export init_mpi,me,global_loc,mpi_grid,mpi_dims,finalize_mpi

"""
    DistributedSimulation{D,T,S}

A distributed simulation object that contains the flow, body, and Poisson solver.
"""
mutable struct DistributedSimulation{D,T,S} #<: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow{D,T,S}
    body :: AbstractBody
    pois :: AbstractPoisson
    function DistributedSimulation(dims::NTuple{N}, u_BC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array, psolver=MultiLevelPoisson) where N
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,zero(T)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
        # make the halos 2 cells wide
        dims = dims.+2
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new{N,T,typeof(flow.p)}(U,L,ϵ,flow,body,psolver(flow.p,flow.μ₀,flow.σ;perdir))
    end
end


time(sim::DistributedSimulation) = time(sim.flow)
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::DistributedSimulation) = time(sim)*sim.U/sim.L

function sim_step!(sim::DistributedSimulation,t_end;remeasure=true,
                    max_steps=typemax(Int),verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        (verbose && me()==0) && println("tU/L=",round(sim_time(sim),digits=4),
                                        ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function sim_step!(sim::DistributedSimulation;remeasure=true)
    remeasure && measure!(sim)
    mom_step!(sim.flow,sim.pois)
end

function measure!(sim::DistributedSimulation,t=sum(sim.flow.Δt))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end

export DistributedSimulation

# default WriteVTK functions
function vtkWriter_d end
function write_d! end
# export
export vtkWriter_d,write_d!

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192" include("../ext/WaterLilyDistributedWriteVTKExt.jl")
    end
    WaterLily.check_nthreads(Val{Threads.nthreads()}())
end

end # module
