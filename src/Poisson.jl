## Poisson.jl

import WaterLily: perBC!,increment!,mult

"""
    L₂/∞(a::MPIArray)

Return the L₂/∞ norm of a the residuals of distributed memory `Poisson{T,a}` object where `a<:MPIArray{T}`.
"""
function WaterLily.L₂(p::Poisson{T,S}) where {T,S<:MPIArray{T}} # should work on the GPU
    MPI.Allreduce(sum(T,@inbounds(p.r[I]*p.r[I]) for I ∈ inside(p.r)),+,mpi_grid().comm)
end
WaterLily.L∞(p::Poisson{T,S}) where {T,S<:MPIArray{T}} = MPI.Allreduce(maximum(abs.(p.r)),Base.max,mpi_grid().comm)

"""
    residual!(p::::Poisson{MPIArray{T}})

Distributed memory resiual `r = z-Ax` and corrects it such that
`r = 0` if `iD==0` which ensures local satisfiability
    and 
`sum(r) = 0` which ensures global satisfiability.

The global correction is done by adjusting all points uniformly, 
minimizing the local effect. This requires a global communication.
"""
function WaterLily.residual!(p::Poisson{T,S}) where {T,S<:MPIArray{T}}
    perBC!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-mult(I,p.L,p.D,p.x))
    # s = sum(p.r)/length(inside(p.r))
    s = MPI.Allreduce(sum(p.r)/length(inside(p.r)),+,mpi_grid().comm)
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother run `it` times. 
Note: This runs for general backends, but is _very_ slow to converge.
"""
@fastmath WaterLily.Jacobi!(p::Poisson{T,S};it=1) where {T,S<:MPIArray{T}} = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    # @TODO is that reqired?
    perBC!(p.ϵ,p.perdir)
    increment!(p)
end

using LinearAlgebra: dot
"""
    ⋅(a,b)

Dot product of two arrays, `a` and `b`. If `a` and `b` are distributed memory arrays,
the dot product is computed globally and a Allreduce is performed.
"""
_dot(a,b) = dot(a,b)
⋅(a,b) = _dot(a,b)
function _dot(a::MPIArray{T},b::MPIArray{T}) where T
    MPI.Allreduce(sum(T,@inbounds(a[I]*b[I]) for I ∈ inside(a)),+,mpi_grid().comm)
end

# need to redefine pcg because of the dot function above
function WaterLily.pcg!(p::Poisson{T,S};it=6) where {T,S<:MPIArray{T}}
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    abs(rho)<10eps(T) && return
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(z⋅ϵ)
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        (i==it || abs(alpha)<1e-2) && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        abs(rho2)<10eps(T) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end