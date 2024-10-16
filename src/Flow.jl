## Flow.jl

import WaterLily: quick,inside_u,∂,ϕ,δ,CI,flux_out

@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])

function WaterLily.conv_diff!(r,u,Φ;ν=0.1,perdir=())
    r .= 0.
    N,n = WaterLily.size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # convection diffusion on inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,0)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,0)
    end
end

function WaterLily.CFL(a::Flow{D,T,S};Δt_max=10) where {D,T,S<:MPIArray{T}}
    @inside a.σ[I] = flux_out(I,a.u)
    MPI.Allreduce(min(Δt_max,inv(maximum(a.σ)+5a.ν)),Base.min,mpi_grid().comm)
end