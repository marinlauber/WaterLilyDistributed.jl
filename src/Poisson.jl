
@fastmath WaterLily.Jacobi!(p;it=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    # @TODO is that reqired?
    perBC!(p.ϵ,p.perdir)
    WaterLily.increment!(p)
end

# needed for MPI later
using LinearAlgebra: dot
_dot(a,b) = dot(a,b)
⋅(a,b) = _dot(a,b)

function WaterLily.pcg!(p::Poisson{T};it=6) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    abs(rho)<10eps(T) && return
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = WaterLily.mult(I,p.L,p.D,ϵ)
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

function WaterLily.solver!(p::Poisson;tol=1e-4,itmx=1e4)
    WaterLily.residual!(p); r₂ = L₂(p)
    nᵖ=0
    while nᵖ<itmx
        WaterLily.smooth!(p); r₂ = L₂(p)
        nᵖ+=1; r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(p.n,nᵖ)
end