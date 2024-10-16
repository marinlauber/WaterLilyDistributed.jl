## MultiLevelPoisson.jl

import WaterLily: restrictL!,restrictL,residual!,Vcycle!,smooth!,size_u,up

# overwrite to use the double ghosts
@inline WaterLily.up(I::CartesianIndex,a=0) = (2I-3oneunit(I)):(2I-2oneunit(I)-δ(a,I))
@inline WaterLily.down(I::CartesianIndex) = CI((I+3oneunit(I)).I .÷2)

function WaterLily.restrictML(b::Poisson{T,S}) where {T,S<:MPIArray{T}}
    N,n = size_u(b.L)
    Na = map(i->2+i÷2,N) # this line changes
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end

function WaterLily.restrictL!(a::MPIArray{T},b;perdir=()) where T
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->3:n-2,Na)) # this line changes
    end
    BC!(a,zeros(SVector{n,T}),false,perdir)  # correct μ₀ @ boundaries
end

@inline WaterLily.divisible(N) = mod(N-2,2)==0 && N>6

# this is just because I want a better solver
function WaterLily.solver!(ml::MultiLevelPoisson;tol=1e-5,itmx=32)
    p = ml.levels[1]
    residual!(p); r₂ = L₂(p)
    nᵖ=0
    while nᵖ<itmx
        Vcycle!(ml)
        smooth!(p); r₂ = L₂(p)
        nᵖ+=1; r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
end
