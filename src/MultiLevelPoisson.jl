##

@inline WaterLily.up(I::CartesianIndex,a=0) = (2I-3oneunit(I)):(2I-2oneunit(I)-δ(a,I))
@inline WaterLily.down(I::CartesianIndex) = CI((I+3oneunit(I)).I .÷2)

@fastmath @inline function WaterLily.restrictL(I::CartesianIndex,i,b)
    s = zero(eltype(b))
    for J ∈ WaterLily.up(I,i)
     s += @inbounds(b[J,i])
    end
    return 0.5s
end

function WaterLily.restrictML(b::Poisson)
    N,n = WaterLily.size_u(b.L)
    Na = map(i->2+i÷2,N)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    WaterLily.restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a::AbstractArray{T},b;perdir=()) where T
    Na,n = WaterLily.size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = WaterLily.restrictL(I,i,b) over I ∈ CartesianIndices(map(n->3:n-2,Na))
    end
    BC!(a,zeros(SVector{n,T}),false,perdir)  # correct μ₀ @ boundaries
end

@inline WaterLily.divisible(N) = mod(N-2,2)==0 && N>6

function update!(ml::MultiLevelPoisson)
    WaterLily.update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        WaterLily.restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        WaterLily.update!(ml.levels[l])
    end
end

function WaterLily.solver!(ml::MultiLevelPoisson;tol=1e-5,itmx=32)
    p = ml.levels[1]
    WaterLily.residual!(p); r₂ = L₂(p)
    nᵖ=0
    while nᵖ<itmx
        WaterLily.Vcycle!(ml)
        WaterLily.smooth!(p); r₂ = L₂(p)
        nᵖ+=1; r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
end
