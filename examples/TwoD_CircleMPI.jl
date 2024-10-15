#mpiexecjl --project=. -n 2 julia TwoD_CircleMPI.jl

using WaterLily
using WaterLilyDistributed
using MPI
using WriteVTK
using StaticArrays

# make a writer with some attributes, now we need to apply the BCs when writting
velocity(a::DistributedSimulation) = a.flow.u |> Array;
pressure(a::DistributedSimulation) = a.flow.p |> Array;
_body(a::DistributedSimulation) = (measure_sdf!(a.flow.σ, a.body);
                                   a.flow.σ |> Array;)
vorticity(a::DistributedSimulation) = (@inside a.flow.σ[I] = 
                                       WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                                       WaterLily.perBC!(a.flow.σ,());
                                       a.flow.σ |> Array;)
_vbody(a::DistributedSimulation) = a.flow.V |> Array;
mu0(a::DistributedSimulation) = a.flow.μ₀ |> Array;
ranks(a::DistributedSimulation) = (a.flow.σ.=0; 
                                   @inside a.flow.σ[I] = me()+1;
                                   WaterLily.perBC!(a.flow.σ,());
                                   a.flow.σ |> Array;)

custom_attrib = Dict(
    "u" => velocity,
    "p" => pressure,
    "d" => _body,
    "ω" => vorticity,
    "v" => _vbody,
    "μ₀" => mu0,
    "rank" => ranks
)# this maps what to write to the name in the file

"""Flow around a circle"""
function circle(dims,center,radius;Re=250,U=1,mem=Array)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    DistributedSimulation(dims, (U,0), radius; ν=U*radius/Re, body, exitBC=false, mem=mem)
end

# local grid size
L = 2^6

# init the MPI grid and the simulation
r = init_mpi((L,2L))
sim = circle((L,2L),SA[L/2,L+2],L/8;mem=MPIArray) #use MPIArray to use extension

# need a distributed writer
wr = vtkWriter_d("WaterLily-MPI-circle";attrib=custom_attrib,dir="vtk_data",
                extents=get_extents(sim.flow.p))
for _ in 1:50
    sim_step!(sim,sim_time(sim)+1.0,verbose=true)
    write_d!(wr,sim)
end
close(wr)
finalize_mpi()