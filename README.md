# WaterLilyDistributed.jl

A parallel distributed (MPI) version of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl)

This package acts as a wrapper around WaterLily.jl, and provides a distributed version of the code. To

```julia
using WaterLily
using WaterLilyDistributed
using MPI

... # your code here

r = init_mpi(dims)
sim = Simulation(dims, ...)

... # your code here

finalize_mpi()
```

See the `examples` folder for more examples.

### @TODO's

- [ ] clean up all the import from `WaterLily`
- [ ] make `DistributedSimulation <: AbstractSimulation` (`WaterLily.jl/master` needs to change)
- [ ] Update `WaterLilyWriteVTKExt.jl` to work with distributed simulations
- [ ] make test work here

### Ideas for future work

Wrapp all these methods inside a `@distributed` macro, so that the user can simply write

```julia
using WaterLily
using WaterLilyDistributed
using MPI

@distributed begin
    sim = Simulation(dims, ...)

    ... # your code here

end
```
