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

- [x] clean up all the import from `WaterLily`
- [ ] make `DistributedSimulation <: AbstractSimulation` (`WaterLily.jl/master` needs to change) this will avoid redefining the `sim_step!` and `sim_time` methods.
- [ ] change `WaterLily.⋅` (dot) to enable type dispatch for MPIArrays (`WaterLily.jl/master` needs to change)
- [ ] Update `WaterLilyWriteVTKExt.jl` to work with distributed simulations (`WaterLily.jl/master` needs to change)
- [ ] make test specific to this package

### Ideas for future work

Wrapp all these methods inside a `@distributed` macro, so that the user can simply write

```julia
using WaterLily
using WaterLilyDistributed
using MPI

@distributed (2,2,2) # mpi grid
    sim = Simulation(dims, ...)

    ... # your code here

end
```
