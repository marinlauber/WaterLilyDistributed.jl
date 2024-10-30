# WaterLilyDistributed.jl

A parallel distributed (MPI) version of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl)

This package acts as a standalone parallel distributed version of WaterLily.jl. To use it, just construct a script like with the following mpi-specifi functions

```julia
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

- [ ] make test specific to this package