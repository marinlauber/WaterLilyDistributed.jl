# WaterLilyDistributed.jl

A parallel distributed (MPI) version of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl)

This package acts as a standalone parallel distributed version of `WaterLily.jl`. To use it, just construct a script using the template below

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

## How to use

Because this package is still in a development state, you need to `dev` it
```julia
] dev git@github.com:WaterLily-jl/WaterLilyDistributed.jl.git
```

### @TODO's

- [ ] make test specific to this package
- [ ] keep track of `WaterLily.jl` source updates