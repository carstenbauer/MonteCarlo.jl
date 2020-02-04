# Getting Started

## Installation

**MonteCarlo.jl** hasn't yet been released. To clone the package execute the following command in the Julia REPL:
```julia
] add https://github.com/crstnbr/MonteCarlo.jl
```

To update to the latest version of the package just do `Pkg.update()` or specifically `Pkg.update("MonteCarlo")`.

!!! warning

    The package is still in pre-alpha phase and shouldn't yet be used for production runs.

## Usage

This is a simple demontration of how to perform a Monte Carlo simulation of the 2D Ising model:

```julia
# load packages
using MonteCarlo

# load your model
m = IsingModel(dims=2, L=8);

# choose a Monte Carlo flavor and run the simulation
mc = MC(m, beta=0.35, sweeps=1000, thermalization=1000);
run!(mc, verbose=false);

# analyze results
obs = measurements(mc)[:ME] # what measurements exist for that simulation?
m = obs[:Magn].m # magnetization
mean(m)
std_error(m) # one-sigma error
```

![](../assets/ts_hist.png)
