# Monte Carlo integration (Integrator)

Basic Monte Carlo integration. It can for example be used to integrate a multidimensional [Gaussian function](@ref GaussianFunction).

You can initialize a Monte Carlo integration of a given `model` (function) by
```julia
mc = Integrator(model)
```
where the integration volume is taken as a box.

Allowed keywords are:

* `min_x`: lower integration bounds vector
* `max_x`: upper integration bounds vector
* `sweeps`: number of measurement sweeps
* `thermalization`: number of thermalization (warmup) sweeps
* `seed`: initialize MC with custom seed

Afterwards, you can run the simulation by
```julia
run!(mc)
```

Note that you can just do another `run!(mc, sweeps=1000)` to continue the simulation.

By default, every Monte Carlo integration stores the energies (function values) and the estimate for the integral as observables. You can for example access the latter via

```julia
mean(mc.obs["integral"])
```

## Model interface

Any model to be integrated by means of Monte Carlo integration must implement the following interface.

### Mandatory methods

 * [`energy`](@ref MonteCarlo.energy): energy (i.e. function value) at location `value`

Precise signatures can be found here: [Interface: Monte Carlo integration (Integrator)](@ref).

### Optional methods

 * `rand`: draw random value in the integration domain
 * `propose`: propose a local move
 * `prepare_observables`: initialize observables
 * `measure_observables!`: measure observables
 * `finish_observables!`: finish measurements

 Precise signatures can be found here: [Interface: Monte Carlo integration (Integrator)](@ref).
