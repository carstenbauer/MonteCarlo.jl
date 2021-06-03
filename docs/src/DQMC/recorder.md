# Configuration Recorder

Sometimes it's useful to keep configurations around, for example to perform new measurements after the original simulation has finished. This is especially true in the case of DQMC where simulations can take weeks to run, but measurement on their own are relatively fast. We currently offer two objects for this purpose - `Discarder` and `ConfigRecorder`.

#### General Interface

Any `AbstractRecorder` follows an array-like interface. Configurations are added via `push!(recorder, mc, model, sweep)` and can be retrieved via `getindex`. The recorder has a `length` and implements `isempty`. Furthermore it can be iterated, saved and loaded.

#### Discarder

As the name suggests this recorder simply discards all configurations. It is used by default for classical Monte-Carlo simulations.

#### ConfigRecorder

This recorder keeps track of compressed configurations in memory. On creation a `rate` can be specified as the last argument in the constructor to reduce the amount of configurations saved. Compression relies on overloads of `compress(mc, model, conf)` and `decompress(mc, model, conf)`. For the available Hubbard models these simply transform from and to `BitArray`.