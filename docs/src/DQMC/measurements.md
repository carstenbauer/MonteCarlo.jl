# Measurements

Measurements in DQMC primarily rely on [Wick's theorem](https://en.wikipedia.org/wiki/Wick%27s_theorem) to express observables in terms of Greens function elements. The greens function in DQMC is a matrix $G_{ij}(k, l) = \langle c_i(k \Delta\tau) c_j^\dagger(l \Delta\tau)\rangle_f$, where $\langle\cdot\rangle_f$ represents the fermion average. 

Measurement are created and added via `dqmc[:name] = measurement(dqmc, model)`. The currently implemented measurements are the following



#### Greens

The equal time greens function can be measured via `greens_measurement(dqmc, model)`. The measurement will take the Monte Carlo or bosonic average of `greens(dqmc, model)`.

#### Occupation

The per-site occupation $\langle n_i \rangle$ can be measured via `occupation(dqmc, model)`. This will average $1 - G_{ii}$.

#### Charge Density

The charge density correlation $\langle \sum_r n(r) n(r+\Delta r) \rangle$ can be measured with `charge_density_correlation(dqmc, model)`. The time integral of that, the charge density susceptibility can be measure with `charge_density_susceptibility(mc, model)`. 
Note that either way the result will be averaged over origin sites and saved by distance vectors. These vectors can be generated with `directions(lattice)`

#### Magnetization

`magnetization(dqmc, model, dir::Symbol)` measures the per-site x-, y- or z-magnetizations.

#### Spin Density

The spin density correlation $\langle S_\gamma S_\gamma \rangle$ in x-, y- or z-direction can be measured with `spin_density_correlation(dqmc, model, dir::Symbol)`. The respective susceptibilities follow from `spin_density_susceptibility(dqmc, model, dir)`.

#### Pairing

The pairing susceptibility $\Delta_v(s_1, t_1)(\tau) \Delta_v^\dagger(s_2, t_2)(0)$ can be calculated with `pairing_susceptibility(dqmc, model)`. The $\tau = 0$ pairing correlation follows from `pairing_correlation(dqmc, model)`.

#### Current-Current Susceptibility

`current_current_susceptibility(dqmc, model)` measures ``\langle j_{t_2 - s_2}(s_2, l) j_{t_1 - s_1}(s_1, 0)\rangle$ with $j_{t - s}(s, \tau) = \langle i \sum_\sigma [T_{ts} c_t^\dagger(\tau) c_s(\tau) - T_{st} c_s^\dagger(\tau) c_t(\tau)] \rangle`` where $i^2 = -1$ and $T$ is the hopping matrix.

#### Superfluid Density

`superfluid_density(dqmc, model, L)` computes the superfluid density using the current current susceptibility for a lattice of linear system size $L$.

#### Energies

The energy can be measured with `total_energy(dqmc, model)`. The interacting and noninteracting parts can be measured independently with `interacting_energy(dqmc, model)` and `noninteracting_energy(dqmc, model)`



## General Notes

All measurements are implemented via

```julia
struct DQMCMeasurement{GI, LI, F <: Function, OT, T} <: AbstractMeasurement
    greens_iterator::GI
    lattice_iterator::LI
    kernel::F
    observable::OT
    temp::T
end
```

The `greens_iterator` controls which Greens functions are used for the measurement. Internally measurements that use the same `greens_iterator` will be bundled to avoid expensive recalculations. The available iterators include:

* `nothing` specifies that no Greens function is needed
* `Greens()` forwards the equal time greens function `G(0, 0)` (which matches all other equal time greens functions)
* `GreensAt(k, l)` forwards the result of `greens(dqmc, k, l)`
* `TimeIntegral([recalculate = 2 mc.parameters.safe_mult])` creates an iterator for calculating imaginary time integral of the form $O_i = \int_0^\beta O_(\tau) d\tau$ as $O_i \approx \sum_{l = 0}^{M-1} 0.5 \Delta\tau (O_(l \Delta\tau) + O_((l+1) \Delta\tau))$. In every step this iterator will generate four greens matrices $G(0, 0)$, $G(0, l\Delta\tau)$, $G(l\Delta\tau, 0)$, $G(l\Delta\tau, l\Delta\tau)$. This internally uses `CombinedGreensIterator(mc[; start, stop, recalculate])` to generate these matrices.

The `lattice_iterator` controls which combination of site indices are passed to a measurements and how they are further combined before saving the measurements. For example, `EachSitePairByDistance` passes any combination of two sites indices to the measurements and sums up site pairs which point in the same direction. See the Lattices section for more detail.

The `kernel` is a function that basically just uses the results from Wicks theorem. For example, the kernel for charge density susceptibilities is given by

```julia
function cdc_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4}, flv::Val{2})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))

    # ⟨n↑(l)n↑⟩
    (1 - Gll[i, i]) * (1 - G00[j, j]) -
    G0l[j, i] * Gl0[i, j] +
    # ⟨n↑(l)n↓⟩
    (1 - Gll[i, i]) * (1 - G00[j+N, j+N]) -
    G0l[j+N, i] * Gl0[i, j+N] +
    # ⟨n↓(l)n↑⟩
    (1 - Gll[i+N, i+N]) * (1 - G00[j, j]) -
    G0l[j, i+N] * Gl0[i+N, j] +
    # ⟨n↓(l)n↓⟩
    (1 - Gll[i+N, i+N]) * (1 - G00[j+N, j+N]) -
    G0l[j+N, i+N] * Gl0[i+N, j+N]
end
```

The passed indices and greens matrices vary depending on the chosen iterators. The indices could be a single integer, a tuple of two or a tuple of four integers. `packed_greens` could be a greens matrix (of type `GreensMatrix`) or a tuple of four greens matrices. 

The `observable` is the final storage of the measured values. By default this is a `LogBinner` from `BinningAnalysis.jl` but that can be changed. The only hard requirement is that the data structure implements `push!`. An initial zero-valued entry can be generated with `_binner_zero_element_(dqmc, lattice_iterator, geltype(dqmc))`.

The `temp` field is a temporary storage Array used as a target for summation before pushing the final result of the measurement. It should be initialized with `MonteCarlo._measurement_buffer(dqmc, lattice_iterator, geltype(dqmc))`. Note that this is often but not always the same as the zero element.