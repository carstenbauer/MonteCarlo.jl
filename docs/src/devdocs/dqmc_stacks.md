# DQMC Stacks

There are currently two "stacks" associated with DQMC which mostly hold matrices for the calculations involved. The first one is `DQMCStack` (in "DQMC/stack.jl") which is responsible for the main simulation and the latter is `UnequalTimeStack` (in "DQMC/unequal_time_stack.jl") which is responsible for time displaced greens functions.

## DQMCStack

The `DQMCStack` struct contains:
- the `field_cache` which includes a set of temporary variables used in local updates (see fields devdocs)
- `u_stack`, `d_stack` and `t_stack` which hold on to a set of UDT decomposed matrices necessary to compute the next Greens function in imaginary time
- a bunch of matrices and vectors which are used as short to mid term temporary buffers
- `greens` which contains the current **effective** greens function (there is further computation needed to get to the correct greens function for measurements, see "DQMC/greens.jl")
- infortmation about the current time slice, the total number, safe_mult blocks, etc
- a buffer for the exponentiated interaction matrix
- constant buffers for different forms of the exponentiated hopping matrix 

#### Matrix product chain

To calculate the (equal time) Greens function at some time slice `l` we need to calculate 

```math
G(l) = [1 + B(l-1) B(l-2) \cdots B(1) B(M) \cdots B(l)]^{-1}
```

where $B(l) = e^{-\Delta\tau H(l)}$ are propagation matrices. Specifically we will need two matrix product chains here, $B(l-1) \cdots B(1)$ and $B(M) \cdots B(l)$. When the simulation runs, we alternative between going from `l = 1` to `l = M` and back. Let us consider the former case for an example. We will need:

- $I$ and $B(M) \cdots B(1)$ at `l = 1`
- $B(1)$ and $B(M) \cdots B(2)$ at `l = 2`
- $B(k-1) \cdots B(1)$ and $B(M) \cdots B(k)$ at `l = k`
- $B(M-1) \cdots B(1)$ and $B(M)$ at `l = M`

Thus at the start we will want a full product chain for the right term and an empty chain i.e. an identity for the left side. In every step we multiply a propagation matrix to the left on the left chain and its inverse to right side of the right chain. The opposite direction then simply does the reverse of this. 

One way to deal with this calculation is to store matrices for each step in a big array, and then update parts of the array as you move from one time to another. Matching the list above we would have

- $[I, B(M) \cdots B(1), B(M) \cdots B(2), \dots, B(M)]$ at `l = 1`
- $[I, B(1), B(M) \cdots B(2), \dots, B(M)]$ at `l = 2`
- $[I, \dots, B(k-1) \cdots B(1), B(M) \cdots B(k), \dots, B(M)]$ at `l = k`
- $[I, \dots, B(M-1) \cdots B(1), B(M)]$ at `l = M`

which contains the necessary matrices at each step. Going backwards in time would rebuild this vector, so only need to fully populate it once. We can also cut a few values: The identity at the start is unnecessary and we can have the new value from the left chain overwrite it's partner from the right. This is effectively what `u_stack`, `d_stack` and `t_stack` contain before considering float precision problems and optimizations.

#### Float Precision

Let us start with float precision errors as they are relevant to most of the other details. Before going into solving the problem we will briefly explain how float precision errors occur. A floating point number is effecitvely a number represented in scientific notation with a certain number of digits for the significand and exponent. Consider for example 3 digits for the significant, e.g. $123 \cdot 10^0$ and $987 \cdot 10^{-6}$. These numbers can be multiplied and divided by each other without issues, however adding or subtracting one from the other will lead to the information stores in the second number to be erased, as it cannot be stored in the three digits available in the first. If further calculations cancel out the $123 \cdot 10^0$ we will no longer have the information from $987 \cdot 10^{-6}$ available, resulting in a significantly wrong result. 

#### UDT Decomposition

(Chains of) Matrix multiplications, which are packed with additions, suffer from such floating point precision errors especially when their eigenvalues are spread across a large range of values. This is the case in DQMC. To avoid precision errors, we need to keep the spread of eigenvalues small in consecutive matrix multiplications. Various matrix decomposition can achieve this to varying degrees, see [StableDQMC.jl](https://github.com/carstenbauer/StableDQMC.jl) for details. In MonteCarlo.jl we use introduced a `UDT` decomposition, which is simply a QR decompoisition where the unitary matrix `U = Q` and the `R` matrix is further split into a Diagonal `D` matrix and the remaining upper triangular `T` matrix.

#### `add_slice_matrix_left!`

Let's get back to our product chain of propagators. The product of two such matrices will not immediately be critical in terms of float precision. Instead there is some empiral number of multiplications `safe_mult` which is fine to do. Thus we perform a stabilization step every `safe_mult` multiplications. The stabilization step looks roughly like this for increasing time:

```julia
# builds up error, stops before it gets too much
cur_U = B[n+safe_mult] * ... * B[n+1] * prev_U 
# Diagonal multiplication involves no addition - no extra error
temp = cur_U * prev_D 
# Increases error, but on the scale we still consider fine
cur_U, cur_D, temp = UDT(temp)
# low error
cur_T = temp * prev_T
```

This happens in `add_slice_sequence_left`. Since we have U, D and T matrices here that need to be kept track of the vector of propagation matrices we introduced earlier now becomes three vectors of U, D and T matrices respectively. For the reverse direction we effectively do the same, however we use adjoint matrices. The reason for this follows from the computation of the Greens matrix, so let us look at that next. 

#### `calculate_greens`

To compute the Greens matrix we also need to be careful about float precision. Inversion specifically can be quite sensetive to such errors. After UDT decompositions we have unitary matrices, which can be inverted exactly by taking the adjoint, Diagonal matrices which can be inverted exactly by inverting each element, triangular matrices which may produce errors and normal matrices which may also produce errors. To make a stable inversion happen we use $U_l D_l T_l = B(l-1) B(l-2) \cdots B(1)$ and $U_r D_r Tr =  B^\dagger(l) \cdots B^\dagger(M)$ which results in

```math
G(l) = [1 + U_l D_l T_l T_r^\dagger D_r U_r^\dagger]^{-1}
```

Here we can combine the left and right scales without introducing significant errors by calculating $X = D_l (T_l T_r^\dagger) D_r$ and decompose it into a single scale matrix $U D T = X$. After this we have

```math
G(l) = [1 + U_l U D T U_r^\dagger]^{-1}
```

we can now pull all the scale-free matrices out of the inverse.

```math
G(l) = U_r T^{-1} [U^\dagger U_l^\dagger U_r T^{-1} + D]^{-1} U^\dagger U_l^\dagger
```

This leaves the scale-full diagonal matrix isolated on the right and a product of scale-free matrices on the left. We now calculate all the matrix products to get to $L [X + D]^{-1} R$ where $L$ and $X$ are generic matrices, $D$ is diagonal and $R$ is unitary. At this point we have nothing left to do but calculate $X + D$, which involves some amount of error. We however do not invert yet, but do another UDT decomposition to get to

```math
G(l) = L [u d t]^{-1} R = L t^{-1} d^{-1} u^\dagger R
```

which resolves without further complications. This is what `calculate_greens` does, obscured behind variable reuse. 

#### Optimizing stack sizes

Let us step away from floating point precision for a bit and look back at our matrix product chain. We now have 3 vectors of matrices `u_stack`, `d_stack` and `t_stack` containing a somewhat unclear amount of matrices. In the worst/naive case, we have one matrix per time slice, which is quite a lot. We can reduce the amount of matrices we need to store by using another formula for updating greens matrices

```math
G(l+1) = B(l) G(l) B^{-1}(l)
```

which follows $A^{-1} B^{-1} = (BA)^{-1}$. This is implemented in MonteCarlo.jl as `wrap_greens!`. Much like our matrix product chain this too will build up a significant amount of float precision error over time, which means that we can only use it for so many steps before we need to stabilize.

The way MonteCarlo.jl handles the different formulas for updating the greens function is as follows:

- At `l = 1` (`l = M`) the greens function is calculated from the product chain. At this point the product chain contains a fresh `UDT` decomposition.
- For the next `safe_mult` time steps, `wrap_greens` is used to calculate the corresponding greens function. This avoids using the rather involved inversion which is part of the other method.
- After `safe_mult` steps the product chains update, writing one new set of UDT decomposed matrices to the relevant vectors. These are then used to recompute the Greens function at `l = 1 + safe_mult`. (There are no U, D and T matrices saved for `l = 2` to `l = safe_mult`.)

This process then repeats until we reach `l = M` (`l = 1`). During the stabilization step (last in the list) we also compared the result from one extra `wrap_greens` with the stabilized greens function to make sure float precision errors aren't getting out of hand. These errors are referred to as `propagation errors`. All of this happens in `propagate`.

#### Propagation Matrix

One more thing we have yet to properly define are the propagation matrices $B(l) = e^{-\Delta\tau H(l)}$. Their exact form depends on the Trotter decomposition and the way local updates are implemented. In MonteCarlo.jl local updates assume the right most matrix in the matrix product chain to be an exponentiated interaction matrix. I.e. for an asymmetric Trotter decomposition we would have

```math
G(l) = [1 + e^{-\Delta\tau T} e^{-\Delta\tau V(l-1)} \cdots e^{-\Delta\tau T} e^{-\Delta\tau V(1)} e^{-\Delta\tau T} e^{-\Delta\tau V(M)} \cdots e^{-\Delta\tau T} e^{-\Delta\tau V(l)}]^{-1}
```

We can (and do) however use the more precise symmetric Trotter decomposition $B(l) = e^{-0.5\Delta\tau T} e^{-\Delta\tau V(l)} \cdots e^{-0.5\Delta\tau T}$. For this we use the same trick used in `wrap_greens` which allows us to rewrite the Greens function as

```math
G(l) = e^{0.5\Delta\tau T} [1 + e^{-0.5\Delta\tau T} e^{-0.5\Delta\tau T} e^{-\Delta\tau V(l-1)} \cdots e^{-0.5\Delta\tau T} e^{-0.5\Delta\tau T} e^{-\Delta\tau V(1)} e^{-0.5\Delta\tau T} e^{-0.5\Delta\tau T} e^{-\Delta\tau V(M)} \cdots e^{-0.5\Delta\tau T} e^{-0.5\Delta\tau T} e^{-\Delta\tau V(l)}]^{-1} e^{-0.5\Delta\tau T}
```

where we consider the inverse `[...]^{-1}` as the effective Greens function saved as `greens` in the `DQMCStacj` and whole expression, i.e. `G(l)` as the true greens function.

### Further Details Caveats 

We are now done discussing the main aspects of the `DQMCStack`. There are however some more details that might be worth mentioning:

- We do not actually adhere to `safe_mult` tightly, as that would reduce the already low number of temperatures we can consider for a given $\Delta\tau$ in a certain temperature range. Instead we generate ranges which contain $\le \mathrm{safe\_mult}$ time slices to identify safe matrix multiplications. When stepping from one range to the next we stabilize.
- We do not actually use $e^{-0.5\Delta\tau T} e^{-0.5\Delta\tau T}$ in the calculations. The formula is equivalent to `e^{-\Delta\tau T}`, which is precomputed as `hopping_matrix_exp_squared`.
- Global updates assume to happen when the stack is at time slice 1, going up. They further assume that `calculate_greens` has just happened. Adjusting how the stack is initialized is therefore likely to break global updates.

---

## Unequal Time Stack

The unequal time stack is more of the same, though more complicated and error prone. The time displaced Greens function is given by

```math
G(k, l) = [B(l+1)^-1 B(l+2)^-1 \cdots B(k)^-1 + B(l) \cdots B(1) B(M) \cdots B(k+1)]^-1
```

if $k \ge l$ or

```math
G(k, l) = [B(l) B(l-1) \cdots B(k+1) + (B(k) \cdots B(1) B(M) \cdots B(l+1))^-1]^-1
```

if $k \le l$. To calculate these, three propagator stacks are introduced in `UnequalTimeStack`:

- the `forward` stack, which calculates $B(n \cdot \mathrm{safe\_mult}) \cdots B(1)$
- the `backward` stack, which calculates $B^\dagger(n \cdot \mathrm{safe\_mult}) \cdots B^\dagger(M)$
- the `inverse` stack, which calculates small blocks of $B^{-1}(n \cdot \mathrm{safe\_mult} + 1) \cdots B^{-1}((n+1) \cdot \mathrm{safe\_mult})$

Their calculation is very similar to the `DQMCStack`. Time displaced Greens function are usually used in measurements to calculate susceptibilities, which requires `G(0, l)` and `G(l, 0)` which in turn require values from a wider range of positions in the forward, backward and inverse stack. Thus we use multiple stacks here, rather than having them overwrite entries in one stack like for `DQMCStack`. Filling these stacks is handled by the `lazy_build_<forward/backward/inverse>!` functions, which only compute as much as is requested.

To then compute the relevant Greens function, `calculate_greens_full` is called. In the first step the `compute_<forward/backward/inverse>_udt_block!` functions are called to get the specific UDT decomposed matrix product chains. Then another stabilized inversion like `calculate_greens` from `DQMCStack` is called. There is another quirk here however, which is that the scale-full diagonal matrices are split into parts smaller and larger than unit scale. This further boosts precision/stability which is necessary here.

Note that the time displaced greens functions used for susceptibilities in measurements don't directly call the methods here. Instead they go through `CombinedGreensIterator`, which essentially implements `wrap_greens` and only refers back to `UnequalTimeStack` every so often to get a stabilized result. You can find the iterator in "DQMC/measurements/greens_iterators.jl".