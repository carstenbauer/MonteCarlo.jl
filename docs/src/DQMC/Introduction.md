# Determinant Quantum Monte Carlo

Determinant Quantum Monte Carlo is a Quantum Monte Carlo algorithm for fermionic Hamiltonians. The general idea is to use the Hubbard-Stranovich transformation to simplify the Hamiltonian to one with only quadratic fermionic degrees of freeedom. This introduces a bosonic fields for quartic terms, which can be sampled by a Monte Carlo procedure.

The minimal working example for a DQMC simulation is the following.

```@example
using MonteCarlo

model = HubbardModelAttractive(4, 2)
dqmc = DQMC(model, beta=1.0)
run(dqmc)
```

This will set up and run a DQMC simulation at inverse temperature $\beta = 1.0$ using an attractive Hubbard model with a four by four square lattice. Note that by default no measurements are taken. 

In the following pages we will discuss the various components that go into a DQMC simulation.

## Derivation

If you are interested in the derivation of DQMC you may check [Introduction to Quantum Monte Carlo Simulations for fermionic Systems](https://doi.org/10.1590/S0103-97332003000100003), the book [Quantum Monte Carlo Methods](https://doi.org/10.1017/CBO9780511902581) or [World-line and Determinantal Quantum Monte Carlo Methods for Spins, Phonons and Electrons](https://doi.org/10.1007/978-3-540-74686-7_10). The first reference is most in-line with the implementation of this package.

If you want to go through the source code, compare it and verify for yourself that it is correct there a couple of things that should be pointed out. Most educational sources use the assymmetric two term Suzuki-Trotter decomposition. We use the symmetric three term version for increased accuracy.

\begin{align}
    B(l) = e^{-\Delta\tau \sum_l T+V(l)} = \prod_j e^{-\Delta\tau T/2} e^{-\Delta\tau V} e^{-\Delta\tau T/2} + \mathcal{O}(\Delta\tau^2)
\end{align}

This change is however no trivial as the first or last element of the $B$ matrix/operator needs to be an exponentiated interaction. To get this we use an effective greens function, which cyclically permutes one exponentiation hopping term to the other end of the chain. This adjustment needs to be undone for the actual greens function, which happens in `greens()`.

Another thing worth mentioning is that depending on the choices made at the start of the derivation, matrix products may have different order and indices may vary. The first source should have the same definitions.