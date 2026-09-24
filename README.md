# Lattice Schwinger model

Reproducing and extending Chen, Cheng & Guo,
[arXiv:2607.02894](https://arxiv.org/abs/2607.02894), on nonequilibrium
dynamics in the 1+1D lattice Schwinger model.

The repository currently provides an exact-diagonalization baseline. Lattice units are $a = m = g = 1$,
and the quench starts from the zero-field ground state.

## Current Results

`src/schwinger.py` builds the Pauli Hamiltonian and the paper's observables. The stagger is $(-1)^{n+1}$, and with that sign the exact-diagonalization curves sit on the paper's.

`python scripts/run.py` writes the exact-diagonalization datasets to `data/`. `python scripts/visualize.py` reads those files and writes the figures.

For $N = 8$, the spectrum, charge dynamics, electric-field energy, vacuum fidelity, and early-time decay rate match the paper's exact-diagonalization figures (Figs. 1 and 3–8). The zero-field ground-state energy is $-4.63805774$, in agreement with Table I. The critical field is computed for $N = 8, 10, 12, 14, 16$ and matches the Fig. 2 markers. A fit through those five sizes intercepts near $0.465$. The paper's quoted intercept, $0.469$, includes $N = 18$.

Documentation:

- [docs/building_hamiltonian.md](docs/building_hamiltonian.md) — the Pauli Hamiltonian, Eqs. (8)–(11): kinetic, mass, and electric-field terms, and how the $N$-qubit matrix is assembled.
- [docs/building_observables.md](docs/building_observables.md) — total, site, and spatial charge, electric-field energy, vacuum fidelity, the chiral condensate, and the exact time evolution.

Figures:

`figures/paper_overlay/` places the archived $N = 8$ curves on the paper's axes. A solid line is this exact diagonalization, a dashed line is the paper's exact diagonalization, and gray dots are the paper's quantum-computing markers.

## Next Steps

- Appendix A at $N = 12$ (Figs. 9–11), and the $N = 18$ ground state so the Fig. 2 extrapolation uses the same sizes as the paper.
- VQE ground states, VQD excited states, and second-order Trotter evolution, checked against this baseline.
- A time-dependent external field.
- Larger lattices, with a tensor-network baseline, then ansatz benchmarks and hardware runs.

## Research Status

The results above are the current exact-diagonalization baseline. The items
under Next Steps are still in progress; they are not part of the results yet.
