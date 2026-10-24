# pauli-hamiltonian

Vectorized construction of a sparse (COO/CSR) Hamiltonian from a sum of
Pauli strings: bit manipulation, parity tracking, and index scatter instead
of row-by-row assembly.

Basis states are n-bit masks. A string is three disjoint masks (x, y, z) plus
a real coefficient; with Y\|0> = i\|1>, Y\|1> = -i\|0>, a string maps |b> to
|b ^ (x|y)> with amplitude `coeff * i^{popcount(y)} * (-1)^{popcount((y|z)&b)}`.
Every factor is a sign flip or a component swap, so the fast path is exact in
FP64 — which is what makes the bitwise gate below meaningful.

## Vendor gap

There is no matrix to multiply, so there is nothing for cuBLAS or cuSPARSE
to do. The operation is parity of bitmasks and index scattering — the most
library-resistant kernel in this portfolio.

## Physics payload

Hamiltonian construction for Hubbard and t–J quantum simulation — the
canonical cuprate models. The output feeds exact diagonalization directly;
the validation here rebuilds Hubbard chains via Jordan-Wigner and checks
spectra against an independent fermionic occupation-basis construction.

## Layout

- `include/ph/pauli.hpp` — the whole library (header-only, dependency-free):
  Pauli string types, naive per-qubit builder (baseline and oracle),
  vectorized builder (popcount-parity diagonal accumulation + O(1)
  off-diagonal phase per (string, basis) pair), shared stable-sort CSR
  assembly, bitwise equality and Hermiticity checks.
- `tests/test_equality.cpp` — the gate: naive vs vectorized must produce
  bitwise-identical CSR (same row pointers, same column order, memcmp-equal
  values) before any timing exists.
- `tests/test_hubbard.cpp` — physics validation; Eigen is linked here only,
  as an oracle, never in the library.
- `benchmarks/bench_build.cpp` — timing sweep, JSON with raw per-rep timings
  and machine/toolchain metadata.

Build and run:

```
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_build          # writes data/pauli_build_bench.json
```

## Correctness

- **Bitwise gate** (`test_equality`): 306 cases — randomized ensembles at
  n = 1..10 (20 seeds each), exhaustive singleton strings at n = 2,3 (all 80
  local combinations), all-diagonal sets, exact-duplicate strings that
  accumulate and cancel, identity-only, zero-coefficient, empty, full-fill
  (dense 64×64), and n = 10 stress. Naive and vectorized CSR are
  bit-identical on every case; every vectorized output is bitwise Hermitian.
- **Hubbard spectra** (`test_hubbard`): 1D open chains, mode m = 2·site+spin.
  Pauli-built vs direct fermionic matrices agree entrywise at 0.0 (dyadic
  t, U) and at 8.9e-16 for t = 1.1, U = 1.3 (L = 4). Spectra (Eigen
  `SelfAdjointEigenSolver` on both) agree at 6.6e-14 worst case, tolerance
  1e-12·‖H‖_F. The U = 0 chain matches analytic tight-binding subset sums at
  3.9e-14.

## Measured

AMD Ryzen 9 8940HX (32 hw threads, single-threaded kernel), GCC 16.2.1,
`-O3 -DNDEBUG`, median of 9 reps, full build = triplet generation + CSR
assembly; raw per-rep timings and full metadata in
[`data/pauli_build_bench.json`](data/pauli_build_bench.json).

End-to-end speedup, vectorized vs naive (strings/s and nnz/s in the JSON):

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random (uniform I/X/Y/Z) | 1.3–1.6x | 1.2–1.5x | 0.9–1.5x | 1.1–1.4x |
| diag-heavy (half I/Z-only) | 2.6–4.2x | 3.0–3.4x | 3.0–3.5x | 2.3–4.1x |

Ranges span string counts 64–1024. The generation kernel alone is 1.8–15.6x
faster; end-to-end is dominated at scale by the deterministic sort/accumulate
assembly, which is shared by both builders and is identical work. At
n = 12 with 1024 random strings the vectorized total ran slower than naive
(0.87x) with overlapping rep spreads — machine variance on a thermally
throttled laptop CPU, not a systematic regression; the generation medians
still favor vectorized 1.4x there. No extrapolation beyond the measured grid.

## Status

CPU phase complete 2026-10-24: library, bitwise gate, Hubbard validation,
measured table; suite green. Next gate: GPU scatter kernel over
(string, basis-state) pairs — the assembly side, not the per-pair phase math,
is where the time now goes.
