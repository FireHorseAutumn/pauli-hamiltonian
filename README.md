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
- `include/ph/pauli_gpu.cuh` — GPU triplet emission, no atomics: every
  triplet lands in a fixed slot reproducing the CPU emission order exactly
  (diagonal block first, one thread per basis state walking the diagonal
  strings in order; then one thread per (off-diagonal string, basis) pair).
  All multiplies are by 0 or ±1 — exact in FP64 under any contraction — so
  the emitted stream feeds the shared `assemble()` into bitwise-identical
  CSR. Also the GPU assembly (`assemble_device`, `build_gpu_full`): one
  block per row gathers the row's entries from their analytic slots, sorts
  them in shared memory by (column, emission index) with a bitonic network,
  sums each equal-column run sequentially at its head thread (the CPU
  accumulation order, hence the same bits), and scatters through an exact
  integer prefix sum into device CSR. Bounds: at most 2048 entries per row
  (one shared tile) and n <= 30; both throw past the limit.
- `tests/cases.hpp` — the shared gate case suite.
- `tests/test_equality.cpp` — the CPU gate: naive vs vectorized must produce
  bitwise-identical CSR (same row pointers, same column order, memcmp-equal
  values) before any timing exists.
- `tests/test_gpu_equality.cu` — the GPU gate: `build_gpu` (GPU emit + CPU
  assemble) and `build_gpu_full` (emit + assembly both on device)
  bitwise-identical to both CPU builders on the same suite, run twice for
  run-to-run determinism.
- `tests/test_hubbard.cpp` — physics validation; Eigen is linked here only,
  as an oracle, never in the library.
- `benchmarks/bench_build.cpp` — CPU timing sweep; `bench_gpu_build.cu` —
  GPU vs CPU emit sweep plus the full-build comparison (CPU emit+assemble
  vs GPU emit+assembly+download). Both write JSON with raw per-rep timings
  and machine/toolchain metadata.

Build and run:

```
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_build          # writes data/pauli_build_bench.json
./build/bench_gpu_build      # writes data/pauli_gpu_build_bench.json
```

## Correctness

- **Bitwise gate** (`test_equality`): 306 cases — randomized ensembles at
  n = 1..10 (20 seeds each), exhaustive singleton strings at n = 2,3 (all 80
  local combinations), all-diagonal sets, exact-duplicate strings that
  accumulate and cancel, identity-only, zero-coefficient, empty, full-fill
  (dense 64×64), and n = 10 stress. Naive and vectorized CSR are
  bit-identical on every case; every vectorized output is bitwise Hermitian.
- **GPU bitwise gate** (`test_gpu_equality`): the same 306 cases through
  `build_gpu` and `build_gpu_full`, bitwise-identical to both CPU builders
  (naive and vectorized), every GPU output bitwise Hermitian, and the whole
  suite run twice with bit-identical results across passes.
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
[`data/pauli_build_bench.json`](data/pauli_build_bench.json). All three
tables below are from the governor-pinned idle-machine rerun of 2026-10-30
(performance governor on all CPUs; bitwise gates passed per case before
timing).

End-to-end speedup, vectorized vs naive (strings/s and nnz/s in the JSON):

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random (uniform I/X/Y/Z) | 1.3–1.5x | 1.1–1.5x | 1.1–1.5x | 1.1–1.2x |
| diag-heavy (half I/Z-only) | 2.5–4.2x | 3.2–3.4x | 3.3–3.5x | 2.3–4.3x |

Ranges span string counts 64–1024. The generation kernel alone is 1.9–15.9x
faster; end-to-end is dominated at scale by the deterministic sort/accumulate
assembly, which is shared by both builders and is identical work. The
previously recorded n = 12 / 1024-string random loss cell (0.87x, unpinned)
does not reproduce under the pinned governor (1.11x) — it was thermal
machine variance, as suspected at the time. No extrapolation beyond the
measured grid.

GPU emit (pinned rerun 2026-10-30, RTX 5060 Laptop sm_120, 26 SMs, nvcc
13.2.51 with g++-15 host; CPU side Ryzen 9 8940HX; median of 9 reps, emit
only — assembly is the shared CPU sort everywhere; raw reps and metadata in
[`data/pauli_gpu_build_bench.json`](data/pauli_gpu_build_bench.json);
bitwise gate passed per case before timing):

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random, kernel vs CPU emit | 2.6–12.1x | 7.1–130.2x | 13.5–142.1x | 142.8–145.9x |
| random, end-to-end vs CPU emit | 0.16–0.27x | 0.21–0.78x | 0.23–0.78x | 0.78–0.80x |
| diag-heavy, kernel vs CPU emit | 1.7–4.1x | 6.4–65.3x | 16.4–116.6x | 129.5–135.6x |
| diag-heavy, end-to-end vs CPU emit | 0.20–0.38x | 0.24–2.74x | 0.41–2.71x | 0.86–2.72x |

Ranges span string counts 64–1024. The kernels win by up to ~146x with tight
spreads — the work is a perfectly parallel fixed-slot scatter. End-to-end,
transfer + allocation dominates below n = 12 and the crossover only lands
around n = 12–14 (several cells near or below 1x — recorded, not hidden).

Full build (pinned rerun 2026-10-30, same protocol): CPU
`emit_vectorized + assemble` vs the full GPU path (compaction, upload,
emit, on-device assembly, CSR download), median-of-9 wall clock ratio:

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random | 2.0–10.4x | 2.5–17.5x | 3.0–7.1x | 5.1–8.9x |
| diag-heavy | 0.9–6.4x | 1.9–9.9x | 2.5–7.1x | 3.0–6.0x |

Ranges span string counts 64–1024; the GPU assembly removes the CPU
stable-sort bottleneck recorded above. One wash cell: diag-heavy n = 8
with 64 strings at 0.94x with overlapping spreads — at 256×256 the whole
build is sub-millisecond on both sides. Wide rep spreads persist at the
largest sizes even pinned (GPU allocation variance) — medians reported,
raw reps in the JSON. The GPU assembly is bitwise-identical to the CPU
assemble on every timed case (checked before timing) and bitwise
deterministic run to run.

## Status

GPU phase complete 2026-10-25: scatter kernels with the bitwise gate re-run
GPU-vs-CPU (306/306 cases, two deterministic passes), measured emit table
above; suite green 3/3. GPU assembly complete 2026-10-26: on-device
bitonic-sort/compact/scan assembly, bitwise gate re-run for the full GPU
build, measured full-build table above (the CPU sort bottleneck is gone at
every size past the sub-millisecond regime); suite green 3/3. Bounds of the
current assembly (<= 2048 entries per row, n <= 30) throw loudly; a
two-level scheme is the extension if they ever bind.
