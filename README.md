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

GPU emit (2026-10-25, RTX 5060 Laptop sm_120, 26 SMs, nvcc 13.2.51 with
g++-15 host; CPU side Ryzen 9 8940HX; median of 9 reps, emit only — assembly
is the shared CPU sort everywhere; raw reps and metadata in
[`data/pauli_gpu_build_bench.json`](data/pauli_gpu_build_bench.json);
bitwise gate passed per case before timing):

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random, kernel vs CPU emit | 2.4–10.2x | 6.3–95.7x | 14.2–146.6x | 127.1–164.5x |
| random, end-to-end vs CPU emit | 0.19–0.40x | 0.25–0.70x | 0.46–1.17x | 0.94–1.02x |
| diag-heavy, kernel vs CPU emit | 1.2–3.8x | 4.8–41.9x | 11.3–78.5x | 94.0–136.6x |
| diag-heavy, end-to-end vs CPU emit | 0.18–0.56x | 0.23–3.23x | 0.37–3.07x | 0.91–4.45x |

Ranges span string counts 64–1024. The kernels win by up to ~165x with tight
spreads (e.g. [1.890, 1.899] ms at the largest case) — the work is a
perfectly parallel fixed-slot scatter. End-to-end, transfer + allocation
dominates below n = 12 and the crossover only lands around n = 12–14
(0.37–4.45x, several cells near or below 1x, some with wide e2e spreads from
allocation variance — recorded, not hidden).

Full build (2026-10-26, same protocol): CPU `emit_vectorized + assemble`
vs the full GPU path (compaction, upload, emit, on-device assembly, CSR
download), median-of-9 wall clock ratio:

| ensemble | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| random | 2.2–11.3x | 2.6–17.6x | 2.4–7.0x | 5.4–11.0x |
| diag-heavy | 1.0–5.8x | 2.2–9.2x | 2.6–7.7x | 2.7–5.8x |

Ranges span string counts 64–1024; the GPU assembly removes the CPU
stable-sort bottleneck recorded above. One wash cell: diag-heavy n = 8
with 64 strings at 0.98x with overlapping spreads — at 256×256 the whole
build is sub-millisecond on both sides. Wide rep spreads on both sides at
the largest sizes (CPU sort variance up to 3x, GPU allocation variance up
to 2x on an unpinned laptop) — medians reported, raw reps in the JSON. The
GPU assembly is bitwise-identical to the CPU assemble on every timed case
(checked before timing) and bitwise deterministic run to run.

## Status

GPU phase complete 2026-10-25: scatter kernels with the bitwise gate re-run
GPU-vs-CPU (306/306 cases, two deterministic passes), measured emit table
above; suite green 3/3. GPU assembly complete 2026-10-26: on-device
bitonic-sort/compact/scan assembly, bitwise gate re-run for the full GPU
build, measured full-build table above (the CPU sort bottleneck is gone at
every size past the sub-millisecond regime); suite green 3/3. Bounds of the
current assembly (<= 2048 entries per row, n <= 30) throw loudly; a
two-level scheme is the extension if they ever bind.
