#pragma once

// pauli_gpu.cuh — GPU triplet emission for the Pauli-string Hamiltonian
// builder, bitwise-compatible with emit_vectorized.
//
// Determinism strategy: no atomics. Every triplet is written to a fixed slot
// reproducing the CPU vectorized emission order exactly:
//   slots [0, dim)                  diagonal block (one per row), iff any_diag
//   slots [diag_off + j*dim, ...)   off-diagonal string j (string order), one
//                                   per basis state
// The diagonal kernel parallelizes over basis states and walks the diagonal
// strings sequentially in string order, so each diagonal accumulator sees the
// identical FP64 addition sequence as the CPU. Off-diagonal values are pure
// table lookups and sign flips (all multiplies are by 0 or +-1, hence exact
// under any contraction), so each off-diagonal triplet is bit-exact per
// (string, basis) pair. Feeding the emitted buffer to the shared assemble()
// therefore yields bitwise-identical CSR, and the kernel is bitwise
// deterministic run to run.

#include "pauli.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ph {
namespace gpu {

struct GpuTriplet {
  std::uint64_t row, col;
  double vre, vim;
};
static_assert(sizeof(GpuTriplet) == sizeof(Triplet), "triplet layout");
static_assert(offsetof(GpuTriplet, col) == offsetof(Triplet, col),
              "triplet layout");

inline void cuda_check(cudaError_t st, const char* what) {
  if (st != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(st));
}

// Diagonal block: one thread per basis state, strings walked in order.
// acc starts at +0.0 and accumulates (parity ? -c : +c); the same sequence
// the CPU vectorized path applies to its dense diagonal.
__global__ void emit_diag_kernel(const std::uint64_t* __restrict__ z,
                                 const double* __restrict__ coeff, int n_diag,
                                 std::uint64_t dim, GpuTriplet* out) {
  const std::uint64_t b =
      (std::uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= dim)
    return;
  double acc = 0.0;
  for (int s = 0; s < n_diag; ++s)
    acc += (__popcll(z[s] & b) & 1) ? -coeff[s] : coeff[s];
  out[b] = {b, b, acc, 0.0};
}

// Off-diagonal block: one thread per (string, basis) pair, fixed slot
// diag_off + j*dim + b.
__global__ void emit_offdiag_kernel(const std::uint64_t* __restrict__ f,
                                    const std::uint64_t* __restrict__ yz,
                                    const double* __restrict__ pre,
                                    const double* __restrict__ pim, int n,
                                    std::uint64_t dim, int n_od,
                                    std::uint64_t diag_off, GpuTriplet* out) {
  const std::uint64_t idx =
      (std::uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t j = idx >> n;
  if (j >= (std::uint64_t)n_od)
    return;
  const std::uint64_t b = idx & (dim - 1);
  const double re = pre[j], im = pim[j];
  const bool neg = (__popcll(yz[j] & b) & 1) != 0;
  out[diag_off + idx] = {b, b ^ f[j], neg ? -re : re, neg ? -im : im};
}

// ---- GPU assembly ------------------------------------------------------
//
// assemble_device() consumes the emitted triplet buffer and produces CSR on
// the device, bitwise-identical to the shared CPU assemble() applied to the
// same stream: stable (row, col) order, equal-key runs summed in emission
// order, zeros canonicalized.
//
// No atomics and no general-purpose sort. Each row's triplets sit at
// analytic slots (diagonal slot r; off-diagonal string j at
// diag_off + j*dim + r), so one block per row gathers its
// K = any_diag + n_od entries and sorts them in shared memory by
// key = (col << 11) | emission_index. Bitonic is not a stable network; the
// emission-index tiebreak in the key makes equal-column runs come out in
// emission order, matching the CPU stable_sort. Run heads sum their runs
// sequentially (the CPU accumulation order, hence the same bits), per-row
// run counts are prefix-summed by an exact integer scan into row_ptr, and
// segments are copied out of the dim*K staging area into the final arrays.
//
// Bounds (checked, hard throws): K <= 2048 (one shared tile), n <= 30
// (one block per row in the launch x dim). Both are far past every shipped
// case; a two-level scheme is the recorded extension if they ever bind.

constexpr int kAsmBlock = 256;
constexpr int kAsmMaxK = 2048; // padded pow2 ceiling for one row's entries
constexpr int kAsmIdxBits = 11; // emission-index field width in the sort key
constexpr int kScanBlock = 1024;

__global__ void asm_sort_compact_kernel(const GpuTriplet* __restrict__ tr,
                                        const std::uint64_t* __restrict__ od_f,
                                        const int n_od,
                                        const std::uint64_t dim,
                                        const std::uint64_t diag_off,
                                        const int p2,
                                        unsigned* __restrict__ nnz,
                                        std::uint64_t* __restrict__ seg_col,
                                        double* __restrict__ seg_vre,
                                        double* __restrict__ seg_vim) {
  __shared__ std::uint64_t key[kAsmMaxK];
  __shared__ unsigned scan[kAsmBlock];
  const std::uint64_t r = blockIdx.x;
  const int K = (diag_off ? 1 : 0) + n_od;

  for (int i = threadIdx.x; i < p2; i += blockDim.x) {
    std::uint64_t k = ~0ull; // padding sorts last
    if (i < K) {
      std::uint64_t col;
      if (diag_off && i == 0)
        col = r;
      else {
        const int j = diag_off ? i - 1 : i;
        col = r ^ od_f[j];
      }
      k = (col << kAsmIdxBits) | static_cast<std::uint64_t>(i);
    }
    key[i] = k;
  }
  __syncthreads();

  for (int k = 2; k <= p2; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < p2; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i && ixj < p2) {
          const bool up = (i & k) == 0;
          if ((key[i] > key[ixj]) == up) {
            const std::uint64_t tmp = key[i];
            key[i] = key[ixj];
            key[ixj] = tmp;
          }
        }
      }
      __syncthreads();
    }

  // Inclusive scan of run-head flags via per-thread contiguous ranges:
  // local sums, one shared scan over the 256 partials, then a re-walk that
  // writes inclusive run indices. Head flags are recomputed from key[] on
  // the second walk, so no extra shared array is needed.
  const int lo = static_cast<int>((static_cast<std::uint64_t>(p2) *
                                   threadIdx.x) /
                                  blockDim.x);
  const int hi = static_cast<int>((static_cast<std::uint64_t>(p2) *
                                   (threadIdx.x + 1)) /
                                  blockDim.x);
  unsigned local = 0;
  for (int i = lo; i < hi; ++i)
    if (i < K && (i == 0 || (key[i] >> kAsmIdxBits) !=
                                (key[i - 1] >> kAsmIdxBits)))
      ++local;
  scan[threadIdx.x] = local;
  __syncthreads();
  for (int off = 1; off < kAsmBlock; off <<= 1) {
    const unsigned v =
        threadIdx.x >= static_cast<unsigned>(off) ? scan[threadIdx.x - off]
                                                  : 0u;
    __syncthreads();
    scan[threadIdx.x] += v;
    __syncthreads();
  }
  const unsigned prefix = scan[threadIdx.x] - local;
  if (threadIdx.x == 0)
    nnz[r] = scan[kAsmBlock - 1]; // total heads; flags are 0 past K
  unsigned running = prefix;
  for (int i = lo; i < hi; ++i) {
    if (i < K && (i == 0 || (key[i] >> kAsmIdxBits) !=
                                (key[i - 1] >> kAsmIdxBits))) {
      // Head thread sums its run sequentially in emission order — the same
      // FP64 addition sequence the CPU assemble applies.
      const std::uint64_t col = key[i] >> kAsmIdxBits;
      double re = 0.0, im = 0.0;
      for (int m = i; m < K; ++m) {
        if ((key[m] >> kAsmIdxBits) != col)
          break;
        const int idx =
            static_cast<int>(key[m] & ((1u << kAsmIdxBits) - 1u));
        const std::uint64_t slot =
            diag_off
                ? (idx == 0 ? r
                            : diag_off +
                                  static_cast<std::uint64_t>(idx - 1) * dim +
                                  r)
                : static_cast<std::uint64_t>(idx) * dim + r;
        if (m == i) {
          re = tr[slot].vre;
          im = tr[slot].vim;
        } else {
          re += tr[slot].vre;
          im += tr[slot].vim;
        }
      }
      if (re == 0.0)
        re = 0.0; // canonical(): no signed zeros in the output
      if (im == 0.0)
        im = 0.0;
      const std::uint64_t dst = r * static_cast<std::uint64_t>(K) + running;
      seg_col[dst] = col;
      seg_vre[dst] = re;
      seg_vim[dst] = im;
      ++running;
    }
  }
}

// row_ptr from per-row run counts: exact integer prefix sum, single block,
// chunked with a running carry.
__global__ void asm_rowptr_kernel(const unsigned* __restrict__ nnz,
                                  const std::uint64_t dim,
                                  std::uint64_t* __restrict__ row_ptr) {
  __shared__ std::uint64_t part[kScanBlock];
  __shared__ std::uint64_t s_carry;
  const int tid = threadIdx.x;
  if (tid == 0) {
    s_carry = 0;
    row_ptr[0] = 0;
  }
  __syncthreads();
  for (std::uint64_t base = 0; base < dim; base += kScanBlock) {
    const std::uint64_t i = base + tid;
    part[tid] = i < dim ? nnz[i] : 0;
    __syncthreads();
    for (int off = 1; off < kScanBlock; off <<= 1) {
      const std::uint64_t v = tid >= off ? part[tid - off] : 0;
      __syncthreads();
      part[tid] += v;
      __syncthreads();
    }
    if (i < dim)
      row_ptr[i + 1] = s_carry + part[tid];
    __syncthreads();
    if (tid == kScanBlock - 1)
      s_carry += part[tid];
    __syncthreads();
  }
}

__global__ void asm_copy_kernel(const std::uint64_t* __restrict__ seg_col,
                                const double* __restrict__ seg_vre,
                                const double* __restrict__ seg_vim,
                                const unsigned* __restrict__ nnz,
                                const std::uint64_t* __restrict__ row_ptr,
                                const int K, std::uint64_t* __restrict__ col,
                                double* __restrict__ vre,
                                double* __restrict__ vim) {
  const std::uint64_t r = blockIdx.x;
  const unsigned cnt = nnz[r];
  const std::uint64_t dst0 = row_ptr[r];
  const std::uint64_t src0 = r * static_cast<std::uint64_t>(K);
  for (unsigned t = threadIdx.x; t < cnt; t += blockDim.x) {
    col[dst0 + t] = seg_col[src0 + t];
    vre[dst0 + t] = seg_vre[src0 + t];
    vim[dst0 + t] = seg_vim[src0 + t];
  }
}

// Compacted device-resident ensemble. Upload once; emit many times
// (benchmarks time the kernels with inputs resident, KPM discipline).
struct Ensemble {
  int n = 0;
  std::uint64_t dim = 0;
  int n_diag = 0, n_od = 0;
  std::uint64_t diag_off = 0; // dim if any_diag else 0
  std::uint64_t n_tr = 0;     // total triplet slots
  std::uint64_t* d_diag_z = nullptr;
  double* d_diag_c = nullptr;
  std::uint64_t* d_od_f = nullptr;
  std::uint64_t* d_od_yz = nullptr;
  double* d_od_pre = nullptr;
  double* d_od_pim = nullptr;
  GpuTriplet* d_out = nullptr;

  Ensemble(const std::vector<PauliString>& strings, int n_) : n(n_) {
    validate(strings, n);
    dim = 1ull << n;
    std::vector<std::uint64_t> dz, of, oyz;
    std::vector<double> dc_, opre, opim;
    // phase via the same complex-multiply formula as the CPU path
    const auto cmul = [](cdouble a, cdouble b) {
      return cdouble(a.real() * b.real() - a.imag() * b.imag(),
                     a.real() * b.imag() + a.imag() * b.real());
    };
    static const cdouble kPowI[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    for (const PauliString& s : strings) {
      const std::uint64_t f = s.x | s.y;
      if (!f) {
        dz.push_back(s.z);
        dc_.push_back(s.coeff);
      } else {
        const cdouble phase =
            cmul(cdouble(s.coeff, 0.0), kPowI[__builtin_popcountll(s.y) & 3]);
        of.push_back(f);
        oyz.push_back(s.y | s.z);
        opre.push_back(phase.real());
        opim.push_back(phase.imag());
      }
    }
    n_diag = (int)dz.size();
    n_od = (int)of.size();
    diag_off = n_diag ? dim : 0;
    n_tr = diag_off + (std::uint64_t)n_od * dim;
    const auto up = [](auto*& dst, const auto& src) {
      using T = std::remove_const_t<
          std::remove_reference_t<decltype(*src.data())>>;
      if (src.empty())
        return;
      cuda_check(cudaMalloc(&dst, src.size() * sizeof(T)), "upload alloc");
      cuda_check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "upload");
    };
    up(d_diag_z, dz);
    up(d_diag_c, dc_);
    up(d_od_f, of);
    up(d_od_yz, oyz);
    up(d_od_pre, opre);
    up(d_od_pim, opim);
    cuda_check(cudaMalloc(&d_out, (n_tr ? n_tr : 1) * sizeof(GpuTriplet)),
               "triplet alloc");
  }
  Ensemble(const Ensemble&) = delete;
  Ensemble& operator=(const Ensemble&) = delete;
  ~Ensemble() {
    cudaFree(d_diag_z);
    cudaFree(d_diag_c);
    cudaFree(d_od_f);
    cudaFree(d_od_yz);
    cudaFree(d_od_pre);
    cudaFree(d_od_pim);
    cudaFree(d_out);
  }

  // Kernels only; inputs and output buffer resident.
  void emit_device() {
    const int T = 256;
    if (n_diag) {
      const std::uint64_t blocks = (dim + T - 1) / T;
      emit_diag_kernel<<<(unsigned)blocks, T>>>(d_diag_z, d_diag_c, n_diag,
                                                dim, d_out);
    }
    if (n_od) {
      const std::uint64_t total = (std::uint64_t)n_od * dim;
      const std::uint64_t blocks = (total + T - 1) / T;
      emit_offdiag_kernel<<<(unsigned)blocks, T>>>(d_od_f, d_od_yz, d_od_pre,
                                                   d_od_pim, n, dim, n_od,
                                                   diag_off, d_out);
    }
    cuda_check(cudaGetLastError(), "emit kernels");
    cuda_check(cudaDeviceSynchronize(), "emit kernels sync");
  }

  std::vector<Triplet> download() const {
    std::vector<Triplet> t(n_tr);
    cuda_check(cudaMemcpy(t.data(), d_out, n_tr * sizeof(GpuTriplet),
                          cudaMemcpyDeviceToHost),
               "triplet download");
    return t;
  }
};

// Device-resident CSR produced by Ensemble::assemble_device. Move-only.
struct GpuCsr {
  int n = 0;
  std::uint64_t dim = 0, nnz = 0;
  std::uint64_t* d_row_ptr = nullptr; // dim + 1 entries
  std::uint64_t* d_col = nullptr;
  double* d_vre = nullptr;
  double* d_vim = nullptr;

  GpuCsr() = default;
  GpuCsr(const GpuCsr&) = delete;
  GpuCsr& operator=(const GpuCsr&) = delete;
  GpuCsr(GpuCsr&& o) noexcept { *this = std::move(o); }
  GpuCsr& operator=(GpuCsr&& o) noexcept {
    n = o.n;
    dim = o.dim;
    nnz = o.nnz;
    d_row_ptr = o.d_row_ptr;
    d_col = o.d_col;
    d_vre = o.d_vre;
    d_vim = o.d_vim;
    o.d_row_ptr = o.d_col = nullptr;
    o.d_vre = o.d_vim = nullptr;
    o.nnz = 0;
    return *this;
  }
  ~GpuCsr() {
    cudaFree(d_row_ptr);
    cudaFree(d_col);
    cudaFree(d_vre);
    cudaFree(d_vim);
  }

  CsrMatrix download() const {
    CsrMatrix m;
    m.n = n;
    m.dim = dim;
    m.row_ptr.resize(dim + 1);
    cuda_check(cudaMemcpy(m.row_ptr.data(), d_row_ptr,
                          (dim + 1) * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost),
               "row_ptr download");
    m.col.resize(nnz);
    m.val.resize(nnz);
    std::vector<double> hre(nnz), him(nnz);
    if (nnz) {
      cuda_check(cudaMemcpy(m.col.data(), d_col, nnz * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost),
                 "col download");
      cuda_check(cudaMemcpy(hre.data(), d_vre, nnz * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "val download");
      cuda_check(cudaMemcpy(him.data(), d_vim, nnz * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "val download");
      for (std::uint64_t i = 0; i < nnz; ++i)
        m.val[i] = cdouble(hre[i], him[i]);
    }
    return m;
  }
};

// GPU CSR assembly on the emitted buffer; see the design note above. Call
// after emit_device(). Bitwise-identical to assemble(download(), n).
inline GpuCsr assemble_device(const Ensemble& e) {
  const int K = (e.n_diag ? 1 : 0) + e.n_od;
  if (K > kAsmMaxK)
    throw std::invalid_argument(
        "assemble_gpu: entries per row above the shared-tile ceiling");
  if (e.n > 30)
    throw std::invalid_argument("assemble_gpu: n above the launch bound");

  GpuCsr g;
  g.n = e.n;
  g.dim = e.dim;
  cuda_check(cudaMalloc(&g.d_row_ptr, (e.dim + 1) * sizeof(std::uint64_t)),
             "row_ptr alloc");
  if (K == 0) {
    cuda_check(cudaMemset(g.d_row_ptr, 0, (e.dim + 1) * sizeof(std::uint64_t)),
               "row_ptr memset");
    g.nnz = 0;
    return g;
  }

  int p2 = 1;
  while (p2 < K)
    p2 <<= 1;
  const std::uint64_t stage =
      e.dim * static_cast<std::uint64_t>(K); // staging slots
  unsigned* d_nnz = nullptr;
  std::uint64_t* d_seg_col = nullptr;
  double *d_seg_vre = nullptr, *d_seg_vim = nullptr;
  cuda_check(cudaMalloc(&d_nnz, e.dim * sizeof(unsigned)), "nnz alloc");
  cuda_check(cudaMalloc(&d_seg_col, stage * sizeof(std::uint64_t)),
             "staging alloc");
  cuda_check(cudaMalloc(&d_seg_vre, stage * sizeof(double)), "staging alloc");
  cuda_check(cudaMalloc(&d_seg_vim, stage * sizeof(double)), "staging alloc");

  asm_sort_compact_kernel<<<static_cast<unsigned>(e.dim), kAsmBlock>>>(
      e.d_out, e.d_od_f, e.n_od, e.dim, e.diag_off, p2, d_nnz, d_seg_col,
      d_seg_vre, d_seg_vim);
  cuda_check(cudaGetLastError(), "asm sort/compact");
  asm_rowptr_kernel<<<1, kScanBlock>>>(d_nnz, e.dim, g.d_row_ptr);
  cuda_check(cudaGetLastError(), "asm rowptr");
  cuda_check(cudaMemcpy(&g.nnz, g.d_row_ptr + e.dim, sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost),
             "nnz readback");

  if (g.nnz) {
    cuda_check(cudaMalloc(&g.d_col, g.nnz * sizeof(std::uint64_t)),
               "col alloc");
    cuda_check(cudaMalloc(&g.d_vre, g.nnz * sizeof(double)), "val alloc");
    cuda_check(cudaMalloc(&g.d_vim, g.nnz * sizeof(double)), "val alloc");
    asm_copy_kernel<<<static_cast<unsigned>(e.dim), kAsmBlock>>>(
        d_seg_col, d_seg_vre, d_seg_vim, d_nnz, g.d_row_ptr, K, g.d_col,
        g.d_vre, g.d_vim);
    cuda_check(cudaGetLastError(), "asm copy");
    cuda_check(cudaDeviceSynchronize(), "asm sync");
  }
  cudaFree(d_nnz);
  cudaFree(d_seg_col);
  cudaFree(d_seg_vre);
  cudaFree(d_seg_vim);
  return g;
}

// Full GPU build: emit + assemble on device, CSR to host. Bitwise identical
// to build_vectorized(strings, n).
inline CsrMatrix build_gpu_full(const std::vector<PauliString>& strings,
                                int n) {
  Ensemble e(strings, n);
  e.emit_device();
  GpuCsr g = assemble_device(e);
  return g.download();
}

// GPU triplet emission, host in/out. Emission order matches
// emit_vectorized exactly, so assemble(emit_gpu(...)) is bitwise identical
// to build_vectorized(...).
inline std::vector<Triplet> emit_gpu(const std::vector<PauliString>& strings,
                                     int n) {
  Ensemble e(strings, n);
  e.emit_device();
  return e.download();
}

inline CsrMatrix build_gpu(const std::vector<PauliString>& strings, int n) {
  std::vector<Triplet> t = emit_gpu(strings, n);
  return assemble(t, n);
}

} // namespace gpu
} // namespace ph
