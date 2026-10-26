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
