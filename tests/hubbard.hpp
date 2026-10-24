// 1D Hubbard chain, open boundaries, as Pauli strings (Jordan-Wigner) and as
// a direct fermionic occupation-basis construction. The two paths share no
// matrix-element code; agreement of the spectra is the physics gate.
//
// Modes: m = 2*site + spin, spin 0 = up, 1 = down; n = 2L qubits.
//   H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_iup n_idn
// JW gives, for a bond between modes m < m':
//   c+_m c_m' + h.c. = (1/2)(X_m X_m' + Y_m Y_m') prod_{m<k<m'} Z_k
// and n_iup n_idn = (1/4)(I - Z_2i - Z_2i+1 + Z_2i Z_2i+1).

#pragma once

#include "ph/pauli.hpp"

#include <algorithm>
#include <numbers>

namespace hub {

inline std::vector<ph::PauliString> hubbard_strings(int L, double t,
                                                    double U) {
  std::vector<ph::PauliString> out;
  for (int i = 0; i + 1 < L; ++i)
    for (int s = 0; s < 2; ++s) {
      const int m = 2 * i + s, mp = m + 2;
      ph::PauliString xx, yy;
      xx.x = (1ull << m) | (1ull << mp);
      yy.y = xx.x;
      for (int k = m + 1; k < mp; ++k) {
        xx.z |= 1ull << k;
        yy.z |= 1ull << k;
      }
      xx.coeff = yy.coeff = -t / 2;
      out.push_back(xx);
      out.push_back(yy);
    }
  for (int i = 0; i < L; ++i) {
    const std::uint64_t a = 1ull << (2 * i), b = 1ull << (2 * i + 1);
    out.push_back({0, 0, 0, U / 4});
    out.push_back({0, 0, a, -U / 4});
    out.push_back({0, 0, b, -U / 4});
    out.push_back({0, 0, a | b, U / 4});
  }
  return out;
}

// Direct fermionic builder in the full 4^L Fock space. Hopping matrix
// element between modes m < m': amplitude -t * (-1)^(occupied modes strictly
// between m and m') — the fermion anticommutation sign, evaluated on the
// occupation bitmask directly.
inline std::vector<ph::Triplet> hubbard_direct_triplets(int L, double t,
                                                        double U) {
  const int modes = 2 * L;
  const std::uint64_t dim = 1ull << modes;
  std::vector<ph::Triplet> out;
  for (std::uint64_t b = 0; b < dim; ++b) {
    double diag = 0.0;
    for (int i = 0; i < L; ++i)
      if ((b & (1ull << (2 * i))) && (b & (1ull << (2 * i + 1))))
        diag += U;
    out.push_back({b, b, {diag, 0.0}});
    for (int i = 0; i + 1 < L; ++i)
      for (int s = 0; s < 2; ++s) {
        const int m = 2 * i + s, mp = m + 2;
        const bool src = (b >> m) & 1, dst = (b >> mp) & 1;
        if (src == dst)
          continue;
        const std::uint64_t between = b & ((1ull << mp) - (2ull << m));
        const double sign = ph::parity64(between) ? -1.0 : 1.0;
        const std::uint64_t b2 = (b ^ (1ull << m)) ^ (1ull << mp);
        out.push_back({b, b2, {-t * sign, 0.0}});
      }
  }
  return out;
}

inline ph::CsrMatrix hubbard_direct(int L, double t, double U) {
  std::vector<ph::Triplet> t3 = hubbard_direct_triplets(L, t, U);
  return ph::assemble(t3, 2 * L);
}

// Analytic U=0 spectrum: two independent spin copies of the open
// tight-binding chain, single-particle levels eps_k = -2t cos(k pi/(L+1)),
// k = 1..L; many-body energies are subset sums over 2L modes.
inline std::vector<double> tb_chain_spectrum(int L, double t) {
  std::vector<double> eps;
  for (int k = 1; k <= L; ++k)
    eps.push_back(-2.0 * t * std::cos(k * std::numbers::pi / (L + 1)));
  const std::uint64_t dim = 1ull << (2 * L);
  std::vector<double> spec(dim, 0.0);
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < L; ++k) {
      const std::uint64_t bit = 1ull << (2 * k + s);
      for (std::uint64_t b = 0; b < dim; ++b)
        if (b & bit)
          spec[b] += eps[k];
    }
  std::sort(spec.begin(), spec.end());
  return spec;
}

} // namespace hub
