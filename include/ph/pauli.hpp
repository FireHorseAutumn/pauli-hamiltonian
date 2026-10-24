// Sparse Hamiltonian construction from a sum of Pauli strings.
//
// Basis states are n-bit masks; bit q is qubit q. A string is three disjoint
// masks (x, y, z) plus a real coefficient. Single-qubit action:
//   I|b> = |b>          Z|b> = (-1)^b |b>
//   X|b> = |1-b>        Y|0> = i|1>,  Y|1> = -i|0>
// so a string maps |b> to |b ^ (x|y)> with amplitude
//   coeff * i^{popcount(y)} * (-1)^{popcount((y|z) & b)}.
// Every factor is a sign flip or a component swap, hence exact in FP64; the
// naive and vectorized paths differ only in how they arrive at it.

#pragma once

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ph {

using cdouble = std::complex<double>;

struct PauliString {
  std::uint64_t x = 0, y = 0, z = 0;
  double coeff = 0.0;
};

struct CsrMatrix {
  int n = 0; // matrix dimension is (1 << n)
  std::uint64_t dim = 0;
  std::vector<std::uint64_t> row_ptr; // dim + 1 entries
  std::vector<std::uint64_t> col;
  std::vector<cdouble> val;
  std::uint64_t nnz() const { return col.size(); }
};

inline void validate(const std::vector<PauliString>& strings, int n) {
  if (n < 1 || n > 62)
    throw std::invalid_argument("pauli: n out of range");
  const std::uint64_t top = (1ull << n) - 1;
  for (const PauliString& s : strings) {
    if ((s.x & s.y) || (s.x & s.z) || (s.y & s.z))
      throw std::invalid_argument("pauli: overlapping masks in one string");
    if ((s.x | s.y | s.z) & ~top)
      throw std::invalid_argument("pauli: mask bit above qubit count");
  }
}

inline int parity64(std::uint64_t v) { return __builtin_parityll(v); }

// Signed zeros differ between evaluation orders; canonicalize so bitwise
// comparison of accumulated values is meaningful. Zero entries are kept.
inline cdouble canonical(cdouble v) {
  double re = v.real(), im = v.imag();
  if (re == 0.0) re = 0.0;
  if (im == 0.0) im = 0.0;
  return {re, im};
}

struct Triplet {
  std::uint64_t row, col;
  cdouble val;
};

// Shared assembly: stable sort by (row, col), accumulate equal keys in
// emission order. Both builders feed this, so accumulation order — and
// therefore the exact bits of every sum — is a property of the triplet
// stream, not of builder internals.
inline CsrMatrix assemble(std::vector<Triplet>& t, int n) {
  const std::uint64_t dim = 1ull << n;
  std::stable_sort(t.begin(), t.end(), [](const Triplet& a, const Triplet& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });
  CsrMatrix m;
  m.n = n;
  m.dim = dim;
  m.row_ptr.assign(dim + 1, 0);
  m.col.reserve(t.size());
  m.val.reserve(t.size());
  for (size_t i = 0; i < t.size();) {
    size_t j = i + 1;
    cdouble acc = t[i].val;
    while (j < t.size() && t[j].row == t[i].row && t[j].col == t[i].col) {
      acc += t[j].val;
      ++j;
    }
    ++m.row_ptr[t[i].row + 1];
    m.col.push_back(t[i].col);
    m.val.push_back(canonical(acc));
    i = j;
  }
  for (std::uint64_t r = 0; r < dim; ++r)
    m.row_ptr[r + 1] += m.row_ptr[r];
  return m;
}

// Baseline and oracle: per string, per basis state, walk qubits 0..n-1 and
// apply each factor individually. O(n) work per (string, basis) pair.
inline std::vector<Triplet> emit_naive(const std::vector<PauliString>& strings,
                                       int n) {
  validate(strings, n);
  const std::uint64_t dim = 1ull << n;
  std::vector<Triplet> out;
  out.reserve(strings.size() * dim);
  for (const PauliString& s : strings) {
    for (std::uint64_t b = 0; b < dim; ++b) {
      std::uint64_t col = b;
      cdouble acc(s.coeff, 0.0);
      for (int q = 0; q < n; ++q) {
        const std::uint64_t bit = 1ull << q;
        if (s.x & bit) {
          col ^= bit;
        } else if (s.y & bit) {
          col ^= bit;
          acc *= cdouble(0.0, (b & bit) ? -1.0 : 1.0);
        } else if (s.z & bit) {
          if (b & bit)
            acc *= cdouble(-1.0, 0.0);
        }
      }
      out.push_back({b, col, acc});
    }
  }
  return out;
}

// Vectorized: O(1) work per (string, basis) pair, no per-qubit loop.
// Diagonal strings (x|y == 0) accumulate into a dense diagonal in string
// order — the same order the shared assembly would apply — then emit one
// triplet per row. Off-diagonal strings emit per (string, basis): column
// flip, one parity for the sign, one table lookup for i^{popcount(y) mod 4}.
inline std::vector<Triplet> emit_vectorized(
    const std::vector<PauliString>& strings, int n) {
  validate(strings, n);
  const std::uint64_t dim = 1ull << n;
  static const cdouble kPowI[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  std::vector<Triplet> out;
  out.reserve(strings.size() * dim);

  std::vector<cdouble> diag(dim, cdouble(0.0, 0.0));
  bool any_diag = false;
  for (const PauliString& s : strings) {
    if (s.x | s.y)
      continue;
    any_diag = true;
    for (std::uint64_t b = 0; b < dim; ++b)
      diag[b] += parity64(s.z & b) ? cdouble(-s.coeff, 0.0)
                                   : cdouble(s.coeff, 0.0);
  }
  if (any_diag)
    for (std::uint64_t b = 0; b < dim; ++b)
      out.push_back({b, b, diag[b]});

  for (const PauliString& s : strings) {
    const std::uint64_t f = s.x | s.y;
    if (!f)
      continue;
    const cdouble phase =
        cdouble(s.coeff, 0.0) * kPowI[__builtin_popcountll(s.y) & 3];
    const std::uint64_t yz = s.y | s.z;
    for (std::uint64_t b = 0; b < dim; ++b)
      out.push_back({b, b ^ f, parity64(yz & b) ? -phase : phase});
  }
  return out;
}

inline CsrMatrix build_naive(const std::vector<PauliString>& strings, int n) {
  std::vector<Triplet> t = emit_naive(strings, n);
  return assemble(t, n);
}

inline CsrMatrix build_vectorized(const std::vector<PauliString>& strings,
                                  int n) {
  std::vector<Triplet> t = emit_vectorized(strings, n);
  return assemble(t, n);
}

// Bitwise equality: same structure, same column order, same value bits
// (memcmp on the complex array; zeros are canonicalized at assembly).
inline bool bitwise_equal(const CsrMatrix& a, const CsrMatrix& b) {
  if (a.n != b.n || a.dim != b.dim)
    return false;
  if (a.row_ptr != b.row_ptr || a.col != b.col)
    return false;
  if (a.val.size() != b.val.size())
    return false;
  return a.val.empty() ||
         std::memcmp(a.val.data(), b.val.data(),
                     a.val.size() * sizeof(cdouble)) == 0;
}

// First differing entry, for diagnostics. Returns false when equal.
inline bool first_mismatch(const CsrMatrix& a, const CsrMatrix& b,
                           std::uint64_t& row, std::uint64_t& col,
                           cdouble& va, cdouble& vb) {
  if (a.dim != b.dim || a.row_ptr != b.row_ptr || a.col != b.col) {
    row = col = ~0ull;
    return true;
  }
  for (std::uint64_t r = 0; r < a.dim; ++r)
    for (std::uint64_t k = a.row_ptr[r]; k < a.row_ptr[r + 1]; ++k)
      if (std::memcmp(&a.val[k], &b.val[k], sizeof(cdouble)) != 0) {
        row = r;
        col = a.col[k];
        va = a.val[k];
        vb = b.val[k];
        return true;
      }
  return false;
}

// Bitwise Hermiticity: H[c][r] == conj(H[r][c]) exactly.
inline bool is_hermitian_bitwise(const CsrMatrix& m) {
  for (std::uint64_t r = 0; r < m.dim; ++r)
    for (std::uint64_t k = m.row_ptr[r]; k < m.row_ptr[r + 1]; ++k) {
      const std::uint64_t c = m.col[k];
      const auto lo = m.col.begin() + m.row_ptr[c];
      const auto hi = m.col.begin() + m.row_ptr[c + 1];
      const auto it = std::lower_bound(lo, hi, r);
      if (it == hi || *it != r)
        return false;
      const cdouble w = m.val[it - m.col.begin()];
      const cdouble cv = canonical(std::conj(m.val[k]));
      if (std::memcmp(&w, &cv, sizeof(cdouble)) != 0)
        return false;
    }
  return true;
}

} // namespace ph
