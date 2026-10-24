// The gate: naive and vectorized builders must produce bitwise-identical CSR
// (same row pointers, same column order, memcmp-equal values) across a
// randomized ensemble plus structured cases. Every case also requires bitwise
// Hermiticity of the vectorized output. Any mismatch exits nonzero.
// No timing happens anywhere until this passes.

#include "ph/pauli.hpp"

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;
std::uint64_t cases_run = 0, strings_run = 0, nnz_run = 0;

void fail(const std::string& what) {
  std::printf("[FAIL] %s\n", what.c_str());
  ++failures;
}

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    s += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double uniform(double lo, double hi) {
    return lo + (hi - lo) * (next() >> 11) * (1.0 / 9007199254740992.0);
  }
};

ph::PauliString random_string(Rng& rng, int n, double p_identity) {
  ph::PauliString s;
  for (int q = 0; q < n; ++q) {
    const double u = rng.uniform(0.0, 1.0);
    if (u < p_identity)
      continue;
    const std::uint64_t bit = 1ull << q;
    switch (rng.next() % 3) {
    case 0: s.x |= bit; break;
    case 1: s.y |= bit; break;
    default: s.z |= bit; break;
    }
  }
  s.coeff = rng.uniform(-2.0, 2.0);
  return s;
}

void run_case(const char* name, const std::vector<ph::PauliString>& strings,
              int n) {
  const ph::CsrMatrix a = ph::build_naive(strings, n);
  const ph::CsrMatrix b = ph::build_vectorized(strings, n);
  ++cases_run;
  strings_run += strings.size();
  nnz_run += b.nnz();
  if (!ph::bitwise_equal(a, b)) {
    std::uint64_t r = 0, c = 0;
    ph::cdouble va, vb;
    ph::first_mismatch(a, b, r, c, va, vb);
    char buf[256];
    if (r == ~0ull)
      std::snprintf(buf, sizeof buf,
                    "%s: structure differs (nnz naive=%" PRIu64
                    " vec=%" PRIu64 ")",
                    name, a.nnz(), b.nnz());
    else
      std::snprintf(buf, sizeof buf,
                    "%s: entry (%" PRIu64 ",%" PRIu64 ") naive=(%.17g,%.17g)"
                    " vec=(%.17g,%.17g)",
                    name, r, c, va.real(), va.imag(), vb.real(), vb.imag());
    fail(buf);
    return;
  }
  if (!ph::is_hermitian_bitwise(b))
    fail(std::string(name) + ": not bitwise Hermitian");
}

} // namespace

int main() {
  // Randomized ensemble: n = 1..10, 20 seeds each, random string counts and
  // coefficients, mixed identity density.
  for (int n = 1; n <= 10; ++n)
    for (int seed = 0; seed < 20; ++seed) {
      Rng rng(0x5eed0000ull + n * 1000 + seed);
      const int count = static_cast<int>(rng.next() % (4 * n + 1));
      const double pI = rng.uniform(0.15, 0.6);
      std::vector<ph::PauliString> ss(count);
      for (auto& s : ss)
        s = random_string(rng, n, pI);
      run_case("random", ss, n);
    }

  // Exhaustive local coverage: every string at n = 2 (16 strings) and
  // n = 3 (64 strings), each as a singleton with a fixed coefficient.
  for (int n = 2; n <= 3; ++n) {
    const int combos = 1 << (2 * n);
    for (int c = 0; c < combos; ++c) {
      ph::PauliString s;
      s.coeff = 0.75;
      for (int q = 0; q < n; ++q) {
        switch ((c >> (2 * q)) & 3) {
        case 1: s.x |= 1ull << q; break;
        case 2: s.y |= 1ull << q; break;
        case 3: s.z |= 1ull << q; break;
        default: break;
        }
      }
      run_case("exhaustive-singleton", {s}, n);
    }
  }

  // All-diagonal ensemble (I/Z only).
  {
    Rng rng(0xd1a9ull);
    for (int trial = 0; trial < 10; ++trial) {
      std::vector<ph::PauliString> ss(25);
      for (auto& s : ss) {
        for (int q = 0; q < 6; ++q)
          if (rng.next() & 1)
            s.z |= 1ull << q;
        s.coeff = rng.uniform(-2.0, 2.0);
      }
      run_case("all-diagonal", ss, 6);
    }
  }

  // Duplicate strings that accumulate, including exact cancellation.
  {
    ph::PauliString s;
    s.x = 0b0101;
    s.y = 0b0010;
    s.z = 0b1000;
    run_case("duplicates-accumulate", {{s.x, s.y, s.z, 0.5},
                                       {s.x, s.y, s.z, 0.25},
                                       {s.x, s.y, s.z, 1.0}},
             4);
    run_case("duplicates-cancel", {{s.x, s.y, s.z, 0.5},
                                   {s.x, s.y, s.z, -0.5},
                                   {s.x, s.y, s.z, 1.0}},
             4);
  }

  // Identity-only and zero-coefficient strings.
  run_case("identity-only", {{0, 0, 0, 1.5}, {0, 0, 0, -0.5}}, 5);
  run_case("zero-coeff", {{0, 0, 0, 0.0}, {0b1, 0, 0, 0.0}, {0b1, 0, 0, 1.0}},
           3);

  // Empty Hamiltonian.
  run_case("empty", {}, 4);

  // Full-fill: enough random X/Y-heavy strings to densify a 64 x 64 matrix.
  {
    Rng rng(0xf111ull);
    std::vector<ph::PauliString> ss(300);
    for (auto& s : ss)
      s = random_string(rng, 6, 0.05);
    run_case("full-fill", ss, 6);
  }

  // Larger randomized stress at n = 10.
  {
    Rng rng(0xb19ull);
    for (int trial = 0; trial < 10; ++trial) {
      std::vector<ph::PauliString> ss(40);
      for (auto& s : ss)
        s = random_string(rng, 10, 0.35);
      run_case("n10-stress", ss, 10);
    }
  }

  std::printf("cases=%" PRIu64 " strings=%" PRIu64 " nnz=%" PRIu64
              " failures=%d\n",
              cases_run, strings_run, nnz_run, failures);
  if (failures == 0)
    std::printf("[pass] naive == vectorized bitwise on all cases\n");
  return failures ? 1 : 0;
}
