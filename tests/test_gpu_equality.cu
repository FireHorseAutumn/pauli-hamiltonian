// GPU gate: build_gpu must produce CSR bitwise-identical to both CPU
// builders across the shared case suite (tests/cases.hpp) — same row
// pointers, same column order, memcmp-equal values. Every case also
// requires bitwise Hermiticity of the GPU output, and the full suite is run
// twice to check the kernels are bitwise deterministic run to run.
// Any mismatch exits nonzero. No timing happens anywhere until this passes.

#include "cases.hpp"
#include "ph/pauli_gpu.cuh"

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

void check_pair(const char* name, const char* tag, const ph::CsrMatrix& a,
                const ph::CsrMatrix& b) {
  if (ph::bitwise_equal(a, b))
    return;
  std::uint64_t r = 0, c = 0;
  ph::cdouble va, vb;
  ph::first_mismatch(a, b, r, c, va, vb);
  char buf[256];
  if (r == ~0ull)
    std::snprintf(buf, sizeof buf,
                  "%s: %s structure differs (nnz %" PRIu64 " vs %" PRIu64 ")",
                  name, tag, a.nnz(), b.nnz());
  else
    std::snprintf(buf, sizeof buf,
                  "%s: %s entry (%" PRIu64 ",%" PRIu64 ") (%.17g,%.17g) vs "
                  "(%.17g,%.17g)",
                  name, tag, r, c, va.real(), va.imag(), vb.real(), vb.imag());
  fail(buf);
}

void run_case(const char* name, const std::vector<ph::PauliString>& strings,
              int n) {
  const ph::CsrMatrix a = ph::build_naive(strings, n);
  const ph::CsrMatrix b = ph::build_vectorized(strings, n);
  const ph::CsrMatrix g = ph::gpu::build_gpu(strings, n);
  ++cases_run;
  strings_run += strings.size();
  nnz_run += b.nnz();
  check_pair(name, "gpu-vs-naive", a, g);
  check_pair(name, "gpu-vs-vectorized", b, g);
  if (!ph::is_hermitian_bitwise(g))
    fail(std::string(name) + ": gpu output not bitwise Hermitian");
}

} // namespace

int main() {
  const std::vector<phtest::Case> cases = phtest::make_cases();
  for (int pass = 0; pass < 2; ++pass)
    for (const phtest::Case& c : cases)
      run_case(c.name.c_str(), c.strings, c.n);
  cases_run /= 2; // second pass was the determinism rerun
  strings_run /= 2;
  nnz_run /= 2;

  std::printf("cases=%" PRIu64 " (x2 passes) strings=%" PRIu64
              " nnz=%" PRIu64 " failures=%d\n",
              cases_run, strings_run, nnz_run, failures);
  if (failures == 0)
    std::printf("[pass] naive == vectorized == gpu bitwise on all cases, "
                "deterministic across passes\n");
  return failures ? 1 : 0;
}
