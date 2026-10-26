// The gate: naive and vectorized builders must produce bitwise-identical CSR
// (same row pointers, same column order, memcmp-equal values) across the
// shared case suite (tests/cases.hpp). Every case also requires bitwise
// Hermiticity of the vectorized output. Any mismatch exits nonzero.
// No timing happens anywhere until this passes.

#include "cases.hpp"

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
  for (const phtest::Case& c : phtest::make_cases())
    run_case(c.name.c_str(), c.strings, c.n);

  std::printf("cases=%" PRIu64 " strings=%" PRIu64 " nnz=%" PRIu64
              " failures=%d\n",
              cases_run, strings_run, nnz_run, failures);
  if (failures == 0)
    std::printf("[pass] naive == vectorized bitwise on all cases\n");
  return failures ? 1 : 0;
}
