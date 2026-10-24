// Physics gate: Hubbard chain via Jordan-Wigner Pauli strings (vectorized
// builder) vs an independent direct fermionic occupation-basis construction.
// Both matrices are diagonalized with Eigen's SelfAdjointEigenSolver and the
// sorted spectra compared; the matrices themselves are also compared entry by
// entry. At U = 0 the spectrum is additionally checked against the analytic
// tight-binding subset sums. Tolerance: 1e-12 * max(1, ||H||_F).

#include "hubbard.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail) {
  std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name.c_str(),
              detail.c_str());
  if (!ok)
    ++failures;
}

Eigen::MatrixXcd dense_of(const ph::CsrMatrix& m) {
  Eigen::MatrixXcd A(m.dim, m.dim);
  A.setZero();
  for (std::uint64_t r = 0; r < m.dim; ++r)
    for (std::uint64_t k = m.row_ptr[r]; k < m.row_ptr[r + 1]; ++k)
      A(r, m.col[k]) = m.val[k];
  return A;
}

double frobenius(const ph::CsrMatrix& m) {
  double acc = 0.0;
  for (const ph::cdouble& v : m.val)
    acc += std::norm(v);
  return std::sqrt(acc);
}

std::vector<double> spectrum(const Eigen::MatrixXcd& A) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("SelfAdjointEigenSolver failed");
  std::vector<double> ev(es.eigenvalues().size());
  for (int i = 0; i < es.eigenvalues().size(); ++i)
    ev[i] = es.eigenvalues()[i];
  return ev; // SelfAdjointEigenSolver returns ascending order
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double w = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    w = std::max(w, std::abs(a[i] - b[i]));
  return w;
}

void run_hubbard(int L, double t, double U, bool analytic) {
  char tag[64];
  std::snprintf(tag, sizeof tag, "L=%d t=%g U=%g", L, t, U);

  const std::vector<ph::PauliString> strings = hub::hubbard_strings(L, t, U);
  const ph::CsrMatrix hp = ph::build_vectorized(strings, 2 * L);
  const ph::CsrMatrix hd = hub::hubbard_direct(L, t, U);

  const Eigen::MatrixXcd Ap = dense_of(hp);
  const Eigen::MatrixXcd Ad = dense_of(hd);
  const double scale = std::max(1.0, frobenius(hp));
  const double tol = 1e-12 * scale;

  const double mdiff = (Ap - Ad).cwiseAbs().maxCoeff();
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "max|H_pauli - H_direct| = %.3e (tol %.3e, nnz %" PRIu64 ")",
                mdiff, 1e-13 * scale, hp.nnz());
  check(mdiff <= 1e-13 * scale, std::string(tag) + " matrix", buf);

  const std::vector<double> ep = spectrum(Ap);
  const std::vector<double> ed = spectrum(Ad);
  const double sdiff = max_diff(ep, ed);
  std::snprintf(buf, sizeof buf,
                "max|dE| = %.3e (tol %.3e, dim %" PRIu64 ")", sdiff, tol,
                hp.dim);
  check(sdiff <= tol, std::string(tag) + " spectrum", buf);

  if (analytic) {
    const std::vector<double> ea = hub::tb_chain_spectrum(L, t);
    const double adiff = max_diff(ep, ea);
    std::snprintf(buf, sizeof buf, "max|dE| vs subset sums = %.3e", adiff);
    check(adiff <= tol, std::string(tag) + " analytic", buf);
  }
}

} // namespace

int main() {
  try {
    run_hubbard(3, 1.0, 4.0, false);
    run_hubbard(4, 1.0, 8.0, false);
    run_hubbard(4, 1.1, 1.3, false); // non-dyadic: sums round differently
    run_hubbard(4, 1.0, 0.0, true); // tight-binding chain, analytic check
    run_hubbard(5, 1.0, 2.0, false);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_hubbard failed: %s\n", e.what());
    return 1;
  }
  std::printf("failures=%d\n", failures);
  return failures ? 1 : 0;
}
