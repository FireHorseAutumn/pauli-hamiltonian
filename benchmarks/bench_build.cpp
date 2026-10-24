// Builder benchmark: naive vs vectorized full builds (triplet generation +
// CSR assembly), random Pauli-string ensembles, median/min/max over reps with
// raw per-rep timings preserved. Throughput reported as strings/second and
// final nonzero entries/second. JSON out with machine/toolchain metadata.
//
// Correctness precondition: each case first checks that the two builders
// produce bitwise-identical CSR; a mismatch aborts the benchmark.
//
// Absolute times vary with machine state; report medians and observed spreads.

#include "ph/pauli.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

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

// diag_heavy: each string is I/Z-only with probability 1/2 (physics
// Hamiltonians are diagonal-dominated); otherwise per-site uniform I/X/Y/Z.
std::vector<ph::PauliString> random_ensemble(std::uint64_t seed, int n,
                                             int count, bool diag_heavy) {
  Rng rng(seed);
  std::vector<ph::PauliString> ss(count);
  for (auto& s : ss) {
    const bool diagonal = diag_heavy && (rng.next() & 1);
    for (int q = 0; q < n; ++q) {
      const double u = rng.uniform(0.0, 1.0);
      if (u < 0.25)
        continue;
      const std::uint64_t bit = 1ull << q;
      if (diagonal) {
        s.z |= bit;
        continue;
      }
      switch (rng.next() % 3) {
      case 0: s.x |= bit; break;
      case 1: s.y |= bit; break;
      default: s.z |= bit; break;
      }
    }
    s.coeff = rng.uniform(-1.0, 1.0);
  }
  return ss;
}

struct BenchResult {
  double median_s = 0, min_s = 0, max_s = 0;
  double emit_median_s = 0, assemble_median_s = 0;
  std::vector<double> reps_s;
};

template <typename EmitFn>
BenchResult bench_one(EmitFn emit, const std::vector<ph::PauliString>& ss,
                      int n, int reps) {
  BenchResult r;
  std::vector<double> emit_s, assemble_s;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ph::Triplet> t = emit(ss, n);
    const auto t1 = std::chrono::steady_clock::now();
    const ph::CsrMatrix m = ph::assemble(t, n);
    const auto t2 = std::chrono::steady_clock::now();
    r.reps_s.push_back(std::chrono::duration<double>(t2 - t0).count());
    emit_s.push_back(std::chrono::duration<double>(t1 - t0).count());
    assemble_s.push_back(std::chrono::duration<double>(t2 - t1).count());
  }
  const auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  r.median_s = med(r.reps_s);
  r.min_s = *std::min_element(r.reps_s.begin(), r.reps_s.end());
  r.max_s = *std::max_element(r.reps_s.begin(), r.reps_s.end());
  r.emit_median_s = med(emit_s);
  r.assemble_median_s = med(assemble_s);
  return r;
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    out.push_back(std::stoi(s.substr(pos, comma - pos)));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

std::string timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

std::string cpu_model() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("model name", 0) == 0)
      return line.substr(line.find(':') + 2);
  return "unknown";
}

std::string cpu_mhz() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("cpu MHz", 0) == 0)
      return line.substr(line.find(':') + 2);
  return "unknown";
}

void json_result(std::ofstream& f, const BenchResult& r, std::uint64_t strings,
                 std::uint64_t nnz, int indent) {
  const std::string pad(indent, ' ');
  f << pad << "{\n"
    << pad << "  \"median_s\": " << r.median_s << ",\n"
    << pad << "  \"min_s\": " << r.min_s << ",\n"
    << pad << "  \"max_s\": " << r.max_s << ",\n"
    << pad << "  \"emit_median_s\": " << r.emit_median_s << ",\n"
    << pad << "  \"assemble_median_s\": " << r.assemble_median_s << ",\n"
    << pad << "  \"strings_per_s\": " << strings / r.median_s << ",\n"
    << pad << "  \"nnz_per_s\": " << nnz / r.median_s << ",\n"
    << pad << "  \"reps_s\": [";
  for (size_t i = 0; i < r.reps_s.size(); ++i)
    f << (i ? ", " : "") << r.reps_s[i];
  f << "]\n" << pad << "}";
}

struct CaseEntry {
  const char* ensemble;
  int n, strings;
  std::uint64_t dim, nnz;
  BenchResult naive, vec;
};

} // namespace

int main(int argc, char** argv) {
  std::vector<int> ns = {8, 10, 12, 14};
  std::vector<int> string_counts = {64, 256, 1024};
  int reps = 9;
  std::uint64_t seed = 0x5eedb00full;
  std::string out_path = std::string(PH_DATA_DIR) + "/pauli_build_bench.json";
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::strncmp(argv[i], "--n=", 4) == 0)
        ns = parse_list(argv[i] + 4);
      else if (std::strncmp(argv[i], "--strings=", 10) == 0)
        string_counts = parse_list(argv[i] + 10);
      else if (std::strncmp(argv[i], "--reps=", 7) == 0)
        reps = std::stoi(argv[i] + 7);
      else if (std::strncmp(argv[i], "--seed=", 7) == 0)
        seed = std::stoull(argv[i] + 7, nullptr, 0);
      else if (std::strncmp(argv[i], "--out=", 6) == 0)
        out_path = argv[i] + 6;
      else
        throw std::invalid_argument(std::string("unknown argument: ") +
                                    argv[i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "bench_build: %s\n", e.what());
      std::fprintf(stderr,
                   "usage: bench_build [--n=8,10,12,14] "
                   "[--strings=64,256,1024] [--reps=9] [--seed=0x...] "
                   "[--out=PATH]\n");
      return 1;
    }
  }

  try {
    std::printf("cpu: %s (%s MHz at bench start), hw threads %u\n",
                cpu_model().c_str(), cpu_mhz().c_str(),
                std::thread::hardware_concurrency());
    std::printf("median of %d reps, full build = emit + assemble\n", reps);
    std::printf("ensembles: random = per-site uniform I/X/Y/Z, p(I)=0.25; "
                "diag-heavy = half the strings I/Z-only\n\n");
    std::printf("%10s %3s %6s %7s | %9s %9s %6s | %10s %10s | %11s %11s\n",
                "ensemble", "n", "dim", "strings", "naive ms", "vec ms",
                "speed", "naive S/s", "vec S/s", "naive nnz/s", "vec nnz/s");

    std::vector<CaseEntry> entries;
    for (const bool dh : {false, true})
      for (const int n : ns)
        for (const int S : string_counts) {
          const std::vector<ph::PauliString> ss =
              random_ensemble(seed + n * 7919 + S, n, S, dh);
          const ph::CsrMatrix a = ph::build_naive(ss, n);
          const ph::CsrMatrix b = ph::build_vectorized(ss, n);
          if (!ph::bitwise_equal(a, b))
            throw std::runtime_error("bitwise mismatch at n=" +
                                     std::to_string(n) + " strings=" +
                                     std::to_string(S));
          CaseEntry e{dh ? "diag-heavy" : "random", n, S, b.dim, b.nnz(),
                      bench_one(ph::emit_naive, ss, n, reps),
                      bench_one(ph::emit_vectorized, ss, n, reps)};
          entries.push_back(e);
          std::printf(
              "%10s %3d %6" PRIu64 " %7d | %9.3f %9.3f %5.2fx | %10.3g "
              "%10.3g | %11.4g %11.4g\n",
              e.ensemble, n, e.dim, S, e.naive.median_s * 1e3,
              e.vec.median_s * 1e3, e.naive.median_s / e.vec.median_s,
              S / e.naive.median_s, S / e.vec.median_s,
              e.nnz / e.naive.median_s, e.nnz / e.vec.median_s);
          std::printf("            spread ms [min,max]: naive [%.3f,%.3f] "
                      "vec [%.3f,%.3f]\n",
                      e.naive.min_s * 1e3, e.naive.max_s * 1e3,
                      e.vec.min_s * 1e3, e.vec.max_s * 1e3);
        }

    std::ofstream f(out_path);
    if (!f)
      throw std::runtime_error("cannot open " + out_path);
    f.precision(17);
    f << "{\n"
      << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
      << "  \"seed\": \"" << seed << "\",\n"
      << "  \"cpu\": \"" << cpu_model() << "\",\n"
      << "  \"cpu_mhz_at_start\": \"" << cpu_mhz() << "\",\n"
      << "  \"hw_threads\": " << std::thread::hardware_concurrency() << ",\n"
      << "  \"build_type\": \"" << PH_BUILD_TYPE << "\",\n"
      << "  \"cxx_flags\": \"" << PH_CXX_FLAGS << "\",\n"
      << "  \"compiler\": \"" << __VERSION__ << "\",\n"
      << "  \"n_reps\": " << reps << ",\n"
      << "  \"cases\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
      const CaseEntry& e = entries[i];
      f << "    {\"ensemble\": \"" << e.ensemble << "\", \"n\": " << e.n
        << ", \"dim\": " << e.dim << ", \"strings\": " << e.strings
        << ", \"nnz\": " << e.nnz << ", \"bitwise_equal\": true,\n"
        << "     \"speedup\": " << e.naive.median_s / e.vec.median_s << ",\n"
        << "     \"naive\": ";
      json_result(f, e.naive, e.strings, e.nnz, 16);
      f << ",\n     \"vectorized\": ";
      json_result(f, e.vec, e.strings, e.nnz, 16);
      f << "}" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    if (!f)
      throw std::runtime_error("write failed: " + out_path);
    std::printf("\nJSON -> %s\n", out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_build failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
