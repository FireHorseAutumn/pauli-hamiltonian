// GPU builder benchmark: CPU vectorized emit vs the GPU scatter kernels,
// same ensembles as bench_build. Three timings per case: CPU vectorized
// emit (wall), GPU kernels only (CUDA events, inputs resident), and GPU
// end-to-end (wall: host-side string compaction, upload incl. allocations,
// kernels, triplet download). Assembly is the shared CPU assemble() for all
// paths and is excluded; it is identical work everywhere.
//
// Correctness precondition: each case first checks build_gpu produces CSR
// bitwise-identical to build_vectorized; a mismatch aborts the benchmark.
//
// Absolute times vary with machine state; report medians and observed
// spreads, raw per-rep timings preserved in the JSON.

#include "ph/pauli_gpu.cuh"

#include <algorithm>
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

// Same generator as bench_build: diag_heavy gives each string a 1/2 chance
// of being I/Z-only; otherwise per-site uniform I/X/Y/Z with p(I) = 0.25.
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

struct Stats {
  std::vector<double> reps;
  double median = 0, min = 0, max = 0;
  void finish() {
    std::sort(reps.begin(), reps.end());
    median = reps[reps.size() / 2];
    min = reps.front();
    max = reps.back();
  }
};

Stats bench_cpu_emit(const std::vector<ph::PauliString>& ss, int n, int reps) {
  Stats r;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ph::Triplet> t = ph::emit_vectorized(ss, n);
    const auto t1 = std::chrono::steady_clock::now();
    r.reps.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  r.finish();
  return r;
}

Stats bench_gpu_kernel(ph::gpu::Ensemble& e, int reps) {
  Stats r;
  cudaEvent_t ev0, ev1;
  ph::gpu::cuda_check(cudaEventCreate(&ev0), "event create");
  ph::gpu::cuda_check(cudaEventCreate(&ev1), "event create");
  for (int i = 0; i < reps; ++i) {
    ph::gpu::cuda_check(cudaDeviceSynchronize(), "pre-rep sync");
    ph::gpu::cuda_check(cudaEventRecord(ev0), "event record");
    e.emit_device();
    ph::gpu::cuda_check(cudaEventRecord(ev1), "event record");
    ph::gpu::cuda_check(cudaDeviceSynchronize(), "post-rep sync");
    float ms = 0.0f;
    ph::gpu::cuda_check(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
    r.reps.push_back((double)ms * 1e-3);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  r.finish();
  return r;
}

// End-to-end: host strings -> compact, allocate, upload, emit, download.
Stats bench_gpu_e2e(const std::vector<ph::PauliString>& ss, int n, int reps) {
  Stats r;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<ph::Triplet> t = ph::gpu::emit_gpu(ss, n);
    const auto t1 = std::chrono::steady_clock::now();
    r.reps.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  r.finish();
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

void json_stats(std::ofstream& f, const Stats& r, int indent) {
  const std::string pad(indent, ' ');
  f << pad << "{\n"
    << pad << "  \"median_s\": " << r.median << ",\n"
    << pad << "  \"min_s\": " << r.min << ",\n"
    << pad << "  \"max_s\": " << r.max << ",\n"
    << pad << "  \"reps_s\": [";
  for (size_t i = 0; i < r.reps.size(); ++i)
    f << (i ? ", " : "") << r.reps[i];
  f << "]\n" << pad << "}";
}

struct CaseEntry {
  const char* ensemble;
  int n, strings;
  std::uint64_t dim, nnz;
  Stats cpu, gpu_kernel, gpu_e2e;
};

} // namespace

int main(int argc, char** argv) {
  std::vector<int> ns = {8, 10, 12, 14};
  std::vector<int> string_counts = {64, 256, 1024};
  int reps = 9;
  std::uint64_t seed = 0x5eedb00full;
  std::string out_path =
      std::string(PH_DATA_DIR) + "/pauli_gpu_build_bench.json";
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
      std::fprintf(stderr, "bench_gpu_build: %s\n", e.what());
      std::fprintf(stderr,
                   "usage: bench_gpu_build [--n=8,10,12,14] "
                   "[--strings=64,256,1024] [--reps=9] [--seed=0x...] "
                   "[--out=PATH]\n");
      return 1;
    }
  }

  try {
    int dev = 0;
    ph::gpu::cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop;
    ph::gpu::cuda_check(cudaGetDeviceProperties(&prop, dev), "props");
    int driver_v = 0, runtime_v = 0;
    ph::gpu::cuda_check(cudaDriverGetVersion(&driver_v), "driver");
    ph::gpu::cuda_check(cudaRuntimeGetVersion(&runtime_v), "runtime");
    std::printf("cpu: %s, hw threads %u\n", cpu_model().c_str(),
                std::thread::hardware_concurrency());
    std::printf("gpu: %s sm_%d%d, %d SMs, %zu MiB, driver %d, runtime %d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem >> 20, driver_v, runtime_v);
    std::printf("median of %d reps; emit only (assembly is the shared CPU "
                "path everywhere)\n\n",
                reps);
    std::printf("%10s %3s %6s %7s | %9s %9s %9s | %7s %7s\n", "ensemble", "n",
                "dim", "strings", "cpu ms", "gpu ms", "e2e ms", "k speed",
                "e2e speed");

    std::vector<CaseEntry> entries;
    for (const bool dh : {false, true})
      for (const int n : ns)
        for (const int S : string_counts) {
          const std::vector<ph::PauliString> ss =
              random_ensemble(seed + n * 7919 + S, n, S, dh);
          const ph::CsrMatrix a = ph::build_vectorized(ss, n);
          const ph::CsrMatrix g = ph::gpu::build_gpu(ss, n);
          if (!ph::bitwise_equal(a, g))
            throw std::runtime_error("bitwise mismatch at n=" +
                                     std::to_string(n) + " strings=" +
                                     std::to_string(S));
          ph::gpu::Ensemble e(ss, n);
          CaseEntry ce{dh ? "diag-heavy" : "random",
                       n,
                       S,
                       a.dim,
                       a.nnz(),
                       bench_cpu_emit(ss, n, reps),
                       bench_gpu_kernel(e, reps),
                       bench_gpu_e2e(ss, n, reps)};
          entries.push_back(ce);
          std::printf(
              "%10s %3d %6" PRIu64 " %7d | %9.3f %9.3f %9.3f | %6.2fx %6.2fx\n",
              ce.ensemble, n, ce.dim, S, ce.cpu.median * 1e3,
              ce.gpu_kernel.median * 1e3, ce.gpu_e2e.median * 1e3,
              ce.cpu.median / ce.gpu_kernel.median,
              ce.cpu.median / ce.gpu_e2e.median);
          std::printf("            spread ms [min,max]: cpu [%.3f,%.3f] gpu "
                      "[%.3f,%.3f] e2e [%.3f,%.3f]\n",
                      ce.cpu.min * 1e3, ce.cpu.max * 1e3,
                      ce.gpu_kernel.min * 1e3, ce.gpu_kernel.max * 1e3,
                      ce.gpu_e2e.min * 1e3, ce.gpu_e2e.max * 1e3);
        }

    std::ofstream f(out_path);
    if (!f)
      throw std::runtime_error("cannot open " + out_path);
    f.precision(17);
    f << "{\n"
      << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
      << "  \"seed\": \"" << seed << "\",\n"
      << "  \"cpu\": \"" << cpu_model() << "\",\n"
      << "  \"hw_threads\": " << std::thread::hardware_concurrency() << ",\n"
      << "  \"gpu\": \"" << prop.name << "\",\n"
      << "  \"compute_capability\": \"" << prop.major << "." << prop.minor
      << "\",\n"
      << "  \"sm_count\": " << prop.multiProcessorCount << ",\n"
      << "  \"total_global_mem\": " << prop.totalGlobalMem << ",\n"
      << "  \"driver_version\": " << driver_v << ",\n"
      << "  \"runtime_version\": " << runtime_v << ",\n"
      << "  \"nvcc\": \"" << __CUDACC_VER_MAJOR__ << "."
      << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << "\",\n"
      << "  \"build_type\": \"" << PH_BUILD_TYPE << "\",\n"
      << "  \"cxx_flags\": \"" << PH_CXX_FLAGS << "\",\n"
      << "  \"n_reps\": " << reps << ",\n"
      << "  \"note\": \"emit only; assembly is the shared CPU assemble() "
         "everywhere. gpu_kernel is CUDA-event timed with resident inputs; "
         "gpu_e2e is wall clock including compaction, allocations, and "
         "transfers.\",\n"
      << "  \"cases\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
      const CaseEntry& e = entries[i];
      f << "    {\"ensemble\": \"" << e.ensemble << "\", \"n\": " << e.n
        << ", \"dim\": " << e.dim << ", \"strings\": " << e.strings
        << ", \"nnz\": " << e.nnz << ", \"bitwise_equal\": true,\n"
        << "     \"kernel_speedup\": " << e.cpu.median / e.gpu_kernel.median
        << ", \"e2e_speedup\": " << e.cpu.median / e.gpu_e2e.median << ",\n"
        << "     \"cpu_vectorized\": ";
      json_stats(f, e.cpu, 16);
      f << ",\n     \"gpu_kernel\": ";
      json_stats(f, e.gpu_kernel, 16);
      f << ",\n     \"gpu_e2e\": ";
      json_stats(f, e.gpu_e2e, 16);
      f << "}" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    if (!f)
      throw std::runtime_error("write failed: " + out_path);
    std::printf("\nJSON -> %s\n", out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_gpu_build failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
