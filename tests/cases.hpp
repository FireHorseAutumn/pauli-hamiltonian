#pragma once

// cases.hpp — shared bitwise-gate case suite: every case the builders are
// checked against. Used by test_equality (naive vs vectorized) and
// test_gpu_equality (GPU vs both). Deterministic; no <random>.

#include "ph/pauli.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace phtest {

struct Case {
  std::string name;
  std::vector<ph::PauliString> strings;
  int n;
};

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

inline ph::PauliString random_string(Rng& rng, int n, double p_identity) {
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

// The full gate suite: randomized ensemble n = 1..10 x 20 seeds, exhaustive
// singletons at n = 2 and 3, all-diagonal, duplicate accumulation and
// cancellation, identity-only, zero-coefficient, empty, full-fill, and an
// n = 10 stress.
inline std::vector<Case> make_cases() {
  std::vector<Case> out;
  const auto add = [&out](std::string name,
                          std::vector<ph::PauliString> strings, int n) {
    out.push_back({std::move(name), std::move(strings), n});
  };

  for (int n = 1; n <= 10; ++n)
    for (int seed = 0; seed < 20; ++seed) {
      Rng rng(0x5eed0000ull + n * 1000 + seed);
      const int count = static_cast<int>(rng.next() % (4 * n + 1));
      const double pI = rng.uniform(0.15, 0.6);
      std::vector<ph::PauliString> ss(count);
      for (auto& s : ss)
        s = random_string(rng, n, pI);
      add("random", ss, n);
    }

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
      add("exhaustive-singleton", {s}, n);
    }
  }

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
      add("all-diagonal", ss, 6);
    }
  }

  {
    ph::PauliString s;
    s.x = 0b0101;
    s.y = 0b0010;
    s.z = 0b1000;
    add("duplicates-accumulate",
        {{s.x, s.y, s.z, 0.5}, {s.x, s.y, s.z, 0.25}, {s.x, s.y, s.z, 1.0}},
        4);
    add("duplicates-cancel",
        {{s.x, s.y, s.z, 0.5}, {s.x, s.y, s.z, -0.5}, {s.x, s.y, s.z, 1.0}},
        4);
  }

  add("identity-only", {{0, 0, 0, 1.5}, {0, 0, 0, -0.5}}, 5);
  add("zero-coeff", {{0, 0, 0, 0.0}, {0b1, 0, 0, 0.0}, {0b1, 0, 0, 1.0}}, 3);
  add("empty", {}, 4);

  {
    Rng rng(0xf111ull);
    std::vector<ph::PauliString> ss(300);
    for (auto& s : ss)
      s = random_string(rng, 6, 0.05);
    add("full-fill", ss, 6);
  }

  {
    Rng rng(0xb19ull);
    for (int trial = 0; trial < 10; ++trial) {
      std::vector<ph::PauliString> ss(40);
      for (auto& s : ss)
        s = random_string(rng, 10, 0.35);
      add("n10-stress", ss, 10);
    }
  }

  return out;
}

} // namespace phtest
