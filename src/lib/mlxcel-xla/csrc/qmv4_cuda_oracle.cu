// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Independent CUDA test harness for the six production QMV4 entry points.
// Build with -I <repo>/src/lib/mlxcel-xla/csrc. No model or service is used.
#include "qmv4.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e));          \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

template <class T> struct Device {
  T *p = nullptr;
  explicit Device(size_t n) { CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  void put(const std::vector<T> &v) {
    CUDA_OK(
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

struct Weights {
  int n, k;
  std::vector<uint32_t> w;
  std::vector<__half> s, b;
  Device<uint32_t> dw;
  Device<__half> ds, db;
  Device<float> dy;
  Weights(int rows, int width, int seed, bool fractional)
      : n(rows), k(width), w(rows * width / 8), s(rows * width / 64),
        b(s.size()), dw(w.size()), ds(s.size()), db(b.size()), dy(rows + 16) {
    uint32_t state = seed;
    auto rng = [&]() {
      state = state * 1664525u + 1013904223u;
      return state;
    };
    for (auto &v : w)
      v = rng();
    for (size_t i = 0; i < s.size(); ++i) {
      s[i] =
          __float2half(float(1 + rng() % 7) / (fractional ? 113.0f : 128.0f));
      b[i] = __float2half(float(int(rng() % 17) - 8) /
                          (fractional ? 127.0f : 128.0f));
    }
    dw.put(w);
    ds.put(s);
    db.put(b);
  }
  void reset() { dy.put(std::vector<float>(n + 16, 1234567.0f)); }
  void verify(const std::vector<__half> &x, bool exact, double &max_error,
              double &max_bound_ratio, size_t &checked) {
    std::vector<float> y(n + 16);
    CUDA_OK(cudaMemcpy(y.data(), dy.p, y.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    for (int r = 0; r < n; ++r) {
      double sum = 0, abs_sum = 0;
      for (int c = 0; c < k; ++c) {
        int q = (w[r * (k / 8) + c / 8] >> (4 * (c % 8))) & 15;
        double weight = q * double(__half2float(s[r * (k / 64) + c / 64])) +
                        double(__half2float(b[r * (k / 64) + c / 64]));
        double term = weight * double(__half2float(x[c]));
        sum += term;
        abs_sum += std::abs(term);
      }
      double error = std::abs(double(y[r]) - sum);
      // Conservative FP32 forward-error envelope for per-lane dot plus
      // shuffle reduction; compare with the *exact half-input* FP64 sum.
      double nu = (2.0 * std::ceil(k / 32.0) + 32) * std::ldexp(1.0, -24);
      double bound = nu / (1 - nu) * abs_sum + 1e-7;
      max_error = std::max(max_error, error);
      max_bound_ratio = std::max(max_bound_ratio, error / bound);
      if (!std::isfinite(y[r]) ||
          (exact ? double(y[r]) != sum : error > bound)) {
        std::fprintf(stderr,
                     "FAIL K=%d N=%d row=%d exact=%d actual=%.10g "
                     "reference=%.17g error=%g bound=%g\n",
                     k, n, r, exact, y[r], sum, error, bound);
        std::exit(3);
      }
      ++checked;
    }
    for (int r = n; r < n + 16; ++r)
      if (y[r] != 1234567.0f) {
        std::fprintf(stderr, "output guard overwritten\n");
        std::exit(4);
      }
  }
};

int main() {
  size_t calls = 0, checked = 0;
  double max_error = 0, max_bound_ratio = 0;
  for (bool fractional : {false, true})
    for (int k : {64, 128, 192, 1024, 2048, 4032, 4096, 4160, 8192, 18944})
      for (int n : {1, 7, 8, 9, 17}) {
        std::vector<__half> x(k);
        for (int c = 0; c < k; ++c)
          x[c] = __float2half(float((c * 13) % 33 - 16) /
                              (fractional ? 17.0f : 16.0f));
        Device<__half> dx(k);
        dx.put(x);
        Weights a(n, k, 17, fractional), b(3, k, 29, fractional),
            c(11, k, 41, fractional);
        for (bool row : {false, true}) {
          a.reset();
          b.reset();
          c.reset();
          int grid = row ? n : (n + 7) / 8;
          if (row)
            qmv4_row_kernel<<<grid, 256>>>(dx.p, a.dw.p, a.ds.p, a.db.p, a.dy.p,
                                           k, n);
          else
            qmv4_kernel<<<grid, 256>>>(dx.p, a.dw.p, a.ds.p, a.db.p, a.dy.p, k,
                                       n);
          CUDA_OK(cudaGetLastError());
          CUDA_OK(cudaDeviceSynchronize());
          a.verify(x, !fractional, max_error, max_bound_ratio, checked);
          ++calls;
          a.reset();
          b.reset();
          grid = row ? n + 3 : (n + 3 + 7) / 8;
          if (row)
            qmv4_gateup_row_kernel<<<grid, 256>>>(dx.p, a.dw.p, a.ds.p, a.db.p,
                                                  b.dw.p, b.ds.p, b.db.p,
                                                  a.dy.p, b.dy.p, k, n, 3);
          else
            qmv4_gateup_kernel<<<grid, 256>>>(dx.p, a.dw.p, a.ds.p, a.db.p,
                                              b.dw.p, b.ds.p, b.db.p, a.dy.p,
                                              b.dy.p, k, n, 3);
          CUDA_OK(cudaGetLastError());
          CUDA_OK(cudaDeviceSynchronize());
          a.verify(x, !fractional, max_error, max_bound_ratio, checked);
          b.verify(x, !fractional, max_error, max_bound_ratio, checked);
          ++calls;
          a.reset();
          b.reset();
          c.reset();
          grid = row ? n + 14 : (n + 14 + 7) / 8;
          if (row)
            qmv4_qkv_row_kernel<<<grid, 256>>>(
                dx.p, a.dw.p, a.ds.p, a.db.p, b.dw.p, b.ds.p, b.db.p, c.dw.p,
                c.ds.p, c.db.p, a.dy.p, b.dy.p, c.dy.p, k, n, 3, 11);
          else
            qmv4_qkv_kernel<<<grid, 256>>>(
                dx.p, a.dw.p, a.ds.p, a.db.p, b.dw.p, b.ds.p, b.db.p, c.dw.p,
                c.ds.p, c.db.p, a.dy.p, b.dy.p, c.dy.p, k, n, 3, 11);
          CUDA_OK(cudaGetLastError());
          CUDA_OK(cudaDeviceSynchronize());
          a.verify(x, !fractional, max_error, max_bound_ratio, checked);
          b.verify(x, !fractional, max_error, max_bound_ratio, checked);
          c.verify(x, !fractional, max_error, max_bound_ratio, checked);
          ++calls;
        }
      }
  std::printf("PASS calls=%zu outputs=%zu max_abs_error=%.9g "
              "max_error_bound_ratio=%.9g\n",
              calls, checked, max_error, max_bound_ratio);
}
