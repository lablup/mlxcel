// Library-level reproduction for lablup/mlxcel#2206. No MLX.
//
// bf16 x bf16 GEMM with f32 compute (HIPBLAS_COMPUTE_32F, f32 alpha/beta, the
// configuration `gemms/hipblaslt_gemm.cpp` uses) through hipBLASLt and
// rocBLAS `gemm_ex`, with bf16 and f32 outputs, against an exact host
// reference. Integer inputs keep every product and partial sum exact, so an
// IEEE GEMM in any order returns the reference.
//
//   ./run.sh                 # integer cases, the first heuristic algorithm
//   ./run.sh --all           # integer cases, every heuristic algorithm
//   ./run.sh --random        # random normal bf16 inputs, 64^3 .. 4096^3
//
// Each line prints the number of wrong outputs, how many of them have an
// exact value of zero, the largest error, a sample wrong value at a zero, the
// largest ratio of the error to the f32 dot-product bound
// K * u * sum_k |a_k b_k| (u = 2^-24), and the Tensile kernel name.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                        \
  do {                                                                  \
    int e_ = (int)(x);                                                  \
    if (e_ != 0) {                                                      \
      fprintf(stderr, "%s:%d %s returned %d\n", __FILE__, __LINE__, #x, \
              e_);                                                      \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

static uint16_t to_bf16(float f) {  // truncation; inputs are bf16-exact
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

static float from_bf16(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// The generator of tests/rocm_hipblaslt_concurrency.rs (`integer_values`).
static std::vector<float> integer_values(size_t len, size_t seed, int half) {
  size_t span = 2 * half + 1;
  std::vector<float> v(len);
  for (size_t i = 0; i < len; ++i)
    v[i] = (float)((int)((i * 31 + seed * 17 + 7) % span) - half);
  return v;
}

// a is [M, K] row-major, b is [N, K] row-major; D = a * b^T, stored
// column-major [M, N] as the BLAS libraries write it.
struct Case {
  std::string name;
  int M, N, K;
  std::vector<float> a, b;
};

struct Reference {
  std::vector<double> value, abs_sum;
};

static Reference reference(const Case& c) {
  Reference r{std::vector<double>((size_t)c.M * c.N),
              std::vector<double>((size_t)c.M * c.N)};
  for (int m = 0; m < c.M; ++m)
    for (int n = 0; n < c.N; ++n) {
      double s = 0, t = 0;
      for (int k = 0; k < c.K; ++k) {
        double p = (double)c.a[(size_t)m * c.K + k] * c.b[(size_t)n * c.K + k];
        s += p;
        t += std::fabs(p);
      }
      r.value[m + (size_t)n * c.M] = s;
      r.abs_sum[m + (size_t)n * c.M] = t;
    }
  return r;
}

static void report(const char* what, const std::vector<float>& got,
                   const Reference& ref, int K, const std::string& kernel) {
  int wrong = 0, wrong_at_zero = 0;
  double max_err = 0, sample = 0, ratio = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double e = (double)got[i] - ref.value[i];
    if (ref.abs_sum[i] > 0)
      ratio = std::max(ratio, std::fabs(e) / (K * std::ldexp(1.0, -24) *
                                              ref.abs_sum[i]));
    if (e == 0) continue;
    ++wrong;
    max_err = std::max(max_err, std::fabs(e));
    if (ref.value[i] == 0) {
      ++wrong_at_zero;
      if (sample == 0) sample = got[i];
    }
  }
  printf("  %-26s wrong=%-8d at_zero=%-6d max_err=%-9.3g zero_sample=%-15.9g "
         "bound_ratio=%.4f  %s\n",
         what, wrong, wrong_at_zero, max_err, sample, ratio,
         kernel.substr(0, 72).c_str());
}

struct Device {
  void *a = nullptr, *b = nullptr, *d = nullptr;
  size_t nd = 0, d_bytes = 0;
  Device(const Case& c, size_t elem) {
    size_t na = (size_t)c.M * c.K, nb = (size_t)c.N * c.K;
    nd = (size_t)c.M * c.N;
    d_bytes = nd * elem;
    std::vector<uint16_t> ha(na), hb(nb);
    for (size_t i = 0; i < na; ++i) ha[i] = to_bf16(c.a[i]);
    for (size_t i = 0; i < nb; ++i) hb[i] = to_bf16(c.b[i]);
    CHECK(hipMalloc(&a, na * 2));
    CHECK(hipMalloc(&b, nb * 2));
    CHECK(hipMalloc(&d, d_bytes));
    CHECK(hipMemcpy(a, ha.data(), na * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b, hb.data(), nb * 2, hipMemcpyHostToDevice));
  }
  std::vector<float> output() const {
    std::vector<float> out(nd);
    if (d_bytes == nd * 4) {
      CHECK(hipMemcpy(out.data(), d, d_bytes, hipMemcpyDeviceToHost));
    } else {
      std::vector<uint16_t> raw(nd);
      CHECK(hipMemcpy(raw.data(), d, d_bytes, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < nd; ++i) out[i] = from_bf16(raw[i]);
    }
    return out;
  }
  ~Device() {
    (void)hipFree(a);
    (void)hipFree(b);
    (void)hipFree(d);
  }
};

static void run_hipblaslt(hipblasLtHandle_t h, const Case& c,
                          const Reference& ref, hipDataType d_type,
                          bool every_algo) {
  Device dev(c, d_type == HIP_R_32F ? 4 : 2);
  size_t ws_bytes = 32 << 20;
  void* ws;
  CHECK(hipMalloc(&ws, ws_bytes));
  hipblasLtMatmulDesc_t desc;
  CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
  CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &ta, sizeof(ta)));
  CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &tb, sizeof(tb)));
  hipblasLtMatrixLayout_t la, lb, ld;
  CHECK(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, c.K, c.M, c.K));
  CHECK(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, c.K, c.N, c.K));
  CHECK(hipblasLtMatrixLayoutCreate(&ld, d_type, c.M, c.N, c.M));
  hipblasLtMatmulPreference_t pref;
  CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
      sizeof(ws_bytes)));
  std::vector<hipblasLtMatmulHeuristicResult_t> algos(64);
  int found = 0;
  CHECK(hipblasLtMatmulAlgoGetHeuristic(h, desc, la, lb, ld, ld, pref,
                                        (int)algos.size(), algos.data(),
                                        &found));
  float alpha = 1.f, beta = 0.f;
  int runs = every_algo ? found : std::min(found, 1);
  for (int i = 0; i < runs; ++i) {
    CHECK(hipMemset(dev.d, 0x7f, dev.d_bytes));
    CHECK(hipblasLtMatmul(h, desc, &alpha, dev.a, la, dev.b, lb, &beta, dev.d,
                          ld, dev.d, ld, &algos[i].algo, ws, ws_bytes, 0));
    CHECK(hipDeviceSynchronize());
    char what[64];
    snprintf(what, sizeof(what), "hipBLASLt D=%s algo %d",
             d_type == HIP_R_32F ? "f32" : "bf16", i);
    report(what, dev.output(), ref, c.K,
           hipblaslt_ext::getKernelNameFromAlgo(h, algos[i].algo));
  }
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(la);
  hipblasLtMatrixLayoutDestroy(lb);
  hipblasLtMatrixLayoutDestroy(ld);
  hipblasLtMatmulDescDestroy(desc);
  (void)hipFree(ws);
}

static void run_rocblas(rocblas_handle h, const Case& c, const Reference& ref,
                        rocblas_datatype d_type) {
  Device dev(c, d_type == rocblas_datatype_f32_r ? 4 : 2);
  CHECK(hipMemset(dev.d, 0, dev.d_bytes));
  float alpha = 1.f, beta = 0.f;
  CHECK(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                        c.M, c.N, c.K, &alpha, dev.a, rocblas_datatype_bf16_r,
                        c.K, dev.b, rocblas_datatype_bf16_r, c.K, &beta, dev.d,
                        d_type, c.M, dev.d, d_type, c.M, rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0));
  CHECK(hipDeviceSynchronize());
  report(d_type == rocblas_datatype_f32_r ? "rocBLAS D=f32" : "rocBLAS D=bf16",
         dev.output(), ref, c.K,
         "(kernel name: rocprofv3 --kernel-trace)");
}

static std::vector<Case> integer_cases() {
  std::vector<Case> cases;
  const int D = 64;
  const size_t n = (size_t)D * D;
  std::vector<float> ones(n, 1.f);
  for (int half : {1, 2})
    cases.push_back({"integer_values -" + std::to_string(half) + "..=" +
                         std::to_string(half) + ", 64^3",
                     D, D, D, integer_values(n, 0, half),
                     integer_values(n, 1000003, half)});
  {
    auto a = integer_values(n, 0, 2), b = integer_values(n, 1000003, 2);
    for (auto& x : a) x = x == 0 ? 1 : x;
    for (auto& x : b) x = x == 0 ? -1 : x;
    cases.push_back({"{-2,-1,1,2} (no zeros), 64^3", D, D, D, a, b});
  }
  cases.push_back({"-2..=2 pattern x ones", D, D, D, integer_values(n, 0, 2), ones});
  cases.push_back({"ones x -2..=2 pattern", D, D, D, ones, integer_values(n, 0, 2)});
  for (float h : {1.f, 2.f}) {
    std::vector<float> a(n, 0.f);
    for (int m = 0; m < D; ++m) {
      a[(size_t)m * D] = h;
      a[(size_t)m * D + 1] = -h;
    }
    cases.push_back({"rows [" + std::to_string((int)h) + ", -" +
                         std::to_string((int)h) + ", 0, ...] x ones",
                     D, D, D, a, ones});
  }
  {
    std::vector<float> a(n, 0.f);
    for (int m = 0; m < D; ++m) a[(size_t)m * D] = 2;
    cases.push_back({"rows [2, 0, ...] x ones (no cancellation)", D, D, D, a, ones});
  }
  return cases;
}

static std::vector<Case> random_cases() {
  srand(2206);
  auto gen = [](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
      double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
      double u2 = rand() / (RAND_MAX + 1.0);
      double z = std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2);
      x = from_bf16(to_bf16((float)z));
    }
    return v;
  };
  std::vector<Case> cases;
  for (int d : {64, 256, 1024, 4096})
    cases.push_back({"random normal, " + std::to_string(d) + "^3", d, d, d,
                     gen((size_t)d * d), gen((size_t)d * d)});
  return cases;
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  bool random_mode = mode == "--random";
  bool every_algo = mode == "--all";
  hipblasLtHandle_t lt;
  CHECK(hipblasLtCreate(&lt));
  rocblas_handle rb;
  CHECK(rocblas_create_handle(&rb));
  for (const Case& c : random_mode ? random_cases() : integer_cases()) {
    printf("%s\n", c.name.c_str());
    Reference ref = reference(c);
    if (!random_mode) run_hipblaslt(lt, c, ref, HIP_R_16BF, every_algo);
    run_hipblaslt(lt, c, ref, HIP_R_32F, every_algo);
    if (!random_mode) run_rocblas(rb, c, ref, rocblas_datatype_bf16_r);
    run_rocblas(rb, c, ref, rocblas_datatype_f32_r);
  }
  hipblasLtDestroy(lt);
  rocblas_destroy_handle(rb);
  return 0;
}
