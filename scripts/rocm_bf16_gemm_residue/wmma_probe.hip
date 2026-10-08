// Instruction-level probe for lablup/mlxcel#2206. No MLX, no BLAS library.
//
// Runs one v_wmma_f32_16x16x16_bf16, one v_wmma_f32_16x16x16_f16 and a chain
// of v_dot2_f32_bf16 on a single wave32, plus a plain f32 FMA loop as the
// IEEE baseline. Row m of A holds a short test vector padded with zeros, B is
// all ones and C is zero, so D[m][n] = sum_k A[m][k]. Every product and every
// partial sum is exactly representable, so an IEEE dot product in any order
// returns the exact sum.
//
//   hipcc --offload-arch=gfx1151 -O2 wmma_probe.hip -o wmma_probe && ./wmma_probe
//
// Exit status is 0 whether or not the hardware is exact; read the WRONG
// markers. On gfx1151 (ROCm 7.15) the three matrix/dot instructions return a
// small negative residue whenever a cancelling term has the opposite sign
// (for example [1, -1] -> -2^-24, [8, -8] -> -2^-21) and the FMA loop is exact.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef short short16 __attribute__((ext_vector_type(16)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef short short2v __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

#define CHECK(x)                                                          \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,           \
              hipGetErrorString(e_));                                     \
      return 1;                                                           \
    }                                                                     \
  } while (0)

// gfx11 wave32 WMMA 16x16x16: lane l supplies row (l % 16) of A and column
// (l % 16) of B as 16 K values (lanes 16..31 repeat lanes 0..15); lane l gets
// D[2 * i + l / 16][l % 16] for i in 0..7.
__global__ void wmma_bf16(const uint16_t* A, const uint16_t* B, float* D) {
  int l = threadIdx.x;
  short16 a, b;
  for (int k = 0; k < 16; ++k) {
    a[k] = (short)A[(l % 16) * 16 + k];
    b[k] = (short)B[(l % 16) * 16 + k];
  }
  float8 c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
  for (int i = 0; i < 8; ++i) D[(2 * i + l / 16) * 16 + l % 16] = c[i];
}

__global__ void wmma_f16(const uint16_t* A, const uint16_t* B, float* D) {
  int l = threadIdx.x;
  half16 a, b;
  for (int k = 0; k < 16; ++k) {
    a[k] = __builtin_bit_cast(_Float16, A[(l % 16) * 16 + k]);
    b[k] = __builtin_bit_cast(_Float16, B[(l % 16) * 16 + k]);
  }
  float8 c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  for (int i = 0; i < 8; ++i) D[(2 * i + l / 16) * 16 + l % 16] = c[i];
}

// acc = dot2(a[k], a[k+1]; b[k], b[k+1]) + acc, k = 0, 2, ..., 14.
__global__ void dot2_bf16(const uint16_t* A, const uint16_t* B, float* D) {
  int m = threadIdx.x;
  if (m >= 16) return;
  float acc = 0.f;
  for (int k = 0; k < 16; k += 2) {
    short2v a = {(short)A[m * 16 + k], (short)A[m * 16 + k + 1]};
    short2v b = {(short)B[m * 16 + k], (short)B[m * 16 + k + 1]};
    acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a),
                                         __builtin_bit_cast(bf16x2, b), acc,
                                         false);
  }
  D[m] = acc;
}

// IEEE baseline: widen bf16 to f32 and accumulate with fmaf.
__global__ void fma_bf16(const uint16_t* A, const uint16_t* B, float* D) {
  int m = threadIdx.x;
  if (m >= 16) return;
  float acc = 0.f;
  for (int k = 0; k < 16; ++k) {
    float a = __uint_as_float((uint32_t)A[m * 16 + k] << 16);
    float b = __uint_as_float((uint32_t)B[m * 16 + k] << 16);
    acc = fmaf(a, b, acc);
  }
  D[m] = acc;
}

static uint16_t to_bf16(float f) {  // exact for every value used below
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

static uint16_t to_f16(float f) {
  _Float16 h = (_Float16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}

int main() {
  const std::vector<std::vector<float>> rows = {
      {1, -1},    {2, -2},      {-1, 1},       {-2, 2},
      {2},        {1, 1, -2},   {2, -1, -1},   {4, -4},
      {0.5f, -0.5f}, {1, -1, 1, -1}, {3, -3}, {1, 2, -3},
      {-1},       {8, -8},      {2, 2, -4},    {1.5f, -1.5f},
  };
  std::vector<float> A(256, 0.f), B(256, 1.f);
  for (int m = 0; m < 16; ++m)
    for (size_t k = 0; k < rows[m].size(); ++k) A[m * 16 + k] = rows[m][k];

  const char* names[4] = {"v_wmma_f32_16x16x16_bf16", "v_wmma_f32_16x16x16_f16",
                          "v_dot2_f32_bf16 chain", "f32 fmaf loop (IEEE baseline)"};
  for (int which = 0; which < 4; ++which) {
    bool f16 = which == 1;
    std::vector<uint16_t> ha(256), hb(256);
    for (int i = 0; i < 256; ++i) {
      ha[i] = f16 ? to_f16(A[i]) : to_bf16(A[i]);
      hb[i] = f16 ? to_f16(B[i]) : to_bf16(B[i]);
    }
    uint16_t *da, *db;
    float* dd;
    CHECK(hipMalloc(&da, 512));
    CHECK(hipMalloc(&db, 512));
    CHECK(hipMalloc(&dd, 1024));
    CHECK(hipMemcpy(da, ha.data(), 512, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, hb.data(), 512, hipMemcpyHostToDevice));
    CHECK(hipMemset(dd, 0, 1024));
    if (which == 0) wmma_bf16<<<1, 32>>>(da, db, dd);
    if (which == 1) wmma_f16<<<1, 32>>>(da, db, dd);
    if (which == 2) dot2_bf16<<<1, 32>>>(da, db, dd);
    if (which == 3) fma_bf16<<<1, 32>>>(da, db, dd);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> d(256);
    CHECK(hipMemcpy(d.data(), dd, 1024, hipMemcpyDeviceToHost));
    printf("%s (B = ones, C = 0)\n", names[which]);
    for (int m = 0; m < 16; ++m) {
      float want = 0;
      for (float x : rows[m]) want += x;
      float got = which >= 2 ? d[m] : d[m * 16];
      printf("  row %-2d sum %-5g got %.9g%s\n", m, want, got,
             got == want ? "" : "   <-- WRONG");
    }
    CHECK(hipFree(da));
    CHECK(hipFree(db));
    CHECK(hipFree(dd));
  }
  return 0;
}
