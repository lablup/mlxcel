// Reproduction for "fix(rocm): run CPU-stream BLAS single-threaded over
// fine-grained memory". No MLX: cblas_sgemm with M=1 (x[1,K] @ W^T, W [N,K]),
// the shape of an MLX CPU-stream matmul(x, transpose(w)), with its buffers in
// malloc (0), fine-grained device memory (1) or hipHostMalloc (2) memory.
//
//   gcc -O2 repro.c -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ \
//       -L/opt/rocm/lib -lamdhip64 -lopenblas -lm -o repro
//   MODE=11 ./repro 50     # inputs and output fine-grained: many bad iterations
//   MODE=01 ./repro 300    # output fine-grained only: some bad iterations
//   MODE=00 ./repro 300    # malloc: 0 bad
//   OPENBLAS_NUM_THREADS=1 MODE=11 ./repro 300   # one thread: 0 bad
//
// Seen on gfx1151 (Radeon 8060S APU) with Debian OpenBLAS 0.3.29, 32 threads.
//
// Standalone check: is OpenBLAS cblas_sgemm with M=1 (x[1,K] @ W^T, W [N,K])
// deterministic on this host? Mirrors MLX CPU matmul(x, transpose(w)).
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
static void* amalloc(size_t n, int mode){ void* p=0; if(mode==0) return malloc(n); if(mode==1){ if(hipExtMallocWithFlags(&p,n,hipDeviceMallocFinegrained)!=hipSuccess){printf("alloc fail\n");exit(1);} return p;} hipHostMalloc(&p,n,hipHostMallocDefault); return p; }

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 300; int N = argc > 2 ? atoi(argv[2]) : 2880; int K = argc > 3 ? atoi(argv[3]) : 2880;
  const char* e=getenv("MODE"); int in_mode = e? e[0]-'0':0; int out_mode = e? e[1]-'0':0;
  float* w = amalloc(sizeof(float) * N * K, in_mode);
  float* x = amalloc(sizeof(float) * K, in_mode);
  float* y = amalloc(sizeof(float) * N, out_mode);
  double* exact = malloc(sizeof(double) * N);
  srand(18081);
  for (int i = 0; i < N * K; i++) w[i] = (float)rand() / RAND_MAX * 2 - 1;
  for (int i = 0; i < K; i++) x[i] = (float)rand() / RAND_MAX * 2 - 1;
  double scale = 0;
  for (int c = 0; c < N; c++) {
    double s = 0;
    for (int k = 0; k < K; k++) s += (double)w[c * K + k] * x[k];
    exact[c] = s;
    if (fabs(s) > scale) scale = fabs(s);
  }
  int bad_iters = 0;
  for (int it = 0; it < iters; it++) {
    memset(y, 0, sizeof(float) * N);
    // C[1,N] = A[1,K] * B^T, B stored row-major [N,K]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, N, K, 1.0f, x, K,
                w, K, 0.0f, y, N);
    int bad = 0, first = -1;
    for (int c = 0; c < N; c++)
      if (fabs(y[c] - exact[c]) / scale > 1e-4) {
        if (first < 0) first = c;
        bad++;
      }
    if (bad) {
      bad_iters++;
      printf("iter %d: %d bad, first col %d got %f exact %f\n", it, bad, first,
             y[first], exact[first]);
    }
  }
  printf("SGEMM SUMMARY bad %d/%d (config: %s)\n", bad_iters, iters,
         openblas_get_config());
  return 0;
}
