// Reproduction for "fix(rocm): set blocking-sync before the first HIP queue
// is created". No MLX and no hipFFT: this is the HIP runtime behavior the
// backend ran into.
//
//   hipcc --offload-arch=gfx1151 -O2 repro.hip -o repro
//   timeout 30 ./repro 1 200   # null stream touched before the flag: hangs
//                              # at the first hipFree (last line "iter 7")
//   ./repro 0 200              # flag first: prints "done"
//
// Seen on gfx1151 with HIP 7.15.26333. With AMD_LOG_LEVEL=4 the hanging run
// logs "hsa_amd_signal_async_handler() failed to set the handler!".
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void k(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.0f;
}

int main(int argc, char** argv) {
  bool early_touch = argc > 1 && argv[1][0] == '1';
  int iters = argc > 2 ? atoi(argv[2]) : 200;
  if (early_touch) (void)hipStreamQuery(nullptr);  // creates the null-stream queue now
  (void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync);  // turns active wait off
  hipStream_t s;
  (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  std::vector<float> h(1024, 1.0f);
  std::vector<void*> live;
  for (int i = 0; i < iters; ++i) {
    float* p;
    (void)hipMalloc(&p, 4096);
    (void)hipMemcpy(p, h.data(), 4096, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k, dim3(4), dim3(256), 0, nullptr, p, 1024);
    live.push_back(p);
    if (live.size() > 8) {
      (void)hipFree(live.front());
      live.erase(live.begin());
    }
    printf("iter %d\n", i);
    fflush(stdout);
  }
  printf("done\n");
}
