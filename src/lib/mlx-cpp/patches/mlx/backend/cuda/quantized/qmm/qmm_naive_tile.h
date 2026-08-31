// Added by mlxcel (lablup/mlxcel#1541). Not an upstream MLX file.
//
// The CTA tile selection for `qmm_naive`, factored out of `qmm_naive.cu` as a
// pure integer function so it can be unit tested on the host with no CUDA
// device, no CUDA toolkit and no CuTe. `qmm_naive.cu` calls it with the value
// of `cudaDevAttrMaxSharedMemoryPerBlockOptin` queried from the running
// device, and `src/lib/mlxcel-core/cpp/qmm_naive_tile_probe.cpp` exposes the
// same function to `qmm_naive_tile_tests.rs` through a C shim. Both callers
// share this one definition, so the tested function is the shipped one.
//
// Upstream sized the N tile with one predicate whose two halves control
// different things:
//
//   bool enough_smem = sm80 && itemsize <= 2 && group_size <= 64;
//
// `itemsize <= 2 && group_size <= 64` is a shared-memory term, and it is now
// written as one: a comparison against the device's real per-block budget
// rather than a shape rule standing in for it. That is what lets the launch
// site know when it must opt into a larger dynamic shared-memory maximum, and
// when no tile fits at all.
//
// `sm80` is not a shared-memory term. A Tesla V100 offers 96 KB per block
// through the opt-in and 48 KB without it, against the 24 KB the widest
// eligible tile needs, so shared memory was never what kept the wide tile off
// pre-Ampere parts. It is a profitability term, and it appears below under the
// name of the thing it actually selects, `tensor_core_mma`: with
// `compute_capability_major() >= 8` and a 16-bit element type,
// `make_tiled_mma` in `device/gemm_sm70.cuh` picks an `SM80_16x8x16` tensor-core
// atom, and below that it picks `UniversalFMA<float, Element, Element>`.
//
// #1541 measured the wide tile on the FMA path rather than assuming either way,
// and upstream's exclusion turns out to be right for a reason its name did not
// give. On a V100, at bf16 and group size 64, the 128-wide instantiation needs
// 255 registers and spills 128 bytes per thread where the 64-wide one needs 224
// and spills nothing; both land on the same 2 blocks per SM, so the wider tile
// buys no occupancy and halves the CTA count on a grid that is already smaller
// than an 80-SM part at realistic prompt lengths. `qmm_naive` measures 1.50x
// slower at a 106-token prompt and 1.5% slower at 1906, at identical launch
// counts. Full record:
// `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
//
// The selected tile is therefore identical to upstream's on every
// architecture, which `qmm_naive_tile_tests.rs` asserts by enumeration over
// every `(itemsize, group_size, m)` combination and every real per-block
// budget. That closes the sm_80+ non-regression item epic #1536 deferred to a
// GB10 host: it is a property of a host-side pure function, not of silicon.
// What did change at the launch site is the shared-memory opt-in, the refusal
// to launch a tile that does not fit, and `MLXCEL_QMM_NAIVE_TILE_N`, which
// pins the width so the sweep can be re-run in one command when #1543 gives
// Volta a tensor-core MMA to re-measure against.

#pragma once

#include <algorithm>
#include <initializer_list>

namespace mlxcel {

// Dynamic shared memory the `qmm_naive` kernel asks for, in bytes.
//
// The kernel stores one A tile and one B tile, both in the element type, laid
// out by `cute::tile_to_shape` over an 8x64 (K-major) or 16x8 (M-major)
// swizzle base. Both tile extents are multiples of those bases for every shape
// this selector can produce, so each layout's cosize is exactly its product and
// the total is the expression below. `qmm_naive.cu` recomputes the same number
// from the real CuTe layouts and refuses to launch if the two disagree, so an
// MLX pin bump that changes the layouts is a loud error here rather than a
// silent under-allocation.
inline long long
qmm_naive_smem_bytes(int itemsize, int tile_m, int tile_n, int tile_k) {
  return static_cast<long long>(itemsize) * tile_k *
      (static_cast<long long>(tile_m) + tile_n);
}

// Bytes of the per-block budget held back from the tile.
//
// Static shared memory in this kernel is zero (everything goes through
// `extern __shared__`), and `cudaDevAttrMaxSharedMemoryPerBlockOptin` already
// excludes the driver's own per-block reservation, so nothing is known to need
// this. It is a margin against both of those assumptions, at a cost of 1 KB out
// of a 96 KB Volta budget.
inline constexpr long long kQmmNaiveSmemReserveBytes = 1024;

// Above this much dynamic shared memory a launch needs
// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` first,
// on every architecture from Volta onward. Omitting it does not raise a
// compile or load error, only a launch failure. The true ceiling is 49152;
// 48000 is what `qmm_naive.cu`'s sibling `qmm_sm80.cu` uses upstream and
// erring low only means opting in when it was not strictly required.
inline constexpr long long kQmmNaiveSmemOptInFreeBytes = 48000;

// `mlx::core::next_power_of_2` without `<cmath>`, so this header stays free of
// MLX and of floating point. Agrees with it for every n >= 0 in range, and
// saturates instead of overflowing above 2^30.
inline int qmm_naive_next_power_of_2(int n) {
  if (n <= 0) {
    return 0;
  }
  int p = 1;
  while (p < n) {
    if (p > (1 << 29)) {
      return p;
    }
    p <<= 1;
  }
  return p;
}

struct QmmNaiveTile {
  int tile_m;
  int tile_n;
  int tile_k;
  // Dynamic shared memory the chosen tile needs.
  long long smem_bytes;
  // The launch must opt into a larger dynamic shared-memory maximum first.
  bool needs_smem_opt_in;
  // The chosen tile fits the device budget with the reserve applied. False
  // means no tile in the candidate set fits and the caller must refuse to
  // launch, which is the only way this selector can fail.
  bool fits;
};

// Pick the CTA tile for one `qmm_naive` launch.
//
// `smem_budget_bytes` is `cudaDevAttrMaxSharedMemoryPerBlockOptin` for the
// device the launch goes to. `tensor_core_mma` is whether the kernel will run
// on an `SM80_16x8x16` MMA atom rather than `UniversalFMA`, which the caller
// gets from `compute_capability_major() >= 8`; see the header comment for why
// the wide tile is confined to that path and what measured it.
// `forced_tile_n` is 64 or 128 to pin the N tile (the
// `MLXCEL_QMM_NAIVE_TILE_N` override, which exists so the occupancy sweep in
// #1541 can measure both widths from one build) and 0 to select it.
//
// tile_m and tile_k are upstream's, unchanged.
inline QmmNaiveTile qmm_naive_choose_tile(
    int itemsize,
    int m,
    int group_size,
    long long smem_budget_bytes,
    bool tensor_core_mma,
    int forced_tile_n = 0) {
  const int tile_m = std::max(16, std::min(64, qmm_naive_next_power_of_2(m)));
  const int tile_k = std::max(64, group_size);

  // The 128-wide tile is considered only on the tensor-core MMA path, for
  // 16-bit elements, at group sizes up to 64.
  const bool wide_eligible =
      tensor_core_mma && itemsize <= 2 && group_size <= 64;

  const auto fits = [&](int tile_n) {
    return qmm_naive_smem_bytes(itemsize, tile_m, tile_n, tile_k) +
        kQmmNaiveSmemReserveBytes <=
        smem_budget_bytes;
  };

  int tile_n = 64;
  if (forced_tile_n == 64 || forced_tile_n == 128) {
    tile_n = forced_tile_n;
  } else {
    // Largest first, so the widest eligible tile that fits wins.
    for (int candidate : {128, 64}) {
      if (candidate == 128 && !wide_eligible) {
        continue;
      }
      if (fits(candidate)) {
        tile_n = candidate;
        break;
      }
    }
  }

  QmmNaiveTile choice{};
  choice.tile_m = tile_m;
  choice.tile_n = tile_n;
  choice.tile_k = tile_k;
  choice.smem_bytes = qmm_naive_smem_bytes(itemsize, tile_m, tile_n, tile_k);
  choice.needs_smem_opt_in = choice.smem_bytes > kQmmNaiveSmemOptInFreeBytes;
  choice.fits = fits(tile_n);
  return choice;
}

} // namespace mlxcel
