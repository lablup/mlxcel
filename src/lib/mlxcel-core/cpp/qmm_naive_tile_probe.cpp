// Copyright 2026 mlxcel authors
//
// C shim over the `qmm_naive` CTA tile selector (lablup/mlxcel#1541), so
// `qmm_naive_tile_tests.rs` can sweep the shipped selection function on the
// host with no CUDA device, no CUDA toolkit and no GPU feature enabled.
//
// The header this includes is the same file the CUDA overlay
// `patches/mlx/backend/cuda/quantized/qmm/qmm_naive.cu` includes; CMake copies
// it into the fetched MLX tree verbatim (`configure_file ... COPYONLY`). There
// is one definition, and the test exercises it rather than a restatement of it.
// It is pure integer arithmetic with no CUDA in it, which is why this file is
// compiled unconditionally rather than behind the `cuda` feature: the sm_80+
// non-regression claim in #1541 has to be checkable on a machine with no
// NVIDIA hardware at all, including the macOS hosts this project also targets.

#include "../../mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmm_naive_tile.h"

extern "C" {

// Mirrored by `qmm_naive_tile_tests::Tile`, which is `#[repr(C)]`. Plain
// fixed-width members only, and `int` rather than `bool` for the two flags so
// the ABI does not depend on how either language sizes a bool.
struct MlxcelQmmNaiveTile {
  int tile_m;
  int tile_n;
  int tile_k;
  long long smem_bytes;
  int needs_smem_opt_in;
  int fits;
};

MlxcelQmmNaiveTile mlxcel_qmm_naive_choose_tile(
    int itemsize,
    int m,
    int group_size,
    long long smem_budget_bytes,
    int tensor_core_mma,
    int forced_tile_n) {
  auto tile = mlxcel::qmm_naive_choose_tile(
      itemsize,
      m,
      group_size,
      smem_budget_bytes,
      tensor_core_mma != 0,
      forced_tile_n);
  return MlxcelQmmNaiveTile{
      tile.tile_m,
      tile.tile_n,
      tile.tile_k,
      tile.smem_bytes,
      tile.needs_smem_opt_in ? 1 : 0,
      tile.fits ? 1 : 0};
}

// The two constants the selector is parameterised by, so a test asserting on
// boundaries reads them from the header rather than restating them.
long long mlxcel_qmm_naive_smem_reserve_bytes() {
  return mlxcel::kQmmNaiveSmemReserveBytes;
}

long long mlxcel_qmm_naive_smem_opt_in_free_bytes() {
  return mlxcel::kQmmNaiveSmemOptInFreeBytes;
}

} // extern "C"
