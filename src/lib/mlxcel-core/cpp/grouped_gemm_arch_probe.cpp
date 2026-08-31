// Copyright 2026 mlxcel authors
//
// C shim over the grouped GEMM CUTLASS architecture tag selection
// (lablup/mlxcel#1544), so `grouped_gemm_arch_tests.rs` can enumerate the
// shipped decision on the host with no CUDA device, no CUDA toolkit and no
// GPU feature enabled.
//
// The header this includes is the same file the CUDA overlay
// `patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu` includes; CMake
// copies it into the fetched MLX tree verbatim (`configure_file ... COPYONLY`
// over a recursive glob of `patches/mlx/backend/cuda`). There is one
// definition, and the test exercises it rather than a restatement of it.
//
// It is a pure integer function with no CUDA in it, which is why this file is
// compiled unconditionally rather than behind the `cuda` feature. #1544 has to
// answer "the sm_80 and sm_90 arms are untouched" on a host that has neither,
// and epic #1536 has no Ampere-or-later part; enumeration here settles it
// everywhere, including the macOS hosts this project also targets.

#include "../../mlx-cpp/patches/mlx/backend/cuda/gemms/grouped_gemm_arch.h"

extern "C" {

// Returns the `mlxcel::GroupedGemmArch` enumerator as its compute capability
// value (70, 80, 90), which is what the enumerators are defined to be. An
// `int` return keeps the ABI independent of how either language sizes an enum.
int mlxcel_grouped_gemm_arch_for(int compute_capability_major) {
  return static_cast<int>(
      mlxcel::grouped_gemm_arch_for(compute_capability_major));
}

} // extern "C"
