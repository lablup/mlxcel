// Added by mlxcel (lablup/mlxcel#1544). Not an upstream MLX file.
//
// The CUTLASS architecture tag selection for the grouped GEMM in
// `grouped_gemm_unaligned.cu`, factored out of that file as a pure integer
// function so it can be enumerated on the host with no CUDA device, no CUDA
// toolkit and no NVIDIA hardware at all. `grouped_gemm_unaligned.cu` calls it
// with `cu::Device::compute_capability_major()`, and
// `src/lib/mlxcel-core/cpp/grouped_gemm_arch_probe.cpp` exposes the same
// function to `grouped_gemm_arch_tests.rs` through a C shim. Both callers
// share this one definition, so the tested function is the shipped one.
//
// # What was wrong
//
// Upstream wrote the pre-Ampere arm as:
//
//   if (device.compute_capability_major() < 8) {
//     f(type_identity<cutlass::arch::Sm75>{});
//
// `cutlass::arch::Sm75` names Turing. A Tesla V100 is compute capability 7.0,
// one generation below Turing, and has no `m16n8k8` MMA at all; its tensor
// cores expose only the `8x8x4` HMMA shape. The tag therefore described
// hardware the device does not have on every Volta part that reached this
// dispatch.
//
// # Why nothing broke
//
// Measured rather than assumed, because "the model still produces correct
// text" is not evidence about a GEMM. On a compute-70 build of the shipped
// translation unit the tag never reaches device code:
//
//   - The configuration this arm selects is `GemmConfiguration`'s primary
//     template, which is `cutlass::arch::OpClassSimt` with
//     `InstructionShape<1, 1, 1>`. That is plain FFMA. The two tensor-core
//     specializations are constrained on `Arch::kMinComputeCapability >= 80`,
//     so no pre-Ampere tag can select an MMA atom, `m16n8k8` included.
//   - Consequently CUTLASS erases the tag. No `cutlass::arch::Sm*` token
//     survives into any of the 51 device symbols the translation unit emits,
//     and retagging the arm from `Sm75` to `Sm70` leaves the same 51 device
//     symbols with byte-identical bodies, 58,211,476 bytes of
//     `cuobjdump --dump-sass` text compared per symbol. Only the order the
//     cubin emits them in moves.
//   - Compiling the file with a separate `Sm70` arm added *alongside* the
//     `Sm75` one is the control: same 51 device symbols again, whole dump
//     byte-identical at 493,538 lines, against 26 extra host-side template
//     instantiations and 194,704 more bytes of object.
//
// So the pre-Ampere arm is one arm for a reason. Splitting Turing back out
// buys nothing but a second host-side copy of instantiations that emit the
// same device code. The tag names the floor of the range the arm covers rather
// than a target, and `Sm70` is the correct floor. Full record:
// `docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`.
//
// # When this stops being safe
//
// The erasure is a property of the configuration, not of CUTLASS. The moment
// the pre-Ampere `GemmConfiguration` selects `OpClassTensorOp` (the obvious
// candidate is Volta `884` MMA work of the kind #1543 is doing for
// `qmm_naive`), the tag becomes load-bearing and a Turing part tagged `Sm70`
// would lose `m16n8k8`. `grouped_gemm_unaligned.cu` carries a `static_assert`
// on that configuration's `OpClass` so the build fails on the day that changes
// rather than silently downgrading Turing, and the assert names this file.

#pragma once

namespace mlxcel {

// The CUTLASS architecture tags the grouped GEMM dispatches over. The values
// are the compute capability each tag names, so a reader does not have to know
// CUTLASS to read a test failure.
//
// `Sm75` is deliberately absent. An enumerator here is a template argument
// `grouped_gemm_unaligned.cu` instantiates the whole GEMM over, so listing a
// tag this function never returns would emit a dead copy of every
// instantiation in the arm: 26 host-side functions and 195 KB of object, for
// device code that is byte identical to the `Sm70` arm's. Turing is covered by
// `Sm70` for the reason given above.
enum class GroupedGemmArch {
  Sm70 = 70,
  Sm80 = 80,
  Sm90 = 90,
};

// Maps a device's compute capability major version to the CUTLASS arch tag the
// grouped GEMM instantiates for it.
//
// Only the major version is consulted, which is the whole reason this is a
// pure function worth testing: the pre-Ampere arm deliberately covers 7.0
// through 7.5 with one tag (see the header comment), so a minor version would
// be an argument the function does not use.
//
// Anything below compute capability 7.0 also lands on `Sm70`. MLX's CUDA
// backend does not support those parts, so this is a floor rather than a
// claim; it is still strictly closer to the truth than the `Sm75` they used to
// receive.
constexpr GroupedGemmArch grouped_gemm_arch_for(int compute_capability_major) {
  if (compute_capability_major > 8) {
    return GroupedGemmArch::Sm90;
  }
  if (compute_capability_major == 8) {
    return GroupedGemmArch::Sm80;
  }
  return GroupedGemmArch::Sm70;
}

} // namespace mlxcel
