# PR #2711: Allow for backend rounding in the ROCm imagegen parity tests

**Date**: 2026-10-28 (written before merge; PR open at `5cbf1bb8`)
**Status**: Verified on ROCm gfx1151 only; Metal and CUDA not run
**Languages**: Rust (tests only)
**Risk Level**: Low (no production code changes; the risk is that the looser bounds hide a future porting bug on non-bit-exact backends)

## Executive Summary

PR #2711 (closes #2709) makes five imagegen parity tests from the Ideogram 4 port (#2703) and the Qwen-Image port (#2706) pass on ROCm gfx1151. Those tests compared against fixtures recorded on mlx 0.32.4 CUDA, and they required either bit equality or a bound that only holds on a backend that reproduces CUDA bit for bit. The investigation in #2709 found no kernel defect and no porting bug: the failures are last-bit differences in transcendental functions and bf16 rounding, and the tiny random models used as fixtures amplify them.

The PR changes test code only. The noise draw is compared within 2 ulps, the RoPE tables within 1e-4 absolute, and the bf16 forward bounds depend on the backend: Metal and CUDA keep the strict 1e-3 (Ideogram 4) and 2e-3 (Qwen-Image), and every other backend (ROCm, and CPU-only builds) gets 5e-2 and 2e-2. Because the looser bf16 bound can no longer detect a device kernel defect on ROCm, a new test runs the Ideogram 4 forward in f32 on the GPU and on the CPU stream and requires them to agree within 1e-4. On ROCm, the cost is that a porting bug smaller than the bf16 noise floor (the fused RMSNorm mutation at 1.5e-2 is the measured example) is caught only on Metal and CUDA.

## 1. Problem Statement

`make verify-rocm-held` failed on main on a gfx1151 host in five tests:

| Test | Assertion | ROCm result |
|---|---|---|
| `ideogram4::pipeline::tests::noise_reproduces_upstream_global_seeding` | exact `assert_eq!` | element 5 is one ulp off (-1.3240628 vs -1.3240626) |
| `ideogram4::transformer::tests::the_rope_tables_match_upstream` | exact `assert_eq!` on cos and sin | max differences 3.2e-6 (cos), 3.0e-5 (sin) |
| `ideogram4::transformer::tests::conditional_and_unconditional_forwards_match_upstream` | rel-RMS < 1e-3 | 1.41e-2 conditional, 2.37e-2 unconditional |
| `qwen_image::transformer::tests::generate_forward_matches_upstream` | rel-RMS < 2e-3 | 5.9e-3 |
| `qwen_image::transformer::tests::edit_forward_with_and_without_the_kv_cache_matches_upstream` | rel-RMS < 2e-3 | 2.8e-3 |

The question #2709 had to answer before any tolerance could change was whether these numbers come from a ROCm overlay kernel defect (which belongs in the mlxcelverse overlay and `LOCAL_FIXES.md`), from a porting bug (which belongs in the model code), or from rounding (which belongs in the tests).

### Why it is rounding

- **Transcendentals.** The noise draw goes through `erfinv` and `log`, and the RoPE tables through `pow`, `sin` and `cos`. These differ in the last bit or two between CUDA, the CPU stream and ROCm. The CPU stream already missed CUDA by one ulp at noise element 510, which the old test allowed for with a 1e-7 absolute tolerance on the tail only. The RoPE arguments are `position * inv_freq` with positions past 65536, so a one-ulp difference in `inv_freq` grows to about 3e-5 in the table. The tables are rounded to bf16 (spacing about 4e-3 near 1) before the model uses them, so this difference never reaches the model.
- **Chaotic tiny models.** The fixtures use tiny models with random bf16 weights. Upstream's own bf16 forward is 0.37 (Ideogram 4) and 2.9e-2 (Qwen-Image) away from the f32 forward of the same weights, so the network amplifies a flipped bf16 bit far more than a trained model would. Two bf16 runs that do not match bit for bit land about 1e-2 apart.
- **CPU control.** The CPU stream also misses the CUDA reference (Ideogram 4 conditional 1.61e-2, Qwen-Image 5.2e-3). The bf16 gap is therefore not specific to ROCm: Metal and CUDA happen to be bit exact at this size, and other backends are not.
- **f32 control.** The f32 forward of the same weights agrees between the ROCm GPU and the CPU stream to 3e-5 (Ideogram 4) and 7e-7 (Qwen-Image). A defective RMSNorm, attention, matmul, RoPE application or mask would show up here, and none did.
- **Localization.** In Ideogram 4 the first divergence between the ROCm GPU and the CPU is in the timestep MLP: the sinusoid going in is identical, and 1.7e-3 comes out, which is bf16 matmul and bias rounding. Every later layer amplifies it. In Qwen-Image the `t = 0.5` cases match upstream exactly on ROCm and the `t = 0.875` cases do not. Computing the timestep frequencies and the sinusoid on the host in f64 did not change that, which rules out the timestep embedding's transcendentals as the source.

## 2. Change Summary

| File | Change |
|---|---|
| `src/imagegen/ideogram4/pipeline_tests.rs` | New `ulps(a, b)` helper (bit distance for same-sign finite values, `u32::MAX` across signs) and `assert_noise`, gated by `MAX_NOISE_ULPS = 2`. All three noise slices (seed 0 head, seed 0 tail at 510, seed 7 head) use it, replacing two exact `assert_eq!`s and the tail-only 1e-7 tolerance. |
| `src/imagegen/ideogram4/transformer_tests.rs` | RoPE table test compares element-wise within `MAX_TABLE_ABS_ERR = 1e-4`, reporting the table name and index. `MAX_REL_RMS` becomes `max_rel_rms()`: 1e-3 when `gpu_backend_kind()` is Metal or CUDA, 5e-2 otherwise, with the measurements in the doc comment. The forward test now holds `lock_default_device()`. New test `the_f32_forward_agrees_between_the_gpu_and_the_cpu`. |
| `src/imagegen/qwen_image/transformer_tests.rs` | `TINY_REL_RMS` becomes `tiny_rel_rms()`: 2e-3 on Metal and CUDA, 2e-2 otherwise, with the measurements in the doc comment. |

No production code, no overlay change, and no `LOCAL_FIXES.md` entry.

### The new f32 test

`the_f32_forward_agrees_between_the_gpu_and_the_cpu` widens every fixture weight (`w.*`) to f32, builds `Ideogram4Transformer` twice, once on the default GPU device and once inside `DefaultDeviceGuard::cpu()`, and runs both the conditional and the unconditional forward. It asserts rel-RMS below 1e-4 for each (measured 3e-5 and 5e-5 on gfx1151). It returns early when no GPU backend is available, since there is nothing to compare. It holds `lock_default_device()` for the whole span because it moves the process-global default device, and the bf16 forward test in the same file takes the same lock so the two cannot interleave under a parallel `cargo test`.

## 3. Technical Decisions

### 3.1 Fix the tests, not the code

The alternative to looser tolerances was to make ROCm reproduce CUDA bit for bit: match the accumulation order of the matmuls, use the same `exp`, `erfinv` and `sin` implementations, and compute the timestep path on the host. The investigation tried the cheapest such step (host f64 timestep frequencies and sinusoid in Qwen-Image) and it did not move the result, so the divergence is spread across ordinary bf16 kernels. Making a whole backend bit-compatible with another is not a goal of the overlay, and the CPU stream, which nobody would call broken, misses CUDA by the same margin. The fixtures are rounding-sensitive by construction (random weights), so the test is the part that was miscalibrated.

### 3.2 Backend-keyed bounds instead of one loose bound

A single 5e-2 bound on every backend would have been simpler, but it would discard the strongest evidence the suite has. On Metal and CUDA the port reproduces upstream bit for bit, and the 1e-3 bound there catches the pairwise (Z-Image) rotation, the fused RMSNorm kernel and the missing mask on the biased projections. Keying the bound on `gpu_backend_kind()` keeps that check where it works and documents why it cannot work elsewhere. `GpuBackendKind::None` (a CPU-only build) falls into the loose branch, which matches the measured CPU-stream gap of 1.61e-2.

### 3.3 Choosing 5e-2 and 2e-2

The bounds sit between the measured noise and the measured mutations:

| Model | Noise on ROCm and CPU | Loose bound | Caught mutations (ROCm) | Missed mutation (ROCm) |
|---|---|---|---|---|
| Ideogram 4 | 1.41e-2 to 2.37e-2 | 5e-2 | pairwise rotation 5.2e-1 and 3.2e-1, missing mask 2.0e-1 | fused RMSNorm 1.5e-2 |
| Qwen-Image | 2.8e-3 to 5.9e-3 | 2e-2 | not measured | not measured |

The Ideogram 4 margin over the worst measured noise is about 2x, and the closest caught mutation is 4x above the bound. Qwen-Image has about 3.4x headroom over its noise, but no mutations were measured against its loose bound, so its detection power on ROCm is unknown.

### 3.4 An f32 cross-device test as the replacement for the lost check

Loosening the bf16 bound on ROCm loses the ability to tell a rounding gap from a broken device kernel. The f32 forward removes most of the bf16 amplification (3e-5 between GPU and CPU instead of about 1e-2), so a 1e-4 bound there is tight enough to catch a wrong norm, attention, matmul or mask in the device path. Comparing GPU against CPU in the same process, instead of against a stored f32 fixture, needs no new fixture and isolates exactly what the overlay owns: the device kernels.

The test checks something narrower than the bf16 parity test did. A porting bug is computed the same way on both devices, so GPU-vs-CPU agreement says nothing about whether the port matches upstream. On ROCm, port fidelity below 5e-2 is now enforced only by the Metal and CUDA runs of the strict bound.

### 3.5 ulps for the noise, absolute error for the tables

The noise values span several orders of magnitude (from 1e-2 to 1.4), so a single absolute tolerance would be either too loose for small values or too tight for large ones. A 2-ulp gate scales with the value and still rejects a wrong key derivation or sampling order, which changes every element completely. The RoPE table entries are cos and sin values in [-1, 1] and the relevant scale is the bf16 spacing they are rounded to, so an absolute 1e-4 bound (40x below that spacing, 3x above the measured 3.0e-5) expresses the requirement directly.

## 4. Validation

- `cargo test --lib --profile test-fast --features rocm imagegen:: -- --test-threads=1` on gfx1151: 374 passed, 0 failed, 30 ignored. Before the change the five tests in section 1 failed.
- `cargo clippy --lib --tests --features rocm -- -D warnings` and `cargo fmt --check` clean.
- Not run: Metal and CUDA. Those backends keep the old strict bounds for the bf16 forwards, but the noise and RoPE tests are looser there too, and the new f32 test runs there for the first time.

## 5. Learning Points

- **Tiny random models are rounding amplifiers.** A 0.37 gap between bf16 and f32 forwards of the same random weights means any non-bit-exact bf16 backend will be about 1e-2 off. A parity bound calibrated on a bit-exact backend says nothing about what other backends can achieve. Measure the bf16-vs-f32 gap of a fixture before choosing a cross-backend bound.
- **Use the CPU stream as a control.** If the CPU stream misses the reference by the same margin as the GPU, the GPU is not the problem. This check is cheap and settled most of the question in #2709.
- **Separate kernel correctness from port fidelity.** Running f32 on GPU and CPU tests the device kernels; the comparison against an upstream fixture tests the port. One loose test cannot do both, and the PR says which backend enforces which property.
- **Large arguments to periodic functions amplify ulps.** `sin(position * inv_freq)` with positions past 65536 turns a one-ulp change in `inv_freq` into 3e-5 of output. Exact equality on RoPE tables is not portable.
- **Tests that move the default device must hold the global lock.** `DefaultDeviceGuard` restores the device but does not exclude other threads; `lock_default_device()` (not reentrant) does. Any test in the same file that depends on the default device must take it too.

## 6. Follow-up and Limitations

- **f32 bound unmeasured on Metal and CUDA.** The 1e-4 bound in `the_f32_forward_agrees_between_the_gpu_and_the_cpu` was measured only on ROCm. A backend that computes f32 matmuls in reduced precision (for example TF32 on CUDA, if enabled) could exceed it. The first Metal or CUDA run should confirm the margin.
- **Qwen-Image has no f32 cross-device test.** On ROCm its bf16 bound is the only check, and its mutation sensitivity at 2e-2 was not measured. The f32 control (7e-7) was measured during the investigation but not committed as a test.
- **Sub-noise port bugs on non-bit-exact backends.** A change like the fused RMSNorm kernel (1.5e-2) passes on ROCm and CPU-only builds. That is acceptable only as long as Metal or CUDA runs the suite regularly; while hosted CI is down, ROCm-only verification does not cover it.
- **Other parity tests in the suite.** Only the five failing tests were recalibrated. Other fixtures recorded on CUDA with exact or bit-exact-calibrated bounds would fail the same way on a new backend; the pattern here (backend-keyed bound plus f32 cross-device control) is the template for them.
- **Default-device lock coverage.** The Qwen-Image tests do not take `lock_default_device()`. Under `--test-threads=1` that does not matter, but a parallel `cargo test --lib` could run them while another module holds the CPU default, measuring them on the wrong device.

## References

- PR #2711: https://github.com/lablup/mlxcel/pull/2711
- Issue #2709: ROCm parity tests for ideogram4 and qwen_image demand CUDA bit equality
- PR #2703: Ideogram 4 text-to-image port (origin of the Ideogram 4 fixtures)
- PR #2706: Qwen-Image-2.1 generation and editing port (origin of the Qwen-Image fixtures)
- Issue #1421: process-wide default-device lock (`lock_default_device`)
- `src/lib/mlxcel-core/src/streams.rs` (`DefaultDeviceGuard`, `lock_default_device`), `src/lib/mlxcel-core/src/hardware.rs` (`GpuBackendKind`)
