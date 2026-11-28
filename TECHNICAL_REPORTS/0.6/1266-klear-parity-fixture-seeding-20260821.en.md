# Technical Report: PR #1266 - Deterministic seeding for synthetic model fixtures

## Executive Summary

Issue #1265 reported that `models::klear::tests::the_prefill_is_causal_without_being_handed_a_mask` produced a different result on every run on GB10, at both precision settings, and that under MLX's default precision the spread straddled the test's own 1e-3 tolerance and failed 8 runs in 10. The issue's leading hypothesis was nondeterministic fp32 accumulation order in the quantized MoE path.

That hypothesis is wrong, and so is the premise it rests on. The variance came from the test fixture, not the backend. `filled_weights` advanced one LCG seed per key while iterating a `WeightMap`, which is a `HashMap`; Rust's `RandomState` randomizes that iteration order per process, so every run assigned a different noise block to a different tensor and each process built a **different random model**. Sorting the key walk removes the variance completely. The same fixture bug was present in three sibling test files and all four are fixed.

## 1. Problem Statement

The test compares a single-token forward against row 0 of a 4-token prefill and asserts they agree within 1e-3. Its subject is prefill causality: a model that builds no causal mask produces a bidirectional prefill, which is fluent and wrong.

Before this change the measured delta wandered over 5.9e-4 to 2.0e-3 at `MLX_ENABLE_TF32=1` and failed 8 of 10 runs, while passing 10 of 10 under the `MLX_ENABLE_TF32=0` pin from #1260. Because the value moved run to run, single observations of it produced two wrong conclusions during #1088's CUDA-runner validation, both of which were written into tracked files and later corrected by #1262, #1263 and #1264.

## 2. Technical Decisions

### 2.1 Fix the fixture, not the tolerance

The issue offered widening the tolerance as one option. Measurement rejected it. Handing the prefill an all-zero additive mask inside the same test binary makes the prefill genuinely bidirectional and moves row 0 by about **1.6**, roughly 1.5e3x the default-precision delta and 3e6x the pinned one. The 1e-3 bound therefore has enormous margin over the property the test names; it was never causality that was marginal. Widening it would have hidden a fixture defect behind a weaker assertion.

### 2.2 Sort the key walk rather than re-key the seed

Deriving each tensor's seed from a hash of its key would make ordering irrelevant by construction and is arguably the stronger remedy. Sorting was chosen because it is a one-line change per site whose effect was verified directly (20 processes, two precision modes, one distinct value each), and because re-keying changes every fixture value in four files and would need the whole verification repeated for no additional guarantee. The comment at each site records why the sort is load-bearing.

### 2.3 Withdraw the claims the nondeterminism had produced

Two statements at the `verify-test-cuda` gate definition in the `Makefile` were inferred from the broken fixture and are withdrawn rather than left standing:

- that the spread was "klear's own, not TF32's", and that it "varies at both precision settings, so the pin bounds the magnitude rather than the variance";
- that klear "runs a whole quantized MoE model rather than an isolated module over small dense tensors".

The second is refuted structurally. `synthetic_weights` emits only `.weight` for the expert stacks and no `.scales`, and `SwitchLinear::from_stacked_parts` selects the non-quantized `Regular` path when scales are absent. Probed directly on the loaded test model, `SwitchGLU::forward_fused_kernel(..).is_some()` is **false**, so neither the fused decode-MoE kernel nor the quantized `gather_qmm` path executes in this test.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/models/klear_tests.rs` | Sort the key walk in `filled_weights`; record what the 1e-3 bound is and is not |
| `src/models/afmoe_tests.rs` | Sort the key walk in `filled_weights` |
| `src/models/phixtral_tests.rs` | Sort the key walk in `filled_weights` |
| `src/models/bailing_moe_linear_tests.rs` | Sort the key walk at both fixture sites |
| `Makefile` | Rewrite the TF32 note at the `verify-test-cuda` gate definition |

A repository sweep for the pattern (an LCG advanced inside a loop over a map-derived key list) finds no sites beyond these four files.

## 4. Review Findings

See section 5 for validation. Review findings are recorded in the PR thread.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64, driver 580.173.02), MLX pin `9a795735`, commit `5334b77d`.

Max abs logit delta between the two arms, 10 processes per mode, after the fix:

| | `MLX_ENABLE_TF32=0` (pin) | `MLX_ENABLE_TF32=1` (MLX default) |
| --- | --- | --- |
| delta | 5.364418e-7 | 1.0485351e-3 |
| distinct values over 10 runs | 1 | 1 |
| verdict over 10 runs | 10 passed, 0 failed | 0 passed, 10 failed |

Both arms are additionally bit-identical to themselves across repeated forwards inside one process (`0e0`), at both precision settings. That scopes the claim: it says nothing about the quantized MoE accumulation-order nondeterminism recorded in #629 and PR #726, which is a different path this fixture never reaches.

Gates on the same tree:

- `make verify-test-cuda`: **8171 passed, 0 failed, 310 ignored**, exit 0.
- `cargo fmt --all -- --check`: clean.
- `cargo clippy --lib --tests --features cuda -- -D warnings`: clean.
- The four touched modules individually: klear 22, afmoe 23, phixtral 18, bailing_moe_linear 35; 98 tests, 0 failed.

## 6. Related Work

- #1265: the issue this closes.
- #1088, #1259, #1260, #1262, #1263, #1264: the TF32 pin and the three successive corrections to the note this PR rewrites.
- #629, PR #726: quantized MoE fp32 accumulation-order nondeterminism, a different path and still a real property.
- #1045: fused decode-MoE byte-identity on the real Klear checkpoint, distinct from this fixture.
- On the runtime exactness probes (#1188, #1189): they compare a verify block against a single-token chain byte for byte inside one process, so run-to-run variance would make the verdict noise-determined. The in-process repeat measurement above shows forwards on this path are bit-reproducible. One asymmetry is recorded without being changed: the Qwen 3.5 gate short-circuits on `!mlxcel_core::metal_is_available()` in both the server and offline CLI paths, while the three Gemma 4 arms carry no such guard and therefore do run the probe on CUDA. Not exercised against a real Gemma 4 checkpoint on CUDA here.
