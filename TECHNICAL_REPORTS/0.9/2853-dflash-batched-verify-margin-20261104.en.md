# Technical Report: PR #2853 - test(speculative): gate batched DFlash parity on measured jitter

**Date**: 2026-11-04
**Status**: In review (open PR; closes #2851)
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #2853 establishes that the batched DFlash verify forward is not byte-identical to a single-row forward, then re-expresses the batched parity gate as a calibrated evaluation-path jitter margin. On an M3 Ultra with `qwen3.5-4b-4bit` and `qwen3.5-4b-dflash` at block 16, the hard byte-equality gate failed at row 0 token 14 (batched 5388 versus B=1 15217). Four identical prompts in one burst also diverged, with `clamped_rounds=0`, no ragged round, and all four rows parting at the same token, so the accept policy and the uniform clamp cannot be the cause. A one-shot B=1 arbiter ranks the batched token first and the B=1 token 0.25 logits behind, the near-tie class the Gemma 4 MTP verifier already gates with `EVAL_PATH_JITTER_MARGIN`. The gate now admits a first mismatch only within `BATCHED_VERIFY_JITTER_MARGIN = 1.0` logits, four times the measured gap, and a committed margin-removal arm rejects all 4 of 4 observed mismatches at margin 0. Outcome (a), byte identity, was rejected on measurement: reproducing B=1 shapes means per-row verify evaluation against per-row KV/GDN cache views, the Gemma 4 per-row architecture (#2190), which would spend the 1.07x batching win measured in #2442. The margin is calibrated on one host and one prompt set; no runtime B>1 exactness probe was implemented.

---

## 1. Problem Statement

### 1.1 Background

Issue #2442 item 1 asks to replace the uniform clamp in the batched DFlash round loop with per-row accepts, gated on `greedy_parity_dflash_qwen35_batched_b4_matches_b1`. That gate failed on unmodified `main`, so the accept policy was not the blocker. Issue #2851 asked for the prerequisite experiment: establish whether the batched `[B, block]` target verify forward is byte-identical to the `[1, block]` forward, and either make it identical (outcome a) or re-express the gate as a measured jitter margin with a structural-defect control (outcome b).

The verify call is `target.verify_forward_with_capture_layers` on a `[B, bs]` input from the batched round loop, reaching `Qwen35Model::forward_speculative` through `SpeculativeTarget`. The server's batched burst drives the same loop from `run_dflash_batched_on_target`. The existing runtime exactness probe, `probe_block_chain_exactness`, compares a block against a chain at B=1 only, so it cannot see a batch-size-dependent difference.

### 1.2 Existing Issues

- **The byte-equality gate never passed on Metal.** Before this PR the B=4 gate reported 96 rounds, 93 clamped, mean accept per row-round 2.938 versus committed 0.323, and row 0 first diverged at token 14 (5388 versus 15217).
- **The accept policy was a plausible but wrong explanation.** A ragged accept or a stale prefix can move a later token, so the failing gate alone did not isolate the cause.
- **The divergence is a near-tie, not a corrupt distribution.** The one-shot B=1 arbiter's argmax is the batched token 5388, with the B=1 token 15217 only 0.25 logits behind. Every other row of both runs is byte-identical.
- **The existing Qwen 3.5 exactness probe cannot cover this class.** It evaluates at B=1 only, so a runtime decision to decline a non-exact batched width had no measurement to stand on.
- **A hard byte gate would gate the whole batching path on an expensive rewrite.** Byte identity at B=4 requires evaluating the target verify forward row by row against per-row views of the rectangular KV/GDN caches.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| The margin hides a real structural parity break | Medium (correctness of the test gate) | Low: the margin admits a mismatch only when both tokens are within 1.0 logit of a third evaluation path, and the margin-removal arm must reject the same mismatches at 0 |
| The margin is calibrated on the wrong host or prompt set | Medium (a different host fails closed, blocking CI) | Medium: calibration covers one M3 Ultra and one prompt set |
| A future change raises the margin without evidence | Medium (tolerance becomes permissive) | Low: the constant carries a do-not-raise instruction and the control is committed |
| The batched path is served at runtime while no runtime exactness probe covers B>1 | Low to Medium (unverified runtime surface) | Certain: this PR deliberately does not add one |

---

## 2. Technical Review

### 2.1 Security

No runtime code, input parsing, authentication, or data path changes. The PR touches two ignored real-model tests, their shared harness, and one documentation paragraph. There is no new input surface and no logging of user data.

### 2.2 Performance

The change is test-only and does not alter the serving path. The performance question is architectural: outcome (b) keeps the batched path at its measured throughput, while outcome (a) would add per-row verify evaluation. On M3 Ultra the clamp arm measured B=1 at 77.70 / 77.23 / 77.83 tok/s and B=4 at 82.96 / 83.00 / 83.00 tok/s aggregate over 512 tokens, about 1.07x. The per-row rewrite needed for byte identity would spend that entire win. The new harness runs one extra one-shot B=1 forward per diverging row at its first mismatch, only inside ignored real-model tests.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Test names are preserved; the new control is an additional ignored test.
- **New Dependencies**: none.
- **Compatibility**: the tests remain `#[ignore]`-gated real-model tests for the CI hardware lane. `MLXCEL_SDPA_DETERMINISTIC=1` produces the same verdict.

### 2.4 Code Quality

The burst runner, the top-gap arbiter, and the margin rule moved into a new shared harness module, `tests/speculative_parity/qwen35_dflash_harness.rs`, so the hard gate and the identical-prompts control cannot drift apart. The constant `BATCHED_VERIFY_JITTER_MARGIN` carries the full calibration record next to its definition. The margin-removal control is part of the test rather than a comment, so a future relaxation that makes the margin non-load-bearing fails instead of passing silently.

---

## 3. Technical Decisions

### 3.1 Outcome (b): measured jitter margin instead of byte identity

**Context:**
The gate demanded byte identity between a `[4, block]` and a `[1, block]` target verify forward. The identical-prompts control showed the divergence survives when every row drafts the same block and every round commits its walk, so the difference is produced inside the batched target forward.

**Alternatives Considered:**

| Option | Pros | Cons |
|---|---|---|
| Option A: force byte identity by pinning kernel dispatch or per-row verification | Exact gate, no tolerance | Per-row evaluation needs per-row KV/GDN cache views (Gemma 4 per-row architecture, #2190); spends the 1.07x batching win from #2442; hard to prove no other shape changes dispatch |
| Option B: calibrated evaluation-path jitter margin | Keeps the batching win, matches the existing Gemma 4 MTP precedent, fails closed on decisive positions | Admits near-ties by construction; calibration is host-specific |
| **Chosen: Option B** | Measured, small relative to the observed gap, protected by a margin-removal control | Does not make the batched forward exact; leaves a host-specific constant |

**Rationale:**
The leading dimension of the verify block changes which quantized-projection and SDPA kernels MLX dispatch and how their reductions tile. That is the near-tie class the project already documents for prefill partitioning and gates with `EVAL_PATH_JITTER_MARGIN` for the Gemma 4 MTP verifier. Under greedy decoding every emitted token is the target's argmax over the accepted prefix, so the drafter's proposals cannot change the emitted chain; only the target forward's evaluation path can. Forcing B=1 shapes therefore means per-row evaluation of the target forward, not a cheaper dispatch tweak.

**Trade-offs:**
The gate is no longer a byte-equality assertion. It becomes a near-tie assertion: a first mismatch passes only when a third evaluation path (one-shot B=1) places both candidate tokens within the margin of its top token. A structural break that moves the distribution further than 1.0 logit still fails.

### 3.2 The margin is 1.0, four times the measured gap

**Context:**
The measured maximum divergence is 0.25 logits: the arbiter's top-gap for the batched token is 0, and for the B=1 token is 0.25. The margin needs to cover the observed evaluation-path noise without swallowing a decisive flip.

**Rationale:**
`BATCHED_VERIFY_JITTER_MARGIN` is set to 1.0, four times the measured maximum. The value is not a free widening because the rule measures both tokens against a third path, and the committed margin-removal control proves the margin is load-bearing: at margin 0 every observed mismatch is rejected (4 of 4 in the identical-prompts control). A decisive position where the batched token wins by more than 1.0 logit, or where either candidate is more than 1.0 logit below the arbiter's top, still fails closed.

**Trade-offs:**
The constant is calibrated to this class of divergence on this hardware. A different quantization, kernel family, or MLX version could produce a first mismatch above 1.0 logit at a legitimate near-tie, in which case the test fails loudly and the constant needs a fresh measurement, not a blind raise.

### 3.3 Measure both tokens against a one-shot B=1 arbiter

**Context:**
Comparing only the batched token's logit to the B=1 token's logit would accept any pair the batched path happens to rank closely, including pairs the arbiter ranks decisively. A one-shot forward is itself a third evaluation path; it can differ from the incremental chain.

**Rationale:**
`b1_arbiter_top_gaps` prefills `prompt + emitted_prefix[..index]` through `Qwen35Model::forward_with_sequence_id` on a fresh sequence, takes the last-position logits, and returns `top - logit[want]`, `top - logit[got]`, and the arbiter's argmax. Both gaps must lie in `[0, margin]`. This distinguishes "two candidates the model cannot separate" from "the batched path moved onto a token the model ranks far behind", which is what a structural defect looks like.

### 3.4 The identical-prompts control isolates the accept policy

**Context:**
The hard gate uses four different prompts, so ragged accepts and the uniform clamp are active. A failure there could be caused by cache layout rather than the target forward.

**Rationale:**
The control runs `[prompts[0]; 4]`. Every row drafts the same block from the same prefix, so every round commits its full walk. The test asserts `clamped_rounds == 0` and `ragged == false` before applying the parity rule, which makes the control discriminating: if the clamp or ragged accepts were the cause, identical rows would not diverge. They do diverge, at the same token and with the same gaps, so the difference is in the batched verify forward itself.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
tests/speculative_parity.rs
  ├── gemma4_batched.rs
  └── qwen35_dflash_batched.rs   (burst runner + byte-equality helper inline)

[After]
tests/speculative_parity.rs
  ├── gemma4_batched.rs
  ├── qwen35_dflash_harness.rs   (burst runner, arbiter, margin rule, constants)
  └── qwen35_dflash_batched.rs   (hard gate + identical-prompts control)
```

`tests/speculative_parity.rs` registers the harness with `#[path = "speculative_parity/qwen35_dflash_harness.rs"] mod qwen35_dflash_harness;` before the test module.

### 4.2 Key Code Changes

**File: `tests/speculative_parity/qwen35_dflash_harness.rs` (new)**

The calibrated constant carries its evidence and a do-not-raise instruction:

```rust
pub(crate) const BATCHED_VERIFY_JITTER_MARGIN: f32 = 1.0;
```

The arbiter runs a fresh one-shot B=1 prefill and returns both top-gaps:

```rust
fn b1_arbiter_top_gaps(
    model: &mlxcel::models::Qwen35Model,
    seq_raw: u64,
    prompt: &[i32],
    emitted_prefix: &[i32],
    want: i32,
    got: i32,
) -> (f32, f32, i32)
```

The margin rule rejects a row when either token's gap exceeds the margin, which is why the same function is the margin-removal control at `margin == 0.0`:

```rust
pub(crate) fn rows_outside_margin(ties: &[NearTie], margin: f32) -> Vec<String>
```

**Reason for change:** Both tests need the same burst runner and the same arbitrated rule. Keeping them in one harness prevents the hard gate and the control from implementing subtly different tolerances.

**File: `tests/speculative_parity/qwen35_dflash_batched.rs`**

The hard gate now collects arbiters and applies the margin instead of asserting byte equality directly:

```rust
let (ties, structural) = first_mismatch_arbiters(
    model, "dflash B=4", &prompts, &batched, &reference, ARBITER_SEQ_BASE_GATE,
);
let outside = rows_outside_margin(&ties, BATCHED_VERIFY_JITTER_MARGIN);
assert!(structural.is_empty(), ...);
assert!(ragged, "no round had ragged accepts; the prompts do not exercise the clamp");
assert!(outside.is_empty(), ...);
```

The new control asserts the two premises, then the parity rule, then the margin-removal arm:

```rust
assert_eq!(batched.clamped_rounds, 0, ...);
assert!(!ragged, ...);
let at_zero = rows_outside_margin(&ties, 0.0);
assert!(!at_zero.is_empty(), "margin is not load-bearing ...");
```

**Reason for change:** `clamped_rounds == 0` and `!ragged` make the control's conclusion attributable to the batched forward. The margin-removal arm prevents the margin from silently becoming a tolerance that admits everything.

**File: `docs/CONTINUOUS_BATCHING.md`**

Records that an equal-length queued DFlash window on Qwen 3.5 is not bitwise identical to sequential service, that the leading dimension changes kernel dispatch, and that the parity gate uses a calibrated 1.0-logit near-tie margin backed by the four-identical-prompts control (#2851).

### 4.3 Data Model Changes

None.

---

## 5. Learning Points

### 5.1 Evaluation-path jitter versus structural parity

**Concept:**
On Metal, the same arithmetic organized into different leading dimensions can select different quantized-projection and SDPA kernels and tile reductions differently. The result is a small, deterministic logit shift, not nondeterminism. Where two candidates are within that shift, a greedy argmax can legitimately flip.

**Application in this PR:**
The shift is 0.25 logits at the observed divergence, and the flip is a near-tie: the arbiter ranks the batched token first. The gate distinguishes the jitter regime from a structural break by requiring both candidates to sit within 1.0 logit of a third evaluation path. A structural defect moves the distribution by far more than this and flips decisive positions, which the same rule rejects.

**Common Use Cases:**
- Batched versus single-row parity tests on quantized GPU backends.
- Comparing incremental and one-shot prefill paths.
- Any tolerance gate where the failure mode to exclude is a positional or cache-layout defect, not floating-point noise.

### 5.2 Identical-input controls isolate policy from execution

**Concept:**
When a batched path can diverge for several reasons, running identical inputs removes the policy variables. Every row drafts the same block and commits the same accept count, so ragged accepts, clamping, and stale prefixes are inert.

**Application in this PR:**
`[prompts[0]; 4]` with asserted `clamped_rounds == 0` and `!ragged_round` turns "the gate failed" into "the batched target forward differs", which is the only conclusion that justifies a margin instead of an accept-policy fix.

**Common Use Cases:**
- Batched decode and speculative-decoding parity.
- Any concurrent pipeline where a shared code path has per-row policy state.

### 5.3 A tolerance needs a removal control

**Concept:**
A margin that is only added, never tested for necessity, can quietly widen until it admits the defects it was meant to catch. The removal control applies the same evidence at a stricter setting and requires it to fail.

**Application in this PR:**
The identical-prompts control reruns the observed mismatches at `margin == 0` and asserts at least one is rejected. Here all 4 of 4 are rejected, so the 1.0 margin is load-bearing rather than decorative.

**Example Code:**
```rust
let at_zero = rows_outside_margin(&ties, 0.0);
assert!(!at_zero.is_empty(), "margin is not load-bearing ...");
```

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---|---|---|
| DFlash | Block-draft speculative decoding with a diffusion-style drafter | The batched verify path under test |
| Evaluation-path jitter | Small deterministic logit shift from different kernel dispatch or reduction tiling | The cause of the 0.25-logit divergence |
| Near-tie | Two candidates within the jitter margin, where a greedy argmax can flip | What the margin gate admits |
| One-shot arbiter | A fresh B=1 full-prefill forward used as a third evaluation path | Decides whether a mismatch is a near-tie |
| Uniform clamp | Round policy that commits the same accept count on every row | Inert in the identical-prompts control, which isolates the forward |
| Margin-removal control | Rerunning the observed mismatches at margin 0 | Proves the margin is load-bearing |

### Related Technologies/Frameworks

- **MLX**: Apple's array framework and Metal kernel dispatch, where leading-dimension-dependent kernel selection produces the divergence.
- **Gemma 4 MTP verifier**: the existing `EVAL_PATH_JITTER_MARGIN` and `check_row_parity_with_near_tie` precedent this PR ports to Qwen 3.5.

### Related PRs/Issues

- Issue #2851: the prerequisite experiment this PR closes.
- Issue #2353: the batched DFlash clamp and the original B=4 versus B=1 gate.
- Issue #2442: the batching win and the per-row accept item this work re-scopes.
- Issue #2190: the Gemma 4 per-row verify architecture that byte identity would require.
- PR #2170: prefill partitioning and the documented near-tie class.

---

## 7. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 4 |
| Lines added | +579 |
| Lines deleted | -194 |
| Tests added | 1 new ignored real-model control, plus a shared harness |
| Runtime behavior changed | None |

### Changes by Category

| Category | Count | Summary |
|---|---|---|
| Tests | 3 files | Shared harness, hard gate re-expressed as a margin gate, identical-prompts control with margin-removal arm |
| Documentation | 1 file | `docs/CONTINUOUS_BATCHING.md` records the non-bitwise window and the margin |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `a3caa55` | test | test(speculative): gate batched DFlash parity on measured jitter |

---

## 8. Follow-up Actions

### Required

- [ ] Calibrate the margin on other backends and checkpoints (GB10/CUDA is the known second backend where a B=2 window of identical prompts already diverged in the 2026-10-18 #2442 comment) before treating 1.0 as portable. The constant must not be raised without a fresh measurement and a rerun of the margin-removal control.
- [ ] Re-scope #2442 item 1 with this measured result. Per-row accepts stay blocked until the exactness question is settled for the shapes they would serve.

### Unvalidated Surface

- The margin is calibrated on one host (M3 Ultra, Metal, macOS 26.6.2) and one prompt set. Its value is not evidence for other hardware, quantization, or MLX versions.
- A first mismatch above 1.0 logit on either token fails closed. On a host with larger evaluation-path divergence, a legitimate near-tie will fail the gate until the constant is re-measured.
- No runtime B>1 exactness probe was implemented. `probe_block_chain_exactness` still compares a block against a chain at B=1 only, so the served batched path has no runtime check that would decline a non-exact batched width. This PR makes the test gate honest; it does not add a runtime gate.

### Monitoring Required

- Watch the CI hardware lane for `greedy_parity_dflash_qwen35_batched_b4_matches_b1` and `identical_prompts_dflash_qwen35_b4_matches_b1` when MLX, quantization, or kernel dispatch changes land.
- The margin-removal arm is a canary: if it starts reporting that all rows are byte-identical, the margin is no longer needed on that tree and the constant should be revisited.

### Future Improvements

- If the runtime needs to decline non-exact batched widths (rather than only gate the test), extend `probe_block_chain_exactness` or add a sibling that compares one `[B, block]` forward against B single-row forwards, memoized per (model, width, batch).
- Revisit byte identity only with a measured throughput budget for per-row verify evaluation; the clamp arm measured about 1.07x over B=1, so the rewrite must be cheaper than that to keep the batching rationale.

---

## Appendix

### A. Test Results

| Check | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --test speculative_parity qwen35_dflash_batched -- --ignored --test-threads=1` | 2 passed (before: 2 failed) |
| Same with `MLXCEL_SDPA_DETERMINISTIC=1` | 2 passed |
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Pass |

### B. Measurement Environment

| Field | Value |
|---|---|
| Host | M3 Ultra, 80-core, 512 GB unified memory, macOS 26.6.2, Metal |
| Target | `models/qwen3.5-4b-4bit` (mlx-community/Qwen3.5-4B-4bit) |
| Drafter | `models/qwen3.5-4b-dflash` (z-lab/Qwen3.5-4B-DFlash) |
| Block size | 16 |
| Generated tokens | 128 |
| Sampling | Temperature 0, greedy |

### C. Evidence Record

| Observation | Value |
|---|---|
| Failing hard gate | Row 0, token 14: batched 5388 versus B=1 15217 |
| Identical-prompts control | `clamped_rounds=0`, `ragged=false`, all four rows at token 14 |
| Arbiter argmax | 5388 (the batched token) |
| Top-gap(batched) | 0.0 |
| Top-gap(B=1) | 0.25 |
| Other rows | Byte-identical |
| Chosen margin | `BATCHED_VERIFY_JITTER_MARGIN = 1.0` (4x measured) |
| Margin-removal arm | 4 of 4 mismatches rejected at margin 0 |

### D. References

- `tests/speculative_parity/qwen35_dflash_harness.rs`
- `tests/speculative_parity/qwen35_dflash_batched.rs`
- `tests/speculative_parity.rs`
- `docs/CONTINUOUS_BATCHING.md`
- Issue #2851, issue #2442, issue #2353, issue #2190
