# Technical Report: PR #1391 - filter on the untempered distribution and scale by temperature last

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust, C++ (cxx bridge), Metal, CUDA (mirror, uncompiled)
**Risk Level**: High (changes the sampled distribution for every request that uses a truncation filter, which the server defaults do)

---

## Executive Summary

mlxcel divided logits by temperature before top-k, top-p and min-p ran, and applied XTC first of all on the raw distribution. The llama-server chain that mlxcel's flags and request schema are built to be compatible with does the opposite: every truncation filter is evaluated on the untempered distribution and temperature is applied last, only to the draw. With the server defaults (`--temp 0.8 --top-k 40 --top-p 0.9 --min-p 0.1`) every default request was affected, and a nucleus a client tuned elsewhere did not transfer.

Two things about this change are worth more than the reorder itself.

**It uncovered a second, larger defect.** The stock chain's compiled min-p filter was a silent no-op on Metal. Because the joint top-k plus top-p routing declines above vocab 32768, every large-vocab model at the server defaults, Qwen3 and Llama 3.1 among them, was running with **no min-p at all**. The rejection kernel applied min-p correctly, which is what hid it.

**And a plausible optimization turned out to be a regression.** Review proposed hoisting a duplicated per-token cast as "free to remove". Measuring it showed the opposite, and the reason generalizes: common-subexpression elimination is not free when it costs kernel fusion.

## 1. Problem Statement

### 1.1 Background

The Rust pre-step applied XTC first, on raw untempered unfiltered logits. The C++ stock chain scaled by temperature before any filter. The rejection kernel received a single already-tempered probability vector and evaluated the top-k count test, the top-p mass test and the min-p threshold against it. Effective order:

```
XTC(raw) -> temperature -> top_k -> top_p (renormalised) -> min_p -> draw
```

Reference order for the same knobs:

```
top_k -> top_p -> min_p -> XTC -> temperature -> draw
```

`min_p`'s test `p_i >= min_p * p_max` is renormalisation-invariant but temperature-variant, so its position mattered.

### 1.2 Existing Issues

At `T = 0.8` the tempered distribution is sharper, so top-p 0.9 and min-p 0.1 kept fewer tokens than the same request keeps on llama-server. Nothing errored; output stayed fluent. The failure was a quiet compatibility gap in exactly the knobs the compatibility surface exists to honor.

### 1.3 Risk Assessment

High. Two invariants had to hold or the change would be a regression for most users: greedy must be byte-identical, and `T = 1.0` must produce identical draws. Both were verified against a pre-change binary rather than argued.

## 2. Change Summary

12 files, roughly 1174 insertions before review fixes, plus 3 test additions after.

| Area | Change |
|---|---|
| `mlx_cxx_bridge.cpp` | Temperature moved to the end of `fused_sample_filter_logits`; XTC ported in after min-p; `rejection_probabilities` split into filter-space and draw-space arrays; min-p replaced with straight-line ops |
| `sampling_rejection.{h,cpp}` | `rejection_sample` takes both probability rows; Metal kernel reads filter space for every threshold test and draw space for the proposal scan; CUDA mirror updated |
| `sampling.rs` | XTC pre-step deleted; `xtc_gate_draw`; the four entry points thread `xtc_*` |
| `speech_layers.rs`, `talker.rs` | Qwen3-Omni talker kept on the reference temperature-first order |
| `docs/` | Chain order, environment variables, speculative acceptance |

## 3. Technical Decisions

### 3.1 The rejection split, and why the monotone argument is not load-bearing

The kernel now takes two rows: `probs_filter = softmax(x)` and `probs_draw = softmax(x / T)`. At `T = 1.0` the second is the same array object, so there is no second softmax.

The issue justified this with a monotone argument: `p_draw = p_filter^(1/T) / Z` is strictly increasing, so a threshold set in filter space is the same index set in draw space. That argument is true but turned out not to be what makes the implementation correct. Review traced every buffer read and found that **no filter test ever reads the draw row**: the row-max sweep, `total`, `tau_min = min_p * p_max`, `pmass_target = top_p * total`, the pivot value and both `(count, mass)` reductions all read filter space, while only the proposal scan, the CDF offsets and `target = u * mass_prop` read draw space, each gated on `probs > low` in filter space.

That makes it textbook rejection sampling: the proposal is the tempered row restricted to a superset of the target support, acceptance is tested in filter space, and the accepted token is distributed as the tempered row truncated to the untempered support. The monotone identity is a consistency statement, not a dependency. Worth recording, because a future change that made a filter test read the draw row would silently break the sampler while the monotone argument still "held".

### 3.2 A pre-existing silent no-op, found because the new tests could not pass without fixing it

The stock chain's min-p filter was built with `mlx::core::compile(fn, shapeless=true)` and did nothing on Metal. Evidence captured on the pre-change tree before any edit: the reported distribution did not move across `min_p` from 0.05 to 0.9 on a fixed row, and 2000 draws through both the production stock chain and the reference arm produced tokens below `min_p * p_max` (317 and 95 of 2000 at `min_p = 0.5`), while the identical ops uncompiled filtered correctly.

Independently confirmed end to end through the server: on the pre-change binary, `min_p = 0.3` and `min_p = 0.7` produced byte-identical output. min-p was not responding to its own magnitude. Post-change they differ correctly.

Folding the fix into this PR was not a scope choice so much as a precondition: `min_p_support_is_temperature_invariant`, one of the issue's own required tests, routes to the stock chain and cannot pass while min-p is a no-op.

The MLX-internal mechanism was bounded but not diagnosed to a line, and that honesty has a cost worth naming: the same file constructs about fifteen sibling `compile(fn, shapeless=true)` sites the same way. The three that share min-p's shape (`compiled_softcap`, `compiled_softcap_sdpa_gqa`, `compiled_clip_residual`) are named in the PR body; the rest are activations and MLP forwards where a no-op would be numerically loud rather than silent. That is an argument, not a check, so an audit is filed as #1392.

### 3.3 The optimization that measured slower

Review proposed hoisting the duplicated `astype(logits, float32)` shared by the two probability helpers, on the reasoning that MLX does not CSE separately-constructed nodes and this is two full-row casts per token on bf16 logits. Plausible, and part of the reported regression.

Implemented, then A/B'd across nine interleaved pairs with alternating order at vocab 151936:

| Arm | Median | Range |
|---|---|---|
| Unhoisted (shipped) | **108.15 tok/s** | 107.89 to 108.20 |
| Hoisted | 107.46 tok/s | 107.32 to 107.53 |

The hoisted arm lost all nine pairs with non-overlapping ranges. The mechanism: one shared cast forces the f32 row to materialize and be read twice, while two separately constructed casts each fuse into their consumer and never materialize a 600 KB intermediate. **CSE is not free when it costs fusion.**

Reverted, with the measurement left in a comment above both helpers so it is not proposed again. The final regression figure on the rejection arm is **2.63%**, re-measured as four interleaved pre/post pairs so drift cancels; the original 2.6% estimate was correct and the better method only tightened it.

### 3.4 Qwen3-Omni's talker kept the reference order

The talker's `sample_logits` calls the shared sampler, so the reorder reached it too, silently, while its doc comment claimed to implement "the reference `top_p_sampling`" and the module header said sampling "mirrors the reference". That claim would have become false.

The talker keeps the mlx-vlm temperature-first order. The reasoning: #1379's contract is llama-server compatibility for the **request-driven** sampler, and the talker's knobs are not request parameters (`talker_temperature` and `talker_top_p` come from the checkpoint's generation config, and the code predictor's `top_p` is fixed at 0.8), so no compatibility argument reaches that path. mlx-vlm is its only correctness oracle, and no audio validation was run. mlx-vlm's `top_p_sampling` was read directly: its ascending-cumsum rule is algebraically equivalent to mlxcel's descending exclusive-cumsum, so temperature order was the only divergence, which made preserving it cheap.

## 4. Verification

### 4.1 The two invariants, against a saved pre-change binary

Pre-change binaries were rebuilt at the merge base and saved before any edit. On `models/qwen3-4b-4bit` with a raw high-entropy prompt:

- **Greedy (`--temp 0`): token-identical.**
- **`T = 1.0`, all min-p-free combinations across all three dispatch arms** (rejection kernel, stock chain, Gumbel): token-identical.
- `T = 1.0` stock-chain min-p combinations: diverge, as the min-p fix requires.

A methodological note worth carrying: the chat-templated prompt was **powerless** as a probe. Even unfiltered at `T = 1.5` it equaled greedy, because the leading tokens are effectively deterministic. A raw prompt through `--no-chat-template` was needed before any sampling difference was observable at all. That cost a false conclusion during orchestration (the CLI's sampling flags were briefly and wrongly judged inert) before the prompt, not the code, turned out to be the problem.

### 4.2 Statistical validation of the rejection sampler

`rejection_kernel_filter_on_untempered_draw_on_tempered`: 100,000 draws at `T = 0.5`, top-p 0.9. Every in-support empirical frequency within 3 sigma of the tempered truncated reference, **zero draws outside the untempered nucleus**, and the stock arm resolves the identical support.

Speculative decoding acceptance, `qwen3-0.6b` drafter at `T = 0.8`: 0.5167 to 0.5739 per position. No degradation, and the direction is consistent with the widened untempered target support.

### 4.3 Coverage the review found missing

The PR originally claimed the C++ XTC port was held against the Rust reference "element for element by the existing eight XTC unit tests". There were seven, and every one called only the Rust functions. The practical gap was not the count: **the C++ allowlist branch was executed by no test at all**, while production populates it with newline plus the merged EOS set. A defect there lets XTC suppress EOS and produce runaway generation.

Three tests were added: an element-for-element comparison against `softmax(apply_xtc_filter(...))` on all four reference rows including the allowlist row, a gate test at `xtc_probability = 0.5` where the compare actually decides, and the only test that calls `fused_sample_xtc` at all, covering the `T = 0` argmax-of-filtered path.

### 4.4 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8379 passed, 0 failed. Local CI (8 of the 10 `ci.yml` jobs; GitHub Actions unavailable this session): 7 pass, 0 fail, 2 skip. Both real-model gates pass. `check_kernel_dtype_keys.py` OK, the kernel having gained a `DrawType` key for the new input.

Two gate runs failed on known flakes, both resolved by the standing procedure and both given fresh occurrence data on their issues: `text_only_forward_produces_finite_logits` (#997) and `e2e_crossover_larger_models_benefit_more` (#1184). The second is worth a note: it reads as a purely analytical computation at the call site, so its failure looks like a deterministic calculation failing nondeterministically. The wall-clock dependency is one level down, where the harness times `std::thread::sleep` with `Instant::now()`.

## 5. What Remains Unverified

**The CUDA mirror.** Already marked UNVALIDATED in-tree before this change (written for parity, never compiled or run, no CUDA hardware and no nvcc). The comment was extended to name this change's edits specifically. No validation is claimed.

**The two OpenXLA CI jobs**, which need a CUDA toolchain.

**The Qwen3-Omni talker change**, which is compile- and lint-verified only. No audio checkpoint was run. It restores the order the branch's base already had, so it moves the audio path back to its prior state rather than into new territory, but it is unvalidated in either direction.

## 6. Learning Points

- A correctness argument can be true and still not be the thing making the code correct. The monotone identity holds, but what actually makes the rejection sampler right is that no filter test reads the draw row. Verify the property the code depends on, not the property the issue offers.
- "Free to remove" is a hypothesis. CSE that removes a duplicate computation can cost more than it saves when the duplicate was fusing into its consumer and the shared version has to materialize.
- A probe can be powerless without being wrong. A chat-templated prompt made unfiltered `T = 1.5` sampling indistinguishable from greedy, which briefly looked like a bug in the sampling flags. Check that the probe can express the difference before concluding anything from its absence.
- Test count is not coverage, and a claim about coverage should be checked against what the tests call. Seven Rust-only tests were described as holding a C++ port they never invoke.
- When a fix is a precondition for the issue's own acceptance tests, folding it in is right, but the larger observable change deserves to be named. Here it is min-p taking effect for the first time on every model above vocab 32768.
