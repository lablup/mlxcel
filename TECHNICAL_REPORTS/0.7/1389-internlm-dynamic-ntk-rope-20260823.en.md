# Technical Report: PR #1389 - stop doubling RoPE positions under dynamic NTK scaling

**Date**: 2026-08-23
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust, Python (committed oracle)
**Risk Level**: Medium (changes InternLM3 decode numerics at every context length; InternLM2 gains a block it previously dropped at parse time)

---

## Executive Summary

`internlm3` passed `scale = 2.0` to `fast_rope` whenever `rope_scaling` was absent or its `rope_type` was `dynamic`, because `rope_scale()` handled only `"linear"` and ended in `.unwrap_or(2.0)`. The validation checkpoint ships `{"factor": 6.0, "rope_type": "dynamic"}`, so every query and key was rotated at twice its true position, at every context length rather than only past `max_position_embeddings`. Correct dynamic NTK never scales positions; only the rotary base grows once the sequence exceeds `max_position_embeddings`.

The fix itself is small. What makes this change worth a report is the reference it had to replace, and what verifying that reference turned up.

**The existing parity test was pinning the defect.** `INTERNLM3_REF_OUT` was captured from a reference that applied the same wrong `scale = 2.0`, and `docs/supported-models.md` advertised the family as token-exact against it. The green check was not weak evidence of correctness; it was positive evidence of the bug wearing a passing badge, and it had been that way since the port landed.

**And the obvious way to fix that would have re-set the same trap.** The orchestrator's instruction was to re-pin from the mlx-lm oracle and never from mlxcel's own output. That instruction was wrong: mlx-lm 0.31.3 carries the identical defect, so stock mlx-lm reproduces the old, defective pin exactly, 24 of 24 ids. The reference had to come from somewhere neither implementation could contaminate.

## 1. Problem Statement

### 1.1 Background

`src/models/internlm3.rs` computed the position scale as:

```rust
self.rope_scaling.as_ref()
    .and_then(|s| if s.rope_type == "linear" { Some(1.0 / s.factor) } else { None })
    .unwrap_or(2.0)
```

Both an absent block and `rope_type == "dynamic"` fall through to `2.0`. `fast_rope` multiplies every position by `scale` before computing the rotation angle, so the token at absolute position `p` was rotated as if it sat at `2p`. Relative angles between any two tokens were doubled as well, so attention differed from training everywhere, not only in the long-context regime the scheme exists for.

`compute_dynamic_ntk_base` was already correct and was moved rather than rewritten.

`internlm2` was worse in a quieter way: its `ModelArgs` did not declare `rope_scaling` at all, so the live `{"type": "dynamic", "factor": 2.0}` its checkpoint ships was dropped at deserialization, one stage earlier than internlm3's parsed-but-unread.

### 1.2 Existing Issues

The defect was load-bearing on a passing test. `tests/causal_prefill_greedy_parity.rs` pinned 24 reference ids for a 56-id prompt, captured from stock mlx-lm, which agreed with mlxcel because mlxcel had ported mlx-lm's expression verbatim. Two implementations of the same mistake agreeing is not corroboration.

### 1.3 Risk Assessment

Medium. The blast radius is two families and nothing else: `internlm3::ModelArgs` and `internlm2::ModelArgs` are each constructed at exactly one site (`src/model_metadata.rs:187` and `:188`), both on the `ConfigBacked` route from a top-level `config.json`. No VLM loader funnels an arbitrary `text_config` into either, and neither family is tensor-parallel enabled. That is what makes the load-error-on-unimplemented-scheme decision safe here, where the sibling issue #1355 had to warn and continue instead.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/dynamic_ntk_rope.rs` (new) | `DynamicNtkRopeMode`, `DynamicNtkRope`, `from_scaling`, `scale()`, `base_for()`, `apply()`. Built on `rope_utils::{RopeScalingSpec, is_usable_scalar, printable_label}` |
| `src/models/internlm3.rs` | Private `RopeScaling` and the `unwrap_or(2.0)` deleted; uses the helper |
| `src/models/internlm2.rs` | Declares `rope_scaling`, uses the helper, honors `rope_traditional` instead of hardcoding `false` |
| `tests/causal_prefill_greedy_parity.rs` | `INTERNLM3_REF_OUT` re-pinned, with the provenance and regeneration recipe in the doc comment |
| `scripts/tools/internlm_rope_oracle.py` (new) | The oracle, committed, with `--mode stock` and `--mode fixed` |
| `docs/supported-models.md` | The token-exactness claim against the defective reference removed |

## 3. Technical Decisions

### 3.1 The oracle carried the same defect, so the reference came from neither implementation

mlx-lm 0.31.3's `internlm3.py` has two independent problems, both confirmed by reading the installed source rather than inferring from behavior:

1. Line 104-108 computes `rope_scale = 1 / factor if rope_type == "linear" else 2.0` and passes it to `mx.fast.rope(..., scale=self.scale, ...)`, so `dynamic` gets a position scale of 2.0. Line 70-72 then reuses that same 2.0 as the NTK factor in the base formula.
2. `DynamicNTKScalingRoPE.__call__` reads `seq_len = x.shape[1] + offset`. After `transpose(0, 2, 1, 3)` the tensor is `[B, n_heads, L, head_dim]`, so `shape[1]` is the head count, not the sequence length. mlxcel always read this correctly.

Because mlxcel had ported the first expression verbatim, stock mlx-lm and pre-change mlxcel agreed exactly, which is precisely why the gate was green.

The pin therefore comes from mlx-lm with only its per-layer rope module replaced, by the schedule the checkpoint's own remote code implements (`modeling_internlm3.py` to transformers' `_compute_dynamic_ntk_parameters`). Weight loading, dequantization, attention and sampling stay mlx-lm's. Two independent corroborations that this direction is the right one: the checkpoint's own remote code, and mlxcel's XLA emitter, which had already mapped `dynamic` to `RopeScaling::Plain` with the comment "identity within the original context." The MLX path was the outlier, not the consensus.

### 3.2 The oracle is committed, because a described reference is not a checkable one

The doc comment explained the construction well enough to reconstruct. That is still not enough, and this PR is the proof: the previous pin was also, presumably, reconstructable by whoever captured it, and it encoded a bug for months.

`scripts/tools/internlm_rope_oracle.py` makes the pin regenerable. `--mode stock` reproduces the defective ids this constant used to hold and `--mode fixed` produces the ones it holds now, so the difference is demonstrable rather than asserted. The doc comment also records the detail most likely to be missed on a hand reconstruction, the `x.shape[1]` versus `x.shape[-2]` axis confusion above, which lived only in prose before.

### 3.3 internlm2 was wired up rather than deferred

The issue described the helper as one "that internlm2 will reuse", which understated the work: internlm2 was not ignoring a parsed block, it was never parsing one.

Shipping a helper shaped for a family that still drops its own config was not worth deferring. Everything inside `max_position_embeddings` is bit-identical, and that is confirmed rather than asserted: the position scale was already a correct hardcoded `1.0`, `Dynamic` returns `1.0`, and at `factor = 2.0` the base is exactly `rope_theta` below the clamp, so `models/internlm2-7b-4bit` generates byte-identical output pre and post.

`rope_traditional` is now honored instead of hardcoded. That is inert on both shipped checkpoints, neither of which declares it, so `#[serde(default)]` yields the previously hardcoded `false`.

### 3.4 A load error here, a warning on the shared Llama path

Unimplemented schemes are a named load error for these two families. #1355 could not do that, because eight VLM loaders route an arbitrary `text_config` into `llama3::ModelArgs` and one of them (`models/internvl3-1b`) declares `dynamic`, so erroring would have stopped a working model from loading.

That constraint does not exist here, and it was verified rather than assumed: both InternLM arg types are constructed at exactly one site each, InternVL parses into `llama3::ModelArgs` rather than InternLM's, and `runtime_kind_for` has no InternLM arm so the tensor-parallel path is a table-totality arm only.

## 4. Verification

### 4.1 Oracle comparison, with the pre-change binary preserved

| Checkpoint / prompt | Pre | Post | Oracle margin at first pre-change mismatch |
|---|---|---|---|
| internlm3, 56-id gate prompt | 2/24 | **24/24** | 0.31 |
| internlm3, 670-id prompt | 9/32 | **32/32** | 0.48 |
| internlm3, 56-id, `max_pos` forced to 32 | 2/24 | **24/24** | 0.39 |
| internlm2, 614-id prompt | unchanged | **24/24** | n/a |

Pre-change mlxcel reproduces stock mlx-lm exactly at 24/24 and 32/32. The port was faithful; what it was faithful to was wrong.

Forcing `max_position_embeddings` to 32 (weights symlinked, config copied) reaches the rescaled-base branch without needing a 32768-token prompt, which is what made the long-context path testable at all on this hardware.

Every probe separates the two binaries, and every discriminating step has a healthy margin. Both halves matter: a probe the buggy binary also passes proves nothing, and a divergence at a near-tie proves nothing either. This batch produced one of each before anyone learned to check both.

The CLI difference is visible without any tooling. Pre-change output read `"centruty... chnage... indusstry... repplacd"` and inverted the semantics, claiming factories were replaced by cottage industry. Post-change is clean and correct.

### 4.2 Formula

Four worked values reproduce, checked independently in f32 and f64: `seq_len` 100 and 32768 both give the unchanged `5e7`; 40000 gives 117777118.66; 65536 gives 360979300.43. The clamp direction is `max(seq_len, max_pos)`, `scale()` returns `1.0` for `Default` and `Dynamic` and `1.0/factor` only for `Linear`, and `factor` is screened positive-and-finite through the shared `is_usable_scalar`, so `factor: 0` is a load error rather than a degenerate base.

### 4.3 Multi-turn gate, and a probe that could not decide

The owner added a merge condition mid-run: a real local model run covering both single-turn and multi-turn, both passing, with GitHub CI skipped while the self-hosted runner pool is in maintenance.

The multi-turn harness compares a three-turn conversation with the prompt cache on against the same conversation with `--no-prompt-cache`. Under its first formulation, which demanded byte-identical transcripts, `models/internlm3-8b-4bit` failed at turn 2 post-change.

It is not a cache regression. Measured with the cache actually adopted (`hits=1`), the two paths agree to within one bf16 ULP on every token in the top-5, and the cache-on side lands on an **exact tie**: `Rep` and `Note` both at -1.640625, so the tie-break decides. Cache-off separates them by 0.0078125, exactly one ULP at that magnitude. The pre-change binary passed the same assertion only because its margin there was 1.28, about 160 times the jitter.

The distinction from a real cache fault is not subtle. For `models/gemma3-1b-4bit`, which is affected by the still-unfixed #1346, the same measurement shows the top token moving by 9.69, with `'Not'` at logprob 0.0000 on one side and absent from the other side's top-5 entirely.

The harness now compares every generated token and excuses a divergence only when that step's own top-2 margin falls below a jitter floor of 0.05, roughly six times one bf16 ULP and two hundred times below a real fault. Under that rule `llama-3.1-8b-4bit` passes with all turns identical, `internlm3-8b-4bit` passes with the tie excused and named, and `gemma3-1b-4bit` still fails, as it must until #1346 lands.

### 4.4 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8362 passed, 0 failed. `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` clean. GitHub CI skipped for this run by owner instruction.

## 5. Findings from Review

Nothing above MEDIUM. The reviewer independently verified the pin's provenance by locating the oracle artifacts and confirming byte-for-byte that the committed ids match its output, that the artifact predates the commit, and that the oracle reads nothing from mlxcel. Three MEDIUM items were applied: committing the oracle, updating three `Used by:` rosters in `rope_utils.rs` for the two new consumers, and correcting the over-broad harness relaxation described above.

Four LOW items were recorded and not fixed: internlm2 now hard-fails a malformed non-object `rope_scaling` shape it previously ignored (no published config does this); `base_for` is evaluated twice per attention block per forward instead of once (tens of nanoseconds against a ~10 ms step); the load-error label names the architecture rather than the checkpoint directory (less important for a hard error than for a deduped warning); and `dims` and `rope_theta` are not screened the way `factor` is, which is pre-existing and shared with the code this replaced.

## 6. What Remains Unverified

A genuine prompt past 32768 tokens was not run. The `max_position_embeddings` shrink reaches the identical code path at a fraction of the wall time, and the rescaled base is additionally pinned against f64 reference values, but the full-length case itself is untested.

Chunked prefill past `max_pos` computes the base per chunk rather than once per prompt. That is the same approximation mlx-lm makes and was not changed here.

Separately, `models/internlm3-8b-4bit` does not stop on `<|im_end|>` in server chat because `eos_token_ids` returns `[2]`. Pre-existing and identical on both binaries.

## 7. Learning Points

- A passing parity test is only as good as the provenance of its reference. When the reference was captured from an implementation that shares the code under test, agreement is tautology, not evidence.
- An oracle can carry the defect it is being used to detect. Before trusting a reference implementation, check whether it is independent of the thing being verified. Here the two agreed because one was ported from the other.
- Instructions can be wrong in the same way issue bodies can. "Re-pin from the oracle, never from our own output" was sound in intent and wrong in fact, and following it literally would have re-encoded the bug.
- Commit the tool that generates a pinned reference. A reference that is described but not regenerable is exactly the artifact this change existed to replace.
- A differential gate needs a jitter floor scoped to the divergence point. Excusing a whole comparison because one step was a coin flip trades a false alarm for a blind spot.
