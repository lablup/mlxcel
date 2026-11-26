# PR #3459: Raon-1 mini correctness and validation follow-ups (issue #3457)

**Date**: 2026-11-27

**Status**: PR open (closes #3457). Validated on an M3 Ultra; the CUDA and ROCm run for item 7 is outstanding. 35 files, +1300 / -67.

**Risk Level**: Medium (load-time checks now run for every quantized family; the server's default presence penalty changes for checkpoints that set one)

## Executive Summary

The four Raon-1 mini PRs (#3441, #3445, #3453, #3455) left nine gaps: a sampling default that was never read, load paths that trusted tensor shapes, a memory estimate that missed recurrent state, an unpinned reference oracle, an unmeasured norm rounding difference, a CUDA rounding hazard, server paths that silently dropped the reasoning effort, and an unbounded error echo. This PR closes all nine in one change. Seven are fixes with tests that fail on main; item 6 was measured and left as is; item 7's fix is in, but its CUDA/ROCm run could not be done from this host.

## 1. Problem Statement

- **Sampling default.** Raon's `generation_config.json` sets `presence_penalty 1.5`; mlxcel read temperature, top_p and top_k from it but not this field, so served Raon ran at 0.
- **Shape trust.** A projection whose input width disagreed with the config, or quantized scales with fewer rows than the weight, loaded and then failed (or read out of bounds) at the first forward. #3453 had added a scales check only for Raon planes.
- **KV estimate.** The admission estimate counted MLA latents but not RDA's per-sequence recurrent and conv state (48.8 MB per sequence on Raon-1 mini).
- **Oracle.** `scripts/raon1_mini_oracle_trace.py` imports the checkpoint's own Python code; it recorded digests without checking them.
- **Paths that ignore effort.** Pipeline-parallel, disaggregated prefill/decode and XLA serving never carry the per-sequence effort, so a steering model would run at the wrong alpha with no error.

## 2. Technical Decisions

### 2.1 presence_penalty is a server default only

The server overlays the config value when `--presence-penalty` was not passed (`presence_penalty_was_set`), and a per-request value still wins. `mlxcel generate` keeps 0.0: it has no flag to switch the penalty off, and greedy oracle comparisons depend on 0. Only Raon checkpoints under `models/` set the field (`raon-1-mini`, `-mlx`, `-nvfp4`, `raon-1-nano`).

Side effect: a nonzero penalty makes `needs_token_history()` true, and `sampling_config_eq` in `speculative_burst.rs` keeps such requests out of batched speculative windows. Concurrent Raon requests that name no penalty now draft as single-request DFlash2 bursts. Output is unchanged; `--presence-penalty 0` or a per-request value restores batching. This interacts with batched DFlash2 work in #3458.

### 2.2 Refuse, do not carry, on paths that drop effort

| Option | Pros | Cons |
|---|---|---|
| Carry the effort through each path | Feature-complete | Dead code today: the disaggregated path only serves pool-backed Fp16 families and Raon has no XLA port |
| **Fail at startup, naming the path and model** | Small, explicit, no silent wrong alpha | A future port must lift the guard |

`src/server/reasoning_steering_startup_guard.rs` runs in `start_server` before the model loads. Its tests sit beside it because `start_server` cannot be unit-tested per path.

### 2.3 Keep fast RMSNorm (item 6)

Teacher-forced traces against the Transformers reference, decode ABAB in one lock hold:

| Arm | 4 x 256 decided disagree | 871-token decided disagree | 871-token ppl delta | Decode tok/s |
|---|---|---|---|---|
| `fast_rms_norm` | 0 / 578 | 1 / 549 | +0.337% | 33.30 |
| `rms_norm_round_once` | 0 / 578 | 0 / 549 | +0.607% | 32.26 |

Round-once removes the single disagreement (row 866, which #3441 had traced to reference-side bf16 noise) but costs 3.1% decode, over the 2% bar, and raises the long-prompt perplexity delta. No family's norm changes; `rms_norm_round_once` stays available.

## 3. Implementation Details

- Items 2 and 3: `UnifiedLinear::check_input_width`, `SwitchLinear::check_input_width` and a shared `check_scales_leading_axes` in core `layers.rs`. The scales check runs in `UnifiedLinear::from_weights_with_mode` and `SwitchLinear::from_stacked_parts` for every quantized family; Raon's `check_scales_match_weight` and `validate_quantized_packing` delegate to it. Raon calls the width check for every RDA, MLA, MLP, expert and `lm_head` projection. `KimiK3MLAAttention::check_input_widths` exists but is not yet called.
- Item 4: `kv_arch` adds `rda_layers * (H*D*D*4 + (K-1)*3*H*D*2)` per sequence to `total_bytes`; the per-token marginal is unchanged.
- Item 5: `PINNED_CODE_DIGESTS` for revision `6691a076`, checked before import; `--allow-unpinned-code` warns and marks the trace header.
- Item 7: `compute_g_lower_bounded` casts `a` to f32 before adding `dt_bias`. Metal already promotes bf16 + f32 to f32, so the output there is bit-identical; mlxcel's CUDA overlay types the sum as bf16 and rounded it.
- Item 9: the invalid-level 400 echoes at most 32 characters, Debug-escaped, with a byte-count marker.

## 4. Validation

New tests fail on main for items 2, 3, 4, 5 and 9; items 1 and 8 add API, so their tests do not compile on main. Targeted suites (raon, generation_config, kv_arch, reasoning_steering, gated_delta, switch_layers, server startup, kimi_k3, core layers and rms_norm) pass. All three Raon checkpoints and two other quantized families load and generate with the new checks. The full local `make verify` passed (17,183 tests), and the repository contract targets passed before the merge. pr-reviewer and pr-security-checker found nothing at CRITICAL or HIGH.

## 5. Follow-up

- Item 7 on CUDA and ROCm: `make test-fast-cuda FILTER=gated_delta` and the Raon tiny-model suites. No such host was reachable.
- Batched speculative windows need per-row token history so penalty-bearing requests can share them (see 2.1).
- Kimi K3 should call its new input-width check.
- LOW, not fixed: a non-string `reasoning_effort` kwarg is serialized in full before truncation (bounded by the request body limit); the config's presence_penalty has no upper bound.
