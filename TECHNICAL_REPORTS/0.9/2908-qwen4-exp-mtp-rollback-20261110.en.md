# Technical Report: PR #2908 - feat(speculative): wire the native MTP head and rollback for qwen4_exp

**Date**: 2026-11-10
**Reviewer**: -
**Status**: Needs Follow-up
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

`qwen4_exp` (Qwen3.8-Flash-Next) can now unwind a rejected speculative block, which enables both the native `qwen4_exp_mtp` drafter and prompt lookup for this family. On the OptiQ 2-bit checkpoint at temperature 0, both are byte-identical to classic decode on four prompts over 256 tokens, and MTP stays identical on an external PLE table. The paired timing benchmark and the adaptive policy verdict are still outstanding, so the MTP default is unchanged.

---

## 1. Problem Statement

### 1.1 Background

The family owns all of its per-sequence state: KV cache, QSA index and block summaries, GatedDeltaNet recurrence, convolution windows, and PLE n-gram contexts. Speculative decoding has to discard rejected tokens from all of it. Nothing did, so `mtp.*` tensors were dropped and `--prompt-lookup` declined.

### 1.2 Existing Issues

- **No rollback path**: GatedDeltaNet and convolution state cannot be truncated, and a drafted then rejected EOS would leave a wrong n-gram context.
- **Different drafter contract**: the head consumes the `[B, L, hc_count * hidden]` streams from before the final mixer. Qwen 3.5's adapter hands over the post-norm hidden, so reuse would silently give the wrong input.

---

## 3. Technical Decisions

### 3.1 Replay what cannot be trimmed, depth-1 log

| State | Strategy |
|-------|----------|
| KV, QSA raw keys and positions | Trim by the rejected count |
| QSA block summaries | Drop incomplete blocks, recompute lazily |
| GatedDeltaNet, both convolution windows | Replay over the accepted prefix |
| PLE n-gram contexts | Recompute from accepted tokens |

The log holds only the newest speculative forward. Depth 4 was estimated at about 450 MB per sequence (not measured). A rewind past the log returns an error and leaves state untouched.

### 3.2 A decode-equivalent verify block

**Context:** Without it, the exactness probe declined the real checkpoint at block 4 in about half of all logit bytes.

**Rationale:** Verify now takes the same arithmetic path as decode: one query per QSA call, convolutions one row at a time, the fused GatedDeltaNet projection, dense gate projections row by row, unsorted expert gathers. The remaining byte source on GPU generation 15 and later is `qmv_wide`, which the existing exactness gate switches off. Prompt lookup on this family now runs that gate too.

**Trade-offs:** `qmv_wide` is switched off process-wide once speculation is configured, so classic baselines on those GPUs must state the flag.

### 3.3 Opt-in rewind for model-owned families

`LanguageModel::supports_speculative_verify_rewind` defaults to `false` and only `qwen4_exp` opts in. An earlier default admitted other model-owned families such as Gemma 3 and Baichuan to prompt lookup.

### 3.4 Keep the adaptive policy as the gate

Qwen 3.5 MTP measures 0.59x to 0.70x classic decode on M1 Ultra, so a positive result on another engine does not justify a default flip. `mtp_policy.rs` is untouched.

---

## 4. Implementation Details

| Area | Files |
|------|-------|
| Rollback | `src/models/qwen4_exp_rollback.rs`, separate from `Qwen4ExpCache` |
| Drafter | `src/models/qwen4_exp_mtp_drafter.rs`, `src/models/qwen4_exp_mtp_config.rs`, in the binary crate because its layer is the target's own |
| Target adapter | `src/models/qwen4_exp_mtp_target.rs` |
| Dispatch | CLI, burst, slice and scheduler arms for `LoadedModel::Qwen4ExpVLM` |

Rollback touches per-sequence state only and never the row table, so it works unchanged with the resident and external PLE stores from #2881.

Defects found during verification and fixed in this PR:

- Prompt lookup diverged after about 25 tokens under `qmv_wide` on GPU generation 15.
- MTP diverged from classic on long prompts because the target prefilled in one call instead of 2048-token chunks.
- Server `prefill_chunk_size` was ignored by MTP.
- An `hc_count` mismatch truncated replies; it is now refused before bind.
- `generate --draft-model` panicked on an empty prompt; it now declines by name.

---

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 40 |
| Lines added | +4456 |
| Lines deleted | -172 |
| `models::qwen4_exp` tests after rebase | 147 passed |

---

## 8. Follow-up Actions

### Required

- [ ] Paired five-run benchmark for classic, MTP at block sizes 2, 3 and 4, and prompt lookup, plus the `/v1/internal/mtp-policy` verdict.

### Future Improvements

- MTP prefill retains pre-mixer streams for the whole prompt (about 20 KB per token at bf16).
- `verify_finalize` only logs a rollback failure, because `MtpTarget` returns no `Result`.
- A draft error silently ends generation in the shared MTP loop, for all families.
- Drafter width (bf16, 8-bit, 4-bit) against cost was not measured. The CUDA and CPU paths are not admitted for this family.

---

## Appendix

### A. Test Results

After rebase onto `21532270` (HEAD `274dc165`): `cargo fmt --check`, `make verify-env-var-docs`, `cargo clippy --lib --tests --bins --features metal,accelerate -- -D warnings`, and `cargo test --profile test-fast --features metal,accelerate --lib -- --test-threads=1 models::qwen4_exp` (147 passed). Workspace-wide tests are left to CI and the final gate.

### B. Real-Model Results

Target `mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit`, drafter `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` at `be6e2bc3`, M3 Ultra, temperature 0, 256 tokens, commit `e2151399` before rebase.

| Prompt | MTP vs classic | MTP acceptance | Prompt lookup vs classic | Lookup acceptance |
|--------|----------------|----------------|--------------------------|-------------------|
| English essay | Identical | 0.479 | Identical | 0.229 |
| Rust quicksort | Identical | 0.630 | Identical | 0.189 |
| Korean seasons | Identical | 0.396 | Identical | 0.206 |
| English file rewrite | Identical | 0.628 | Identical | 0.411 |

After the rebase (`274dc165`, 128 tokens, Rust and Korean prompts): MTP and prompt lookup are identical to classic, and MTP on an external `safetensors_ranges` PLE view with `MLXCEL_PLE_IO=mmap` is identical to classic on the resident checkpoint for the Rust prompt. Every speculative arm rejected drafts, so rollback ran.

Decode rates observed on a host running sibling builds are not a benchmark and are omitted.

### C. References

- Issue #2880
- Blaizzy/mlx-vlm#2040, Blaizzy/mlx-vlm#2046
- Strata `docs/MAC.md` at `146df3c`
