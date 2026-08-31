# Technical Report: PR #1540 - Inkling native MTP

**Date**: 2026-08-31

**Author**: mlxcel maintainers

**Status**: Implementation and deterministic validation completed; real-checkpoint validation remains deferred

**Languages**: Rust, Markdown

**Risk Level**: High

---

## Executive Summary

PR #1540 adds native B=1 speculative decoding for Inkling's chained multi-token-prediction head. It loads the original `model.mtp.layers.*` tensors, reuses the same Inkling decoder-layer implementation as the target, binds the target embedding, final norm, and LM head, captures pre-norm target hidden states, and restores exact KV plus four-convolution state by snapshot and accepted-prefix replay.

The change also registers offline and server dispatch for both standalone Inkling text targets and the `InklingVLM` variant produced by the public multimodal checkpoint, derives the default verify width as `num_nextn_predict_layers + 2`, adds a safe selective downloader path for the separate 4.46 GB `mtp.safetensors`, and rejects a head-only directory as a standalone target with an actionable diagnostic. Text-only speculative decode on the wrapper uses `vlm.text`; image requests retain classic HMLP prepared-embedding prefill. Deterministic tiny-model tests establish chained-forward finiteness, target block-versus-incremental parity, bitwise continuation equality after partial-accept rollback, wrapper dispatch completeness, and image-prefill preservation without claiming results for checkpoints that were not available on the validation host.

## 1. Problem Statement

Inkling-Small publishes its MTP head as a separate safetensors file next to 32 target shards, while the common MLX conversion drops the head. The existing MTP implementations did not support Inkling's learned relative-position attention and four recurrent short-convolution states, and tail-trimming KV alone cannot roll those recurrent states back after a speculative verify forward.

The implementation needed to distinguish real MTP tensors from a config flag, avoid opening hundreds of gigabytes of target shards when the full original repository is supplied as the drafter directory, and preserve every B=1 entry point. Missing any repeated server adapter match would silently route one scheduler path back to classic decoding.

## 2. Change Summary

| Area | Result |
| --- | --- |
| Shared decoder | Moved Inkling attention, short convolution, cache, decoder shell, and dense MLP primitives into `mlxcel-core`; the target injects its dense-or-sparse MLP while every MTP block injects the dense implementation |
| Drafter | Added validated config fallbacks, raw checkpoint sanitization, chained block execution, target module binding, prompt seed prefill, round snapshots, and accepted-token replay |
| Target | Added pre-norm hidden capture, exact pre-verify state capture, block verify, snapshot restore, accepted-plus-bonus replay, and B=1 linear-only capability boundaries |
| Loading | Added index-aware filtered loading and header-only tensor detection that inspect unindexed auxiliary safetensors without opening indexed target shards |
| Dispatch | Registered both `LoadedModel::Inkling` and `LoadedModel::InklingVLM` in offline generation, legacy server burst, tick-cooperative slice start/step/park/finalize, drafter binding, and source-scanned coverage tests |
| CLI | Added default `K = n + 2`, explicit override precedence, and repeatable `--include` glob filtering after the downloader's existing safe path/type allow-list |
| Detection | Required actual MTP tensors for drafter auto-detection and rejected `config.json` plus `mtp.safetensors` as a standalone model directory |
| Documentation | Added exact selective-download and invocation examples, B=1/tree limits, rollback semantics, and validation limits |

## 3. Technical Decisions

### 3.1 Reuse one decoder implementation

The MTP transform after `hidden_norm`, `embed_norm`, concatenation, and the `2H -> H` projection is an ordinary Inkling decoder layer. A generic `InklingDecoderLayer<M>` now owns attention, residual short convolutions, norms, and cache behavior, while `InklingFeedForward` selects the target's dense/sparse plane or the MTP head's dense plane. This removes a second copy of the recurrence and relative-attention logic, so target and drafter fixes cannot drift independently.

### 3.2 Restore and replay instead of trimming

Each target layer carries a KV cache and four causal convolution states. A verify block changes all five state components, and the convolution tails cannot be reconstructed by reducing a token count. The adapter captures the complete pre-verify state, runs the block once, restores that snapshot, and replays exactly `accepted + 1` inputs. The drafter independently snapshots all block caches before proposing and restores them before replaying accepted drafts plus the new bonus through block 0.

This costs one short replay per round but preserves the greedy invariant. The deterministic regression compares the next hidden state and logits bit for bit with a target that never speculated.

### 3.3 Treat the model index as an optimization, not a complete inventory

The public Inkling-Small index describes only the 32 target shards; `mtp.safetensors` is a top-level unindexed auxiliary file. The filtered loader first selects matching indexed shards, then inspects only top-level safetensors absent from the index. On the public layout this opens the MTP head and skips every indexed target shard. Header detection is bounded, shard names selected from an index retain plain-filename path-traversal validation, and downloader include patterns are applied only after the existing safe relative-path and file-type checks.

### 3.4 Keep unsupported shapes explicit

Inkling MTP implements B=1 only. Batches larger than one decline to classic decode, and the target advertises no tree-aware verify capability because recurrent convolution state cannot be separated across sibling branches with an attention mask. Requested block sizes remain linear chains, with the last MTP block reused when the requested proposal depth exceeds the native chain.

## 4. Review and Hardening

Correctness, security, performance, and finalizer review produced the following fixes before handoff:

- Fixed loading and detection for the real target-only index plus unindexed `mtp.safetensors` layout; the initial indexed-only fixture would have missed the public full repository.
- Added Inkling to every tick-cooperative scheduler site after the source-scanned dispatch test identified the repeated adapter boundary.
- Kept glob includes subordinate to the downloader's safe allow-list and compiled invalid patterns before cache or network reuse.
- Preserved exact absolute KV offsets together with visible sliding-window slabs and all four optional convolution tensors.
- Replaced an avoidable raw-pointer concatenate with the safe wrapper and documented the remaining attention FFI lifetime invariant.
- Split the implementation into files below 500 lines and reused target primitives instead of duplicating the decoder.
- Reconciled the branch with PR #1535 so HMLP image detection/loading and MTP-only detection both remain active.
- Fixed a post-submission HIGH finding: public checkpoints load as `LoadedModel::InklingVLM`, while the original MTP gates accepted only `LoadedModel::Inkling`. A dedicated wrapper adapter now routes text-only speculative work through `vlm.text` across CLI, legacy burst, cooperative slice start/step, drafter bind/return, and finalization.
- Kept image-bearing requests outside MTP through the existing multimodal request gate, and added a tiny real HMLP regression proving the classic wrapper forwards merged prepared embeddings directly to `vlm.text` without replacing or renormalizing them.

No unresolved CRITICAL or HIGH correctness, security, or performance findings remained in the focused review.

## 5. Validation

| Gate | Result |
| --- | --- |
| `cargo check --lib` | Pass |
| `cargo test -p mlxcel-core inkling_mtp --lib` | Pass, 7/7 |
| `cargo test --lib inkling -- --test-threads=1` | Pass, 48/48; combined target, HMLP, detection, both-variant MTP dispatch, raw-image classic gate, and prepared-image prefill |
| `cargo test --lib every_mtp_dispatch_site_covers_every_capable_variant` | Pass |
| `cargo test --lib burst_declined_for_vlm_embeddings` | Pass; multimodal requests retain classic prefill |
| `cargo test --lib resolve_draft_block_size_derives_inkling_default_from_the_mtp_layer_count` | Pass |
| `cargo test --lib isolated_inkling_mtp_download_is_not_a_standalone_model` | Pass |
| `cargo test --lib include_globs_select_only_safe_allow_list_files` | Pass |
| `cargo clippy -p mlxcel-core --lib -- -D warnings` | Pass |
| `cargo clippy --lib -- -D warnings` | Pass |
| `cargo clippy --lib --tests -- -D warnings` | Pass |
| `cargo fmt --all -- --check` | Pass |
| `git diff --check` | Pass |

The core suite covers config fallback and local/global block derivation, config-only negative detection, actual-tensor detection, real-layout index skipping, sanitizer mapping, five-token finite logits, and exact flat KV plus four-convolution restoration. The target suite covers block verify versus incremental argmax and bitwise continuation after partial acceptance.

## 6. Validation Limits and Follow-up

The approximately 153.5 GB Inkling-Small target, 4.46 GB MTP head, and 0.6B checkpoint were not available on the validation host. Real-checkpoint 128-token greedy parity, mean accepted length, throughput, peak memory, and Apple GPU behavior therefore remain unverified and are not claimed by this report.

The broad workspace Metal/Accelerate test and all-target clippy gates were intentionally left to epic-level final verification. B > 1 acceptance, tree verification, a standalone split tool, and performance tuning remain outside issue #1315.

## References

- Epic #1313, issue #1315, and prerequisite issue #1318
- PR #1540
- [Public mlx-vlm Inkling MTP reference](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/inkling_mtp/inkling_mtp.py)
- [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small)
- `docs/supported-models.md`
