# Qwen3.8-27B snapshot capacity default — issue #1167

Date: 2026-08-30
Host for this validation: Linux spark-101, aarch64, kernel 6.17.0-1029-nvidia
Load sample before local checks: 1.99, 2.48, 2.30

## Scope

This record documents the implementation-side validation for issue #1167:
making the prompt-cache snapshot capacity default model-aware for large
snapshot-only hybrid families such as Qwen3.8-27B.

The original Apple Silicon / Metal Qwen3.8-27B measurement that motivated the
issue was not rerun in this worktree. This host did not provide the same
Apple/Metal target, Qwen3.8 checkpoint, and serving benchmark environment, so
no new decode latency median is claimed here.

## External baseline preserved from the issue

Issue #1167 reports a Qwen3.8-27B 4-bit multi-turn run where the old 512 MiB
snapshot capacity held only one large snapshot at a time. The issue's measured
snapshot footprint was approximately:

- fixed recurrent/conv state: 91,251,006 bytes
- per-token growth: 65,637.6 bytes/token
- architectural minimum per token: 65,536 bytes/token

That explains why the 512 MiB default could accept one representative entry and
then evict the live session chain, producing a 0% multi-turn hit rate.

## New default sizing contract

When the operator does not set `--prompt-cache-snapshot-capacity-bytes` or
`MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES`, startup derives an implicit
snapshot capacity from `config.json`:

- representative length: `min(context_size, 8192)` tokens
- per-entry size: architecture-aware attention KV at that representative length
  plus fixed recurrent state where the family exposes it in config
- target retention: six representative snapshots
- resource bound: clamp the implicit raise to one quarter of detected available
  host/unified memory
- override rule: explicit CLI/env capacity remains authoritative

For the deterministic Qwen3.8-shaped unit fixture used in this change, the
computed representative entry is 615,317,504 bytes:

- attention KV at 8192 tokens: 536,870,912 bytes
- fixed gated-delta state estimate: 78,446,592 bytes
- implicit target for six entries before memory clamp: 3,691,905,024 bytes

## Local validation result

This worktree validated the sizing logic with deterministic unit tests rather
than a live Apple/Metal model run:

- Qwen3.8-shaped config raises the fallback above 512 MiB and holds multiple
  representative snapshots.
- detected available memory clamps the implicit raise.
- standard full-attention models keep the legacy default path.
- explicit snapshot capacity from CLI/env is marked authoritative and is not
  eligible for model-aware replacement.
- cache stats expose `snapshot_bytes_per_entry` and `snapshot_self_evictions`.
- snapshot self-eviction is counted for same-session capacity thrash while
  warning only once per affected session.

No small-prompt decode regression benchmark was rerun on this host. The local
gate is limited to compile-time and deterministic cache-store behavior.
