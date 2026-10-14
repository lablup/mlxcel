# Technical Report: PR #2129 - Model-aware snapshot sizing for attention-cache families (Issue #1761)

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (tests only), Markdown

**Risk**: Low (no runtime code change; tests and documentation)

## Summary

Issue #1761 reported that Gemma 3, Llama 4 and AFMoE fell back to the fixed 512 MiB snapshot store, whose entries are O(context) for these attention-cache families, and that `model_type()` missed the `_text` spellings. PR #1978 (`d7279c52`) had already fixed both in `snapshot_sizing.rs`. What was missing was coverage with the geometry real checkpoints carry and a measurement of the default memory change. This PR adds both and documents the policy.

## Policy

`capacity = max(512 MiB, min(6 * entry, left / 4))`, raised to one `entry` when it fits in `left / 2`. `entry` is the unwindowed snapshot at `min(ctx, 8192)` tokens; `left` is memory after the model's estimated footprint. The value caps the store; memory is used only as snapshots are inserted.

## Tests

The module's tests moved to `snapshot_sizing_tests.rs`. New cases use reduced on-disk configs of gemma-3-4b (`gemma3_text`, head fields omitted so Gemma 3 defaults apply), llama-4-scout (`llama4_text`), gemma-4-12b (`gemma4_unified_text`), muse-glimmer-30b (`muse_glimmer_text`) and an AFMoE config. Each must take the model-aware path with an unwindowed 8192-token entry and must not get the dense KV-store raise. Per-token estimates are asserted within 2% of first-turn boundary snapshots measured on GB10 (all three within 1.2%), and Scout within 1% of the #1752 measurement.

## Measurement

Default-launch multi-turn `mlxcel-server` on GB10; "before" is the same binary with `MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES=536870912`.

| Checkpoint | Capacity | Oversized rejects | Hits | Last turn |
|---|---|---|---|---|
| gemma-3-4b, 31.8k tokens | 512 MiB -> 6.375 GiB | 13 -> 0 | reuse stalls at 19.9k -> continues | 15.5 s -> 2.07 s |
| gemma-4-12b, 15.8k tokens | 512 MiB -> 15.75 GiB | 8 -> 0 | 0/12 -> 11/12 | 37.7 s -> 4.3 s |
| muse-glimmer-30b, 17.4k tokens | 512 MiB -> 2.44 GiB | 0 -> 0 | 5/6 both | 7.4 s both |

Peak stored bytes after the change were 2.02 GB, 4.39 GB and 739 MB respectively.

## Not verified

Llama 4 Scout (57 GB, above the host's 40 GB limit) and AFMoE (no local checkpoint) were covered by unit tests only. Metal was not run.
