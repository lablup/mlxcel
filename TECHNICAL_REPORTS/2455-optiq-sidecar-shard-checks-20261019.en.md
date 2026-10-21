# PR #2455: Accept the declared OptiQ vision sidecar outside the loader

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with real OptiQ checkpoints; PR open

## Summary

Closes #2451. OptiQ conversions index their vision tensors to the sidecar that `config.json` `optiq_vision.sidecar` declares (`optiq/optiq_vision.safetensors`). #2446 made the loader accept that entry, but the downloader completeness check, the WebUI catalog and `weight_footprint_bytes` still treated it as a missing or unsafe shard. This PR adds one shared helper for the config read and uses it at all four sites. The issue asks for all three items in one PR with one verification pass at the end.

## Problem

- **Snapshot classified incomplete.** `missing_indexed_shards` sends the entry to `shard_present`, which rejects any `/`. `model.safetensors` is indexed, so `has_unindexed` is false and the snapshot is `Incomplete`. On main, a store copy of an OptiQ checkpoint therefore fails an `--offline` run, and a normal run re-downloads it.
- **WebUI catalog reports an unsafe shard path.** `indexed_completeness` and `push_index_shard_facts` reject any multi-component name, so the catalog marks the checkpoint incomplete with "safetensors index contains an unsafe shard path".
- **Footprint leaves out the sidecar.** `weight_footprint_bytes` fails `validate_index_shards` on this entry and falls back to the `model.safetensors` header. `metadata.total_size` also omits the sidecar, so `mlxcel inspect` leaves the vision tower out of the weight footprint.

## Technical decisions

- **Shared helper.** `mlxcel_core::weights::declared_optiq_vision_sidecar(dir)` is split out of #2446's private helper. It does the Infinity/NaN-tolerant config read plus `optiq_vision_sidecar_path` validation, and accepts subdirectory paths only. The loader's `collect_shard_paths` and the three new sites all call it, so the validation is not copied.
- **Bounded config read.** The read is capped at 16 MiB and declares nothing past that, because the catalog calls the helper on directories it does not control and its completeness path had no size check before reading the config. This was added in the second commit as a review follow-up.
- **Downloader.** In `missing_indexed_shards`, the declared sidecar counts as present when `dir.join(sidecar)` is non-empty, and is reported missing under its relative name otherwise. Undeclared subdirectory and traversal entries are still missing. The config is read only when the index names a multi-component entry.
- **WebUI catalog.** `indexed_completeness` and `push_index_shard_facts` skip the unsafe-path refusal for the declared sidecar only. Completeness is then decided by the shared rule above, and the content fingerprint now covers the sidecar file.
- **Footprint.** `weight_footprint_bytes` drops the sidecar before index validation and adds its header bytes when the file exists. With a sidecar declared, it sums the main shard headers instead of trusting `metadata.total_size`, which OptiQ writes without the sidecar; a converter that counted the sidecar would otherwise have it counted twice. The empty-shard-list guard is dropped, so non-OptiQ checkpoints take exactly the old path.

## Changes

- `src/lib/mlxcel-core/src/weights.rs`: `declared_optiq_vision_sidecar`, bounded config read, footprint change, tests.
- `src/downloader/completeness.rs`: declared sidecar handling in `missing_indexed_shards`, tests.
- `src/server/webui/catalog_fs.rs`, `src/server/webui/catalog_fs_tests.rs`: unsafe-path refusal skipped for the declared sidecar, fingerprint coverage, tests.

## Verification

GB10, checkpoints under `~/models_archive/mlx`.

| Check | main (fixes disabled) | this PR |
|---|---|---|
| `mlxcel inspect -m lfm2.5-vl-3b-optiq-4bit` | 1958475776 bytes | 2811046368 bytes |
| `mlxcel inspect -m muse-glimmer-30b-optiq-4bit` | 20003849216 bytes | 23847540736 bytes |
| `mlxcel-server --offline -m mlx-community/<name> --model-store-root <store>` (store entry for each OptiQ checkpoint) | "no cached snapshot" error | resolves from the store (LFM serves `/health`; Muse reaches model loading) |
| `GET /ui-api/v1/catalog` with `--models-dir ~/models_archive/mlx --webui` | both OptiQ entries `complete: false`, unsafe shard path | both `complete: true`, no reason |

The expected totals equal main header plus sidecar header, computed independently from the safetensors headers (sidecars 852570592 and 3843691520 bytes). Non-OptiQ checks are unchanged: the other 191 catalog entries have identical completeness and fingerprints, inspect footprints for gemma-3-4b-it-4bit, glm-4.5v-4bit (stale index), llama-3.2-1b-instruct-bf16 and qwen3.5-0.8b-optiq-4bit (no sidecar) match main, and gemma-3-4b-it-4bit still resolves offline from the store.

Five new tests fail with their fix disabled and pass with it: completeness present/absent, catalog OptiQ layout, catalog fingerprint, footprint (including a `total_size` that already counts the sidecar, and a stale index), and the bounded config read. The two undeclared/traversal-path tests are guards and pass both ways. `downloader::`, `webui::catalog`, `execution::memory_estimate`, mlxcel-core `weights::`, clippy (`--lib --tests -D warnings`, main crate and mlxcel-core), fmt and the contract scripts pass locally.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | 331 |
| Lines deleted | 19 |

| Commit | Message |
|--------|---------|
| `1440ad55` | fix: accept the declared OptiQ vision sidecar outside the loader |
| `205cb110` | fix: bound the sidecar config read and stop trusting total_size |

## Related

- #2446: made the loader accept the declared sidecar and introduced `optiq_vision_sidecar_path` in `mlxcel_core::weights`.
- #2451: the issue this PR closes.
