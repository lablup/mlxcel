# Technical Report: PR #2906 - feat(models): read the qwen4_exp n-gram table from disk (ple_storage)

**Date**: 2026-11-10
**Status**: In review (open PR; closes #2881)
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

PR #2906 lets a qwen4_exp checkpoint that declares `text_config.ple_storage` serve its roughly 30 GiB per-layer-embedding (PLE) n-gram table from disk, row by row, instead of holding it as MLX parameters. Checkpoints without `ple_storage` keep the resident path unchanged. On the OptiQ 2-bit checkpoint, load-time resident memory is 45.39 GB (`ranges` view). A new `PleRowStore` has a `Resident` and an `External` variant. `External` deduplicates the token ids of a request, serves hits from a byte-bounded LRU row cache (default 0.5 GB, `MLXCEL_PLE_CACHE_BYTES`), reads the misses, dequantizes each row once, and restores the caller's order. The mlx-vlm v2 manifest is accepted as written, and `mlxcel prepare-external-ple` builds the view without Python. A whole-model trace after 4,608 context tokens showed 0 of 128 top-1 disagreements and a logit delta of 0.0000 between resident and external. The default I/O mode (`mmap`) is chosen for simplicity, not measurement, and per-mode benchmarks, the peak-memory comparison, and the 64/96 GB hardware runs are not done.

---

## 1. Problem Statement

### 1.1 Background

qwen4_exp stores a very large n-gram embedding table (the PLE table) that each decode or prefill step reads only a few rows of. Loading it as parameters costs about 30 GiB of resident memory even though a step touches a few thousand rows. mlx-vlm already solved this with an external-storage manifest (`ple_storage`), so checkpoints prepared by it exist. Issue #2881 asks mlxcel to read those checkpoints and to reach the smaller footprint, so the model fits on 64 GB and 96 GB Macs.

### 1.2 Existing Issues

- **The table is a parameter.** The resident path holds every row in MLX arrays, so memory scales with the table, not with the working set.
- **A checkpoint directory is untrusted input.** The manifest names files, offsets and shapes. A reader that follows them naively can be steered outside the model directory or past the end of a tensor.
- **Memory accounting assumed residency.** `mlxcel inspect` and `--estimate-memory` counted the table as resident, which would reject the model on machines where it now fits.
- **The prefill loop had no seam for I/O overlap.** Reading rows on demand adds latency per step unless the next piece's rows are fetched while the GPU runs the current one.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A crafted manifest reads or links files outside the checkpoint | High (information disclosure, file clobber) | Medium for third-party checkpoints, mitigated in this PR |
| External rows differ numerically from resident rows | High (wrong logits) | Low, tested bit for bit on fixtures and on one real model trace |
| `mmap` mode stalls or faults under memory pressure | Medium (latency, SIGBUS on truncation) | Medium, default unmeasured |
| Disk read error panics the forward pass | Medium (process abort) | Low, no error channel exists in the forward |

---

## 2. Change Summary

### 2.1 Row store and cache

- `qwen4_exp_ple_storage.rs` defines `PleRowStore` (`Resident` or `External`). `External::gather` deduplicates ids, looks them up in the cache, reads misses, dequantizes once per unique row, and scatters back to caller order. Dequantizing per unique row, not per requested row, is what keeps repeated n-gram ids cheap.
- `qwen4_exp_ple_cache.rs` is a slab-based LRU. Records live in 64 MiB slabs (one allocation per step, eviction reuses a slot in place). A per-record bookkeeping charge (56 bytes) is counted against the budget so slabs plus bookkeeping never exceed it.
- Growth goes through the `RowCacheGrowth::can_grow` trait, which always agrees today. #2883 is expected to plug an availability check in here, so the cache stops growing and evicts when memory is short.

### 2.2 I/O modes

- `qwen4_exp_ple_io.rs` and `qwen4_exp_ple_direct.rs` implement `MLXCEL_PLE_IO=direct|mmap|ram`. `mmap` is the default, `direct` avoids the page cache and SIGBUS on truncation, and `ram` loads the table into memory.
- The default follows what mlx-vlm uses and is the simplest. No measurement backs it.

### 2.3 Manifest, view builder and trust

- `qwen4_exp_ple_manifest.rs` accepts the mlx-vlm v2 manifest (`safetensors_ranges`, `interleaved_q4`, affine 4/32, nvfp4 4/16). Every validation failure names the offending field. `ple_storage` with more than one PLE layer is refused, since one manifest describes one table.
- `qwen4_exp_ple_prepare.rs` and `commands/prepare_external_ple.rs` add `mlxcel prepare-external-ple`, which builds a view directory without Python.
- `qwen4_exp_ple_trust.rs` holds the path confinement rules (section 3.3).

### 2.4 Gather-ahead

- `prefill_lookahead.rs` and the hook in `planned_prefill.rs` and `engine/direct.rs` let the next prefill piece's rows be read while the GPU runs the current piece. A hint can only change what is cached, never the values computed, so a wrong or late hint costs speed and not correctness.

### 2.5 Accounting and observability

- `qwen4_exp_ple_footprint.rs` makes `inspect` and `--estimate-memory` count the row cache (or the table under `ram`), not the on-disk table.
- `qwen4_exp_ple_stats.rs` prints hit, miss, bytes and time on a stats line after generation.

---

## 3. Technical Decisions

### 3.1 Dedup then cache, dequantize once

Alternatives were dequantizing all requested rows each step or caching dequantized rows. The chosen design caches the stored (packed) record and dequantizes a unique row once per gather. This keeps cache capacity high per byte, at the cost of repeated dequantization on hits across steps. The cost was not measured against caching dequantized rows.

### 3.2 mmap default, three modes kept

All three modes are kept because the right default depends on disk and memory pressure that this host could not isolate. `mmap` was picked as default only because it is simplest and matches upstream. Treat it as provisional until the benchmarks in section 5 exist.

### 3.3 Path confinement instead of trusting the manifest

- The manifest and row file must resolve inside the model directory, so an absolute `ple_storage.manifest` is refused.
- `source_root` cannot be confined, because upstream writes it relative to and outside the view. The compensating control is that each `safetensors_ranges` tensor must match its file's own safetensors header (offsets and shape), so a manifest cannot address bytes the file does not declare.
- `prepare-external-ple` refuses index paths that leave the source, symlinks and reserved names, and hard-links only `.safetensors` files. A later commit stopped it hard-linking an OptiQ sidecar twice.
- This rules out accepting upstream views by path alone. Interop with upstream is kept by validating content, not by trusting location.

### 3.4 Budget includes bookkeeping

The first cache version could exceed its byte budget through hash map and list overhead. Charging 56 bytes per record (16 for key and links, 39 for the worst-case map entry) makes the budget a hard bound. The map figure assumes the current std HashMap load factor, so a change in the standard library's hash map would need this constant rechecked.

---

## 4. Verification

- `models::qwen4_exp` lib tests: 128 passed. Clippy on lib and tests and `fmt --check` pass. There is one test per manifest rule, and External equals Resident bit for bit for both layouts, both quantizations and all three I/O modes.
- Whole-model trace on OptiQ 2-bit after 4,608 context tokens, resident against external: 0 of 128 top-1 disagreements, logit delta 0.0000. It used a `safetensors_ranges` view and predates the review fixes, which did not touch row decoding.
- After the review fixes, a fresh `ranges` view generated 24 tokens in both `mmap` and `direct`.
- Interoperability with mlx-vlm 0.7.11 `ple_storage.py` on tiny checkpoints, at manifest and row level, in both directions.

---

## 5. Unverified and Known Limits

- `interleaved_q4` is proven on tiny fixtures only. A 30 GiB row file would disturb sibling work on this shared host.
- Real-model interop is not run: an mlx-vlm-prepared view generating in mlxcel, and the helper's output generating in mlx-vlm.
- Per-mode paired five-run benchmarks and the peak-memory comparison are deferred to a quiet machine. No default is claimed from measurement, and the 0.5 GB cache size is likewise unmeasured.
- 96 GB and 64 GB Mac runs: hardware absent. The headline 45.39 GB is a load-time resident figure on this host, not a peak under load.
- Truncating a file while `mmap` mode maps it raises SIGBUS (`direct` avoids it).
- A disk read error panics the forward, which has no error channel. A fallible forward would be needed to fix this.
- `can_grow` is always true, so the cache is bounded only by its configured budget until #2883 lands.

---

## 6. Learning Points

- When a format must interoperate with another project but is read from an untrusted directory, validate the content against a source of truth the file carries (here, the safetensors header) when the path cannot be confined.
- Make a prefetch hint structurally unable to change results (it only touches the cache), so a wrong hint is a performance bug only.
- Count bookkeeping inside a byte budget from the start, or the budget stops being a bound.
- Defaults chosen for simplicity should be labelled as such in the PR so they are not mistaken for tuned values.

## 7. Related

- Closes #2881. Follow-up hook for #2883 (growth availability check).
