# Technical Report: PR #1119 - docs: add the six paged-decode v2 variables and correct the NATIVE row

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1119 closes issue #1104. `docs/environment-variables.md` claims to be the index for `MLXCEL_*` variables, but six paged-decode v2 variables that are read from Rust were absent from it, and the one paged-attention row that was present told operators the opposite of what `README.md` and `docs/CONTINUOUS_BATCHING.md` told them.

The six variables were added with defaults read from their definition sites rather than from the issue text, and the `MLXCEL_PAGED_ATTENTION_NATIVE` row was rewritten to describe both of its consumers. Docs only, one file, no behavior change.

---

## 1. Problem Statement

### 1.1 Background

Issue #899 made the fused paged-decode v2 kernel the production decode path for pool-backed continuous batching. That promotion added a second consumer to an existing variable and introduced five new ones, but the reference page was not updated with it. The result is the worst failure mode a reference page has: not a gap, but a confident wrong answer.

### 1.2 Existing Issues

- **Six variables read from Rust, none listed.** `MLXCEL_PAGED_ATTENTION_V2`, `MLXCEL_PAGED_DECODE_V2_CHUNK`, `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`, `MLXCEL_PAGED_SLAB_BLOCKS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS` and `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`. The last three are already documented as operator-facing knobs in `docs/CONTINUOUS_BATCHING.md`, and `MLXCEL_CASCADE_ATTENTION`, a sibling from the same feature area, was already on the page. That pattern makes the omission drift rather than a deliberate withholding of internals.
- **A row that contradicted two other documents.** The `MLXCEL_PAGED_ATTENTION_NATIVE` row ended by calling the variable "a control for external mlxcel-core consumers and `examples/paged_attention_kernel_bench.rs`, not a server knob", which was true when #710 retired the library entry point and false after #899. `README.md` and `docs/CONTINUOUS_BATCHING.md` both describe `MLXCEL_PAGED_ATTENTION_NATIVE=0` as the server-side kill switch. An operator who opened the reference page specifically to find that kill switch was told it did not apply to them.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Operator cannot disable the v2 decode path during an incident because the reference page says the variable is not a server knob | High | Medium |
| Operator sizes the paged pool by guesswork because `MLXCEL_PAGED_SLAB_BLOCKS` is undocumented, and the fused path silently stays unreachable | Medium | Medium |
| Contributor treats the three diagnostic variables as supported tuning knobs because nothing marks them diagnostic | Low | Medium |

---

## 2. Technical Review

### 2.1 Defaults Traced to Definition Sites

The issue supplied some defaults in prose. None of them were copied. Each was read at its definition site, which is what caught the discrepancy in section 3.2 below.

| Variable | Default as documented | Source |
|---|---|---|
| `MLXCEL_PAGED_ATTENTION_V2` | off | `paged_v2/mod.rs` (`V2_ENV`, `parse_v2_enabled`) |
| `MLXCEL_PAGED_DECODE_V2_CHUNK` | unset (planned or autotuned chunk size) | `autotune/ops/paged_decode_v2_chunk.rs` (`CHUNK_ENV`) |
| `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` | derived: Apple `gpu_core_count * 8` floored at 64, every other host including CUDA `512` | `paged_v2/plan.rs` (`device_target_ctas`) |
| `MLXCEL_PAGED_SLAB_BLOCKS` | derived `ceil(per_slot_ctx / block_size) * batch`, floored at the 32-block pool default, capped at the per-layer budget share | `execution/memory_estimate.rs` (`resolve_paged_slab_blocks`), `cache/paged.rs` (`POOL_SLAB_BLOCKS = 32`) |
| `MLXCEL_PAGED_V2_MIN_KV_TOKENS` | `4096` | `paged_v2/dispatch.rs` (`MIN_SINGLE_REQUEST_KV_TOKENS`) |
| `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST` | `512` | `paged_v2/dispatch.rs` (`MIN_BATCHED_KV_TOKENS_PER_REQUEST`) |

### 2.2 The Rewritten NATIVE Row

The replacement row is written against `resolve_dispatch_decision` and `resolve_paged_v2_dispatch` in `src/lib/mlxcel-core/src/layers.rs`, which is where the two-consumer structure actually lives. It states both consumers, keeps the #710 retirement as history rather than as the current claim, and notes that the override is checked before the selector in both consumers while a forced dispatch still goes through the kernel's structural declines (single-slab layer, servable geometry, non-empty batch). The Default cell was widened to cover the server floors as well as the library selector's regime, since one cell now describes two dispatch policies.

### 2.3 Scope Containment

One file changed, +26/-1. `README.md` and `docs/CONTINUOUS_BATCHING.md` already stated the correct behavior and were deliberately not touched: the reference page was the document that disagreed with them, so editing the other two would have moved the inconsistency rather than closed it.

---

## 3. Technical Decisions

### 3.1 Split the Six Across Two Sections Rather Than One

| Option | Pros | Cons |
|---|---|---|
| One new section holding all six | Keeps the feature area in one place | Puts three kernel-development switches in front of operators as if they were tuning knobs |
| **Chosen: three in a new operator-facing section, three in the existing diagnostic section marked Diagnostic** | Audience matches the section; the diagnostic marker survives even if the row is read out of context | The feature area is now described in two places on one page |

The new `## Paged decode v2 variables` section holds `MLXCEL_PAGED_SLAB_BLOCKS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS` and `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`, placed after the KV cache section. `MLXCEL_PAGED_ATTENTION_V2`, `MLXCEL_PAGED_DECODE_V2_CHUNK` and `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` went into `## Hardware and kernel diagnostic variables`, each prefixed **Diagnostic**. The split is reconnected by a lead-in that names `MLXCEL_PAGED_ATTENTION_NATIVE` as the kill switch and links to the diagnostic section by anchor.

`MLXCEL_PAGED_ATTENTION_V2` in particular needed the marker: it is issue #898's comparison gate for the library-only v1 entry point, and it does not gate the server's production v2 decode at all. Without an explicit statement of that, its name reads like the master switch for the whole feature.

### 3.2 Document the Behavior, Not the Warning Text

`resolve_paged_slab_blocks` in `src/execution/memory_estimate.rs` matches on the parsed value: `Ok(0)` returns `None`, `Ok(n)` returns `Some(n)`, and `Err(_)` logs `"... is not a non-negative integer; using the derived slab size"` and then also returns `None`. `None` means no override, and the pool falls back to `POOL_SLAB_BLOCKS = 32`, so an unparseable value takes the same path as `0` and lands on the historical 32-block default, not on the derived size the warning names.

The row documents what happens ("a value that is not a non-negative integer warns and leaves that same 32-block default in place") rather than what the warning says. Correcting the warning string is a code change and was out of scope for a docs issue; see section 5.

### 3.3 Cross-Link Instead of Duplicating

The dispatch policy, the per-outcome startup log lines, and the list of shapes the fused kernel declines are all already written in `docs/CONTINUOUS_BATCHING.md`. The new section links `CONTINUOUS_BATCHING.md#seeing-which-path-ran` instead of restating them, which is the same instruction the issue gave. A reference page that duplicates a guide acquires its own drift, which is precisely the failure this PR was fixing.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 1 |
| Lines added | +26 |
| Lines deleted | -1 |
| Variables newly documented | 6 |
| Rows rewritten | 1 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Operator reference | `docs/environment-variables.md` | New `## Paged decode v2 variables` section with three operator knobs, a lead-in naming the kill switch, and a cross-link to the continuous-batching guide |
| Diagnostic reference | `docs/environment-variables.md` | Three rows added to `## Hardware and kernel diagnostic variables`, each marked **Diagnostic** |
| Correctness | `docs/environment-variables.md` | `MLXCEL_PAGED_ATTENTION_NATIVE` row rewritten to describe both consumers; #710 kept as history; Default cell widened to cover the server floors |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `cf4e22cd` | docs | docs: add the six paged-decode v2 variables and correct the NATIVE row |

---

## 5. Validation and Follow-up

### Passed

- Every default traced to its definition site rather than to the issue text (table in 2.1).
- `python3 scripts/ci/check_cross_repo_refs.py` passes.
- Added rows verified to keep the 4-column shape of the sections they joined.
- `git diff --stat` is `docs/environment-variables.md` only, so no build or test surface is involved.

### Found and Deliberately Not Fixed

- **`docs/turbo-kv-cache.md` carries the same stale framing.** Around lines 298 to 312 it still says #710 retired the pooled entry point so that "`MLXCEL_PAGED_ATTENTION_NATIVE` is a control for external mlxcel-core consumers and the kernel bench, not a server knob". That is the identical pre-#899 claim this PR corrected on the reference page. Issue #1104 scoped one file, so the second site is a separate follow-up.
- **The `MLXCEL_PAGED_SLAB_BLOCKS` warning text is wrong in the code.** As traced in 3.2, the `Err(_)` arm warns "using the derived slab size" and then returns `None`, which keeps the 32-block pool default instead. Fixing the message is a Rust change and was out of scope here.

### Follow-up Candidates

- The CI check the issue suggested: diff `MLXCEL_*` occurrences in `src/**/*.rs` against the table in `docs/environment-variables.md`, modeled on `scripts/ci/check_kernel_dtype_keys.py`. This PR closed today's gap by hand; nothing prevents the next feature from reopening it.
- `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` is documented as unvalidated on CUDA, where the fixed `512` target should probably be derived from the SM count. The row records that honestly rather than presenting the constant as calibrated.
