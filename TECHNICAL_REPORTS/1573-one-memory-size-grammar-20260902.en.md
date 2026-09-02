# Technical Report: PR #1573 - chore(execution): parse memory sizes with one shared grammar

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: implementation and unit-test cycle
**Status**: Completed (unit-verified; the real-binary preflight acceptance run is handed to the merge orchestrator, see section 4.2)
**Languages**: Rust, Markdown
**Risk Level**: Medium (the shared parser also governs `MLXCEL_WIRED_LIMIT` and `MLXCEL_CACHE_LIMIT`, so it sits on the allocator-cap path of every run, not only on the estimate path the defect was reported against)

---

## Executive Summary

`MLXCEL_MEMORY_LIMIT` had two readers with two different grammars. `parse_memory_size` in `src/execution/runtime.rs` set the MLX allocator cap and accepted `4G`, `4GB`, `512M`, `512MB` and plain bytes; the memory-estimation preflight in `src/execution/memory_estimate.rs` carried its own `parse_optional_memory_size_bytes` plus `parse_scaled_memory_size`, which took `GB` and `MB` only. `MLXCEL_MEMORY_LIMIT=4G` therefore capped the allocator at 4 GiB while `mlxcel inspect` and `--estimate-memory` dropped the value on the floor and reported availability from the machine's total unified memory, which on a 128 GB host is a 32x overstatement of what the process will actually be allowed to allocate. This PR makes `parse_memory_size` the single grammar, exposes it to the preflight, extends it with `K`/`KB`, and pins every accepted spelling with tests on both sides. The defect's real cause was narrower than "two parsers": the two grammars agreed on every spelling any test ever exercised, and diverged only on the short suffixes that no test on either side had ever used.

---

## 1. Problem Statement

### 1.1 Background: one variable, two readers, and only one of them runs late

`MLXCEL_MEMORY_LIMIT` (issue #55) is a soft cap on the MLX allocator. `initialize_runtime()` reads it through `resolve_memory_limit()` and calls `mlxcel_core::memory::set_memory_limit(...)`, so MLX raises rather than thrashing once the working set would pass the cap.

The memory-estimation preflight (issue #56) has to answer the same question, "how much memory will this process actually be allowed to use", but it runs on a path where the allocator cap has not been applied yet. `resolve_available_memory` in `memory_estimate.rs` documents a five-step precedence for that, and the first step is deliberately the environment variable rather than `mlxcel_core::memory::memory_limit()`:

```
1. MLXCEL_MEMORY_LIMIT when set to a nonzero value  <- catches estimate-only commands
2. mlxcel_core::memory::memory_limit() when nonzero <- the already-applied MLX cap
3. HardwareCapabilities::unified_memory_gb << 30    <- the machine figure
4. /proc/meminfo MemAvailable, then MemTotal (Linux)
5. 0
```

Step 1 exists precisely because `mlxcel inspect` and `serve --estimate-memory` estimate before runtime bring-up, so step 2 reads zero there. That is the correct design. The defect was that step 1 read the variable through a second parser.

### 1.2 The divergence, and what a user saw

| Input | Runtime allocator cap (old) | Preflight availability (old) |
|---|---|---|
| `4GB` | 4 GiB | 4 GiB |
| `4G` | 4 GiB | **machine total** |
| `512MB` | 512 MiB | 512 MiB |
| `512M` | 512 MiB | **machine total** |
| `4096` | 4096 bytes | 4096 bytes |

The failure is silent in both directions. Nothing warns, and the preflight's output looks entirely normal because falling through to `hw.unified_memory_gb` is a legitimate branch that every unconfigured run takes. A user capping a run at 4 GiB and asking `inspect` whether a model fits was told yes on the strength of a 128 GB figure, and then met the cap at load.

### 1.3 Why no test caught it

This is the part worth carrying forward. The two grammars agreed on `GB`, `MB` and bare bytes, and diverged only on the bare `G` and `M` suffixes. Every test that existed used a spelling from the agreeing half:

- `runtime_tests.rs` covered `"64GB"`, `"128gb"`, `"512MB"`, `"1073741824"`, `"1.5GB"`, `"abc"`. The bare `G` and `M` suffixes its own parser accepted were never asserted anywhere.
- `memory_estimate.rs` covered `"512MB"` end-to-end through `estimate_total_memory`, and `"0"`, `"none"`, `"-1GB"`, `"NaNGB"`, `"1.5GB"` at the parser.

The documentation had already drifted the same way, and disagreed with itself. Three rows of `docs/environment-variables.md` describe three variables that shared one parser, and two of them described different grammars: `MLXCEL_WIRED_LIMIT` and `MLXCEL_MEMORY_LIMIT` were documented as `bytes, NGB, NMB`, while `MLXCEL_CACHE_LIMIT`, four lines below and reading the same function, was documented as `bytes, NG/NGB, NM/NMB`. The `mlxcel --help` block said `supports GB, MB, or bytes` for both of the first two. A reader following the `MLXCEL_MEMORY_LIMIT` documentation would never have written `4G` in the first place; a reader following the `MLXCEL_CACHE_LIMIT` row, or reading the parser, would.

Two parsers is the structural fault. An accepted spelling that no test asserts and no consistent document names is what let the structural fault produce a wrong number for as long as it did. Pinning every spelling, including the ones a reader might assume are obviously covered, is the check that would have failed on day one.

### 1.4 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Preflight overstates available memory by the ratio of machine RAM to the requested cap, and reports `fits = true` for a model that will hit the cap | Medium (a wrong go/no-go answer, not a crash) | Certain for any run using `4G`-style spelling |
| A third reader of a size-valued variable is added with a third grammar | Low, now that there is one `pub(crate)` owner | Was moderate: the pattern had already reproduced once |
| Consolidating the parser silently moves `MLXCEL_WIRED_LIMIT` or `MLXCEL_CACHE_LIMIT` for existing deployments | High if it happened (allocator caps on every run) | Low, and bounded by the audit in section 2.2 |

---

## 2. Technical Review

### 2.1 Blast radius

`parse_memory_size` is not the `MLXCEL_MEMORY_LIMIT` parser. It is also the parser for `MLXCEL_WIRED_LIMIT` (which drives `mlxcel_core::set_wired_limit` and defaults to `gpu_max_memory_size()`) and `MLXCEL_CACHE_LIMIT` (issue #627, the CUDA buffer-cache bound). A change to its grammar or its return type moves allocator caps for every run of both binaries, so the pre-existing tests were kept as regression pins and the change was audited spelling by spelling rather than reasoned about in the abstract.

### 2.2 Behavior audit

Every accepted input keeps its value. The exhaustive set of outcome changes:

| Input class | Old runtime | New runtime | Old preflight | New preflight |
|---|---|---|---|---|
| `4GB`, `4gb`, ` 4 GB `, `1.5GB`, `4.1GB` | 4 GiB / floor | same | same | same |
| `4G`, `512M` | accepted | same | **ignored** | **accepted, same value** |
| `8K`, `8KB` | rejected | **accepted** | rejected | **accepted** |
| `1024` (bare) | 1024 bytes | same | same | same |
| `1.5` (bare) | rejected | rejected | rejected | rejected |
| `abc`, `GB` | `None` | `None` | `None` | `None` |
| `-1GB`, `NaNGB`, `infGB` | `Some(0)` | **`None`** | `None` | `None` |
| `1e30GB` | `usize::MAX` | `u64::MAX` | `u64::MAX` | `u64::MAX` |
| `0GB` | `Some(0)` | `Some(0)` | `None` | `None` (via the resolver's filter) |

Two rows deserve a note.

**The `-1GB` row is the only real behavior change, and only for one variable.** Rust's float-to-int `as` cast saturates, so `(-1.0 * 2^30) as usize` was `0`, not a wrap. `resolve_memory_limit` and `resolve_cache_limit` both mapped a zero to `None` already, so their end state is unchanged. `resolve_wired_limit` did not: it fell into `if limit > 0 { ... } else { None }` and so a garbage `MLXCEL_WIRED_LIMIT=-1GB` silently *disabled* the wired limit, while `MLXCEL_WIRED_LIMIT=abc` fell back to `gpu_max_memory_size()`. Both now fall back to the default. That is a consistency repair, not a regression, but it is the one line in this PR a deployment could notice.

**The `1e30GB` row is not a change.** Both old parsers already saturated: the runtime one implicitly through the saturating cast, the preflight one explicitly through `bytes.min(u64::MAX as f64)`. The new code states it (`if bytes >= u64::MAX as f64 { return Some(u64::MAX); }`) and tests it rather than leaving it to a language guarantee a reader has to know. Describing this as "fixing a wrap" would be wrong.

### 2.3 Types

The parser returns `u64`, which is what the `mlxcel_core::memory::set_memory_limit` / `set_cache_limit` FFI takes and what the preflight computes in. The wired-limit path and `RuntimeSetup`'s three `Option<usize>` fields still speak `usize`, so the narrowing happens once, at the boundary, in `clamp_to_usize`. `usize::try_from(bytes).unwrap_or(usize::MAX)` is lossless on every 64-bit target mlxcel builds for; the saturation is there so a hypothetical 32-bit build clamps rather than truncating a large cap into a small one.

### 2.4 Code quality

`parse_scaled_memory_size` is deleted and `parse_optional_memory_size_bytes` shrinks to its unset check plus a delegation, so the preflight no longer owns any size arithmetic. The `0` / `none` / empty handling stays where it was, in each resolver, because the three resolvers do not agree on it: `resolve_wired_limit` additionally maps `max` and `""` to `gpu_max_memory_size()`, which is a resolver policy rather than a grammar rule.

---

## 3. Technical Decisions

### 3.1 One parser in `runtime.rs`, not a new shared module

**Context**: the natural instinct is a new `src/execution/size_grammar.rs` owned by neither caller.

**Decision**: keep it in `runtime.rs` as `pub(crate)`.

**Why**: `runtime.rs` already owns all three environment-variable constants and all three resolvers. Moving the parser out would leave the constants and their policy in one file and their grammar in another, which is a second place to look for exactly the kind of question ("what does `4G` mean here") that this defect came from. A new module earns its keep when there is a third owner; today the preflight is a consumer, not a co-owner. The `pub(crate)` doc comment says so explicitly, so a future reader knows why the visibility is wider than the module.

### 3.2 `u64` at the parser, `usize` at the boundary

**Context**: the parser could have stayed `usize` and let the preflight widen.

**Decision**: `u64` return, narrowed by `clamp_to_usize` at the three call sites.

**Why**: byte counts are a fixed-width quantity, not a pointer-width one. The FFI setters and the whole `memory_estimate` module already speak `u64`; `usize` appears only because `set_wired_limit` and `RuntimeSetup` predate that. Widening at one point and narrowing at three is the direction that keeps the arithmetic in the type the consumers actually use, and it puts the single lossy operation somewhere named and commented instead of at an implicit `as`.

### 3.3 A bare byte count stays integer-only

**Context**: the suffixed branch parses `f64`, so `1.5` could have become 1 byte for symmetry.

**Decision**: the no-suffix branch keeps `parse::<u64>()`, so `1.5` is rejected.

**Why**: a fractional byte is not a quantity anyone means to write, and accepting it would turn a typo (`1.5` where `1.5G` was intended) into a 1-byte allocator cap that fails every allocation, rather than into a `None` that falls back to a sane default. This also preserves the old behavior exactly; the old bare branch was `parse::<usize>()`.

### 3.4 Add `K`/`KB` now

**Context**: the issue only required that the two existing grammars agree.

**Decision**: add the kilobyte suffix in the same change.

**Why**: the point of the change is that there is one grammar to learn. Leaving a gap at `K` would mean the answer to "what suffixes are accepted" is still "look at the code", which is the state the change exists to end. It cannot regress anything, because a `K`-suffixed value parsed to `None` under both old parsers.

### 3.5 `resolve_paged_slab_blocks` is untouched

`memory_estimate.rs` contains a second env-driven size-ish resolver. It is owned by issue #1137 and is pending a maintainer decision, so it is deliberately outside this diff even though it sits in the same file.

---

## 4. Validation

### 4.1 What was run

| Command | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib execution::` | 126 passed, 0 failed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `python3 scripts/ci/check_cross_repo_refs.py` | clean |

Thirteen of those 126 are the parser and preflight tests:

- `parse_memory_size_gb`, `_mb`, `_bytes`, `_fractional_gb`, `_invalid`: pre-existing, kept verbatim apart from one `usize` to `u64` literal, as the regression pins for the old spellings.
- `parse_memory_size_accepts_every_suffix_spelling`: `4G` == `4GB` == `4gb` == `" 4 GB "`, `512M` == `512MB`, `8K` == `8KB` == 8192, `1024` bare.
- `parse_memory_size_fractional_is_exact_floor`: `1.5GB` = 1610612736, `4.1GB` = 4402341478, `0.5M` = 524288.
- `parse_memory_size_rejects_garbage`: `-1GB`, `NaNGB`, `infGB`, `abc`, `GB`, bare `1.5` all `None`; `0` is `Some(0)`.
- `parse_memory_size_saturates_instead_of_wrapping`: `1e30GB` is `u64::MAX`.
- `available_memory_honors_short_suffix_env_limit`: drives `MLXCEL_MEMORY_LIMIT=512M` through `estimate_total_memory` and asserts the same 512 MiB its `512MB` sibling asserts. This is the end-to-end test for the reported defect; it fails on `main`.
- `parse_optional_memory_size_accepts_the_runtime_grammar`: `4G` == `4GB` == `4gb`, `512M` == `512MB`, `8K`, bare `1024`, and `0GB` still unset.

### 4.2 What was not run, and why

The real-binary acceptance run needs a `cargo build --release` that this unit deliberately did not perform, so it is handed to the merge orchestrator and recorded in the PR body:

```
MLXCEL_MEMORY_LIMIT=4G    ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
MLXCEL_MEMORY_LIMIT=4GB   ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
MLXCEL_MEMORY_LIMIT=4096M ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
                          ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
```

The first three must print an identical 4.00 GB figure and the fourth the machine figure. On `main`, the `4G` and `4096M` invocations print the machine figure, which is the defect.

Note that `mlxcel inspect` takes its model through `-m/--model`, not positionally. The issue body's `mlxcel inspect models/qwen3-0.6b-4bit` would not have run at all, and the checkpoint it names is at `models/mlx/qwen3-0.6b-4bit` in this tree.

---

## 5. Learning Points

**Grep for the consumers of an environment variable, not for its constant.** Both readers here referenced the same string literal `"MLXCEL_MEMORY_LIMIT"` through two separately declared constants (`runtime.rs:32` and `memory_estimate.rs:135`). A search for the constant name finds one file. A search for the variable's spelling finds both. This is the same shape as the `rope_scaling` class already in the project's memory: a config key that is parsed in one place and consumed in another, where only a search for *consumers* shows the gap.

**A grammar is only as tested as its least-used spelling.** The divergence lived exactly in the spellings that no test asserted, and the documentation had drifted into the same shape: three variables reading one function were described with two different grammars in adjacent rows of one table. When a parser accepts N spellings, N assertions is the floor rather than thoroughness. Untested spellings are free to mean different things in different callers, and untested spellings are also the ones documentation stops tracking.

**A fallback that is also a legitimate default hides its own failure.** The preflight's fall-through to `hw.unified_memory_gb` is correct behavior for an unconfigured run, which is why an ignored `MLXCEL_MEMORY_LIMIT` produced output that looked completely normal. Precedence chains where a later step is the common case need a test that pins the *earlier* step, because the symptom of the earlier step not firing is indistinguishable from the ordinary path.

**Multiplying by a power of two is exact in binary floating point.** Both parsers scale by 1024^n, so a fractional value like `4.1GB` loses nothing before the floor and resolves to the same 4402341478 in every caller. There was no precision problem to fix here, and adding decimal-aware parsing would have changed values that are currently identical across the two paths.

**Trust the tree over the issue body's line numbers.** The issue cited `runtime.rs:217-232`; the parser was at `:254-269`. It also cited a `docs/environment-variables.md` line that had moved, and gave an acceptance command with the wrong argument form and the wrong checkpoint path. The issue's *analysis* was accurate in every particular; only its coordinates had drifted.

---

## 6. Follow-up

- The real-binary acceptance run in section 4.2 is outstanding and belongs to whoever merges this.
- `resolve_paged_slab_blocks` in `memory_estimate.rs` remains a separate env-driven sizing resolver with its own conventions, owned by issue #1137.
- `resolve_wired_limit`'s unset matching is still exact-string (`Some("0") | Some("none") | Some("NONE")`), so `" none "` with surrounding whitespace falls through to the default rather than disabling the limit. That asymmetry with the preflight's trimmed, case-insensitive check predates this change and was left alone to keep the diff to one concern.
