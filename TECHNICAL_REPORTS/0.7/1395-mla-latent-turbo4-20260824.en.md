# Technical Report: PR #1395 - stop symmetric Turbo4 reading past its sign vectors on MLA latent caches

**Date**: 2026-08-24
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: High before the fix (an out-of-bounds read in shipping binaries, and a hard panic on one supported checkpoint)

---

## Executive Summary

`KVCache::update_turbo4_sym` built one `TurboQuantParams` from the **V** head dimension and used it for K as well, guarded only by a `debug_assert_eq!`. Neither `[profile.release]` nor `[profile.test-fast]` enables `debug-assertions`, so that guard never ran in a shipped binary or in the test gate.

The MLA-latent families cache a `(kv_latent, k_pe)` pair through the shared `KVCache`, so K is `kv_lora_rank` wide (512, or 640 on the families that concatenate a DSA indexer key) and V is `qk_rope_head_dim` wide (64). The K quantizer therefore ran a 64-entry sign vector over 512 or 640 coordinates: `from_slice_f32(signs1, &[1,1,1,512])` read 448 floats past the end of a 64-element slice, and 576 past on the indexer-carrying families.

Reproduced before the fix: `glm4-flash-4bit --kv-cache-mode turbo4` emitted 40 tokens of `!` where `fp16` answered normally.

Three things this change turned up that the issue did not describe.

**A hard panic on a supported checkpoint.** `deepseek_v2` under `--kv-cache-mode turbo4` crashed with `wht: last axis must be a non-zero power of 2; got shape=[1, 16, 18, 192]`, by a different route: its decompressed path has K=192 and V=128, so it over-read a 128-entry sign vector by 256 bytes and died in the Walsh-Hadamard transform.

**A silent contract inversion that outlived the crash.** Under the asymmetric modes the "V" slot of these caches holds `k_pe`, the RoPE **key** stream, so the mode quantized a key to 3, 4 or 8 bits while the latent (which is both K and V after absorption) stayed exact. Output stayed finite, which is why it went unnoticed, and `docs/turbo-kv-cache.md` documented the contract as "FP16 K, 4-bit V", which is not what happened.

**A banner that promised a fallback nobody implemented.** The `turbo4` banner has always said "non-allowlisted models fall back to Turbo4Asym". `is_symmetric_turbo_allowed`'s only non-test caller was the advisor, and `cache.rs` asked callers to consult it in a **comment**. A documented precondition with no enforcement.

## 1. Problem Statement

### 1.1 Background

Four families store a latent/rope pair in one `KVCache`: `glm4_moe_lite`, `deepseek_v3`, `kimi_linear`, `longcat_flash_ngram`. Review found a fifth call site, `deepseek_v32.rs:381`, reached by three more `model_type` strings (`deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`, the last wrapping `DeepSeekV32Model`), with no guard of any kind.

`deepseek_v2` is the one MLA family that already guarded: `MlaLatentCache::supports` returns `Err` for any non-Fp16 mode with the reasoning this issue wanted generalised.

### 1.2 Existing Issues

`ffi::from_slice_f32` builds an MLX array from a raw pointer and performs an eager copy with no length check of its own, so a dimension mismatch is an out-of-bounds read rather than a wrong answer. The only thing standing between a shipped binary and that read was an assertion the profile compiled out.

### 1.3 Risk Assessment

High. Two shipping checkpoint families produced garbage, one panicked, and three more silently inverted the documented quantization contract. The fix itself is a refusal, so its own risk is the opposite one: refusing too much would delete working configurations.

## 2. Change Summary

| Area | Change |
|---|---|
| `cache/turbo/quant.rs` | Dimension and sign-vector-length checks promoted from `debug_assert` to `assert`, on both `quantize_into_packed` and `dequantize_from_packed` |
| `cache/turbo/quant3.rs`, `sparse_v.rs` | Same promotion at three more `from_slice_f32(signs, ...)` sites |
| `mla/cache.rs` | Mode rule extracted to `latent_layout_supports_mode`; `MLA_LATENT_CACHE_FAMILIES` and `caches_mla_latent_pair` |
| `cache.rs` | Unconditional K/V head-dim `assert_eq!` in `update_turbo4_sym` |
| CLI and server | Real allowlist fallback; effective mode reported everywhere a mode is announced |
| `server/routes/props.rs`, `types/response.rs` | `/props` now carries `kv_cache_mode` and `kv_bits` |
| `commands/serve.rs`, `inspect.rs` | Memory preflights resolve the effective mode |
| `execution/kv_cache_advisor.rs` | No longer recommends a mode the resolver refuses |
| `docs/turbo-kv-cache.md` | The latent rule, the inverted contract, and the family list |

## 3. Technical Decisions

### 3.1 An assertion the profile deletes is not a guard

The check was `debug_assert_eq!` and both shipping profiles leave `debug-assertions` off, so the precondition was documented and unenforced. It is now `assert!`, which is defensible specifically because the alternative is an out-of-bounds read: turning a wrong answer into a crash is a bad trade, but turning a memory-safety violation into a crash is the right one, and the check is O(1) against a call dominated by a Walsh-Hadamard transform and a host readback.

Review found the same hazard at three sibling sites the issue never named (`quantize_v_turbo3`, `dequantize_v_turbo3`, and three `sparse_v` launchers). None is reachable through mlxcel's own paths today, because those modes build params from the V head dim and quantize only V. They were promoted anyway: `TurboQuantParams` has all-`pub` fields and no `#[non_exhaustive]`, so a short `signs1` is constructible by a struct literal, and a stated rule applied to one file and not its siblings is worse than not stating it.

One placement decision worth recording: in `sparse_v` the assert goes **after** the existing power-of-two gate, not before. That gate is a legitimate graceful fallback for Gemma 4's 192-dimension heads, and asserting ahead of it would convert a working fallback into a panic.

### 3.2 One rule, not two

Rather than write a parallel rule for the latent families, the mode check was lifted out of `MlaLatentCache::supports` into `latent_layout_supports_mode`, which both `supports` and `resolve_kv_cache_mode_for_model` now call. The rule text and its reasoning exist once. `supports` is behaviourally identical for its original caller: same error text, same ordering.

### 3.3 The issue asked for a change that would have deleted a working configuration

The issue's step 3 asks for `deepseek_v2` to be declared a latent family. It was deliberately left off.

`deepseek_v2` asks `supports` per forward call and falls through to a decompressed per-head layout when the mode is quantized. On that path K is `qk_nope_head_dim + qk_rope_head_dim` and V is `v_head_dim`, and both `Int8` and `Turbo4Asym` work, because params come from V only and K never reaches the WHT. Stronger still, its absorbed path is opt-in behind an environment variable and off by default, so it runs decompressed regardless. Listing it would have refused a mode that works. `turbo4` is separately fixed for it through the allowlist, which is what stops the panic in 3.4.

A regression test pins the distinction so the next reader does not "complete" the list.

### 3.4 A hard panic the issue did not know about

`deepseek_v2` under `turbo4` crashed before this PR. DeepSeek-V2-Lite gives K=192 and V=128 on the decompressed path, so params built from V=128 drove `quantize_k_turbo4` over a 192-wide K: a 256-byte over-read at the copy, and then the observable failure one line later in the Walsh-Hadamard transform, whose own assert produced the quoted message.

The fix prevents rather than relocates it: the resolver downgrades `Turbo4` to `Turbo4Asym`, which never calls `quantize_k_turbo4` and never sends a 192-wide K through the WHT.

### 3.5 The doc was wrong about the latent width, for a reason nobody had noticed

Review flagged `glm_moe_dsa`'s `kv_lora_rank` as possibly 128 rather than 512. That premise did not hold: the 128 sits inside a `#[cfg(test)]` fixture, and the real serde default is 512, matching its siblings.

But the doc comment **was** wrong, differently. `deepseek_v32.rs:377-380` concatenates the DSA indexer key onto the latent before caching it, so on those three families the "K" slot is `kv_lora_rank + index_head_dim`, which is 640 on the one local `glm_moe_dsa` config, not 512. The over-read figure is correspondingly 576 floats there rather than 448. Both the doc comment and `docs/turbo-kv-cache.md` were corrected.

Worth recording because the review's finding was wrong and still productive: checking a claim that turned out false is what surfaced the true one next to it.

### 3.6 Announcing the effective mode, and a mirror-image bug in the server

The `turbo4` banner promised a fallback that no code performed. Making it real meant every surface that announces a mode had to report the **effective** one.

Implementing that turned up the same class of problem one layer down: `into_startup_config` runs **before** `initialize_server_logging`, so the new warning went to no subscriber. The server did the right fallback and said nothing about it, which is the original defect mirrored. Notices now ride on `ServerStartupConfig::kv_cache_mode_notices` and are emitted after logging is up, with an `effective KV cache mode` line.

Two more surfaces were resolving the requested mode rather than the effective one, both memory preflights (`serve.rs`, `inspect.rs`), the same bug already fixed in `generate.rs`. The consequence was measurable: on `glm4-flash-4bit` at 32768 tokens, `--kv-cache-mode int8` reported **14.69 GiB** of KV against a real **29.38 GiB**, a 2x under-count, and that estimate can hard-abort startup. Both now report 29.38, while the `llama-3.1-8b-4bit` control still halves correctly.

`mlxcel recommend` was also advising `Int8` for MLA classes that the resolver now refuses, so it would have told operators to use a mode the binary rejects. It now branches on the latent-family predicate.

## 4. Verification

### 4.1 The reproduction, before and after

`models/glm4-flash-4bit`, `-n 40 -t 0 --seed 1350`, against a baseline binary built from the same merge base with the change reverted by path:

| Mode | Before | After |
|---|---|---|
| `fp16` | coherent, 46.16 tok/s | coherent, 46.67 tok/s |
| `turbo4` | **40 tokens of `!`**, 12.06 tok/s | coherent, banner says `fp16 (requested turbo4; not supported on this model family)` |
| `turbo4-asym` | coherent but **different from fp16** | identical to fp16 |

Computed, not eyeballed: after the change `fp16 vs turbo4` and `fp16 vs turbo4-asym` are both IDENTICAL; before, both were DIFFERENT. That last comparison is the evidence for the quieter defect, which produced fluent output and so left no visible trace.

### 4.2 What must not change

`models/qwen3.5-0.8b-4bit` (`qwen3_5`, on the symmetric allowlist) keeps symmetric Turbo4 and is **byte-identical** before and after, as is its `fp16`. Only the banner changed, because that arm can no longer take a fallback. A full re-capture across all seven mode/model combinations after the review fixes was diffed against the first round and is identical modulo timing lines.

### 4.3 Gate

`cargo test --workspace --profile test-fast --features metal,accelerate`: 8399 passed, 0 failed before the review fixes. Local CI (8 of the 10 `ci.yml` jobs; GitHub Actions unavailable this session): 7 pass, 0 fail, 2 skip, the skips needing CUDA and this change touching no XLA path. Real-model gate on `glm4-flash-4bit` and `llama-3.1-8b-4bit`: both PASS.

Clippy was run on `-p mlxcel --bins` as well as `--lib --tests`, which matters here: `serve.rs` and `inspect.rs` are binary-target modules that `--lib --tests` never compiles.

## 5. Findings from Review

One HIGH, four MEDIUM, two LOW, all applied.

The HIGH was a **fifth latent call site**. `MLA_LATENT_CACHE_FAMILIES` listed four families; `deepseek_v32.rs:381` caches the same latent/rope pair with zero guards, and three `model_type` strings reach it. Because `caches_mla_latent_pair` matches exactly by design, `"deepseek_v3"` did not cover `"deepseek_v32"`.

The memory-safety half was incidentally covered, since those families are not on the symmetric allowlist and so were already downgraded. But the PR's **own** second-defect claim was not true for them: the asymmetric modes and the server `--kv-bits` route still quantized their `k_pe` key stream, silently, because `k_pe` is 64 wide and therefore a valid power of two that faults on nothing.

That is worth generalising: a claim of the form "this change closes defect X" needs the same enumeration discipline as the fix itself. Fixing four of five call sites and describing it as closed is the more dangerous outcome, because the description stops the next person from looking.

## 6. What Remains Unverified

**No `deepseek_v32` or `deepseek_v3.2` checkpoint exists locally.** The only `glm_moe_dsa` checkpoint, `models/glm-5-4bit`, is a partial download with no weights and no tokenizer. The refusal path is fully exercised on it, because the resolver runs before model load, but end-to-end generation on those three families could not be driven. Their behaviour is inferred from `glm4_moe_lite`, which shares the identical code path and was measured directly.

`deepseek_v3`, `kimi_linear` and `longcat_flash_ngram` likewise have no local checkpoint; the latent rule is verified on real hardware only through `glm4_moe_lite`, with unit tests covering the rest.

No perplexity or long-context quality measurement. Every real-model check here is short-prompt.

## 7. Learning Points

- An assertion that the shipping profile deletes is documentation, not a guard. If the precondition it protects is memory safety, the check belongs in the shipped binary, and the O(1) cost is not the deciding factor.
- Enumerate call sites mechanically, not from the issue's list. Four of five were found by following the issue; the fifth was found by grepping for the pattern.
- A "closes defect X" claim needs the same rigour as the fix. Closing it on four of five families and saying so is worse than saying nothing, because the claim stops the next reader from checking.
- A refusal-shaped fix has an inverted risk profile: the danger is refusing too much. The issue asked for `deepseek_v2` to be listed, which would have deleted a configuration that works, and the reason it works only becomes visible by reading its per-call fallback.
- A wrong review finding can still be productive. The `kv_lora_rank: 128` premise was false, and checking it surfaced the true error next to it: the indexer key makes the latent 640 wide, not 512.
