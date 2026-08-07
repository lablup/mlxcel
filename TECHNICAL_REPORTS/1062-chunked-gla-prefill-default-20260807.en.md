# Technical Report: PR #1062 - perf(models): make chunked GLA prefill the bailing_moe_linear default

**Date**: 2026-08-07
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (with one listed measurement deliberately not performed and recorded as such)
**Languages**: Rust, Markdown
**Risk Level**: Medium (changes decoded output for one family from the same checkpoint)

---

## Executive Summary

PR #1039 shipped two evaluations of the same GLA recurrence for the Ling / Ring linear-attention MoE family and left the promotion of the faster one to a measurement. This is that measurement, and it lands the opposite way from what the opt-in assumed. Chunked is not a different-but-equal answer traded for speed: it is 2.06x to 2.39x faster on prefill *and* lower perplexity at every window length measured, by 1.50% at 128 tokens rising to 13.75% at 512. It becomes the default, with the environment variable inverted into an opt-out.

---

## 1. Problem Statement

### 1.1 Background

Two paths, same recurrence, both in `src/models/bailing_moe_linear.rs`:

- `gla_sequential`, upstream mlx-lm's per-token loop transcribed step for step, was the default.
- `gla_chunked`, the closed form of that recurrence evaluated over 64-token chunks, was opt-in behind `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=1`.

PR #1039's reasoning for keeping the slower path as the default was explicit and careful: reassociating the sum changes results by well under one bf16 ulp per layer, but this stack amplifies that. Layer 0's mean absolute activation moves by 0.04% on a 101-token prompt, layer 3 multiplies its input magnitude by roughly 23x, and the 256-expert top-8 router converts a sub-ulp score difference into a different selected expert. The two paths therefore decode different continuations from one checkpoint, and rather than assume the faster path was fine, #1039 shipped the reference's own arithmetic and left the promotion to a measurement.

### 1.2 Existing Issues

- **The assumption underneath the default was untested.** "Different, not worse" was reasonable but unmeasured. If chunked were in fact better, the tree was shipping the slower *and* less accurate path by default.
- **The framing understated the effect size.** "Well under one bf16 ulp per layer" describes the *per-step* difference. The sequential path accumulates in bf16 across the entire prefill, so the end-of-sequence difference is not sub-ulp at all.
- **The variable had no `docs/environment-variables.md` row**, so it was discoverable only by reading the model source.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Shipping the slower and less accurate path by default on every Ring / Ling deployment | Medium | Certain, if the measurement favours chunked |
| Promoting on speed alone and regressing quality | High | This is what the opt-in correctly guarded against |
| Losing token-for-token comparability with mlx-lm for this family | Medium | Certain on promotion; mitigated by the opt-out |

---

## 2. Technical Review

### 2.1 Perplexity

Teacher-forced, `examples/perplexity` over `tests/fixtures/wikitext2_excerpt.txt`, on `mlx-community/Ring-mini-linear-2.0-4bit`, M1 Ultra.

| window x windows | tokens | sequential | chunked | delta |
|---|---|---|---|---|
| 128 x 1 | 128 | 115.83 | 114.09 | -1.50% |
| 1024 x 1 | 1024 | 39.39 | 36.39 | **-7.62%** |
| 256 x 32 | 8192 | 155.47 | 151.42 | -2.60% |
| 512 x 32 | 16384 | 85.96 | 74.14 | **-13.75%** |

Chunked wins in every configuration. Within a fixed scoring shape the advantage grows with the window (128 to 1024 at one window; 256 to 512 across 32 windows), which is what a bf16 running state compounding its error over a longer recurrence predicts, and it is the mechanism the closed form avoids by landing the intra-chunk sum in a matmul accumulator.

Absolute perplexity is not comparable across rows, since each scores a different corpus slice and window length. Only the within-row comparison is.

### 2.2 On determinism and the near-miss

The harness is deterministic: teacher-forced scoring with no sampling. A repeated configuration reproduces its number exactly, which was confirmed accidentally. A first attempt at the robustness sweep used `for cfg in "256 32" "1024 1"; do set -- $cfg` in zsh, which does not word-split unquoted parameter expansions, so `$1` became the whole string, argument parsing fell back to defaults, and all three "different" configurations re-ran the default 512 x 32 and produced `85.9565` three times.

The identical numbers were the tell, and the echoed header (`CHUNK_TOKENS=256 32 MAX_CHUNKS=`) named the cause. Rerun with explicit arguments, the sweep produced the table above. The accident is worth recording twice over: it nearly became a false confirmation of robustness, and it independently established that run-to-run noise is not a factor here, so varying the configuration is the robustness check that matters.

### 2.3 Prefill throughput

Wall time of `mlxcel generate -n 1 -t 0 --no-chat-template`, median of 5 runs with the min-max spread, prompts drawn from the same corpus. Wall includes a common ~0.15s load, so these slightly understate the pure-prefill ratio.

| prompt tokens | sequential | chunked | speedup |
|---|---|---|---|
| ~512 | 3.854s (2.381-4.484) | 1.612s (1.589-2.042) | 2.39x |
| ~2048 | 5.785s (5.514-6.229) | 2.814s (2.808-2.829) | 2.06x |
| ~8192 | 18.168s (18.044-18.291) | 8.539s (8.522-8.554) | 2.13x |

Consistent with #1040's 2.6x to 4x, at the lower end. This host carries heavy background load, which shows in the ~512 sequential row's spread; the two larger sizes are tight.

### 2.4 Decode throughput

Unchanged by construction, since both paths take the same single-step recurrence at `L == 1`. Confirmed rather than assumed, as the issue asked: median 142.87 tok/s sequential against 143.58 chunked over 5 runs of 64 tokens.

### 2.5 Compatibility

- **Breaking changes**: decoded output changes for `bailing_moe_linear` from the same checkpoint. No API, config schema, or CLI surface moves.
- **Environment variable**: inverted, not renamed. A pre-#1040 `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=1` still selects chunked rather than silently becoming an opt-out.

---

## 3. Technical Decisions

### 3.1 Invert the variable rather than rename or remove it

**Context:** The issue allowed "the environment variable is inverted or removed".

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Remove it | Simplest surface; no dead knob | Deletes the escape hatch that makes a mlx-lm reference diff possible for this family |
| Rename to `MLXCEL_BAILING_LINEAR_SEQUENTIAL_PREFILL=1` | Name matches behaviour exactly | Breaks a pre-existing `=1`, and worse, silently: the old setting would become an unrecognised variable rather than an error |
| **Chosen: invert in place, `=0` disables** | Mirrors `MLXCEL_FUSED_MOE`'s established shape; a pre-existing `=1` still means chunked | The name reads as opt-in while the default is on, which the doc has to state |

**Rationale:** the silent-breakage property of the rename decided it. Someone who set `=1` did so to get chunked; after a rename they would still get chunked (the new default) but their configuration line would be meaningless, and if the default were ever revisited they would be back on sequential without having changed anything. Inverting in place keeps their intent expressed and honoured.

### 3.2 Split the parsing into a pure function

`chunked_prefill_enabled()` now delegates to `chunked_prefill_enabled_from(Option<&str>)`. This mirrors `switch_layers::fused_moe_enabled_from` and makes the default testable without mutating the process environment, which would otherwise need `test_support::env_lock` and add a serialization point to the suite.

`chunked_prefill_is_on_unless_explicitly_switched_off` pins the default and both spelling sets. That is the assertion that catches the default flipping back.

---

## 4. What Was Not Measured

`inclusionAI/Ring-flash-linear-2.0`, the larger sibling issue #1040 lists. It is not on this host, and fetching it was out of proportion to what it would settle: the mechanism is a per-layer accumulation error, and the advantage already grows monotonically with sequence length within the measured checkpoint, so a deeper sibling would be expected to move the same way.

Recorded as not done rather than assumed in either direction. A session with that checkpoint available should confirm the direction holds and, if the advantage scales with depth as expected, note the magnitude.

---

## 5. The Cost, Stated Plainly

A checkpoint decoded here no longer matches the same checkpoint decoded under mlx-lm token for token, because upstream implements the sequential path. That was the original reason for the default and it is a real loss, not a rounding detail.

It is worth paying for two reasons. The measurement says the divergence is toward the better answer, so the comparability being given up is comparability with a less accurate reference. And this family already could not be gated on token-exactness against mlx-lm: as `docs/supported-models.md` records, the reference disagrees with *itself* between its cached and full-recompute paths, flipping a 0.0625-wide top-2 tie, exactly one bf16 ulp at that magnitude, six tokens into a 12-token prompt.

Anyone who needs the comparability sets `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=0`.

---

## 6. Lessons

- **A carefully reasoned default is still a hypothesis until measured.** #1039's argument for keeping sequential was correct in every particular and still reached the wrong conclusion, because it reasoned about the per-step difference and the decision depends on the accumulated one.
- **"Different, not worse" deserves the same scrutiny as "worse".** It is the comfortable reading of a divergence, and comfortable readings are the ones that go unmeasured.
- **Identical numbers across supposedly different configurations are a bug report about the harness.** The zsh word-splitting slip would have been read as "robust across configurations" if the header had not echoed the malformed argument.
