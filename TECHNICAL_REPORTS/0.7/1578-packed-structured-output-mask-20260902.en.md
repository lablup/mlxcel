# Technical Report: PR #1578 - perf(server): pack structured-output masks as u32 bitmasks

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: n/a
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

Constrained decoding expanded the grammar engine's packed token bitmask into a `Vec<bool>` the width of the vocabulary, walked the logits axis a second time to turn that into a `Vec<f32>` bias, uploaded the bias and added it to the logits, once per token per constrained sequence, on the scheduler thread between the forward pass and sampling. This PR keeps the mask packed all the way to the device and expands it there with a broadcast shift. At the Qwen3.8-27B geometry the scheduler-thread cost of preparing and uploading the mask falls from 711 us to 7.4 us per step, and the per-step host-to-device copy falls from 970 KiB to 30 KiB. The output array is unchanged, verified elementwise against the implementation it replaces.

---

## 1. Problem Statement

### 1.1 Background

An OpenAI-style `response_format: {"type": "json_schema", ...}` request compiles its schema into an `llguidance` grammar and attaches a `StructuredOutputConstraint` to the sequence. Before sampling each token, the scheduler asks the matcher which token ids keep the partial output conforming and masks every other logit to negative infinity, so the sampler cannot emit a token that breaks the schema.

`llguidance` answers that question in the form it computes it: a `SimpleVob`, a bitset with one bit per token id, 32 ids per `u32` word. The mask application then had to get that answer onto the GPU as something MLX can combine with a logits row.

### 1.2 Existing Issues

- **The bitset was expanded twice on the host.** `compute_mask` resized `mask_buf` to `vocab_size` and ran `iter_set_entries` to write one `bool` per allowed id. `apply_structured_mask_to_logits` then resized `bias_buf` to `vocab_size_hint` and walked it entry by entry, writing `0.0` at allowed positions and `f32::NEG_INFINITY` everywhere else. For a 248320-row head that is two loops of roughly a quarter million iterations each, per token, per constrained sequence.
- **The upload was 32 times larger than the information it carried.** An f32 bias whose only two values are `0.0` and `-inf` is one bit of information per token stored in 32 bits. At 248320 rows that is a 970 KiB host-to-device copy per step, against the 30 KiB the packed form needs.
- **All of it sat on the scheduler thread.** `apply_structured_mask` is called from `decode_tick` and `prefill` between the forward pass and sampling, once per constrained sequence per tick, so with B concurrent constrained sequences the cost is paid B times per tick and serializes against the batch's own progress.
- **The `mask_buf` / `bias_buf` reuse only removed the allocator cost.** Pre-allocating both buffers at construction avoided a fresh `Vec` per token but did nothing about the two walks or the copy, and cost about 1 MB of resident host memory per constrained sequence at that vocabulary.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Constrained decode throughput degrades as vocabulary grows, with no bound in sight | Medium | High |
| Cost scales linearly with concurrent constrained sequences, so structured output degrades exactly under load | Medium | High |
| A rewrite of the mask path silently changes which token the sampler selects | High | Low |
| A rewrite drops the padded-head masking rule and lets an unnameable row be sampled | High | Low |

The last two are the risks this change itself introduces; sections 2 and 3 describe how they are constrained.

---

## 2. Technical Review

### 2.1 Security

The mask is the only thing standing between a constrained request and non-conforming output, so a mask that is too permissive is a correctness and a trust problem, not merely a performance one. Two rules had to survive the rewrite and both are now covered by explicit tests:

- A logits row past the matcher's vocabulary must stay masked. Qwen3.5 and Qwen3.8 pad `lm_head` beyond the tokenizer: 248320 rows against 248077 tokenizer entries. Those 243 rows name no token, so the grammar has never seen them and they can never satisfy it. In the packed form they read a zero word and are therefore disallowed by construction.
- The matcher's own bitset has excess bits. `SimpleVob` stores `ceil(size / 32)` words, so a vocabulary of 248077 leaves 19 bits in the final word that name no token. `toktrie` clears them on some paths (`set_all`, `negated`) but the invariant is not guaranteed at the API boundary, so `pack_mask_words` masks the final partial word to its valid bit range rather than trusting it.

No input parsing, authentication, or logging surface changed. The public error message for an empty mask is byte-identical to the previous one.

### 2.2 Performance

This is the point of the change; the measurements are in section 8 and Appendix B. The short version: the host work is gone, the copy is 1/32 the size, and the device-side expansion is four elementwise ops over the vocabulary that fuse into the same graph as the selection.

### 2.3 Compatibility and Dependencies

No new dependencies. Every MLX op used (`from_slice_u32`, `right_shift`, `bitwise_and`, `reshape`, `slice`, `astype`, `where_cond`) was already bound in the cxx bridge, so `src/lib/mlxcel-core` is untouched and no CUDA kernel is added, which keeps `make verify-kernel-dtype-keys` trivially green.

`apply_structured_mask_to_logits` keeps its exact signature, so the call site in `src/server/batch/scheduler/run_loop.rs` and its three callers in `decode_tick.rs` and `prefill.rs` are unchanged. `compute_mask` is kept as the `Vec<bool>` accessor; nine call sites in `tests/structured_outputs.rs` depend on it, and it remains the readable way to inspect individual entries. It is simply no longer on the hot path.

### 2.4 Code Quality

The bit algebra is factored into `pack_mask_words`, a pure function over slices with no matcher and no device involved, which is what makes the randomized equivalence tests possible at all. The device expansion is `expand_packed_mask`, also free-standing. `apply_structured_mask_to_logits` is now short enough to read as a policy statement: gate check, pack, emptiness check, select.

`cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` are clean.

---

## 3. Technical Decisions

### 3.1 Broadcast the bit positions instead of gathering per-token index tables

Issue #1316 proposed precomputing two device arrays per constraint, `word_idx = i32[V]` holding `i >> 5` and `bit_idx = u32[V]` holding `i & 31`, and expanding with `bitwise_and(right_shift(take(words, word_idx, 0), bit_idx), 1)`.

The implementation broadcasts instead. The packed words go up as a `u32[n_words, 1]` column and the 32 bit positions as a `u32[1, 32]` row, so a single `right_shift` broadcasts to `[n_words, 32]` where element `(w, b)` is bit `b` of word `w`. Row-major, that flat index is `w * 32 + b`, which is exactly the token id. A reshape to `[1, n_words * 32]` and a trim to the logits width recover the mask in token order.

Why this rather than the plan:

- **No per-constraint index tables.** The gather design needed `2 * 4 * V` bytes of device memory per constrained sequence, about 2 MB at 248320, allocated for the constraint's lifetime. The broadcast needs a 32-element row.
- **No gather.** `take` over `V` indices reads a `V`-element index array and writes a `V`-element result before the shift even starts. The broadcast reads `n_words` words, 1/32 as much.
- **No cached state keyed on width.** The plan's arrays had to be rebuilt when `vocab_size_hint` or the logits dtype changed, which is why its validation list includes `packed_apply_rebuilds_index_arrays_on_width_change`. Nothing here depends on the width, so that failure mode does not exist; the corresponding test became `packed_apply_handles_a_width_change_between_calls`, which drives six widths through one buffer and checks each.
- **It is MLX's own idiom.** `dequantize` unpacks non-power-of-two quantized weights with literally `bitwise_and(right_shift(w, arange(32, uint32)), 1)`. Reusing a shape MLX exercises on every quantized forward is a better bet than a novel one.

### 3.2 Do not cache the device constants on the constraint

The plan also stored `neg_inf` and the index arrays on `StructuredOutputConstraint` as `Option<UniquePtr<MlxArray>>`. That does not compile, and the reason is worth recording because it is not obvious from reading the struct.

`StructuredOutputConstraint` lives in `SequenceInfo` as `Arc<Mutex<StructuredOutputConstraint>>` and is moved onto the scheduler thread, so it must be `Send`. cxx does not derive `Send` for an opaque `extern "C++"` type, and `mlxcel-core` declares no `unsafe impl Send for MlxArray`, so `UniquePtr<MlxArray>` is not `Send`:

```
error[E0277]: `*const cxx::void` cannot be sent between threads safely
   = help: within `MlxArray`, the trait `Send` is not implemented for `*const cxx::void`
   = note: required for `UniquePtr<MlxArray>` to implement `Send`
```

Adding such a field would have required an `unsafe impl Send` wrapper of the kind `src/server/prompt_cache/entry.rs` uses for `DetachedKvSetHolder`, with a real safety argument about MLX array handles crossing threads. The three constants this design needs (32 bit positions, the scalar `1`, the scalar `-inf`) total 152 bytes, so they are simply constructed per call. The measurement in section 8 is of the whole prepare-and-upload phase including those three constructions, at 7.4 us, so the decision costs nothing observable.

### 3.3 Select with `where_cond` rather than adding a bias

The previous code added an f32 array of `0.0` and `-inf`. The new code selects between the logit and a scalar `-inf`. The reason the output is identical rather than merely equivalent:

- `mlx::core::where` computes `promote_types(b.dtype(), c.dtype())` and casts both operands to it, exactly as `add` does. Pairing f16 or bf16 logits with an f32 `-inf` therefore yields the same f32 output the f32 bias produced. No dtype behavior changed for any logits precision.
- `where` broadcasts through `broadcast_arrays`, the same rules `add` used, so `[1, V]` from prefill and `[1, 1, V]` from the decode tick both come back with the shape they had before.
- At an allowed position the logit is passed through instead of having `0.0` added to it. For IEEE floats `x + 0.0 == x`, so this is the same value, and it is one fewer rounding opportunity rather than one more.

### 3.4 Test the packing algebra without a matcher

The unit test module has only `MlxcelTokenizer::stub()`, whose vocabulary is empty, so a matcher-driven unit test cannot produce an interesting mask. Rather than duplicating the 80-line inline `tokenizer.json` from `tests/structured_outputs.rs`, the packing was factored into a pure function and tested against explicitly constructed bitsets.

That turned out to be the stronger option, not the cheaper one: it allows randomized allow sets over thirteen width pairs, all-ones sources that expose excess-bit leakage, and a direct elementwise comparison against the old bias routine kept as a test-only reference. A matcher-driven test can only assert what that particular grammar happens to allow. The matcher-driven behavior stays covered by the twenty-one runnable tests in `tests/structured_outputs.rs`, which pass unchanged.

### 3.5 Compute emptiness from the packed words

The old code counted allowed entries in `[0, vocab_size_hint)` over the `Vec<bool>` and raised `StructuredOutputError::Matcher` when the count was zero. The new code tests whether every packed word is zero. Because `compute_packed_mask` packs to exactly the logits width and zeroes everything outside it, "all words zero" is precisely "no matcher-allowed token is reachable in the model's logits vocabulary", so no separate bound is needed and the error message is unchanged. A stopped matcher returns an empty slice, and `[].iter().all(...)` is `true`, so it takes the same error path it did before.

---

## 4. Implementation Details

### 4.1 Architecture Changes

The per-step path before this change:

```
matcher.compute_mask_or_eos() -> SimpleVob (packed u32)
  -> iter_set_entries        -> Vec<bool>, vocab_size entries      [host loop 1]
  -> bias fill               -> Vec<f32>, vocab_size_hint entries  [host loop 2]
  -> from_slice_f32          -> 970 KiB upload
  -> add(logits, bias)
```

and after:

```
matcher.compute_mask_or_eos() -> SimpleVob (packed u32)
  -> pack_mask_words         -> Vec<u32>, ceil(V / 32) words       [word copy]
  -> from_slice_u32          -> 30 KiB upload
  -> right_shift / bitwise_and / reshape+slice / astype -> bool[1, V] on device
  -> where_cond(allowed, logits, -inf)
```

### 4.2 Key Code Changes

`StructuredOutputConstraint::compute_packed_mask(vocab_size_hint)` drives the matcher exactly as `compute_mask` does, including the `get_error` check, then packs instead of expanding. It takes the width as a parameter, unlike the `compute_mask` it parallels, because the padding width is a property of the model's logits axis and not of the matcher.

The bit that decides what survives:

```rust
let matcher_vocab = self.vocab_size.min(vob.len());
pack_mask_words(vob.as_slice(), matcher_vocab, vocab_size_hint, &mut self.packed_buf);
```

`compute_mask` dropped every entry at or past `self.vocab_size`, and a bitset shorter than that has no bit to read; taking the smaller of the two reproduces that rule in one expression.

`pack_mask_words` is the whole packing rule:

```rust
let n_words = vocab_size_hint.div_ceil(32);
out.clear();
out.resize(n_words, 0);

let valid_bits = matcher_vocab.min(vocab_size_hint).min(src.len() * 32);
let full_words = valid_bits / 32;
let rem = valid_bits % 32;

out[..full_words].copy_from_slice(&src[..full_words]);
if rem != 0 {
    out[full_words] = src[full_words] & ((1u32 << rem) - 1);
}
```

The triple `min` is the safety argument as well as the semantics: `full_words <= n_words` follows from `valid_bits <= vocab_size_hint`, and `full_words < src.len()` in the `rem != 0` branch follows from `valid_bits <= src.len() * 32` plus `valid_bits` not being a word multiple. Both indexing sites are therefore in bounds without a runtime check.

`expand_packed_mask` is the device half:

```rust
let packed  = from_slice_u32(words, &[n_words, 1]);
let bit_pos = from_slice_u32(&PACKED_MASK_BIT_POSITIONS, &[1, 32]);
let bits    = bitwise_and(&right_shift(&packed, &bit_pos), &one);
let flat    = reshape(&bits, &[1, n_words * 32]);
let trimmed = if flat_len as usize == vocab_size { flat } else { slice(&flat, &[0, 0], &[1, vocab_size as i32]) };
astype(&trimmed, dtype::BOOL)
```

A `debug_assert_eq!` pins `words.len() == vocab_size.div_ceil(32)`, the invariant the trim depends on, so a future caller that packs to one width and expands to another fails loudly in debug rather than slicing past the flat length.

`apply_structured_mask_to_logits` reduces to the policy:

```rust
if constraint.is_gated() { return Ok(copy(logits)); }
let words = constraint.compute_packed_mask(vocab_size_hint)?;
if words.iter().all(|word| *word == 0) { return Err(StructuredOutputError::Matcher(...)); }
Ok(apply_packed_mask_to_logits(words, vocab_size_hint, logits))
```

### 4.3 Data Model Changes

`StructuredOutputConstraint::bias_buf: Vec<f32>` is removed and `packed_buf: Vec<u32>` takes its place. At a 248320-entry vocabulary that is about 1 MB less resident host memory per constrained sequence, and the constructor reserves 1/32 as much as before. `mask_buf` is unchanged and still backs `compute_mask`.

---

## 5. Learning Points

### 5.1 A mask is one bit per token; anything wider is a transport choice

The bias array was carrying one bit of information per token in 32 bits, and the two host loops existed only to perform that widening. Once the question is framed as "what is the narrowest thing that answers this", the answer is the form the producer already computed, and both loops turn out to be pure overhead rather than necessary work. The matcher never stopped producing a bitset; the code just kept unpacking it because the consumer wanted floats.

The general shape: when a producer and a consumer disagree about representation, check whether the consumer can be taught the producer's form before you write a converter that runs per token.

### 5.2 Broadcast beats gather when the index pattern is affine

The natural way to expand a bitset to one entry per token is to look up, for each token, the word it lives in. That is a gather, and it needs an index array as wide as the output. But the mapping token to (word, bit) is not arbitrary, it is `(i >> 5, i & 31)`, which is exactly what a 2-D reshape expresses for free. Laying the words along one axis and the bit positions along the other makes the index table implicit in the shape.

This is worth recognising on sight: an index array whose contents are an arithmetic function of its position is usually a reshape in disguise. MLX arrives at the same conclusion in `dequantize` for the same reason.

### 5.3 `UniquePtr<MlxArray>` is not `Send`, so it cannot be cached on a per-sequence struct

Anything reachable from `SequenceInfo` crosses to the scheduler thread and must be `Send`. cxx does not derive `Send` for opaque `extern "C++"` types and `mlxcel-core` declares no manual impl, so an MLX array handle cannot be stored on a per-request struct without an explicit `unsafe impl Send` wrapper and the safety argument that goes with it (`src/server/prompt_cache/entry.rs` has two such wrappers for exactly this reason). Any design that wants to keep a device-side lookup table alive across steps runs into this first. It is a compile error rather than a subtle bug, but it invalidates a plan late if it is discovered during implementation instead of during design.

### 5.4 An A/B against the code you are deleting is worth keeping as a test

The old bias routine was preserved as a test-only reference, which made two things possible at once: an elementwise equivalence assertion across randomized allow sets and seven vocabulary geometries, and a microbenchmark that times both arms in one process under identical conditions. Neither is available once the old path is gone, and reconstructing it later from the diff is strictly worse because the reconstruction is untested. The reference costs 25 lines.

---

## 6. Further Learning

### Key Terms

- **`SimpleVob`**: `toktrie`'s bitset over token ids, `Vec<u32>` plus a size. Bit `i % 32` of word `i / 32` is token `i`, and `as_slice()` exposes the words directly. The backing store can be wider than the declared size, so the excess bits are not guaranteed to be clear.
- **Padded `lm_head`**: a model whose output projection has more rows than the tokenizer has tokens. Qwen3.8-27B declares `vocab_size: 248320` against 248077 tokenizer entries, leaving 243 rows that no token id names.
- **`vocab_size_hint`**: the model's logits width, as read from the logits array's last axis at the call site. It is deliberately not the matcher's vocabulary, because the mask has to broadcast onto the logits.
- **Broadcast shift**: `right_shift` on `[n, 1]` against `[1, 32]`, producing `[n, 32]`. MLX's binary ops broadcast under NumPy rules, so no explicit `broadcast_to` is needed.

### Related Technologies and Frameworks

- `llguidance` 1.7 / `toktrie` 1.8, the grammar engine behind `response_format`, chosen upstream in Blaizzy/mlx-vlm#1047 and shared by mlxcel.
- MLX's `Select` primitive, which backs `where_cond` and promotes its value operands to their common dtype.
- MLX `dequantize`, the in-tree precedent for the broadcast bit-unpack shape: https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp

### Related PRs and Issues

- #1316, the issue this PR closes.
- Blaizzy/mlx-vlm#1805, the upstream analogue of the padded-`lm_head` sizing rule that `apply_mask_covers_the_qwen3_8_padded_lm_head` pins.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|--------|-------|
| Files changed | 2 |
| Lines added | 698 |
| Lines removed | 65 |
| New unit tests | 6 plus 1 ignored benchmark |
| Crates touched | `mlxcel` only |

### Changes by Category

| File | Change |
|------|--------|
| `src/server/structured.rs` | +218 / -65. Adds `compute_packed_mask`, `pack_mask_words`, `expand_packed_mask`, `apply_packed_mask_to_logits` and `PACKED_MASK_BIT_POSITIONS`; rewrites `apply_structured_mask_to_logits`; replaces `bias_buf` with `packed_buf`. |
| `src/server/structured_tests.rs` | +480. Six unit tests, the test-only bias reference, a deterministic xorshift, array readback and argmax helpers, and the ignored microbenchmark. |

### Related Commits

- `ff3d8bd` perf(server): pack structured-output masks as u32 bitmasks

---

## 8. Follow-up Actions

### Required

- Run the live-server check in the PR body against `models/mlx/qwen3-4b-4bit`: a `response_format: json_schema` request must return a schema-conforming object with `finish_reason: stop`, and at `temperature: 0` the completion must be token-identical to the pre-change binary, with an unconstrained control unchanged.
- Run the whole-workspace gate. This change was validated with narrow commands (`--lib server::structured`, `--test structured_outputs`, `clippy --lib --tests`); `cargo test --workspace --profile test-fast --features metal,accelerate` and `cargo clippy --workspace --all-targets -- -D warnings` have not been run here.

### Monitoring Required

- Constrained decode throughput on a wide-vocabulary checkpoint under concurrency. The removed cost scaled with the number of concurrent constrained sequences, so the gain should be most visible there, and that is also where a regression would show first.

### Future Improvements

- Issue #1316 lists a fused kernel, `logits[i] = (words[i >> 5] >> (i & 31)) & 1 ? logits[i] : -inf`, as an optional follow-up if the four-op expansion ever dominates. The measurement below says it does not: the whole prepare-and-upload phase is 7.4 us at a 248k vocabulary, so the remaining cost is the elementwise pass over the logits, which any implementation has to pay.
- The issue's own out-of-scope list still stands: a fused masked-argmax sampler for greedy constrained decoding, and sharing one mask upload across sequences that hold the same grammar state.

---

## Appendix

### A. Test Results

```
cargo test --profile test-fast --features metal,accelerate --lib server::structured
  21 passed, 0 failed, 1 ignored (the microbenchmark)

cargo test --profile test-fast --features metal,accelerate --test structured_outputs
  21 passed, 0 failed, 2 ignored (require local model weights)

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
  clean

cargo fmt --all -- --check
  clean
```

New unit tests:

| Test | What it pins |
|------|--------------|
| `packed_mask_matches_bool_mask` | Thirteen width pairs, randomized allow sets. Every bit matches the boolean mask, and every lane past the logits width is zero. |
| `packed_mask_trims_the_matchers_own_excess_bits` | An all-ones source with a 77-token matcher: the 19 bits naming no token must not survive. |
| `packed_mask_zero_pads_past_the_matcher_vocabulary` | The padded-head direction, 40 tokens into a 200-wide axis. |
| `packed_mask_of_an_empty_allow_set_is_all_zero` | The exact predicate the empty-mask error reads, plus a lone allowed token in the final partial word. |
| `packed_apply_matches_bias_apply` | Seven geometries: the packed output equals the f32 bias output under IEEE equality at every position, an allowed position carries the exact input logit, a disallowed one is `-inf`, and `argmax` agrees. |
| `packed_apply_handles_the_all_allowed_and_single_allowed_edges` | An all-allowed mask is a no-op; a single allowed token in the last partial word wins greedy decoding at any logit value. |
| `packed_apply_handles_a_width_change_between_calls` | Six widths driven through one buffer, each checked in full. |

### B. Performance Benchmarks

```
cargo test --profile test-fast --features metal,accelerate --lib \
  server::structured::tests::bench_packed_mask_apply -- --ignored --nocapture
```

M1 Ultra, Time Machine stopped, 248320 logits rows against 248077 tokenizer entries, min-of-60 per trial, median of four runs:

| Arm | Prepare + upload | With eval | Upload |
|-----|------------------|-----------|--------|
| f32 bias (before) | 711 us | 1002 us | 970 KiB |
| packed u32 (after) | 7.4 us | 322 us | 30 KiB |

Observed spread across the four runs: 707 to 714 us, 6.7 to 8.9 us, 986 to 1009 us, 300 to 326 us.

"Prepare + upload" is the scheduler-thread cost: mask preparation plus the host-to-device copy, up to having the output graph node. "With eval" adds a forced `mlxcel_core::eval` of the result. The packed arm's eval figure sits at the known MLX single-shot eval floor of roughly 300 us, so it understates the device-side saving rather than overstating it; the honest claim from this measurement is the roughly 0.7 ms per token per constrained sequence removed from the scheduler thread, which is inside the 0.5 to 1.5 ms the issue predicted.

Both arms are timed in the same process under identical conditions, with the reference arm being the code this PR deletes.

### C. References

- Issue #1316, which specifies the change and its acceptance criteria.
- `src/server/structured.rs`, `compute_packed_mask` / `pack_mask_words` / `expand_packed_mask`.
- `tests/structured_outputs.rs`, `apply_mask_covers_the_qwen3_8_padded_lm_head`, the production-magnitude partial-word case.
- MLX `dequantize`, the in-tree precedent for the broadcast bit-unpack: https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp
- `docs/code-guidelines.md` for the shared-function convention; no shared function was widened here.
