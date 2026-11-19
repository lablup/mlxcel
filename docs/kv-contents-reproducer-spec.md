# KV contents reproducer spec (#2778 criterion 1, second half)

Status: specification written 2026-11-07 on M1 Pro, plus measurements from 2026-11-08. The standalone harness `examples/kv_history_divergence.rs` is in the tree and produced the history-build numbers below. The `MLXCEL_TRACE_KV_DUMP` dump and `scripts/compare_kv_contents.py` comparator in "Design" are still proposals; neither exists. Source locations are given by symbol name (file, type, function) rather than line number, because line numbers drift.

One premise has since been measured. Using the prebuilt 2026-11-06 `logit_trace` binary (no rebuild, so the Metal toolchain is not needed) a width-only control at fixed KV contents puts the fp16 logit difference at **exactly 1 ULP (0.015625)**, top-1 and top-5 order unchanged (`W=1,8,16,32` at context 197). Details in the `Measured bound` section below. That bounds the width effect; it does not satisfy criterion 1, because it holds the KV contents identical rather than building them with blocks.

## What this is for

Issue #2778 criterion 1 asks for "a target-only reproducer for the Qwen3 position 197 swing, comparing the logits and KV contents of plain and block-built histories". That criterion has two halves with very different blockers.

| half | blocker | tractable now? |
|---|---|---|
| reproduce the pos 197 swing | the original prompt text and KV mode are not in the issue; the report was measured on M1 Ultra | no |
| compare KV contents of plain vs block-built histories | none: the question is prompt-independent | yes |

The blocker is recorded in the #2778 thread. The status comment on the criteria says of this one: "**Target-only reproducer for the pos 197 swing, left open.** Not reproduced: the prompt text and KV mode of the original runs are not in the issue, and the runs here are on M1 Pro, not M1 Ultra." The original report names the prompt only as `Qwen3 copy` (no token ids, no text, no length) and its only stated measurement condition is "M1 Ultra Metal", with "Other backends not checked" and "The root cause is unknown". So the first half is waiting on a human, and no amount of coding unblocks it.

The second half is the one that actually decides the open question. The decision to treat the flip as expected behaviour is guarded by a named reopening trigger, `docs/temperature-0-block-history-flip.md` "Limits of this evidence": *a reproducer that shows corrupted KV contents*. Until `examples/kv_history_divergence.rs`, nothing in the tree read the KV cache contents. That is the gap this spec addresses, and it does not need the original prompt.

## The question, stated so it can be answered

A block verify at width `M` writes K/V for `M` tokens in one forward; a decode step writes one token at `M = 1`. MLX selects a different kernel per `M`, so the two paths round differently. Three outcomes are possible, and only the third is a bug:

1. **Equal.** KV bytes match. Then the flip is not in the cache at all and the search moves back to the projections.
2. **Rounding-equivalent.** KV differs, but every element's difference is bounded by the storage resolution at that position, and the error does not grow with block index or with `M`. Then the cache is damaged only in the sense any reordered reduction damages it, and the doc's "expected behaviour" reading holds with evidence rather than inference.
3. **Corrupted.** KV differs by more than rounding can explain, or the error grows along the block, or a position is written that a sequential history never writes. This trips the reopening trigger and reopens the 0.8.0 decision.

The harness must be able to tell these apart. Reporting "KV differs" without a rounding bound would be the same failure as reporting "logits differ", an observation with no threshold.

### Measured bound on the width axis (2026-11-08, M1 Pro)

Before building anything, the width axis was measured on the existing prebuilt binary, isolating it from every other variable. `MLXCEL_TRACE_START_TOKEN=197`, `PREFILL=197`, `MAX_CHUNKS=1`, with `CHUNK_TOKENS=W ∈ {1,8,16,32}` on Qwen3-0.6B. The context pass is one unchunked forward (`context_split`, `logit_trace/kv_mode.rs`, which returns `context_len` unchanged unless `MLXCEL_TRACE_DECODE_STEP` is set), so the 197-token history and its KV contents are identical across all four arms, and only the traced forward's width `W` differs.

| `W` | top-1 | max abs logit delta vs `W=1` | top-5 order |
|---|---|---|---|
| 1 | 483 | n/a | baseline |
| 8 | 483 | 0.015625 | unchanged |
| 16 | 483 | 0.015625 | unchanged |
| 32 | 483 | 0.015625 | unchanged |

`0.015625 = 2^-6`, one fp16 ULP at this magnitude.

Two consequences, and they point in opposite directions.

First, this **sets the `ULP_BOUND` prior**. Difference on this axis is 1 ULP, so a `ULP_BOUND` far above 1 would pass a genuinely damaged arm, and one below 1 would fail a clean one. Start at 1 and justify any widening explicitly.

Second, and more important: **width alone cannot produce the reported swing.** The issue says plain decode prefers its token by 2.0 while most block histories prefer the other by about 35. A 1-ULP width effect is two to three orders of magnitude short of those gaps (128x and 2240x). So whatever produces the swing is *not* the kernel-selection story in isolation; it must be in how the block-built history's *contents* differ, or in a mechanism this control does not exercise. That raises the value of the KV comparison below: it is now the leading hypothesis rather than one of three equal candidates.

A caveat that must travel with this number: a control that holds KV identical proves the *width* effect is small. It does not prove block-built histories round the same way, because the histories themselves were never built by blocks here. That is exactly the gap the harness must close, and it is why this measurement bounds the spec's premise without discharging criterion 1.

#### The complementary control: width applied to the *history build*

The measurement above varies the width of the **traced forward** while holding the history fixed. `examples/kv_history_divergence.rs` varies the width of the **history build** instead, on the same model with a 64-token context and an identical corpus. That isolates the other half of the mechanism, and the two controls disagree by an order of magnitude:

| manipulated width | max abs logit delta |
|---|---|
| traced forward only (history identical) | 0.015625 |
| history build (plain vs block), 64-token span | **0.343750** |
| history build (plain vs block), 448-token span | **2.757812** |

A byte-level sweep on the same binary localizes where the history arm changes character. Holding `context = 64`, `max_n = 64` and the corpus fixed, widths 1, 2, 4 and 8 all report **zero differing KV bytes** (`kv_diff_bytes = 0`, zero traced positions with any KV difference). That is a behavioral control and it says the harness does not manufacture divergence. The first divergence appears at `W = 16`:

| `W` | traced positions | differing KV bytes | `kv_max_abs` | `kv_rms_rel` max | max logit delta |
|---|---|---|---|---|---|
| 1 / 2 / 4 / 8 | 64 / 32 / 16 / 8 | 0 | 0 | 0 | 0 |
| 16 | 4 | 7946151 | 3.801758 | 0.028196 | 0.343750 |
| 32 | 2 | 4778963 | 3.801758 | 0.028196 | 0.343750 |
| 64 | 1 | 3201507 | 3.750000 | 0.027346 | 0.218750 |

Two controls underwrite this table, and they must not be conflated. The first is internal: `# arm_repeats_identical true` means the harness re-ran the incremental arm at every traced position and got **identical logits**, bailing otherwise with "negative control failed". Read its scope precisely: it compares logits, not KV bytes, and it proves one arm reproduces *itself*; it does **not** prove the two arms wrote the same tokens into the same slots. The second is the `W <= 8` row above. Neither control licenses the claim that the divergence is fully explained.

The jump between `W = 8` and `W = 16` is a discontinuity, not a trend, which is what a kernel-selection threshold looks like. `W = 64` is measurably distinct from `W = 32`, so **`W = 32` is not saturation**, but read that comparison with care: the arms trace `max_n / W` positions respectively, so the right-hand column maximises over a different position set at each width. The table is reported to show the discontinuity and the width-dependence, not as an ordered sweep.

That contrast is the finding this spec exists to serve, but note how it changed once the span was extended. Over a 64-token span the history-build effect is **22x** the single-forward effect (0.343750 against 0.015625). Over a 448-token span it reaches **2.757812**, the same order as the reported 2.0 plain-side preference. So an earlier draft of this section claimed the width mechanism was "two to three orders of magnitude too small" to be the cause, and **that claim was wrong**: it generalised from a short span. The effect grows with context length, which is what a content-accumulation mechanism looks like.

What survives is narrower and still useful: the single-forward width effect is 1 ULP and is *not* the lever, while the history-build width effect is where the contents hypothesis lives. What the measurement does **not** yet explain is the flip; see the margin section below, where the large deltas turn out to miss the top-1 logit entirely.

## Prior work this builds on

- `examples/logit_trace.rs`: teacher-forced per-position logit trace, with the chunk width explicitly described as selecting the kernel under test. It already has the two arms needed (`CHUNK_TOKENS` is the block width) and the `logit_trace/kv_mode.rs` submodule that installs a KV storage mode into model-owned caches.
- `scripts/matched_width_trace.sh`: runs `w<W>.tsv` (block) and `w1_p<P+j>.tsv` (width-1 reference per row).
- `scripts/compare_matched_width.py`: diffs those traces row by row. Row 0 of a block sees only a sequentially built history; rows 1..W-1 see K/V the block itself computed. That row-0 / rows-1.. distinction is exactly the separator this harness needs.
- `docs/temperature-0-block-history-flip.md`: the measured probe results and the reopening trigger.

Nothing in that set reads KV cache contents. `dump_kv`, `kv_digest`, and `TRACE_KV_DUMP` have no matches in the tree.

## Design

### Where it plugs in

A new submodule `examples/logit_trace/kv_dump.rs`, following the existing `#[path = "logit_trace/kv_mode.rs"] mod kv_mode;` convention, plus a hook in `main()` behind an environment variable. `logit_trace` already has the two arms, the anchoring rules, and the `PREFILL`/chunk-width semantics that make a per-position comparison meaningful. The standalone `examples/kv_history_divergence.rs` exists because `logit_trace` cannot keep one cache slice alive across appends (see its module comment); it answers the byte-identity question but not the per-element classification below.

    MLXCEL_TRACE_KV_DUMP=DIR     # opt-in; unset means no dump, no cost

Off by default. The dump is large and the trace is the hot path.

### What is read

Only public API, because each example is a separate crate that links the `mlxcel` and `mlxcel_core` libraries:

| field | access | notes |
|---|---|---|
| keys | `KVCache::keys: Option<UniquePtr<MlxArray>>` | `None` before the first write |
| values | `KVCache::values: Option<UniquePtr<MlxArray>>` | same |
| offset | `KVCache::offset: i32` | monotonic absolute write position |
| live_len | `KVCache::live_len() -> i32` | `offset - live_start`, **public** |
| mode | `KVCache::mode: KVCacheMode` | the storage arm actually in force |
| bytes | `KVCache::nbytes() -> usize` | cheap size-only pre-check |
| key_scales | **`pub(crate)`**, no accessor | INT8/Turbo per-token K scales |
| val_scales | **`pub(crate)`**, no accessor | INT8/Turbo per-token V scales |
| v_packed | `KVCache::v_packed() -> Option<&MlxArray>` | **public**: packed V indices |
| v_norms | `KVCache::v_norms() -> Option<&MlxArray>` | **public**: per-token V norms |
| v_rescale | `KVCache::v_rescale() -> Option<&MlxArray>` | **public**: fused-kernel rescale |
| cold_offset | `KVCache::cold_offset() -> i32` | **public**: tokens folded into cold V |
| k_packed, k_norms | **`pub(crate)`**, no accessor | symmetric `Turbo4` only: packed K + its norms |

The scales are not uniformly a blocker: it depends on how each mode stores its payload. In `int8` mode the extractor hands back a genuine `Int8` array: `quantize_per_token` in `cache.rs` computes `round(x / absmax * 127)`, and the round/clip/cast trio ends in `ffi::astype(&x_clipped, dtype::INT8)`; the doc comment on its caller calls the result "INT8 tensors with per-token scale factors". So one storage step is exactly **one integer LSB**: `delta = |q_a - q_b|`, and `ulp_span` is computable in integer units with no scale access at all. The scales are needed only for **float-space** magnitudes: the absolute logit-space error, or the fp16-arm cross-comparison. That makes `int8` the one quantized arm that needs no new API.

The scales do remain a blocker for the **packed** modes: `Turbo3Asym`, `Turbo4Asym` and symmetric `Turbo4` store several elements per byte, so a byte-level delta is not a value-level step. The V side is in better shape than an earlier draft assumed: `v_packed`, `v_norms` and `v_rescale` all have public accessors, so V-side packed arms are measurable today. Only the symmetric `Turbo4` **K** side (`k_packed` / `k_norms`) has no public path.

`live_start` itself is `pub(crate)`, but `live_len()` exposes the derived quantity that matters, so the live window *is* reachable from the example crate. An earlier draft of this spec wrongly treated it as unavailable and planned a partial `offset`-only alignment. That is not needed.

`visible_len()` is a `RotatingKVCache` method, not a `KVCache` one. That no longer blocks a rotating arm, but not for the reason an earlier draft gave: the public rotating extractor does **not** narrow the live window. It returns the whole pre-allocated ring and the harness must map `logical_start()` / `visible_len()` itself (see "Limits").

### Do not invent a second definition of "the KV contents"

The main crate already extracts K/V for the disaggregated prefill→decode handoff, and it already had to answer "which bytes are the cache". The answer is in `src/distributed/kv_cache_serde/serialize.rs`:

```rust
let live_len = cache.live_len();
// slice the pre-allocated buffer down to the filled portion
let filled_k = mlxcel_core::slice(keys, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]);
```

followed by `extract_mlx_array_data` → `RawTensorData { data, shape, dtype }` via `array_to_raw_bytes`. That module's own comment states the intent: with `live_start == 0` the live-window slice is *bit-identical* to slicing at `offset`, so the wire format and a naive dump agree, and when they do not, the live window is the correct one.

Both of those helpers are private to the module: `extract_mlx_array_data` is declared `fn`, not `pub fn`. The harness therefore does not reimplement them and does not need to: it calls the enclosing public extractor and receives the same `RawTensorData` the handoff path would send. The task is not to reproduce that slicing logic, it is to not have any of its own.

The requirement is on the **bytes**, not on the steps: a dump that disagrees with the wire format would be indistinguishable from corruption. A mirrored copy of the slicing could drift from the wire format, and the drift would look like corruption. The harness does not need one, because the extractors are public API:

- `src/lib.rs` declares `pub mod distributed;` unconditionally.
- `src/distributed/kv_cache_serde/mod.rs` declares `pub mod serialize;` and re-exports `extract_kv_cache_entry`, `extract_rotating_cache_entry`, and `extract_chunked_cache_entry` at `mlxcel::distributed::kv_cache_serde::*`.
- The return type is usable from outside: `SerializableCacheEntry` holds `pub keys`/`pub values` of `RawTensorData`, whose `data`, `shape`, and `dtype` are all `pub`.

So the harness stores exactly the bytes the handoff path stores, by calling the same function the handoff path calls:

```rust
let entry = mlxcel::distributed::kv_cache_serde::extract_kv_cache_entry(cache);
```

**But the extractor covers only two of six modes, and it fails silently on the rest.** Its body is a two-armed `match`:

```rust
match (&cache.keys, &cache.values) {
    (Some(keys), Some(values)) if live_len > 0 => { /* slice + extract both */ }
    _ => SerializableCacheEntry { keys: None, values: None },
}
```

Neither branch can express a packed cache. `Turbo4Asym` and `Turbo3Asym` keep K in `keys` but store V only in `v_packed`/`v_norms`, leaving `values: None`; symmetric `Turbo4` is the mirror image, leaving `keys: None` and storing both sides packed. Both fall into the `_` arm and return `(None, None)`. There is no error, no warning, and no `mode` field on `SerializableCacheEntry` (`types.rs`, which has exactly `keys` and `values`) to reveal it, so a harness that trusted this call would record an empty dump for those modes and could easily read that as "the KV caches agree, zero bytes differ". That is exactly the corruption-vs-rounding question this task exists to settle, answered by a false negative.

`Turbo4Delegated` is worse than empty, because it takes the *first* branch and returns the wrong bytes. In its default path `values` is not a unified V buffer; it is the **hot ring only**, deliberately shrunk to `hot_threshold`-sized chunks so that decode never touches cold storage (the `KVCache` field docs in `cache.rs`: "`values`: V hot ring FP16, `[B, H, v_hot_capacity, V_dim]`. Visible length is `hot_offset() = offset - cold_offset`"). The extractor slices it with `&[.., live_len]` instead of `&[.., offset - cold_offset]` (`serialize.rs`, where the same `live_len` is used for both `keys` and `values`), so whenever `cold_offset > 0` the read either overruns the hot ring or traps a stale tail, and the tokens actually folded into `v_packed` are never serialized at all. Only the opt-in FP16 fast path (`MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`, read by `delegated_fp16_fast_path_enabled()` in `cache/turbo/mod.rs` into the `pub(crate)` field `KVCache::delegated_fp16_fast_path`, which has no accessor) restores a unified `values` buffer with visible length `offset`, and that flag is off by default (the helper returns `false` when the variable is unset).

The ordering of that read is a trap, and it fails silently. The helper caches its answer forever: `cache/turbo/mod.rs` declares `static DELEGATED_FP16_FAST_PATH_ENABLED: OnceLock<bool>`, and `delegated_fp16_fast_path_enabled()` fills it with `get_or_init`: the env var is read on first call and never again. Each `KVCache` then snapshots the answer into its own field at construction. So the variable must be set before the first cache is built. Setting it later, even before the forward pass that matters, leaves the `OnceLock` and every existing cache at `false`, and the arm silently measures the hot-ring path it was trying to avoid. The harness therefore sets the variable at process start and then asserts the flag actually took effect, by checking that the extracted `values` third dimension equals `offset` rather than `hot_offset()`: that shape difference is the only externally visible distinction between the two regimes, since the flag and the buffer it selects have no public accessor.

This is a real defect in the handoff path, not merely a harness inconvenience, and it is worth reporting separately from the `ulp_span` question: a disaggregated prefill→decode handoff with `Turbo4Delegated` active silently transfers a truncated V history whenever a fold has occurred. It is out of scope for the reproducer itself, which only needs to not depend on that path.

Only two modes are therefore wire-extractable as the code stands: `Fp16`, and `Turbo4Delegated` **with the fast-path flag on**.

So the shared-call argument holds only where the shared call can see the data. The harness therefore uses `extract_kv_cache_entry` for `Fp16` and for `Turbo4Delegated` with the fast path enabled, and reads the packed sidecars through their **public accessors** for the packed arms: `v_packed()`, `v_norms()`, `v_rescale()`, plus `cold_offset()` to know where cold storage ends and the hot ring begins. The symmetric `Turbo4` K side has no public accessor (`k_packed`/`k_norms` are `pub(crate)`), so that one arm still needs either a read-only accessor pair or an explicit "not measured" note in the issue.

A rotating arm calls `extract_rotating_cache_entry` and a chunked arm calls `extract_chunked_cache_entry`. Both differ from the dense extractor in a way that cuts against convenience, not for it: neither has a `live_len > 0` guard and neither slices at all. They hand back the entire pre-allocated buffer (unwritten slots included) so the harness must narrow the live window itself with `logical_start()` / `visible_len()` before comparing anything. See the rotating note under "Limits".

`array_to_raw_bytes` is already imported by `examples/logit_trace.rs`.

### Dump format

One file per (arm, layer), plus a manifest. Binary, not text: these tensors are `[B, H, L, D]` fp16 and a text dump would be slower than the forward pass.

    <DIR>/manifest.tsv
    <DIR>/c<chunk>_l<layer>_k.bin       # raw evaluated key bytes, current live buffer
    <DIR>/c<chunk>_l<layer>_v.bin       # raw evaluated value bytes

`manifest.tsv`, one row per dumped tensor, tab-separated, no header comment so the reader stays dumb:

    chunk  layer  offset  mode  dtype  byte_order  shape  bytes  path

- `chunk`: the same chunk index the trace rows use, so a KV row joins to its logit rows on `(chunk, offset)`.
- `offset`: `KVCache::offset` at dump time. The join key.
- `mode`: `KVCacheMode` as the model stored it, not as requested. `kv_mode.rs` already asserts requested equals applied via `kv_cache_layer_modes()`, and the dump should re-assert rather than trust. This column is not uniform across arms, and the reader must not assume it: `KVCache` and `RotatingKVCache` both carry a `pub mode: KVCacheMode`, but `ChunkedKVCache` has **no mode field and no mode method at all**; it is FP16 by construction. So the chunked arm writes a literal `fp16` here, and a comparator that reads `mode` as proof of which arm produced the bytes would be reading a constant on that arm.
- `dtype`/`shape`: so the comparator does not hardcode a layout. `dtype` is the **integer MLX dtype code**, not a name: `RawTensorData` declares `pub dtype: i32` with the comment "MLX dtype code (matches `mlxcel_core::dtype` constants)". The manifest must therefore render it as a name (`float16`, `int8`) on write, or the reader has to join against the same constants table, and a wrong table would make an int8 arm look like float16 without any error. Writing the name keeps the format self-describing, which is the whole point of the column. Note there is **no** dtype-name helper anywhere in `mlxcel-core` (a search for `dtype_name`/`dtype_to_string` finds nothing), so the harness carries its own small `i32 -> &'static str` table and it must be an exhaustive `match`, not a lookup that silently falls through to `"unknown"`: the code arrives via `ffi::array_dtype`, the same call `extract_mlx_array_data` makes internally.
- `byte_order`: `le` or `be`, as written. Present so the reader never assumes host order; see the note on byte order below.
- `bytes`: must equal the file length; a truncated write is otherwise silent.

Evaluation order matters: call `eval()` on the array before reading bytes, or the dump can observe a lazy node instead of materialized K/V. This is the one place in the design where a mistake produces plausible-looking garbage.

Byte order is part of the format, not an implementation detail, because these files are read back by a Python comparator on a different toolchain. Pin it here or the first reader that reinterprets the payload will disagree with the writer silently:

- `array_to_raw_bytes` returns the bytes as the host laid them out. The two existing consumers already disagree about what that means: `examples/logit_trace.rs` decodes with `f32::from_le_bytes` while `examples/paged_decode_v2_correctness.rs` decodes with `f32::from_ne_bytes`. On a little-endian host those two are indistinguishable, which is why the disagreement has survived.
- The dump therefore records the byte order it wrote in the manifest, and the reader dispatches on that column instead of assuming. The one-line way to make it moot is to declare the dump **little-endian and convert on write** when `cfg!(target_endian = "big")`; then the manifest column is a constant and the Python reader is unconditional.
- This is the same class of defect as the un-evaluated array above: correct on the developer's machine, wrong elsewhere, and invisible until someone looks at the numbers.

### Cost control

`MLXCEL_TRACE_KV_DUMP` stays unset for normal traces. When set, dumping every layer of every chunk for a mixed-width sweep is gigabytes. Two knobs:

    MLXCEL_TRACE_KV_DUMP_LAYERS=0,17,27   # default: first, middle, last
    MLXCEL_TRACE_KV_DUMP_CHUNKS=0,1       # default: first two chunks

Three layers catches a per-layer mode boundary (`MLXCEL_KV_BOUNDARY_V_LAYERS`); first and last catch a layer-0 wiring bug and a depth-accumulated error. Two chunks catch a first-chunk seeding bug without a full sweep.

### The comparator

`scripts/compare_kv_contents.py`, in the style of `compare_matched_width.py`: read two dump directories, print a table, exit non-zero only on classification 3.

Inputs are a block arm `w<W>/` and a width-1 arm `w1/`. Alignment is on `(layer, offset)`: for each monotonic position, the block arm's element must be compared with the width-1 arm's element for the *same written token*, and the token identity check comes from the existing logit trace (`pos` and `tok` columns already assert it: `compare_matched_width.py` has `assert tok1 == tokb` for precisely this reason).

Per `(layer, position, side)`:

- `exact`: byte-identical.
- `max_abs_delta`, `rms_delta`: over the head_dim axis.
- `ulp_span`: the delta divided by the storage resolution at that magnitude. This, not raw delta, is the rounding test.
- `n_clipped`: elements at the mode's saturation bound (turbo codebooks and int8 both clip; a clipped element can move without any rounding story).

`ulp_span` is computable from public API for `fp16`/`bf16` (resolution is a function of the dtype alone) and, contrary to the earlier draft of this section, for **`int8`** as well, without the scales. The quantizer is

```rust
let absmax = ffi::max_axis(&ffi::abs(x), -1, true);       // [B, H, T, 1]
let scale = divide_scalar(&absmax, 127.0);                // [B, H, T, 1]
let is_zero = ffi::equal(&scale, &zero);
let safe_scale = ffi::where_cond(&is_zero, &one, &scale);  // 1.0 for all-zero tokens
let x_div = ffi::divide(x, &safe_scale);
let x_rounded = ffi::round(&x_div);
let x_clipped = ffi::clip(&x_rounded, &lo, &hi);           // -128 ..= 127
let x_int8 = ffi::astype(&x_clipped, dtype::INT8);
(x_int8, safe_scale)
```

(`quantize_per_token` in `cache.rs`, abridged). The stored element is an integer `q`, and the value it represents is `q * safe_scale`. So the storage resolution, in the stored domain, is exactly **one integer step**, and a delta between two `int8` stores is a delta in integer units. `ulp_span` on the int8 arm is therefore the raw integer delta and needs no scale at all, which is the *stronger* statement, because it does not depend on a scale the two arms might compute differently.

The scales matter for a different quantity: converting an integer delta into absolute error, `|Δq| * scale`, and for asking whether the two arms chose *different grids* for the same token. The second one is worth recording separately, and it is also readable: a differing grid moves the element to a different lattice, and comparing integer domains across unequal grids is where `int8` can look corrupted when it is not. That is a comparator rule (the scale-relative comparison is only valid when both arms stored the same `absmax`), not a blocker.

What genuinely remains blocked is the **K side of packed `turbo*`**. Those modes store packed indices plus per-token norms: `v_packed`/`v_norms`/ `v_rescale` on the V side, `k_packed`/`k_norms` on the K side. All of those are `pub(crate)` *fields* (see the `KVCache` struct in `cache.rs`, from `key_scales` through `k_norms`), but field visibility is not the test, accessor visibility is. The V side passes it: `v_packed()`, `v_norms()` and `v_rescale()` are `pub fn`, so a V-side packed delta is readable today. The K side does not: `k_packed` and `k_norms` are `pub(crate)` with no accessor anywhere in the impl, and those exist only in symmetric `Turbo4`. The codebook index is also meaningless as a resolution without the norm, which is why raw delta alone is the fallback rather than the plan. For the K side of symmetric `Turbo4` the honest position is:

1. Report `ulp_span` on the fp16 and int8 arms and on all V-side packed arms, and downgrade symmetric-`Turbo4` K to `max_abs_delta`. The weaker claim is confined to arms where it is the only defensible one.
2. Add a read-only `pub fn` pair for `k_packed`/`k_norms` (or a `pub` re-export). That makes `ulp_span` uniform. Small and reviewable.

The spec does not choose for the maintainers, but the question is now much narrower than it was: it asks about **two accessors in one mode**, not about "all quantized arms" and not about a five-accessor sweep.

For completeness, the extractor does not dissolve this: `SerializableCacheEntry` carries `keys` and `values` and nothing else, and `RawTensorData` has `data`, `shape`, `dtype`. Scales and norms are dropped before the entry is constructed. For `int8` that no longer matters, because the integer lattice is sufficient; for `turbo*` it is exactly the missing piece.

One trap this makes visible, and it silently defeats an int8 arm on any sliding-window model:

- `RotatingKVCache::update_and_fetch` matches on mode, and only `Fp16` and `Turbo4Asym` reach a real implementation. `Int8`, `Turbo4`, `Turbo4Delegated` and `Turbo3Asym` all call `update_and_fetch_fp16`: the comments say so explicitly, e.g. "Fall back to FP16 storage so the path is correct, even if mis-configured" and "Turbo3Asym is not wired into RotatingKVCache in this PR" (`cache.rs`, the `match self.mode` block; `fp16` and `Turbo4Asym` are the only arms that do not delegate).
- So on a sliding-window model the **only** measurable storage arms are `fp16` and `Turbo4Asym`. Every other requested mode is silently storage-FP16.
- The `mode` field is *not* reset, so `KVCache::mode` still reads `Int8` while the bytes are FP16.
- A dump on such a model would record `mode=int8`, `dtype=float16`, and an int8-calibrated comparator would classify correctly by `dtype` but a reader trusting the `mode` column would conclude the int8 arm was exercised when it was not. The manifest therefore records `cache.mode` **and** the extracted `dtype`, and the comparator asserts they form an expected pairing (fp16↔ float16, int8↔int8); a mismatch is reported as "arm not actually exercised" rather than as a pass. This is the same check as step 5, stated in the manifest's own terms.

Classification needs a stated bound, or it is not a test:

- **equal**: `max_abs_delta == 0` for every position.
- **rounding-equivalent**: every `ulp_span <= ULP_BOUND` *and* `n_clipped == 0` *and* `max_abs_delta` does not grow monotonically with position index. The growth clause is what separates "each write is independently rounded" from "error compounds along the block", and only the latter is corruption.
- **corrupted**: any of: `ulp_span > ULP_BOUND`; growth with position; a position written in one arm and absent in the other; `bytes` disagreeing with file length; `byte_order` disagreeing between the arms (the reader would be comparing numbers it misread); `mode` disagreeing with the requested arm.

`ULP_BOUND` must be calibrated on the `fp16` arm first, where the arithmetic difference is smallest, and then applied unchanged to `int8`/`turbo4`. Choosing it per arm would let each arm define its own pass. The width-axis measurement above already supplies the prior: start at `ULP_BOUND = 1` and widen only with a stated reason, because that is the bound the logits actually showed.

### What the harness must also record, or the result is unfalsifiable

- **Absolute paths and `MLXCEL_*` environment** in the manifest. The existing doc's weakness is that the original run's configuration was lost; repeating that mistake in a new harness would be worse than not building it.
- **Backend and chip.** The original report is M1 Ultra; a harness run proving absence of corruption on M1 Pro does not transfer. The classification must carry a `platform` field and the write-up must not generalise across it.
- **Prompt identity.** Even though the harness does not need *the* original prompt, a run must name the corpus file and its hash, so a later reader can re-derive what was compared.

## Limits

- **This does not satisfy criterion 1 as written.** It produces the KV half. The pos 197 half still needs the prompt from `inureyes`, or a substitute prompt that is shown to trigger a top-2 near-tie on a comparable model, and that substitution would have to be argued for in the issue, not assumed.
- **Rotating and chunked caches need harness-side windowing.** An earlier draft called them uncovered on the strength of `live_len()` being a `KVCache`-only method, then over-corrected and claimed the public extractors "answer the live-window question themselves". The extractor body says otherwise: `extract_rotating_cache_entry` passes `cache.keys` and `cache.values` straight to `extract_mlx_array_data` with no `slice` call at all, and `extract_chunked_cache_entry` is byte-for-byte the same shape. Both return the **whole pre-allocated buffer, unsliced**: unwritten slots included and the ring origin unrecoverable from the returned shape (its test tolerates `shape[2]` anywhere in `3..=4` for a 4-slot ring). A rotating arm must therefore map the live window itself with `visible_len()` / `logical_start()` *inside* the harness, and the regimes differ: `visible_len()` is `idx.max(0)` for a fixed-buffer cache (`buffer_size > 0`) and `seq_len().min(offset).max(0)` otherwise; `logical_start()` is `0` while `buffer_size > 0` *or* while `offset <= visible_len` (i.e. before the ring wraps), and only then becomes `idx.rem_euclid(visible_len())`. Assuming either regime alone is a silent off-by-a-rotation in the diff. What is genuinely correct in the extractor is the *convention*: the returned bytes are exactly what the handoff path ships, so the comparator's alignment question for a ring that overwrites (`(layer, offset)` stops being a key once `offset > buffer_size`) is still the one real gap in the dump family.
- **Scope.** `logit_trace` installs a KV mode only for models that own their caches (`kv_cache_layer_modes()` errors otherwise). Families whose caches the harness's own `make_caches()` holds are covered by the same code path but not by that assertion.
- **The wire extractor is not a general KV reader.** `extract_kv_cache_entry` is a handoff serializer for the shapes the handoff path ships, and it degrades silently outside them: `(None, None)` for the two asymmetric turbo modes and symmetric `Turbo4`, and *truncated* V for `Turbo4Delegated` once a fold has advanced `cold_offset`. See "Do not invent a second definition of 'the KV contents'". Packed arms read public sidecar accessors instead.
- **`ulp_span` needs an API decision only for symmetric `Turbo4`.** `int8` resolves in integer LSBs and `v_packed`/`v_norms`/`v_rescale` are public, so those arms need nothing new. Only `k_packed`/`k_norms` (symmetric `Turbo4` K side) are `pub(crate)` with no accessor; that arm is raw-delta-only or unmeasured until a read-only accessor exists.
- **Verify what you cite, and cite what you ran.** The source offsets in this document were read from source, not observed. The *measurements* in the two 2026-11-08 sections are different: they come from `examples/kv_history_divergence.rs`, built with `--features metal,accelerate` and run on Metal (`# device Apple GPU (Metal)`), and every number there was read out of a TSV whose footers are quoted with it. Keep the two kinds of claim apart when editing this document: a source offset can be wrong in a way a TSV row cannot, and vice versa.

## Suggested next actions, in order

1. Ask `inureyes` for the `Qwen3 copy` prompt text and KV mode in #2778. One comment; unblocks the half that is currently impossible. (The earlier framing of criterion 1 as "a code-level fix" has been corrected in the thread, and PR #2879 now says `Refs #2778`: criterion 1 is a reproducer that compares logits **and KV contents**, and the issue must not be auto-closed while it is open.)
2. Decide the K-side accessor question for symmetric **`Turbo4`** only (`pub(crate)` `k_packed`/`k_norms` in `cache.rs`, no accessor). The V side no longer needs a decision: `v_packed()`, `v_norms()` and `v_rescale()` are already `pub fn`. Either add a read-only accessor pair for the K side, or state in the issue that symmetric `Turbo4` is judged on raw delta only. `int8` needs no accessor at all: its resolution is one integer step, readable from the extracted `Int8` bytes.
3. Land the KV dump behind `MLXCEL_TRACE_KV_DUMP`: small, off by default, and useful on its own for any cache-shape investigation. It is now smaller than this spec originally assumed: on the dense `Fp16` arm one call to the public `extract_kv_cache_entry` per layer replaces the hand-written `live_len()` / `slice` / `eval` sequence, so there is no cache-shape arithmetic to review and no second definition of "the KV contents" to keep in sync. Packed arms still read their sidecars directly, and the dump records which source produced each row.
4. ~~Run it and calibrate `ULP_BOUND` on `fp16`.~~ **Done**, on Qwen3-0.6B-4bit rather than the Gemma 3 probe, using a standalone harness that needs no new API. The answer is **not** byte-identical: `kv_rms_rel` up to 0.0455, up to 73.97% of elements perturbed. The fp16 `ULP_BOUND` prior inherited from the width-axis control (1 ULP) is therefore **wrong for the history arm**: it is a bound on a different quantity. Calibrate it against the history arm's own measured distribution.
5. Extend to `int8` with no API change at all (integer-unit `ulp_span`), and to the K side of symmetric `Turbo4` only if step 2 added the accessors. Which arms are even reachable depends on the cache class: on a dense model all six modes store as declared, but `RotatingKVCache` silently falls back to FP16 for `Int8`, `Turbo4`, `Turbo4Delegated` and `Turbo3Asym`, leaving only `fp16` and `Turbo4Asym` genuinely measurable there. So record per row which mode produced the bytes, and assert it harness-side: the extracted entry carries no `mode`, so this is not a round-trip check. `KVCache` and `RotatingKVCache` expose `pub mode`, but `ChunkedKVCache` exposes neither field nor method, so on that arm a literal `fp16` is the only available value and it proves nothing. Where a mode is readable, check it agrees with the extracted `dtype`; where it is not, say so in the row rather than inventing a value. Then report a mismatch as "arm not actually exercised" instead of counting it as a pass.
6. Post the classification to #2778. The **byte-identity** question is now answered: the float KV is *not* byte-identical across plain and block-built histories from `W = 16` up (outcome 1 is ruled out at those widths). Whether the difference is rounding-equivalent or corrupted (outcomes 2 and 3) is not yet decided, because that needs the per-element comparator and a calibrated `ULP_BOUND`; until then the reopening trigger is open, not tripped. What is **not** answered is the flip: on this corpus the perturbation misses the top-1 logit, `positions_with_flip` is 0, and the report's 2.0-versus-35 reversal is therefore *not reproduced*. Post exactly that split: not byte-identical, not yet classified, flip unreproduced, absolute reproduction blocked on the prompt text, rather than reporting either half alone.

## What the 2026-11-08 measurement already changes about the plan

Step 1 said the swing half is waiting on a human, and that is still true for a *literal* reproduction of the reported run. But the width control does narrow the search without needing the original prompt: varying the width of a *single traced forward* moves the logits by 1 ULP (0.015625), which cannot produce a 2.0-versus-35 swing. Varying the width of the *history build* is a different story: it moves logits by 0.343750 over 64 tokens and **2.757812 over 448 tokens**, the same order as the reported 2.0 plain-side preference. So the remaining candidate is the block-built history's *contents*, and magnitude does not rule it out at long context.

That makes step 4 the most valuable step, not a calibration chore. It has now been run: the float KV is **not** byte-identical across plain and block-built histories: `kv_rms_rel` reaches 0.0455 with up to 73.97% of elements perturbed. That rules out outcome 1 but does not by itself separate outcome 2 from outcome 3: a rounding difference in an early layer changes every later layer's input, so wide perturbation in deep layers is expected under either. What remains unexplained is why the flip does not follow, and the margin measurement below answers that on this corpus.

One thing this measurement does not license: claiming the flip is explained. It is a bound on one axis, on one backend, on M1 Pro, with a substitute prompt. The issue's own conditions (M1 Ultra, the original `Qwen3 copy` prompt) remain unreproduced, and the write-up must keep saying so.

## What the margin measurement settles (2026-11-08, second run)

Step 4 asked whether the KV divergence is large enough to matter. It was run, and the answer inverts the question. On Qwen3-0.6B-4bit, `W = 16`, 64-token context, 448-token span, the divergence is real and large: every one of the **28** traced positions carries differing KV bytes, `kv_rms_rel` reaches 0.0455, up to 73.97% of elements perturbed, and a **2.757812** whole-vocabulary logit delta at pos 352. That is the same order as the issue's 2.0 plain-side preference, so magnitude was never the missing piece.

(The harness traces every `W`-th history length, so a 448-token span at `W = 16` gives `positions_traced` = 28, and `positions_with_kv_diff` was also 28. Per-position maxima below are maxima over those 28.)

What was missing is *where* the perturbation lands. Adding a `d_top1` column (the shift at the incremental arm's own winning logit, rather than the maximum over all 151936 of them) makes the picture unambiguous:

| pos | `d_top1` | `margin_inc` | `d_top1` / margin | `max_abs_delta` |
|---|---|---|---|---|
| 352 | 0.500 | 12.000 | 4.17% | 2.757812 |
| 448 | 0.375 | 6.875 | 5.45% | 0.437500 |

Across the 28 diverging positions the top-1 perturbation consumes at most **5.455%** of the margin (pos 448). `d_top1` is signed, so the honest summary uses its magnitude: the mean is **1.53%** of margin over all 28 positions and **2.38%** over the 18 where `d_top1` is non-zero. `margin_consumed_positions` is **0**: none of the 28 traced positions came close to an argmax contest. At pos 352 the entire 2.76 of whole-vocabulary motion misses the top-1 logit, which moves 0.5 against a 12.0 lead.

So **not flipping is the correct, expected outcome on this corpus**, not an anomaly awaiting an explanation. The reported swing therefore cannot arise from an argmax contest at this margin scale: whatever produced a 2.0-versus-35 preference reversal must have restructured the upper vocabulary, used a different prompt, or run under a KV mode this harness cannot install on Qwen3 (`kv_mode.rs`: the model does not own its caches, so the int8-KV hypothesis is untestable here, and the dump reads harness-owned `make_caches()` caches).

A methodological correction worth carrying: the first version of this column compared the whole-vocabulary maximum against the top-1 gap and reported one `margin_consumed = true` at pos 64, where the delta was 0.21875 against a 0.125 gap yet `flip` stayed `false`. That was a false positive from comparing quantities on different supports: a logit ranked thousands down can move by 3.0 without entering contention. The exact test uses `|log_blk[ai] - log_inc[ai]| > margin_inc`, which dropped the count from 1 to 0. Any downstream reader of this harness must use the 17-column layout: `d_top1` is field 11 and `flip` moved to field 17.
