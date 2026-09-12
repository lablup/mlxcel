# Bucketing the cuDNN SDPA plan-cache key across verify rounds (GB10, 2026-09-12)

Issue #1820. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64, MLX pin `81ba1c6a`, CUDA release build (`MLX_CUDA_ARCHITECTURES=121`), toolchain 1.97.1. Base tree `60341873` (main, which already carries PR #1817, the #1799 fix). Pairing: `models/mlx/laguna-xs-2.1-nvfp4` with `models/mlx/laguna-xs-2.1-dflash`, the pairing #1799 attributed the defect on. Prompt: the 152-token raw Python source header of #1782 (`harness/prompt_code0.txt`), no chat template. Harness: `harness/` in this directory, the #1798 and #1799 harness with a `sweep_laguna.sh` driver and a `summarize_trace.py` added.

## Which cache-key fields actually move, per shape class

Before designing around the key, measure it. `MLXCEL_SDPA_PLAN_DEBUG=1` writes one line per cuDNN SDPA call with the fields `build_sdpa_cache_key` reads. Run: block 4, 60 tokens, 25 verify rounds, upstream dispatch restored (`MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0 MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0`), so every verify SDPA reaches cuDNN as it did before #1817. Raw trace: `trace_upstream_w4.txt` (1401 calls).

| shape class | layers | calls | plan builds | distinct `k_len` | distinct mask columns | distinct mask row stride | k/v buffer extent | k/v row stride | mask leading strides | sinks |
|---|---|---|---|---|---|---|---|---|---|---|
| target full attention (48 query heads) | 10 | 260 | 25 | 25 | 25 | 25 | 256, constant | 128, constant | 0, 0 | no |
| target sliding attention (64 query heads) | 30 | 780 | 25 | 25 | 25 | 25 | 768, constant | 128, constant | 0, 0 | no |
| drafter (64 query heads) | 5 | 120 | 24 | 23 | 23 | 23 | none (not a cache slice) | 128, constant | 0, 0 | no |

Three plan builds per round, one per class, measured directly rather than inferred from the round's host time: 75 builds over 25 rounds, plus 7 more for the prefill and the exactness probe, 82 in the process. That is #1799's inference confirmed.

Four facts the design rests on, and two corrections to the record this issue inherited:

- **Only three key fields move**: the k and v sequence length, the mask's column count, and the mask's row stride. The k and v strides do not move (the row stride is the head dim in every class and the buffer extent is constant), and the mask's two leading strides are 0 in every class because `fast::scaled_dot_product_attention` broadcasts a `[L, k_len]` plane up to `[B, H, L, k_len]`. So bucketing the k/v shape alone is not enough (the mask's own two fields would still move) and the mask can be widened for the cost of one `[L, bucket]` plane rather than `B * H` of them.
- **The sliding class does not hold `k_len` constant.** #1820 asked whether a saturated rotating window would. It does not here: the window is 512 and the run reaches 212 keys, so `k_len` grows every round in all three classes. A generation long enough to saturate the window would stop that class missing on its own; no run in this record reaches it.
- **Correction: this checkpoint has no attention sinks.** #1799 describes the sliding layers as carrying per-head sinks. `model.safetensors.index.json` has no `sink` tensor and every traced call reports `sinks=0`. The sinks question is therefore not on the Laguna path at all; it is still handled (the capability flag below is kept per sinks setting) but it is unmeasured here.
- **Correction: the three classes are not "full, sliding, drafter" by head count the way the earlier record reads.** The target's full-attention layers run 48 query heads and its sliding layers 64; both are the target. The drafter is the third class and is distinguished by its k/v not being a cache slice at all.
- **The drafter's k and v are not a slice of any cache buffer.** The other two classes hand SDPA a leading slice of a fixed-size buffer (extent 256 for the dense full-attention cache, 768 for the speculative-buffered rotating cache, both constant across the run), which MLX's own decode canonicalization already knows how to widen. The drafter concatenates its proposal keys onto the cache window before attending, so its k and v are freshly built arrays with no room past `k_len`. Any scheme that only unslices cannot bucket that class, and that class is one of the three builds per round.

## The change

MLX already implements the canonicalization this needs, and gates it to a one-row call with no array mask (`use_cudnn_for_decoding`, `mlx/backend/cuda/scaled_dot_product_attention.cpp:72-107` at the pinned `81ba1c6a`). A decode step unslices k and v to the whole cache buffer, whose extent is fixed across steps, and tells cuDNN the true lengths through `set_padding_mask(true)` with `set_seq_len_q` / `set_seq_len_kv` (`build_sdpa_graph:240-243`), and `build_sdpa_cache_key:168-174` then writes the buffer extent into the key in place of the live length and zeroes the k/v strides. One plan serves every decode step. The overlay extends that to a small array-masked multi-row call, which is the verify shape, and adds the one piece the decode path does not need:

- **The mask is widened too.** Its column count and row stride are separate key fields, so bucketing k and v alone leaves two fields moving and the key still misses. The `[L, k_len]` plane is copied into an `[L, bucket]` plane whose new columns are `-inf`, and re-presented as `[B, H, L, bucket]` with the same stride-0 leading axes the input had. Cost is one `[L, bucket]` plane per call, not `B * H` of them. The alternative the issue named, expressing the band natively with `set_causal_mask_bottom_right` plus a sliding bound and dropping the array mask, was rejected: the overlay is handed an opaque additive mask and would have to read it back to the host to prove it is exactly a causal band, and the callers that build it (`create_causal_mask`, `create_causal_mask_with_window_full`) are shared with the Metal and CPU paths, so converting them is a change to every backend's arithmetic rather than to the CUDA plan cache.
- **k and v reach the bucket two ways.** Unsliced, when they are a leading slice of one contiguous cache buffer with room past the live length: free, and it is what the target's two classes are. Copied into a zero-padded buffer when they are not: that is the drafter class, whose arrays are built fresh by concatenating the proposal keys onto the cache window and so have no room to unslice into. The copy is bounded by `MLXCEL_SDPA_PLAN_BUCKET_MAX_MB` (64 per tensor, so about 32k keys on an 8-head 128-dim bf16 cache); above the bound the call keeps its exact shape and pays the build, because moving hundreds of MB per layer per round would cost more than the 22 ms it saves.
- **The bucket is 256** where a copy is made, and the cache buffer's own allocated extent where one is not. 256 is what MLX's decode canonicalization assumes (`kv_cache_step`) and what mlxcel's KV cache grows by (`cache.rs:499-506`), so the unsliced arm never reallocates and the copied arm rounds to the same grid. Measured extents on this pairing are 256 for the dense full-attention cache and 768 for the speculative-buffered rotating cache.

### How the widened region is kept out of the result

Widening points the kernel at key positions the caller never wrote, so this has to be stated rather than assumed. What those positions contain differs by arm: on the unsliced arm they are the cache buffer's unwritten tail, which is whatever the allocator last left there and can be any bit pattern including a NaN payload; on the copied arm they are zeros, written by the `fill_gpu` that precedes the copy.

Two mechanisms exclude them, and they are independent:

1. `set_padding_mask(true)` with `set_seq_len_kv` set to the true `k_len`. This is cuDNN's own contract and it is the same one MLX's shipped decode path already relies on for exactly this hazard, on the same unwritten cache tail.
2. The widened mask's new columns are `-inf`, so even a bias applied before the padding mask drives those scores to zero weight.

Neither is load-bearing alone for the copied arm, where the padded keys are zeros. For the unsliced arm mechanism 1 is the one that matters, and the evidence that it holds is the identity below rather than the argument.

The backward primitive is untouched: `ScaledDotProductAttentionVJP` has its own cache and its own `use_fallback`, and nothing here reaches it. `force_fused=True` still raises where no fused kernel exists, because the gate only ever moves a call between two paths that both exist. `output_logsumexp` calls are excluded from bucketing outright.

## Identity: the widened path returns the same tokens

Compared as token ids (`MLXCEL_PRINT_TOKEN_IDS=1`), not as text, per #1782. Laguna DFlash block 4, 200 tokens, greedy, 152-token code prompt. Three arms, two binaries: `base` is the tree at `60341873` (main, which carries #1817), `new` is this branch. "exact-shape cuDNN" is `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0 MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0`, which restores upstream dispatch on either binary.

| arm | binary | dispatch | ids |
|---|---|---|---|
| exact-shape cuDNN | base (`60341873`) | one plan per round, per class | `7d044d85` |
| exact-shape cuDNN | new | one plan per round, per class | `7d044d85` |
| bucketed cuDNN (the new default) | new | one plan per class per bucket | `7d044d85` |
| ops fallback (main's default, #1817) | base | no cuDNN | `baa2c6b5` |

All 200 ids identical across the three cuDNN arms. That is the proof the widening asks for: same kernel, same inputs, one arm reading a widened tensor and one not, byte-identical output. Each arm was also run three times and is deterministic run to run.

The fourth row is the one to read carefully. This branch's default output differs from **main's** default output, and that is a dispatch change, not a padding error: #1817 moved these calls from cuDNN's flash kernel to MLX's ops fallback, and #1799 already recorded that the fallback "shifts the greedy path only at ties" (its acceptance moved 0.76 against 0.77 at block 2). Bucketing moves them back, so the branch reproduces the pre-#1817 cuDNN answer exactly. There is no third answer.

### Non-speculative arms

`models/mlx/qwen3-1.7b-4bit` (dense, head_dim 128, so cuDNN-eligible exactly like Laguna), same binary pair.

| workload | result |
|---|---|
| classic decode, 152-token prompt, 200 tokens | 200 ids identical, base against new |
| classic decode, 2634-token prompt (chunked prefill), 120 tokens | not stable enough to compare: the **baseline binary alone**, same env, produced two different id sequences in three repeats (`ddca7360`, `7c42e164`, `ddca7360`), and both values also appear on the new binary. Host nondeterminism, the same signature #1799 recorded for Laguna classic decode, and not attributable either way. |

The stronger statement for the non-speculative case is structural rather than statistical. With `MLXCEL_SDPA_PLAN_DEBUG=1` the 2634-token run traces 3444 cuDNN SDPA calls in three classes, and **not one of them buckets**: the two prefill chunks are 2048 and 586 query rows, both far above the 32-row bound, and the 3332 decode calls take MLX's own one-row canonicalization (one plan for 119 distinct key lengths, which is that canonicalization working). Three plan builds in the process. A non-speculative generation does not enter the new path at all.

The same trace answers the "is the trailing chunk of a chunked prefill affected" question this change inherited from #1799: at the default 2048-token chunk the trailing chunk is hundreds of rows, not tens, so it stays outside the bound. A prompt whose trailing chunk lands under 32 rows would enter, and none is measured here.

## Throughput against the #1817 state, which is what is on main

Same binary on every arm, idle host gated per run (the harness blocks while any compiler process is alive and records `load1` before each run; the self-hosted CI runner was building during parts of this session and the gate held the sweep until it went quiet). Widths interleaved round-robin with a rotating start, n = 3, one discarded warm-up. `fb-*` sets `MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0`, which is exactly main's dispatch (#1817's fallback claims the verify block); the unprefixed arms are the branch default (bucketed cuDNN). Data: `laguna_cli.jsonl`.

| width | bucketed, tok/s (min to max) | #1817 fallback, tok/s (min to max) | bucketed vs fallback | bucketed device sync ms/round | fallback device sync ms/round | bucketed draft host ms/round | fallback draft host ms/round |
|---|---|---|---|---|---|---|---|
| classic | 33.39 (33.14 to 33.63) | 34.13 (33.37 to 35.06) | 0.98x | | | | |
| 2 | 35.70 (34.77 to 37.16) | 35.26 (34.94 to 35.46) | 1.01x | 38.3 | 38.4 | 7.3 | 7.4 |
| 4 | 39.85 (38.74 to 40.54) | 39.61 (39.34 to 39.80) | 1.01x | 47.7 | 47.8 | 9.4 | 9.1 |
| 6 | 36.77 (36.14 to 37.25) | 38.12 (37.81 to 38.67) | 0.96x | 57.2 | 56.8 | 11.7 | 10.3 |
| 8 | 32.33 (32.22 to 32.47) | 33.31 (32.91 to 33.80) | 0.97x | 65.6 | 64.3 | 11.7 | 10.1 |
| 16 | 22.92 (22.65 to 23.29) | 24.43 (24.04 to 24.75) | 0.94x | 94.5 | 91.0 | 13.0 | 10.1 |

The classic arm is untouched either way (the 2% is host noise; a one-row call cannot reach the new path, and the debug trace confirms zero bucketed calls on a classic run). Both arms have removed the plan-build floor: the drafter host build is 7 to 13 ms in both, against 30 ms before #1817.

At this prompt length the two fixes are a wash at widths 2 and 4 (ranges overlap) and bucketing is 3 to 6% behind at 6, 8 and 16 (ranges disjoint). The cost is visible in the columns: the bucketed arm's drafter host build grows with the block width (7.3 to 13.0 ms against 7.4 to 10.1), which is the extra kernel enqueues the widening costs (a fill and a copy per widened mask over 45 attention calls, plus two more per drafter k and v), and its device sync is 1 to 3.5 ms higher at the wide blocks, which is cuDNN reading up to 255 padded key positions per call.

### Long context: measured but not clean, and not used

The prediction this branch is really chasing is #1799's: the ops fallback materializes a `[B, heads, q_len, k_len]` score matrix, so its cost grows with the key length while cuDNN's flash kernel does not, and somewhere past a few hundred keys bucketing should pull ahead. A sweep on the same pairing with a 2634-token prompt (widths 4 and 8, bucketing on and off) was started to test exactly that.

**It is recorded here and used for nothing.** Partway through it a second session outside this record's control loaded its own model on the same GPU, and the host's `NVRM: NV_ERR_NO_MEMORY` count went from 0 to 356 with 351 of those inside two minutes. The run was stopped, the GPU released, and the rows are kept only so the gap is visible: `laguna_long_cli.jsonl`, 10 rows, rounds 0 and 1 of five configurations and no round 2. The idle-host precondition was violated without the harness being able to see it (its gate watches for compiler processes, not for another session's inference), so no number from that file belongs in a comparison. The question it was asked stays open and is the first thing to re-run on a clean host.

Everything above it is clean by construction: the whole 152-token sweep finished at 12:44:29 and the foreign process did not exist until after 12:45:54 (its pid is above the sweep driver's, and the host was checked for foreign model processes before the sweep started and again before the identity runs). The `load1` recorded before every run of that sweep is 0.72 to 2.29, and the four runs with a CI job alive are inside their arm's range.

## Plan-cache growth under bucketing

Measured on the same block-4 run, from the trace's resident plan count (`plans=`, the LRU's `size()`):

| | exact-shape cuDNN | bucketed |
|---|---|---|
| plan builds per verify round | 3 (one per shape class) | 0 after the first round of each class |
| plan builds in a 25-round generation | 82 | 11 |
| resident plans at the end | 82 | 11 |

The bound is **shape classes x query widths x buckets crossed**, not rounds. A class builds one plan per distinct `(q_len, bucket)`, so a generation of `N` new keys crosses `ceil(N / 256)` buckets per class and a block width contributes at most two query lengths (the full block, and the short final one). Laguna at block 4: 3 classes x 2 query lengths x 1 bucket = 6 in the observed run, plus the prefill and probe shapes, measured 11. A generation ten times longer adds one plan per class per 256 new keys, so 100 rounds at block 4 (about 400 new keys) is 2 buckets, not 100 plan builds.

Against the `2 * MLX_CUDA_SDPA_CACHE_SIZE` lifetime-miss abort (4000 at mlxcel's CUDA default of 2000, per #1817): exact-shape cuDNN reaches it in about 1300 rounds and, at MLX's own default of 256, in about 170, which is the abort #1799 hit. Bucketed, the same budget covers roughly 340k new key positions per shape class. The abort stops being reachable by generation length; only genuine shape diversity (prompt lengths through prefill, which is untouched here) can still reach it.

Memory: one plan per entry as before, plus the transient buffers the widening allocates and frees inside each call. Those are a `[L, bucket]` mask plane per widened call (4 x 768 bf16 = 6 KiB on this pairing) and, on the copied arm only, a `[B, H, bucket, D]` k and v per drafter layer (about 0.5 MiB each at bucket 256), bounded by `MLXCEL_SDPA_PLAN_BUCKET_MAX_MB` = 64 per tensor. No preallocation and no growth in the KV cache itself: the unsliced arm addresses only the buffer the cache had already allocated.

## Blast radius

**Measured**: the Laguna DFlash pairing (the sweep and the identity above), and `qwen3-1.7b-4bit` as a dense head_dim-128 non-speculative control, where the trace shows the path is not entered at all.

**Shares the path without a number here**: every other CUDA model whose array-masked SDPA with 2 to 32 query rows over a longer key sequence reaches this file on Ampere or later. In tree that is the speculative verify of any head_dim-128-or-less pairing (Muse Glimmer DFlash, LFM2 and LFM2.5 DSpark, Inkling MTP, Qwen 3.5's own DFlash drafter, whose 5 layers are head_dim 128 even though its target's 256 never enters cuDNN), a short incremental prefill over a reused prefix-cache prefix, and the trailing chunk of a chunked prefill when that chunk lands under 32 rows. None re-probed here. Prefill at the default 2048-token chunk stays outside the bound, measured.

**Untouched**: the one-row decode step (its own upstream canonicalization, unchanged), the backward primitive, Metal, ROCm and CPU. The `metal,accelerate` gate is not runnable on this Linux/CUDA host and was not run; the patched file compiles only into the CUDA backend.

Attention sinks are **not** exercised: this checkpoint carries none, and no sinks-bearing model was measured. The code keeps sinks on the bucketed path and the capability flag that degrades a cuDNN refusal is kept per sinks setting so a refusal on a sinks graph cannot take bucketing away from the shapes that have none, but that arm is unmeasured.

## What is unresolved, and what the #1817 interaction should be

Two of this issue's questions are answered and one is not.

**Answered: the key can be made to hit, and the widening is exact.** Three plan builds per verify round become zero after warm-up, the abort stops being reachable by generation length, and the widened path returns byte-identical token ids to the exact-shape cuDNN path.

**Answered: at a short prompt it does not beat #1817.** On the 152-token prompt the two are a wash at widths 2 and 4 and bucketing is 3 to 6% behind at 6, 8 and 16. So the roughly 10 ms per round of extra GPU time #1799 attributed to the fallback at block 2 is not recovered here; what the widening costs (extra kernel enqueues, and cuDNN reading up to 255 padded key positions across 45 calls) is about what cuDNN's flash kernel saves at 156 to 360 keys.

**Not answered: whether it wins at a long context**, which is where #1799 predicted the fallback would lose because its score matrix is `[B, heads, q_len, k_len]` and cuDNN's flash is not. The sweep that would have decided it was stopped by the host condition above and its rows are unusable.

That makes the #1817 interaction a decision that this record cannot close on its own. On the evidence here the two mechanisms are alternatives with the same host-side effect and slightly different constant factors, and the one that is measured to be faster at the prompt length that was measured is #1817's. The branch's default keeps the two composed rather than making bucketing unconditional: `MLXCEL_SDPA_FALLBACK_MAX_QUERIES` is narrowed so it no longer claims a call whose key can be bucketed, and `MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES=0` hands every such call back to it without a rebuild. Whichever way the long-context arm lands, one environment variable selects it and no rebuild is needed to A/B them again.
