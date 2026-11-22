# Affine int8 MLA latent on the DSA families: measured, not enabled (M3 Ultra, 2026-11-19)

Issue #3163, Phase 2 of epic #3158. `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa` cache `[kv_latent (kv_lora_rank) | DSA indexer key (index_head_dim)]` in one slot, 640 wide at the defaults, with `k_pe` beside it. #3163 makes an `int8` cache able to hold that slot as the #3155 affine group-64 triple (808 bytes per token per layer against 1408 for fp16): the indexer dequantizes only its own two groups, the latent stays packed, sparse decode gathers the packed rows, and the attention runs `quantized_absorbed_attention`.

**Outcome: the three model types are not added to `QUANTIZED_LATENT_FAMILIES`.** On DeepSeek-V3.2-4bit the teacher-forced trace gate passes (zero decided disagreements behind 1024 and 4096) and the top-k overlap gate does not: behind 4096 the mean overlap of the indexer's selection between the int8 and the fp16 cache is under 0.99 on 49 of 61 layers (0.966 on the worst), and behind 16384 on 55 of 61 (0.744 on the worst). A quantized request on these families therefore still resolves to fp16. The code path, the sizing fix and the measurement tooling ship; the int8 arm below was run through the trace harness, which sets the cache mode directly, and through a measurement build with the three types listed.

Two things about this checkpoint qualify the numbers and are stated where they apply: fp16's own kernel choice already moves the selection below 0.99 on 29 layers behind 4096, and behind 16384 the fp16 reference itself is degenerate (perplexity 2080), so the 16384 row does not isolate the codec.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Sibling builds shared the host: load average 0.7 to 5.9 across the runs (`uptime` at each start and end, in the run logs).
- **Model:** `mlx-community/DeepSeek-V3.2-4bit` (`model_type` `deepseek_v32`, 671.9B parameters, 378.09 GB loaded). 61 layers, 128 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `index_head_dim` 128, `index_n_heads` 64, `index_topk` 2048, 163840-token context. One model instance at a time, every run under the shared GPU lock (`lockf -k .mlxcel-gpu.lock`).
- **Binary:** release build (`--features metal,accelerate`) of this branch on `3f5679a4`. The decode sweep and `mlxcel generate` / `inspect` rows used a measurement build of the same tree with the three DSA types added to `QUANTIZED_LATENT_FAMILIES`, since the shipped resolver substitutes fp16.
- **Arms:** `fp16 latent` (default) and `int8 latent` (`MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness, `--kv-cache-mode int8` on the measurement build).
- **Loader and MoE fixes this needed.** The checkpoint did not run on `3f5679a4`: the loader looked for per-expert `mlp.experts.N.*` keys where the conversion ships the stacked `mlp.switch_mlp.*` plane, the routed-expert output kept a singleton axis that the weighted sum could not broadcast, and grouped routing (`n_group` 8, `topk_group` 4) was skipped because the router scores are rank 3. All three are fixed in this change, for both arms; with them a short `mlxcel generate` prompt completes with coherent, on-topic text in both modes.

## Top-k overlap gate (`MLXCEL_TRACE_DSA_TOPK`, `scripts/compare_dsa_topk.py`)

Each decode step of each layer selects `index_topk` = 2048 cached positions. Both arms are teacher-forced over the same tokens, and the overlap of a step is the share of the fp16 selection that the other arm also selected. The gate is a mean of at least 0.99 on every layer.

| context | comparison | steps per layer | layers at or above 0.99 | mean of layer means | worst layer | lowest single step |
|---|---|---|---|---|---|---|
| 4096 | int8 vs fp16, single-sequence forward | 256 | 12 / 61 | 0.9793 | 0.9662 (layer 33) | 0.917 |
| 4096 | int8 vs fp16, two-row batched decode | 512 | 12 / 61 | 0.9794 | 0.9659 (layer 33) | 0.821 |
| 4096 | **fp16 fused vs fp16 composed**, two-row batched decode (noise floor) | 512 | 32 / 61 | 0.9911 | 0.9854 (layer 33) | 0.891 |
| 16384 | int8 vs fp16, single-sequence forward | 64 | 6 / 61 | 0.9021 | 0.7438 (layer 50) | 0.521 |

Mean overlap by layer band:

| layers | 4096, int8 vs fp16 | 4096, fp16 fused vs composed | 16384, int8 vs fp16 |
|---|---|---|---|
| 0 to 5 | 0.996 to 0.998 | 0.997 to 1.000 | 0.991 to 0.997 |
| 6 to 15 | 0.988 to 0.996 | 0.992 to 0.997 | 0.929 to 0.990 |
| 16 to 30 | 0.971 to 0.989 | 0.987 to 0.994 | 0.811 to 0.970 |
| 31 to 45 | 0.966 to 0.979 | 0.985 to 0.991 | 0.767 to 0.950 |
| 46 to 60 | 0.966 to 0.979 | 0.986 to 0.992 | 0.744 to 0.944 |

- **The gate is missed at both contexts.** Per-layer rows are in [topk_overlap.csv](mla-quantized-latent-dsa-m3ultra-2026-11-19/topk_overlap.csv).
- **Layer 0 isolates the 8-bit indexer keys.** Its input is the token embedding, identical in both arms, so its 0.998 (4096) and 0.997 (16384) is what quantizing the indexer keys alone costs. The decline with depth is the two arms' hidden states drifting apart, which moves a selection whose boundary is dense: at 4096 half of the cached positions are selected.
- **fp16 against itself already misses 0.99 on 29 layers.** The fused and the composed fp16 kernels differ only in summation order, and their selections overlap 0.985 to 0.992 on layers 24 and deeper. The int8 arm's shortfall from 1.0 is two to three times that floor on layers 16 and deeper. A 0.99 bar on every layer is tighter than this checkpoint's own kernel noise; whether the bar or the mode should change is a decision for the issue, not settled here.
- **16384 is measured on a degenerate reference** (see the trace section): the figure is recorded, and it does not separate the codec from whatever is wrong beyond 4096 tokens of context.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Reference is the fp16 latent of the same checkpoint. The corpus is the #3155 corpus (`docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` from `3c0d74ec`, 45981 tokens with this tokenizer). Width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1` (the single-sequence forward `mlxcel generate` runs), and `MLXCEL_TRACE_BATCH_ROWS=2` for the batched decode entry `mlxcel serve` runs.

| entry point | shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | int8 path counter |
|---|---|---|---|---|---|---|---|---|
| single-sequence | behind 1024 | 0 / 216 | 20 / 512 | 0.750 | 14 | 8.7577 / 8.7713 | +0.156% | decompressed=488 absorbed_fused=31232 |
| single-sequence | behind 4096 | 0 / 126 | 18 / 256 | 0.500 | 14 | 7.8962 / 7.7047 | -2.425% | decompressed=244 absorbed_fused=15616 |
| single-sequence | behind 16384 | 0 / 13 | 23 / 64 | 1.875 | 22 | 2079.72 / 1573.39 | -24.3% | decompressed=488 absorbed_fused=3904 |
| two-row batched | behind 1024 | 0 / 220 | 25 / 512 | 0.625 | 21 | 8.7718 / 8.7454 | -0.300% | decompressed=976 absorbed_fused=62464 |
| two-row batched | behind 4096 | 0 / 127 | 17 / 256 | 0.500 | 9 | 7.9841 / 7.7606 | -2.799% | decompressed=488 absorbed_fused=31232 |

The fp16 fused-versus-composed noise floor on the same checkpoint (`MLXCEL_MLA_FUSED_DECODE=0` as reference, the default fused kernel as candidate, two-row batched decode):

| shape | decided disagree | top-1 disagree | largest gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta |
|---|---|---|---|---|---|---|
| behind 1024 | 0 / 218 | 19 / 512 | 0.875 | 10 | 8.7193 / 8.7718 | +0.602% |
| behind 4096 | 0 / 126 | 15 / 256 | 0.250 | 11 | 7.8856 / 7.9841 | +1.249% |

- **The hard trace gate passes:** no decided-position disagreement behind 1024 or 4096, on either entry point.
- **Flips above 0.11, with their gaps.** Single-sequence behind 1024: 0.750, 0.625, six at 0.250 and six at 0.125. Behind 4096: 0.500, seven at 0.250 and six at 0.125. Two-row behind 1024: 0.625, two at 0.500, seven at 0.250 and eleven at 0.125; behind 4096: 0.500, 0.375, three at 0.250 and four at 0.125. The full lists are in [logit_trace.csv](mla-quantized-latent-dsa-m3ultra-2026-11-19/logit_trace.csv).
- **The 0.11 bar is below this model's logit resolution.** Its fp16 logits are large enough that adjacent values are 0.125 apart, so every nonzero gap is at least 0.125 and counts as above the bar. The noise-floor rows show the scale: fp16's own kernel choice flips 19 / 512 positions behind 1024 with a largest gap of 0.875, and 15 / 256 behind 4096. The int8 arm's flips (20 and 18 single-sequence, 25 and 17 batched; largest 0.750) are the same size as that floor, and its perplexity moves less than the kernel choice does behind 1024.
- Every flip behind 1024 and 4096 left the reference's choice inside the candidate's top six (rank 2 for 70 to 84% of them). Logit delta on the reference's choice: p50 0.125, p99 2.25 (behind 1024) and 1.0 (behind 4096).
- **The path counter proves the path:** every traced int8 decode step records `absorbed_fused`. Behind 1024 the window is under 2048 keys; beyond `index_topk` the gathered window is exactly 2048 keys, which is the quantized kernel's limit, so sparse int8 decode stays on the fused kernel at any context. No int8 decode step is `decompressed`; the `decompressed` count is the context pass, one multi-token forward per chunk per layer, which runs over the dequantized window.
- **fp16 on this family decodes composed in the single-sequence forward** (`absorbed_composed=31232`, `absorbed_fused=0`) and in a one-row batch, which routes to the same forward: there `MLXCEL_MLA_FUSED_DECODE=0` changes nothing and the two fp16 traces carry identical logits at all 512 positions. The fused fp16 kernel is reached from two rows up, which is why the noise floor is measured there.
- **Behind 16384 the fp16 reference is not usable.** Its perplexity is 2080 on text where the same model reads 7.5 to 13.5 behind 4096, and the int8 arm's is 1573. Only 13 of 64 positions are decided. The context there was fed in 2048-token forwards (`MLXCEL_TRACE_PREFILL_CHUNK=2048`), as the generator prefills, because one 16384-token forward on a 128-head model scores 34 billion pairs per layer. A diagnostic pair on the fp16 arm separates the two candidates: with the same 2048-token forwards, perplexity over 64 positions is 7.32 behind 4096 tokens (7.51 with one forward over the same positions) and 1403 behind 8192. Chunking the context is therefore not the cause: on this implementation and checkpoint the fp16 model stops predicting somewhere between 4096 and 8192 tokens of context. That is independent of the latent's storage, could not have been seen before this change (the checkpoint did not load), and is not addressed here.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Measurement build (`--kv-cache-mode int8` kept), under `scripts/with_indexers_paused.sh`, the two arms interleaved round by round inside one lock hold per prompt length.

| prompt | fp16 latent | int8 latent | int8 vs fp16 | prefill fp16 / int8 (tok/s) | fp16 path counter | int8 path counter |
|---|---|---|---|---|---|---|
| 512 (3 rounds) | 20.76 | 19.75 | -4.9% | 74.2 / 63.5 (-14.5%) | absorbed_composed=8662 | absorbed_fused=8662 |
| 2048 (3 rounds) | 17.54 | 15.44 | -12.0% | 74.5 / 64.7 (-13.2%) | absorbed_composed=8662 | absorbed_fused=8662 |
| 16384 (2 rounds) | 15.68 | 13.63 | -13.1% | 62.2 / 50.0 (-19.6%) | absorbed_composed=8662 | absorbed_fused=8662 |

- Round-to-round spread is under 2.4% at 512, 2.9% at 2048 and 0.1% at 16384 for every arm. Load average at the start of a run was 3.3 to 5.9 during the 512 sweep, 1.8 to 5.3 during 2048 and 0.6 to 2.5 during 16384.
- **Decode is slower, not faster.** The fp16 arm on this family runs composed SDPA over at most 2048 gathered keys; the int8 arm runs the fused quantized kernel over the same keys and adds, per layer per step, the dequantize of the indexer columns over the whole window and four packed gathers instead of two dense ones. The gap is smallest at 512, where nothing is gathered.
- **Prefill is 13 to 20% slower.** Each multi-token forward quantizes its fresh rows and dequantizes the whole 640-wide window in every layer before the fp16 attention runs.
- The 16384-token measured-phase peak is 393.2 GB for int8 against 395.9 GB for fp16; the cache itself is 0.77 GiB against 1.31 GiB beside 378 GB of weights.
- The 16384 row times a context at which this checkpoint's output is degenerate in both arms (trace section), so it is a kernel-cost figure only.

## Cache size and sizing

| layout | bytes per token per layer | 61 layers at 16384 tokens |
|---|---|---|
| fp16 slot, as sized before this change (512 + 64 elements) | 1152 | 1.07 GiB |
| fp16 slot, as stored (512 + 128 + 64 elements) | 1408 | 1.31 GiB |
| int8 slot (640 packed bytes, 10 groups of fp16 scale and bias, fp16 `k_pe`) | 808 | 770.1 MiB |

- **Sizing fix, fp16 included.** `kv_arch` counted `kv_lora_rank + qk_rope_head_dim` for these families and left out the indexer key that shares the latent's slot, so `mlxcel inspect`, the preflights and the prompt-cache budget under-counted the cache by 256 bytes per token per layer. `mlxcel inspect --max-tokens 16384` now reports 1,407,188,992 bytes (`704 elems/token/layer`), which is 1408 x 61 x 16384.
- On the measurement build `mlxcel inspect --max-tokens 16384 --kv-cache-mode int8` reports `MLA compressed latent, affine-quantized (808 bytes/token/layer, shared across heads)` and 807,534,592 bytes. On the shipped build the request is sized as fp16, which is what the run stores.
- **Runtime bytes.** The tiny-model test pins a quantized DSA cache's `bytes_per_reserved_token` to `quantized_latent_bytes_per_token(stored width, rope, 8, 64)`, the function the estimate calls. Runtime `nbytes` were not read on the 378 GB checkpoint.

## Integration checks

- **`mlxcel generate`** (`MLXCEL_TRACE_MLA_STATS=1`, short prompt, 64 tokens). fp16: `decompressed=122 absorbed_composed=3843`, 19.67 tok/s. Measurement build with `--kv-cache-mode int8`: banner `int8 (affine group-64 MLA latent, fp16 rope keys) (applied to 61 of 61 layers)`, no substitution warning, `decompressed=122 absorbed_fused=3843`, 17.93 tok/s. Both continue the prompt with coherent text; the two continuations differ after the first tokens.
- **Shipped resolver:** `int8` and every Turbo mode on `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa` resolve to fp16 with a warning, pinned by `mla::latent_keys::tests`, `cli::turbo_args` and `execution::kv_arch` tests.
- **Stored width that cannot be grouped:** a config whose `kv_lora_rank` or `kv_lora_rank + index_head_dim` is not a multiple of 64 keeps fp16 from startup with a warning naming the width, and the latent view declines such a cache at runtime.

## Not measured

- **GLM-5 (`glm_moe_dsa`):** the checkpoint is larger than DeepSeek-V3.2 and was not downloaded on this host. It wraps `DeepSeekV32Model` wholesale (its indexer RoPE is interleaved, the only difference in this path), so it inherits this measurement and this decision.
- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one, and neither can hold a 378 GB model.
- **`mlxcel serve` end to end** and the prompt-cache store on this checkpoint: not run. The batched decode entry is covered by the two-row traces above.
- **fp16 against a pre-change binary:** the checkpoint does not load on `3f5679a4`, so there is no pre-change trace to compare with. The fp16 path runs the same operations as before; the tiny-model tests hold a one-row batch to exact equality with the single-sequence forward.

## Commands

```
M=/Users/devops/Development/mlxcel/models/deepseek-v3.2-4bit
E="MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1"
env $E logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv
env $E MLXCEL_TRACE_KV_CACHE_MODE=int8 logit_trace $M corpus.txt 64 8 8 1024 > int8-1k.tsv
env $E MLXCEL_TRACE_DSA_TOPK=fp16-4k.topk logit_trace $M corpus.txt 64 4 8 4096 > fp16-4k.tsv     # and the int8 arm
env $E MLXCEL_TRACE_PREFILL_CHUNK=2048 MLXCEL_TRACE_DSA_TOPK=fp16-16k.topk logit_trace $M corpus.txt 64 1 8 16384 > fp16-16k.tsv
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=2 MLXCEL_MLA_FUSED_DECODE=0 logit_trace $M corpus.txt 64 4 8 4096 > fp16-composed-4k.tsv
python3 scripts/compare_logit_traces.py fp16-1k.tsv int8-1k.tsv --decided 2
python3 scripts/compare_dsa_topk.py fp16-4k.topk int8-4k.topk
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; add --kv-cache-mode int8
mlxcel inspect -m $M --max-tokens 16384
```

Rows: [logit_trace.csv](mla-quantized-latent-dsa-m3ultra-2026-11-19/logit_trace.csv), [topk_overlap.csv](mla-quantized-latent-dsa-m3ultra-2026-11-19/topk_overlap.csv) and [decode.csv](mla-quantized-latent-dsa-m3ultra-2026-11-19/decode.csv).
