# Experimental `--mla-latent-quant` on five MLA checkpoints (M3 Ultra, 2026-11-20)

Issue #3179, follow-up to epic #3158. The epic shipped the affine int8 MLA latent on the families that passed its quality gate and kept GLM-4.7-Flash and the DSA families at fp16. `--mla-latent-quant <auto|fp16|int8|int4>` (also `MLXCEL_MLA_LATENT_QUANT`) lets an operator choose the latent precision explicitly, including a 4-bit latent on any MLA family and the int8 latent on the gated-off ones. This document records what the operator opts into. None of it gates the change: `auto` is the default and is unchanged.

**Outcome.** A 4-bit latent has 0 decided-position disagreements against the fp16 latent on Moonlight-16B-A3B, LongCat-Flash-Lite, Kimi-Linear-48B-A3B and DeepSeek-V2-Lite, behind 1024 and 4096 tokens. It fails the epic's near-tie bar on all four: top-1 flips at reference top-two gaps up to 0.84 (Moonlight), 1.13 (LongCat), 0.63 (Kimi Linear) and 0.86 (DeepSeek-V2-Lite) logits, against a bar of 0.11. On GLM-4.7-Flash it fails the decided gate too: 3 decided disagreements behind each length, with perplexity moving by -19.5% and -16.1%. So `int4` stays experimental everywhere and is a poor fit for GLM-4.7-Flash. Decode on the 4-bit latent always takes the composed path; it is 3% to 9% faster than the int8 latent where both are composed and 6% to 8% slower than int8 where int8 has the fused kernel (up to 2048 keys). The int8 arms, run through the new option, reproduce the epic's published numbers digit for digit, GLM-4.7-Flash's failing 1 / 193 included.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Sibling units shared the host and the GPU lock: load average 1.3 to 7.9 during the traces, 0.8 to 2.6 during the Moonlight decode sweep and 1.1 to 4.3 during the LongCat sweep (`uptime` at each run start, in the run logs).
- **Binary:** release build (`--features metal,accelerate`) of `062b003d` for the traces and the decode sweeps; the generate checks and `mlxcel inspect` rows are on the branch head.
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweeps ran under `scripts/with_indexers_paused.sh`, the three arms interleaved round by round inside one lock hold per model.
- **Arms:** `fp16` (default), `int8` and `int4`. The trace harness (`examples/logit_trace`) sets the storage mode itself, so its arms are `MLXCEL_TRACE_KV_CACHE_MODE=int8` with `MLXCEL_MLA_LATENT_QUANT=int8` or `int4`; the CLI arms are `--kv-cache-mode int8` and `--mla-latent-quant int4`.
- **Corpus and shapes:** as in the epic's documents: `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` at `3c0d74ec`, concatenated (167,657 bytes); `64 8 8 1024` (8 chunks of 64 traced positions behind a 1024-token context) and `64 4 8 4096`, with `MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, so every traced position is a one-token decode step.
- **Models:** `moonlight-16b-a3b-instruct-4bit` (`deepseek_v3`), `longcat-flash-lite-4bit` (`longcat_flash_ngram`), `kimi-linear-48b-a3b-instruct-4bit` (`kimi_linear`), `deepseek-v2-lite-chat-4bit-mlx` (`deepseek_v2`), `glm-4.7-flash-4bit` (`glm4_moe_lite`). All have `kv_lora_rank` 512 and `qk_rope_head_dim` 64.

## int4 against fp16: teacher-forced trace (`--decided 2`)

| model | shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter (int4) |
|---|---|---|---|---|---|---|---|---|
| Moonlight (`deepseek_v3`) | behind 1024 | **0 / 192** | 37 / 512 | 0.844 | 18 | 18.6816 / 19.1152 | +2.321% | decompressed=216 absorbed_composed=13824 |
| Moonlight (`deepseek_v3`) | behind 4096 | **0 / 58** | 24 / 256 | 0.766 | 13 | 22.9467 / 23.2242 | +1.209% | decompressed=108 absorbed_composed=6912 |
| LongCat-Flash-Lite (`longcat_flash_ngram`) | behind 1024 | **0 / 239** | 50 / 512 | 1.125 | 38 | 14.1036 / 13.8004 | -2.150% | decompressed=224 absorbed_composed=14336 |
| LongCat-Flash-Lite (`longcat_flash_ngram`) | behind 4096 | **0 / 121** | 20 / 256 | 1.062 | 18 | 9.9146 / 9.5284 | -3.895% | decompressed=112 absorbed_composed=7168 |
| Kimi-Linear-48B (`kimi_linear`) | behind 1024 | **0 / 201** | 18 / 512 | 0.625 | 9 | 12.4213 / 12.3638 | -0.463% | decompressed=56 absorbed_composed=3584 |
| Kimi-Linear-48B (`kimi_linear`) | behind 4096 | **0 / 52** | 17 / 256 | 0.312 | 6 | 24.5971 / 24.6546 | +0.234% | decompressed=28 absorbed_composed=1792 |
| DeepSeek-V2-Lite (`deepseek_v2`) | behind 1024 | **0 / 209** | 45 / 512 | 0.859 | 24 | 14.6613 / 14.6607 | -0.004% | decompressed=216 absorbed_composed=13824 |
| DeepSeek-V2-Lite (`deepseek_v2`) | behind 4096 | **0 / 105** | 14 / 256 | 0.859 | 10 | 12.9589 / 13.2174 | +1.995% | decompressed=108 absorbed_composed=6912 |
| GLM-4.7-Flash (`glm4_moe_lite`) | behind 1024 | **3 / 193** | 57 / 512 | 4.000 | 32 | 21.4881 / 17.2944 | -19.516% | decompressed=376 absorbed_composed=24064 |
| GLM-4.7-Flash (`glm4_moe_lite`) | behind 4096 | **3 / 95** | 38 / 256 | 4.000 | 28 | 53.6498 / 45.0188 | -16.088% | decompressed=188 absorbed_composed=12032 |

`decompressed` is the context pass. Every traced decode step reads the 4-bit latent on the composed path (`absorbed_composed`), behind 1024 as well as behind 4096: the fused Metal kernel, which the int8 arm takes behind 1024, reads 8-bit codes only and never runs for int4.

Logit delta on the reference's own choice (p50 / p90 / p99): Moonlight 0.16 / 0.47 / 1.22 behind 1024 and 0.17 / 0.52 / 1.17 behind 4096; LongCat 0.50 / 1.69 / 3.13 and 0.50 / 1.31 / 2.63; Kimi Linear 0.13 / 0.25 / 0.88 and 0.13 / 0.25 / 0.38; GLM-4.7-Flash 2.50 / 8.25 / 31.1 and 2.00 / 8.38 / 32.5. GLM's logits sit around 100 to 130 in bf16, where adjacent values are 0.5 apart, so its deltas are large in absolute terms on any arm; its decided count and perplexity shift are what set int4 apart from int8 there.

Reading it against the #3155 gate (zero decided disagreements, and every top-1 flip within 0.11 logits of the reference's top two):

- **Decided gate:** passes on Moonlight, LongCat, Kimi Linear and DeepSeek-V2-Lite at both lengths. Fails on GLM-4.7-Flash at both (3 and 3), where int8 already fails with 1 behind 1024.
- **Near-tie bar:** fails everywhere. Between 6 and 38 flips per trace lie above 0.11, the largest from 0.31 (Kimi Linear behind 4096) to 1.13 (LongCat behind 1024). The int8 latent has 0 to 9 such flips on the same traces, the largest 0.5.
- **DeepSeek-V2-Lite** repeats the #3155 measurement that kept 4-bit out of the default (8.8% and 5.5% top-1 flips, gap up to 0.86), now reached through the option: 45 / 512 = 8.8%, 14 / 256 = 5.5%, largest gap 0.859.

## int8 through the option: same numbers as the epic

The `int8` arm of each trace is the control. On the listed families `--mla-latent-quant int8` is the default int8 latent; on GLM-4.7-Flash it is the forced one the default gate withholds.

| model | shape | decided disagree | top-1 disagree | largest gap at a disagreement | flips above 0.11 | ppl delta | path counter (int8) | published |
|---|---|---|---|---|---|---|---|---|
| Moonlight | behind 1024 | 0 / 192 | 7 / 512 | 0.328 | 2 | +0.761% | absorbed_fused=13824 | same (#3160, after #3156) |
| Moonlight | behind 4096 | 0 / 58 | 5 / 256 | 0.039 | 0 | -0.035% | absorbed_composed=6912 | same |
| LongCat-Flash-Lite | behind 1024 | 0 / 239 | 15 / 512 | 0.375 | 9 | -0.727% | absorbed_fused=14336 | same (#3161) |
| LongCat-Flash-Lite | behind 4096 | 0 / 121 | 9 / 256 | 0.250 | 5 | -1.228% | absorbed_composed=7168 | same |
| Kimi-Linear-48B | behind 1024 | 0 / 201 | 12 / 512 | 0.500 | 5 | +0.295% | absorbed_fused=3584 | same (#3162) |
| Kimi-Linear-48B | behind 4096 | 0 / 52 | 13 / 256 | 0.312 | 5 | -0.398% | absorbed_composed=1792 | same |
| DeepSeek-V2-Lite | behind 1024 | 0 / 209 | 10 / 512 | 0.078 | 0 | -0.098% | absorbed_fused=13824 | not compared |
| DeepSeek-V2-Lite | behind 4096 | 0 / 105 | 3 / 256 | 0.109 | 0 | -0.634% | absorbed_composed=6912 | not compared |
| GLM-4.7-Flash (forced) | behind 1024 | **1 / 193** | 40 / 512 | 2.750 | 18 | -3.068% | absorbed_fused=24064 | same (#3161, the failing row) |
| GLM-4.7-Flash (forced) | behind 4096 | 0 / 95 | 25 / 256 | 1.000 | 15 | -0.430% | absorbed_composed=12032 | same |

"Same" means every count, gap and perplexity in the row equals the figure in the epic's results document for that checkpoint. The forced GLM-4.7-Flash rows were measured in #3161 on a build with `glm4_moe_lite` temporarily added to the allowlist; here the shipped binary reaches the same state through the option. The DeepSeek-V2-Lite int8 rows were not checked against #3155, whose int8 trace predates the fused kernel.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

| model | prompt | fp16 latent | int8 latent | int4 latent | int8 vs fp16 | int4 vs fp16 | int4 vs int8 | path (fp16 / int8 / int4) |
|---|---|---|---|---|---|---|---|---|
| LongCat-Flash-Lite | 512 (3 rounds) | 124.71 | 135.48 | 124.30 | +8.6% | -0.3% | -8.3% | composed / fused / composed |
| LongCat-Flash-Lite | 2048 (3 rounds) | 125.11 | 106.89 | 110.02 | -14.6% | -12.1% | +2.9% | composed / composed / composed |
| LongCat-Flash-Lite | 16384 (2 rounds) | 99.27 | 50.20 | 54.50 | -49.4% | -45.1% | +8.6% | composed / composed / composed |
| Moonlight | 512 (3 rounds) | 156.02 | 140.52 | 131.98 | -9.9% | -15.4% | -6.1% | fused / fused / composed |
| Moonlight | 2048 (3 rounds) | 139.81 | 121.26 | 123.61 | -13.3% | -11.6% | +1.9% | fused / composed / composed |
| Moonlight | 8064 (2 rounds, requested 16384) | 99.66 | 93.12 | 98.28 | -6.6% | -1.4% | +5.5% | fused / composed / composed |

Round-to-round spread is under 1.3% for every arm. Moonlight's context is 8192, so the bench clamps a 16384-token request to 8064 tokens; LongCat-Flash-Lite is the checkpoint with a real 16384 row. Prefill medians are within 1.7% across the three arms (LongCat 1996 to 2029 tok/s at 512, 1228 to 1229 at 16384), and peak MLX memory differs by under 0.4 GB at these lengths, where the weights dominate.

Two things decide the int4 column. Where the int8 latent is on the fused kernel (a window of at most 2048 keys, so the 512 rows), int4 has no kernel and runs composed, 6% to 8% behind int8. Where both are composed, int4 is ahead of int8 by 2% to 9%, more at the longer window: the same `quantized_matmul` reads half the packed bytes. The int8 column is within 0.7% of the epic's LongCat rows (135.53, 107.59, 50.39) and within 1.4% of its Moonlight rows after #3156 (142.45, 122.97); the int8 against fp16 ratios agree to within one point on both.

## Sizing (`mlxcel inspect --max-tokens 16384`)

| model | flags | KV cache | detail line |
|---|---|---|---|
| Moonlight | (none) | 486.0 MiB | MLA compressed latent (576 elems/token/layer) |
| Moonlight | `--kv-cache-mode int8` | 283.5 MiB | affine-quantized (672 bytes/token/layer) |
| Moonlight | `--mla-latent-quant int4` | 175.5 MiB | affine-quantized (416 bytes/token/layer) |
| Moonlight | `--mla-latent-quant fp16 --kv-cache-mode int8` | 486.0 MiB | MLA compressed latent (576 elems/token/layer) |
| LongCat-Flash-Lite | `--mla-latent-quant int4` | 182.0 MiB | affine-quantized (416 bytes/token/layer), two caches per layer |
| DeepSeek-V2-Lite | `--mla-latent-quant int4` | 175.5 MiB | affine-quantized (416 bytes/token/layer) |
| GLM-4.7-Flash | `--kv-cache-mode int8` | 846.0 MiB | MLA compressed latent (576 elems/token/layer): the default gate keeps fp16 |
| GLM-4.7-Flash | `--mla-latent-quant int8` | 493.5 MiB | affine-quantized (672 bytes/token/layer) |
| GLM-4.7-Flash | `--mla-latent-quant int4` | 305.5 MiB | affine-quantized (416 bytes/token/layer) |
| DeepSeek-V3.2 | (none) | 1.31 GiB | MLA compressed latent (704 elems/token/layer) |
| DeepSeek-V3.2 | `--mla-latent-quant int8` | 770.1 MiB | affine-quantized (808 bytes/token/layer) |
| DeepSeek-V3.2 | `--mla-latent-quant int4` | 526.1 MiB | affine-quantized (552 bytes/token/layer): 416 for the latent, 136 for the int8 indexer key |

`MLXCEL_MLA_LATENT_QUANT=int4` with no flag gives the Moonlight 175.5 MiB row, and adding `--mla-latent-quant int8` to it gives 283.5 MiB: the flag wins. `--mla-latent-quant int4 --kv-cache-mode int8` on `qwen3.5-4b-4bit` prints `--mla-latent-quant int4 is ignored for this model (model_type="qwen3_5"): it has no MLA latent cache` and sizes the per-token int8 cache it would build anyway. An unknown value fails with `invalid MLA latent quantization 'q4': expected one of auto, fp16, int8, int4`, from the flag and from the environment variable alike.

## Generate checks (`mlxcel generate`, greedy, `MLXCEL_TRACE_MLA_STATS=1`)

Prompt `Explain in two sentences why the sky is blue.`, branch head.

- **GLM-4.7-Flash, `--mla-latent-quant int8`** (200 tokens, `--show-reasoning`): one startup warning, `--mla-latent-quant int8 is experimental (model_type="glm4_moe_lite")` with the measured line `GLM-4.7-Flash with an int8 latent showed 1 decided-position disagreement against the fp16 latent behind 1024 tokens`; banner `KV cache mode: int8 (affine group-64 8-bit MLA latent, fp16 rope keys; --mla-latent-quant int8) (requested fp16; effective int8; applied to 47 of 47 layers)`; counters `decompressed=94 absorbed_fused=9353`, 86.8 tok/s. The output is coherent: it restates the two-sentence constraint, drafts `The sun emits light that contains all the colors of the rainbow. When this light hits the Earth's atmosphere, the gas molecules scatter the blue light in all directions, making the sky appear blue to our eyes.` and critiques it. The fp16 run of the same prompt (`absorbed_composed=9353`, 82.5 tok/s) reasons the same way with different wording.
- **GLM-4.7-Flash, `--mla-latent-quant int4`**: the warning carries four measured lines (GLM int8, GLM int4, the DeepSeek-V2-Lite 4-bit result, the Moonlight and LongCat 4-bit result); banner `int8 (affine group-64 4-bit MLA latent, fp16 rope keys; --mla-latent-quant int4)`; counters `decompressed=94 absorbed_composed=9353`, 81.3 tok/s. Coherent over these 200 tokens (`The sky appears blue because of a phenomenon called Rayleigh scattering. This occurs when sunlight hits the atmosphere and scatters in all directions, with shorter blue wavelengths scattering more than longer red wavelengths.`). A 200-token greedy run neither confirms nor refutes the trace's 3 decided disagreements; the trace is the measurement.
- **GLM-4.7-Flash, `--kv-cache-mode int8 --mla-latent-quant fp16`**: one warning, `--kv-cache-mode int8 and --mla-latent-quant fp16 disagree on the MLA latent precision (model_type="glm4_moe_lite"); the explicit --mla-latent-quant wins and the latent is stored as fp16`; banner `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)`; `absorbed_composed` only.
- **Moonlight, `--mla-latent-quant int4`** (64 tokens): `absorbed_composed=1485`, no fused step, 125.8 tok/s; answer `The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules and tiny particles. Blue light, which has a shorter wavelength, is scattered in all directions more than other colors, making the sky appear blue.`
- **LongCat-Flash-Lite, `--mla-latent-quant int4`** (64 tokens): `decompressed=56 absorbed_composed=1764`, 121.0 tok/s; all 64 tokens went to the reasoning channel, so only the path and the banner were checked there.

Each warning appeared once per run, on stderr.

### DeepSeek-V3.2 (`deepseek_v32`, 352 GB resident, 61 layers)

`mlxcel generate` feeds this checkpoint the prompt as raw text, so all three arms continue it as a document rather than answer a chat turn. 200 tokens each, 19.3 to 19.4 tok/s.

- **fp16:** `absorbed_composed=12139`. `(2 marks) The sky is blue because of the way the atmosphere scatters sunlight. The short wavelengths of blue light are scattered more than the other colours, so we see blue light coming from all directions in the sky.`, then a repeating `Figure 1.n` caption.
- **`--mla-latent-quant int8`:** warning with the measured line `DeepSeek-V3.2 with an int8 key slot passed the logit trace, but the DSA indexer's top-k overlap with fp16 fell below 0.99 on 49 of 61 layers behind 4096 tokens`; banner `int8 (affine group-64 8-bit MLA latent, fp16 rope keys; --mla-latent-quant int8) (requested fp16; effective int8; applied to 61 of 61 layers)`; `decompressed=122 absorbed_fused=12139`. Coherent: it continues into a dataset record whose answer reads `Rayleigh scattering is the scattering of light by particles in the atmosphere that are much smaller than the wavelength of the light. This scattering is more effective for shorter wavelengths, such as blue light, which is why the sky appears blue.`
- **`--mla-latent-quant int4`:** the warning adds the two 4-bit lines and `The DSA indexer key stored beside the latent stays int8.`; banner `int8 (affine group-64 4-bit MLA latent, ...)`; `decompressed=122 absorbed_composed=12139`. Coherent: `(2 marks) The sky is blue because the atmosphere scatters blue light more than red light. This is known as Rayleigh scattering.`, then the same question and answer repeated, as the fp16 arm repeats its caption.
- **`mlxcel-bench-decode --prompt-tokens 4096 --mla-latent-quant int4`** (64 tokens, 8 warmup): the prompt is longer than `index_topk` (2048), so every decode step scores the cached indexer keys at 8 bits, takes a top-k and gathers packed rows from the split slot. It ran to completion on the composed path: `decompressed=244 absorbed_composed=4270`, 14.31 tok/s decode, 55.9 tok/s prefill, 384.7 GB peak. One round, no fp16 or int8 row beside it, so this is a functional check and not a throughput comparison.

## Not measured

- **DeepSeek-V3.2 traces and top-k overlap under the option.** Neither the teacher-forced trace nor the indexer top-k overlap run was repeated for the forced int8 slot, whose numbers are in `mla-quantized-latent-dsa-m3ultra-2026-11-19.md`, and neither was taken for the int4 split slot. The split slot's correctness rests on the tiny-model tests (`mla::latent_int4_tests`, `models::deepseek_v32::latent_tests::latent_option`) and the functional run above; its quality on the real checkpoint is unmeasured.
- **GLM-5 (`glm_moe_dsa`), LongCat-Flash (`longcat_flash`), Kimi K3, Kimi-VL and the Youtu checkpoints.** Not on this host, or not run. They share the code paths measured here and are covered by the tiny-model tests.
- **Decode throughput on GLM-4.7-Flash, Kimi Linear and DeepSeek-V2-Lite under int4.** Only LongCat-Flash-Lite and Moonlight were swept.
- **CUDA (GB10) and ROCm (gfx1151).** Neither host is reachable from this one. The int4 path is the composed `quantized_matmul` decode those backends already run for int8, at 4 bits; no trace or throughput row exists for them.
- **Server prompt-cache adoption on a real checkpoint.** The precision mismatch check is covered by `dense_detached_latent_precision_mismatch_is_declined`, not by a serve run.

## Commands

```
M=models/longcat-flash-lite-4bit   # or one of the other four
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv     # 64 4 8 4096 for the 4k rows
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_MLA_LATENT_QUANT=int4 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > int4-1k.tsv     # MLXCEL_MLA_LATENT_QUANT=int8 for the control arm
python3 scripts/compare_logit_traces.py fp16-1k.tsv int4-1k.tsv --decided 2
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; add --kv-cache-mode int8 or --mla-latent-quant int4
mlxcel inspect -m $M --max-tokens 16384 --mla-latent-quant int4
mlxcel generate -m models/glm-4.7-flash-4bit --mla-latent-quant int8 -p "Explain in two sentences why the sky is blue." -n 64 --temp 0
```

Rows: [logit_trace.csv](mla-latent-quant-option-m3ultra-2026-11-20/logit_trace.csv) and [decode.csv](mla-latent-quant-option-m3ultra-2026-11-20/decode.csv).
