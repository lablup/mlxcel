# Affine int8 MLA latent on Kimi Linear and Kimi K3 (M3 Ultra, 2026-11-19)

Issue #3162, epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent), and #3160 moved the `DeepSeekV3Attention` families onto it. Kimi Linear and Kimi K3 are hybrid KDA + MLA stacks whose caches are model-owned and were always built fp16, so every quantized request on them was substituted with fp16. This change gives both models the per-layer mode table, builds their MLA caches in the table's mode, and writes the latent through `mla::MlaLatentCache` in the single-sequence forward and the shared batched decode helper. Only int8 exists; 4-bit failed the #3155 gate and is not a mode.

**Gate (decided for epic #3158):** zero decided-position disagreements (`--decided 2`) against the fp16 latent behind 1024 and 4096 tokens. A decided disagreement keeps the family on fp16. Top-1 flips outside the 0.11-logit near-tie bar are recorded with their gap, next to the fp16 kernel noise floor and the perplexity delta, and do not block.

**Outcome:** the gate passes. Kimi-Linear-48B-A3B has zero decided-position disagreements behind 1024 (0 / 201) and 4096 (0 / 52), and every top-1 disagreement is at the reference's rank 2 to 5. Five flips behind each prefix lie outside the 0.11-logit near-tie bar (largest 0.500 and 0.312); the int8 latent's own kernel choice (fused against composed) flips 16 positions with seven outside the bar (largest 0.438), so these flips are at the scale of the int8 decode's kernel noise. Perplexity moves +0.29% behind 1024 and -0.40% behind 4096. Decode is 1.5% faster than fp16 behind 512 tokens (fused int8 kernel against Kimi's composed fp16 decode), 4.7% slower behind 2048 and 23.3% slower behind 16384, both on the composed int8 path. Kimi K3 is not measured.

The first sweep found a bug that this change fixes: a bf16 checkpoint promoted to float32 over a dequantized window and decoded 61% slower. See [bf16 promotion](#bf16-promotion-found-and-fixed-during-measurement).

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal.
- **Model:** `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` (`model_type` `kimi_linear`, 27 layers of which 7 are MLA (`full_attn_layers` 4, 8, 12, 16, 20, 24, 27) and 20 KDA, 32 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64).
- **Binary:** release build (`--features metal,accelerate`) of this branch on top of `3f5679a4`, with the bf16 fix below.
- **Load:** `uptime` load average 1.5 to 3.0 during the traces and 0.7 to 4.5 during the decode sweep (sibling builds and traces queued on the same lock).
- **Isolation:** every GPU run holds the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep runs under `scripts/with_indexers_paused.sh`, arms interleaved round by round inside one lock hold.
- **Arms:** `fp16 latent` (default) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness).

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Reference is the fp16 latent of the same checkpoint. The corpus is the #3155 corpus: `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` from `3c0d74ec` concatenated. Width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`; shapes `64 8 8 1024` (512 positions behind 1024) and `64 4 8 4096` (256 positions behind 4096).

| prefix | candidate | top-1 disagree | decided disagree | largest gap at a disagreement | flips over 0.11 | worst rank | ppl fp16 | ppl candidate | candidate path counter |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | int8 latent | 12 / 512 | 0 / 201 | 0.500 | 5 | 3 | 12.4213 | 12.4579 (+0.29%) | absorbed_fused=3584 |
| 4096 | int8 latent | 13 / 256 | 0 / 52 | 0.312 | 5 | 5 | 24.5971 | 24.4992 (-0.40%) | absorbed_composed=1792 |
| 1024 | int8 latent, `MLXCEL_MLA_FUSED_DECODE=0` | 12 / 512 | 0 / 201 | 0.250 | 5 | 3 | 12.4213 | 12.4115 (-0.08%) | absorbed_composed=3584 |

Flips over 0.11 with their reference top-two gap: behind 1024, 0.500, 0.375, 0.250, 0.125, 0.125, each at the reference's rank 2; behind 4096, 0.312, 0.250, 0.250, 0.125, 0.125, one at rank 3 and the rest at rank 2. Logits are bf16 on this checkpoint, so gaps come in steps of 1/16 to 1/8.

**Noise floor.** Kimi's single-sequence fp16 decode is a hand-written absorbed matmul that never takes the fused kernel, so its kernel choice has no noise to measure: `MLXCEL_MLA_FUSED_DECODE=0` and the batched-rows arm are byte-identical to the fp16 reference (0 / 512, every logit equal, path counter `absorbed_composed=3584` in all three). The kernel choice that exists is on the int8 side: the int8 fused arm against the int8 composed arm flips 16 / 512 positions, 0 decided, seven over 0.11 (largest 0.438). The int8 against fp16 flips (12, five over the bar) are within that.

**Path counters.** Every int8 decode step behind 1024 records `absorbed_fused` (3584 = 512 positions x 7 MLA layers, `absorbed_composed=0`); behind 4096 the window exceeds the fused kernel's 2048 keys and every step records `absorbed_composed`. `decompressed=56` / `28` is the context pass (one multi-token forward per chunk per MLA layer). No decode step is decompressed.

Rows: [logit_trace.csv](mla-quantized-latent-kimi-m3ultra-2026-11-19/logit_trace.csv).

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Arms interleaved round by round inside one lock hold, under `scripts/with_indexers_paused.sh`.

| prompt | fp16 latent | int8 latent | int8 vs fp16 | int8 path counter | measured-phase peak fp16 / int8 |
|---|---|---|---|---|---|
| 512 (3 rounds) | 127.49 | 129.34 | +1.5% | absorbed_fused=994 | 28.53 / 28.53 GB |
| 2048 (3 rounds) | 127.43 | 121.43 | -4.7% | absorbed_composed=994 | 28.88 / 28.86 GB |
| 16384 (2 rounds) | 115.94 | 88.97 | -23.3% | absorbed_composed=994 | 32.98 / 32.81 GB |

- Round-to-round spread is at most 1.4% for every arm. Prefill is within 0.6% between arms (2489 to 2516 tok/s at 512, 2981 to 2989 at 2048, 2543 to 2548 at 16384).
- The fp16 arm is `absorbed_composed` at every length (Kimi's hand-written decode), so at 512 the int8 arm is the fused kernel against a composed fp16 decode and comes out ahead. From 2048 keys the int8 arm is composed too and pays the `quantized_matmul` cost, as Kimi-VL did on the #3160 sweep (-35.8% at 16384). Only 7 of 27 layers are MLA here, which is why the loss is smaller.
- `decompressed=14` is the prefill and warmup forward over the 7 MLA layers. Every decode step is absorbed.

Rows: [decode.csv](mla-quantized-latent-kimi-m3ultra-2026-11-19/decode.csv) (`build=fixed`; the `before-dtype-fix` rows are the first sweep).

## bf16 promotion, found and fixed during measurement

The first sweep, on the branch as opened, decoded int8 at 50.09 / 48.65 / 42.53 tok/s against 127.58 / 126.04 / 116.00 for fp16 (-61% to -63%), with prefill 14% to 21% slower and MLX's buffer cache growing by 7.9 GB during the measured pass. The checkpoint is bf16. `LatentWindow::into_dense` returns the dequantized latent in FP16 (its `k_pe` is stored FP16), and MLX promotes a bf16 query against an FP16 window to float32. The multi-token context pass therefore returned float32 from each MLA layer into the residual stream; the KDA layers after it carried that float32 into their recurrent state, and every later decode step, MoE included, ran in float32. The decode attention itself was not at fault: `quantized_absorbed_attention` already returns the query dtype.

The fix casts the dense window back to the activation dtype (`mla_batched::dense_latent_in`) in both Kimi forwards. `models::kimi_linear::latent_tests::a_bf16_layer_keeps_bf16_over_an_int8_latent` fails without it (prefill output float32) and passes with it. The first sweep's trace also passed the gate (0 decided at both prefixes) but measured a float32 decode against a bf16 reference, so the tables above are from the re-run. The `DeepSeekV3Attention` families and DeepSeek V2 share `into_dense` and so may promote a multi-token step on a bf16 checkpoint the same way, but they have no recurrent state to carry it into decode, and their #3160 sweep showed prefill and peak unchanged between arms. This change keeps to the Kimi forwards; checking those families' multi-token step on a bf16 checkpoint is left open.

## Cache size and sizing

| layout | bytes per token per MLA layer | 7 MLA layers at 16384 tokens |
|---|---|---|
| fp16 latent | 1152 | 126.0 MiB |
| int8 latent | 672 | 73.5 MiB |

Before this change `kv_arch` charged the latent on all 27 layers of Kimi-Linear-48B (486.0 MiB at 16384 tokens for fp16); it now reads `linear_attn_config.kda_layers` and charges the 7 MLA layers. The runtime side is pinned by `models::kimi_linear::latent_tests::runtime_bytes_match_the_inspect_rate_on_the_mla_layers`: a rank-512, rope-64 int8 cache allocates 672 bytes per reserved row, which is the `kv_arch` marginal rate of a one-MLA-layer config.

## Not measured

- **Kimi K3:** the only MLX export on the Hub is `mlx-community/Kimi-K3-mlx-reap160-2bit` (181 GB, expert-pruned, 2-bit weights). A pruned 2-bit model would mix weight-quantization and pruning error into a cache-mode comparison, so it was not downloaded. Kimi K3 reuses Kimi Linear's absorbed MLA and is covered by `models::kimi_k3::latent_tests` (int8 table, masked and unmasked decode against the fp16 latent, batched against single-row decode).
- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The path is the #3155 composed decode, which has no Metal-only call, but no trace or throughput row exists for those backends.
