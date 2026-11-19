# Affine int8 MLA latent on Kimi Linear and Kimi K3 (M3 Ultra, 2026-11-19)

Issue #3162, epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent), and #3160 moved the `DeepSeekV3Attention` families onto it. Kimi Linear and Kimi K3 are hybrid KDA + MLA stacks whose caches are model-owned and were always built fp16, so every quantized request on them was substituted with fp16. This change gives both models the per-layer mode table, builds their MLA caches in the table's mode, and writes the latent through `mla::MlaLatentCache` in the single-sequence forward and the shared batched decode helper. Only int8 exists; 4-bit failed the #3155 gate and is not a mode.

**Gate (decided for epic #3158):** zero decided-position disagreements (`--decided 2`) against the fp16 latent behind 1024 and 4096 tokens. A decided disagreement keeps the family on fp16. Top-1 flips outside the 0.11-logit near-tie bar are recorded with their gap, next to the fp16 kernel noise floor and the perplexity delta, and do not block.

**Outcome:** pending. The Kimi-Linear-48B-A3B checkpoint was still downloading when this change was opened; the sections below fill in once it is on the host.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal.
- **Model:** `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` (`model_type` `kimi_linear`, 27 layers of which 7 are MLA (`full_attn_layers` 4, 8, 12, 16, 20, 24, 27) and 20 KDA, 32 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64).
- **Binary:** release build (`--features metal,accelerate`) of this branch on top of `3f5679a4`.
- **Isolation:** every GPU run holds the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep runs under `scripts/with_indexers_paused.sh`, arms interleaved round by round inside one lock hold.
- **Arms:** `fp16 latent` (default) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness).

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Pending.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Pending.

## Cache size and sizing

| layout | bytes per token per MLA layer | 7 MLA layers at 16384 tokens |
|---|---|---|
| fp16 latent | 1152 | 126.0 MiB |
| int8 latent | 672 | 73.5 MiB |

Before this change `kv_arch` charged the latent on all 27 layers of Kimi-Linear-48B (486.0 MiB at 16384 tokens for fp16); it now reads `linear_attn_config.kda_layers` and charges the 7 MLA layers. The runtime side is pinned by `models::kimi_linear::latent_tests::runtime_bytes_match_the_inspect_rate_on_the_mla_layers`: a rank-512, rope-64 int8 cache allocates 672 bytes per reserved row, which is the `kv_arch` marginal rate of a one-MLA-layer config.

## Not measured

- **Kimi K3:** the only MLX export on the Hub is `mlx-community/Kimi-K3-mlx-reap160-2bit` (181 GB, expert-pruned, 2-bit weights). A pruned 2-bit model would mix weight-quantization and pruning error into a cache-mode comparison, so it was not downloaded. Kimi K3 reuses Kimi Linear's absorbed MLA and is covered by `models::kimi_k3::latent_tests` (int8 table, masked and unmasked decode against the fp16 latent, batched against single-row decode).
- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The path is the #3155 composed decode, which has no Metal-only call, but no trace or throughput row exists for those backends.
