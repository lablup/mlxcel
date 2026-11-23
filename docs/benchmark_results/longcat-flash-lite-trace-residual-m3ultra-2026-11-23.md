# LongCat-Flash-Lite: the long-context trace residual (M3 Ultra, 2026-11-23)

Issue #3318. After #3306 applied YaRN to the LongCat ngram MLA, the teacher-forced trace of `longcat-flash-lite-4bit` against mlx-lm still had 39 decided disagreements over 36864 positions. This file records the A/B arms the issue asked for and the arm that explains the residual.

## Setup

- **Host:** Mac Studio M3 Ultra, 512 GB, macOS 27.2, shared with other jobs; every GPU run held `/Users/devops/Development/.mlxcel-gpu.lock`. These are quality runs, so load does not enter the numbers.
- **Checkpoint:** `longcat-flash-lite-4bit` (`longcat_flash_ngram`, affine 4-bit group 64, routers 8-bit, `torch_dtype` bfloat16, YaRN `factor` 10 over 32768, `rope_theta` 5e6). It ships the fused MLA `kv_b_proj` (4-bit), not `embed_q` / `unembed_out`.
- **Corpus:** #3306's corpus, `docs/architecture.md`, `docs/supported-models.md` and `docs/environment-variables.md` concatenated (174,458 tokens, sha256 `5089b8f83ddc4fbb4bdc45c31fb09de084d958db0569aff7fafa672d31ea99fa`). Plan `4096 9 8 40000`: 9 chunks of 4096 traced positions, each behind all earlier tokens, 36864 positions in total.
- **Reference:** mlx-lm at `1bf8da0` (vendored `references/mlx-lm`), mlx 0.32.4, prefilled through its KV cache in 2048-token steps as in #3306, because one 36864-token mlx-lm forward returns uniform logits past 16384. Variants: the model cast to float16 or float32 (`set_dtype` on the bf16 tensors only), and a dense split, where the reference's `kv_b_proj` split keeps the dequantized `embed_q` / `unembed_out` instead of quantizing them again.
- **Candidate:** `examples/logit_trace` built from main at `67dc9090` (release, `metal,accelerate`).
- **Metric:** `scripts/compare_logit_traces.py`, decided positions are those where the reference's top two are at least 2 logits apart. Per-chunk counts are decided disagreements in each 4096-position chunk.

## Arms against the stock bf16 reference

| Arm | Per chunk (0 to 8) | Decided disagreements | Largest gap | Perplexity cand. / ref. |
| --- | --- | --- | --- | --- |
| mlxcel, default | 0 1 0 4 10 4 6 8 6 | 39 / 20486 | 4.500 | 25.4517 / 25.8153 |
| 1. `MLXCEL_KEEP_BF16=1` | 0 1 0 4 10 4 6 8 6 | 39 / 20486 | 4.500 | 25.4517 / 25.8153 |
| 2. `MLXCEL_TRACE_KV_CACHE_MODE=fp16` | 0 1 0 4 10 4 6 8 6 | 39 / 20486 | 4.500 | 25.4517 / 25.8153 |
| 3. rope term as an additive mask (fused 128-wide SDPA, mlx-lm's formulation; local experiment) | 0 0 0 1 6 5 8 8 5 | 33 / 20486 | 3.500 | 25.3950 / 25.8153 |
| float16 reference (dtype noise floor) | 0 0 0 1 1 0 0 1 2 | 5 / 20486 | 4.000 | 25.8895 / 25.8153 |
| float32 reference | 0 0 0 1 2 1 0 1 2 | 7 / 20486 | | 25.9347 / 25.8153 |

- Arm 1 is byte-identical to the default: `bf16_to_f16_at_load` already keeps a quantized checkpoint's bf16 side data, so nothing is converted on this checkpoint and the two runtimes run the same activation dtype.
- Arm 2 gives identical rows; the default dense cache already stores the bf16 activations, as mlx-lm's `KVCache` does.
- Arm 3: `MLXCEL_TRACE_MLA_STATS=1` shows every step on the materializing path (`decompressed=476`, no absorbed step), so there is no path mix to force. MLX runs the folded 192-wide prefill SDPA unfused; the experiment put the rope term back as a bf16 additive mask over a fused 128-wide SDPA, as mlx-lm does. The count moves from 39 to 33 and stays far above the floor; against the dense-split reference below, this arm has 2 / 20489, like the default.

## The weight arm

mlx-lm's `longcat_flash` sanitize dequantizes `kv_b_proj`, splits it into `embed_q` (transposed) and `unembed_out`, and quantizes both halves again at 4 bits, group 64; `embed_q` is quantized along the axis the original groups did not run along. mlxcel's `sanitize_weights` keeps the dequantized halves dense. The two runtimes therefore run different `embed_q` / `unembed_out` weights.

| Reference | Candidate | Per chunk (0 to 8) | Decided disagreements | Perplexity cand. / ref. |
| --- | --- | --- | --- | --- |
| bf16, dense split | mlxcel, default | 0 0 0 2 1 0 0 0 1 | 4 / 20489 | 25.4517 / 25.6342 |
| float32, dense split | mlxcel, default | 0 1 0 1 0 0 0 0 0 | 2 / 20208 | 25.4517 / 25.8198 |
| float32, dense split | bf16, dense split | 0 1 1 0 0 0 0 1 0 | 3 / 20208 | 25.6342 / 25.8198 |
| bf16, dense split | stock bf16 reference | 0 3 0 5 4 3 4 7 6 | 32 / 20489 | 25.8153 / 25.6342 |
| float32, stock split | mlxcel, default | 0 3 1 3 4 3 4 3 7 | 28 / 20174 | 25.4517 / 25.9347 |

Against a reference that keeps the split dense, mlxcel has 4 decided disagreements, below the 5 between the bf16 and float16 references, and 2 against the float32 dense-split reference, which is less than the bf16 reference itself shows against it (3). Changing only the reference's split moves it by 32 from itself, the size of the original residual. Changing only the reference's dtype to float32 leaves the residual at 28. Perplexity spreads by 0.7% between the bf16 and float32 dense-split references, so mlxcel's 25.45 sits inside the spread of the reference's own dtype arms.

The stock and dense-split references differ from each other with the same position profile as the original residual (0 to 3 per chunk below 12288, 3 to 7 from 12288 on), so the growth with position belongs to the weight difference too.

No mlxcel defect was found, so no runtime code changed; the sparse variant and the `deepseek_v3` traces are untouched by construction. `scripts/mlx_lm_logit_trace.py`'s docstring now says that mlxcel converts bf16 to f16 only for unquantized checkpoints, and that a fused quantized `kv_b_proj` loads different weights in the two runtimes.

## No-context check

The same 9 chunks traced with no history (`4096 9 8 0`, positions 0 to 4095 each; stock reference, one forward per chunk): 6 / 17982 decided disagreements (`0 0 2 0 0 1 0 1 2`), perplexity 33.5948 against 33.9605. The stock reference already sits apart from mlxcel at short context; the long-context arms above add more decided positions per chunk, not a new mechanism.
