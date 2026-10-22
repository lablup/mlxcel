# GELU erf vs tanh per family, GB10 (2026-10-23)

Issue #2423: `mlxcel_core::gelu_approx` is the exact erf GELU, while the mlx-vlm ports of several vision towers use MLX's `nn.gelu_approx` / `approx="precise"`, the tanh approximation. This records the bf16 stability re-test, the per-family logit-trace measurements, and the decisions taken from them. The call-site inventory is [../gelu-call-sites.md](../gelu-call-sites.md).

## Environment

- Host: NVIDIA GB10 (sm_121), kernel `7.0.0-1019-nvidia`, driver `580.178.04`, under `gpu-lock`, `MLX_ENABLE_TF32=0` on every GPU arm.
- Build: `MLX_CUDA_ARCHITECTURES=121 cargo build --profile test-fast --features cuda --example logit_trace` at da965b22 (the selector commit) plus a temporary, uncommitted override: `MLXCEL_GELU_2423_<SITE>=erf|tanh|tanh_f32` replaced the activation at exactly one family's sites, and `MLXCEL_GELU_2423_PROBE=1` evaluated every probed activation and printed its dtype, shape, max |input|, NaN count and Inf count. Each A/B pair is the same binary, the same checkpoint and the same token stream; only the activation differs. The override was removed before the behavior commit.
- Image: `tests/fixtures/test_image_shapes.png` (448x448), prompt `Describe the image.`, teacher-forced through `MLXCEL_TRACE_IMAGE` (`examples/logit_trace MODEL REPLY 64 1 8`, so forward width 64 behind the image prompt).
- Reply text: for InternVL, MiniCPM-V 4.6 and Gemma 3 the greedy reply mlx-vlm 0.7.8 produced in float32 on CPU; for PaliGemma2 and Gemma 4 the InternVL reply (the token stream only has to be shared between arms).
- Reference arms: mlx-vlm 0.7.8 on CPU, dequantized to float32, scoring the same target tokens (the logic of `scripts/mlx_vlm_image_logit_trace.py --float32`), once with its own activations (`ref_keep`) and once with the vision-side GELU modules flipped to erf (`ref_erf`). For Gemma 3 and PaliGemma the mlx-vlm forward builds an all-ones 4D mask from the attention mask; the reference arm instead passes the merged embeddings to the language model with `mask=None`, so teacher-forced rows stay causal, which is also what mlxcel does.
- Gate: `scripts/compare_logit_traces.py --decided 2`, disagreement on positions where the reference's top two are more than 2 logits apart.

## bf16 stability re-test (criterion 2)

The comment on `gelu_approx` said tanh GELU produced NaN in the bf16 SigLIP MLP at 4096 patches. Re-run with the probe on every vision MLP activation:

| Checkpoint | Tower dtype | Activation shape (per layer) | Arm | Max \|input\| | NaN | Inf | Logits |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it-4bit (SigLIP 896px, unquantized bf16 tower) | bf16 | [1, 4096, 4304] x 27 | erf | 161 | 0 | 0 | finite |
| same | bf16 | [1, 4096, 4304] x 27 | tanh, input dtype (`x*x*x`) | 161 | 0 | 0 | finite |
| same | bf16 | [1, 4096, 4304] x 27 | tanh, f32 then cast back | 160 | 0 | 0 | finite |
| same, 1240x1754 page image | bf16 | [1, 4096, 4304] x 27 | tanh, input dtype | 177 | 0 | 0 | finite |
| minicpm-v-4.6-bf16, page image | bf16 | [1, 1120, 4304] x 27 plus both mergers | tanh, input dtype | 214 | 0 | 0 | finite |
| same | bf16 | same | tanh, f32 | 214 | 0 | 0 | finite |
| paligemma2-3b-6bit (448px) | f16 | [1, 1024, 4352] x 27 | tanh, input dtype | 70.4 | 0 | 0 | finite |

No NaN or Inf in any arm. bf16 has f32's exponent range, so `x*x*x` at |x| = 214 (about 1e7) is far from overflow; the NaN the comment describes came from `power(x, 3.0f)`, which MLX evaluates as `exp(3*log(x))` and which is undefined for negative x. f16 is the dtype where the cube overflows (|x| > about 40, reached by paligemma2's tower at 70.4); tanh saturates to +-1, so the output is x or 0 there rather than NaN, which `utils_gelu_tests.rs` also checks over |x| up to 1e4. The vision changes below use the f32 variant anyway, which is what the existing SigLIP `gelu_pytorch_tanh` path already computed.

## Logit-trace measurements (criterion 3)

Within mlxcel, erf against tanh on the same build (the reverted arm is the erf row):

| Checkpoint | Sites switched | Positions (decided) | Decided disagreements | Logit delta on ref choice p50 / max | Perplexity erf -> tanh f32 (tanh input dtype) |
|---|---|---|---|---|---|
| internvl3-1b (f16 tower) | InternViT MLP | 64 (38) | 0 | 0.000 / 0.031 | 1.7881 -> 1.7888 (1.7867) |
| minicpm-v-4.6-bf16 | encoder MLP + both mergers | 60 (47) | 0 | 0.125 / 0.500 | 4.4507 -> 4.6071 (4.3991) |
| minicpm-v-4.6-bf16 | encoder MLP + window merger (HF set) | 60 (47) | 0 | 0.125 / 0.500 | 4.4507 -> 4.6649 (4.6389) |
| gemma-3-4b-it-4bit | SigLIP MLP | 64 (50) | 0 | 0.250 / 1.750 | 4.6269 -> 4.7430 (5.0241) |
| paligemma2-3b-6bit | SigLIP MLP | 64 (25) | 0 | 0.008 / 0.188 | 3.0804 -> 3.0838 (3.0946) |
| gemma-4-e2b-it-4bit | vision GeGLU | 64 (45) | 0 | 0.125 / 0.500 | 6.1237 -> 6.1111 (6.0889) |
| gemma3n-e2b-4bit (text, `docs/architecture.md`, 4 x 128) | sparse `gelu_topk`, layers 0-9 | 512 (244) | 0 | 0.125 / 1.000 | 130.44 -> (130.80) |

Within the float32 mlx-vlm reference, its own activation against erf:

| Checkpoint | Positions (decided) | Decided disagreements | Logit delta p50 / max | Perplexity |
|---|---|---|---|---|
| internvl3-1b | 64 (38) | 0 | 0.004 / 0.016 | 1.3916 vs 1.3915 |
| minicpm-v-4.6-bf16 (all vision GELUs) | 60 (39) | 0 | 0.014 / 0.112 | 1.2208 vs 1.2218 |
| minicpm-v-4.6-bf16 (downsample merger only) | 60 (39) | 0 | 0.001 / 0.004 | 1.2208 vs 1.2209 |
| gemma-4-e2b-it-4bit (vision GeGLU) | 64 (46) | 0 | 0.001 / 0.012 | 5.3607 vs 5.3636 |

Against the float32 mlx-vlm reference (`ref_keep`, tanh):

| Checkpoint | `# prompt_tokens` mlxcel / mlx-vlm | erf arm: decided disagreements, ppl | tanh f32 arm | tanh input-dtype arm |
|---|---|---|---|---|
| internvl3-1b | 262 / 268 (invalid) | 0 / 38, 1.7881 | 0 / 38, 1.7888 | 0 / 38, 1.7867 (ref 1.3916) |
| minicpm-v-4.6-bf16 | 78 / 76 (invalid) | 2 / 39, 4.4507 | 2 / 39, 4.6071 | 2 / 39, 4.3991 (ref 1.2208) |
| gemma-4-e2b-it-4bit | 275 / 275 | 1 / 46, 6.1237 | 1 / 46, 6.1111 | 1 / 46, 6.0889 (ref 5.3607) |
| gemma-3-4b-it-4bit | 273 / 274 (invalid, not run) | | | |

What this shows:

- The reverted arm does not fail for any family. Swapping erf for tanh moves logits by at most the bf16 quantum (0.125 to 0.5, 1.75 once on Gemma 3) and never changes a decided token, while the mlxcel-vs-reference gap from the 4-bit weights, bf16 activations and, where they differ, the prompt is 1 to 7 logits. The decided-position gate cannot distinguish erf from tanh at any of these families, so it cannot certify either choice. The perplexity movements in the first table go both ways and differ between the two tanh precisions by as much as tanh differs from erf; they are bf16 noise, not a signal.
- `# prompt_tokens` disagree for InternVL (262 vs 268), MiniCPM-V 4.6 (78 vs 76) and Gemma 3 (273 vs 274), so by `docs/benchmarks.md`'s own rule those cross-runtime comparisons are invalid; the large perplexity gaps there (1.79 vs 1.39, 4.45 vs 1.22) come from the prompt and the reference-greedy reply, not from GELU. Gemma 4 matches (275) and shows the same one decided disagreement in every arm.

## Decisions (criterion 4)

The gate being blind, each family was decided by the activation its checkpoint was trained with, which Hugging Face's code and the checkpoint's `config.json` state, and the mlx-vlm port was used as a cross-check. A family changes only where both say tanh and the measurements above show no regression.

| Family | Checkpoint / HF | mlx-vlm | Decision |
|---|---|---|---|
| MiniCPM-V 4.6 encoder MLP | `hidden_act = gelu_pytorch_tanh` | tanh | tanh (f32), read from `hidden_act` |
| MiniCPM-V 4.6 window merger | `gelu_pytorch_tanh`, hard-coded in `MiniCPMV4_6ViTWindowAttentionMerger` | tanh | tanh (f32) |
| MiniCPM-V 4.6 downsample merger | `nn.GELU()` in `MiniCPMV4_6DownsampleMLP` (erf) | tanh | keep erf |
| InternViT | `hidden_act = gelu`, `ACT2FN["gelu"]` (erf) | tanh | keep erf |
| Gemma 3 VLM SigLIP | key absent, `SiglipVisionConfig` default `gelu_pytorch_tanh` | tanh | tanh (f32); the loader resolves the missing key |
| PaliGemma SigLIP | key absent, same default | tanh | tanh (f32); the loader resolves the missing key (the XLA Gemma 3 vision emitter resolves it the same way) |
| Gemma 4 vision GeGLU | `hidden_activation = gelu_pytorch_tanh` (or absent, same default) | tanh | tanh (f32), read from `hidden_activation` |
| Gemma3n sparse layers | `gelu_pytorch_tanh` | tanh | keep erf here; follow-up (the compiled `gelu_topk` kernel, the fused `gemma3n_mlp_forward` sparse branch and the XLA Gemma3n host oracle have to move together) |

`TanhF32` rather than input-dtype tanh: both were finite, the f32 form is the one the SigLIP `gelu_pytorch_tanh` path and its StableHLO twin were qualified with, and it is how PyTorch evaluates `gelu(approximate="tanh")` for half inputs (in `opmath_t`, f32), whereas mlx-vlm evaluates in the input dtype. It costs prefill time, measured below. The Gemma 4 vision tower is also used by DiffusionGemma and EmbeddingGemma2, whose mlx-vlm ports reuse `gemma4.vision`, so they change with it.

## Prefill cost of the f32 tanh (PR #2552 review)

The eager tanh GELU is unfused: `TanhF32` is about 11 elementwise passes (cast, cube, polynomial, tanh, products, cast back) mostly over f32 intermediates, against about 6 for the f32 erf in `gelu_approx`. Gemma 3's SigLIP runs it on 27 x [1, 4096, 4304].

Same binary (038d8101 plus, for the input-dtype arm only, a temporary uncommitted env switch that routed `TanhF32` to `Tanh`), `gemma-3-4b-it-4bit`, erf arm through a copy of `config.json` declaring `vision_config.hidden_act = "gelu"` (all other files symlinked), `mlxcel generate --image tests/fixtures/test_image_shapes.png -p "Describe the image." -n 8 --temp 0 --profile`, 273 prompt tokens, `MLX_ENABLE_TF32=0`, under `gpu-lock`, arms interleaved with alternating order. The host is also a CI runner, and one round had all three arms disturbed.

| Arm | Runs | Prefill median (ms) | Paired delta vs erf, median |
|---|---|---|---|
| erf (previous behavior) | 12 | 872 | |
| tanh, f32 (`TanhF32`, this PR) | 12 | 1025 | +172 ms (run 1, 6 pairs), +151 ms (run 2, 6 rounds) |
| tanh, input dtype (`Tanh`) | 6 | 916 | +42 ms |

So the switch adds about 150 to 170 ms (+17 to 20 %) to a Gemma 3 image prefill; input-dtype tanh would recover about two thirds of it. The f32 form was kept: input-dtype tanh gave Gemma 3 the worst perplexity of the three arms in the logit-trace table (5.02 against 4.74 and 4.63, inside bf16 noise but not evidence for it), and the cost belongs to the unfused eager evaluation rather than to the precision. A fused (compiled) tanh GELU would remove most of it for every caller, including the f32 erf path, but changes rounding (FMA contraction) and therefore breaks the bit-exact folds, so it is left to the #2548 follow-up. PaliGemma (1024 patches), MiniCPM-V 4.6 and the Gemma 4 vision GeGLU pay the same per-element cost scaled by their tower size; they were not timed.

## Real-checkpoint smoke tests (criterion 6)

On the committed build (c09491b0, `cargo build --profile test-fast --features cuda --bin mlxcel --bin mlxcel-server`, no override in the binary), `mlxcel generate -m <ckpt> -p "Describe the image." --image tests/fixtures/test_image_shapes.png -n 48 --temp 0` and one `/v1/chat/completions` request with the same image as a data URL to `mlxcel-server -m <ckpt>`:

| Checkpoint | Change exercised | CLI | Server |
|---|---|---|---|
| minicpm-v-4.6-bf16 | encoder MLP + window merger tanh | red square, blue circle, green triangle | same shapes, 84 prompt tokens |
| gemma-3-4b-it-4bit | SigLIP tanh (missing key resolved) | describes the three shapes | same, 273 prompt tokens |
| paligemma2-3b-6bit | SigLIP tanh (missing key resolved) | describes the three shapes | same, 1036 prompt tokens |
| gemma-4-e2b-it-4bit | vision GeGLU tanh | describes the three shapes | same, 275 prompt tokens |
| diffusiongemma-26b-a4b-it-4bit | Gemma 4 vision tower, tanh | describes the three shapes (256 soft tokens) | 501: the server rejects image input for this checkpoint ("has no image tower"), a pre-existing server limitation, so the tower is reached only through the CLI |
| internvl3-1b | none (erf kept; comment only) | describes the three shapes | same |

EmbeddingGemma2 also reuses the Gemma 4 vision tower; no local checkpoint, not smoke-tested.
