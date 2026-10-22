# PR #2552: Add an explicit GELU selector and use tanh where the checkpoint declares it

**Date**: 2026-10-23
**Status**: PR open; Refs #2423, which stays open (remaining families in #2548, the mlx-vlm prompt-length mismatch in #2547)

## Executive Summary

Issue #2423 found that `mlxcel_core::gelu_approx` computes the exact erf GELU while MLX's `nn.gelu_approx` and `nn.GELU(approx="precise")` are the tanh approximation, so several vision ports that call `gelu_approx` compute a different formula from their mlx-vlm reference. PR #2552 adds `GeluActivation { Erf, Tanh, TanhF32, TanhF32Compiled, Fast }` with `from_hf_name`, folds six private tanh copies into it with their precision kept (bit-equal in the fold tests; BERT for f32 input), corrects the comments that claimed erf matched `approx="precise"`, and switches the vision families whose checkpoint and mlx-vlm port both use tanh: the MiniCPM-V 4.6 encoder MLP and window merger, the Gemma 3 VLM and PaliGemma SigLIP towers, and the Gemma 4 vision GeGLU (also reached by DiffusionGemma and EmbeddingGemma2). The headline finding is negative: the teacher-forced logit-trace gate could not distinguish erf from tanh in any measured family (0 decided disagreements; logits moved by at most the bf16 quantum, 0.125 to 0.5, and 1.75 once on Gemma 3), so each family was decided by the activation its checkpoint declares, cross-checked against mlx-vlm. The bf16 stability re-test found no NaN or Inf in any arm; the NaN the old comment described came from `power(x, 3)`, not from tanh. Evaluated eagerly in f32, tanh cost Gemma 3 image prefill +17.0%; a compiled kernel brought that to +2.8% pooled, a small cost with a consistent sign. The PR also fixes a NAME_MAX bug in the #2538 PTX cache overlay that kept long-named kernels out of the cache. 29 files, +1353/-178.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcel-core/src/utils.rs`, `cpp/mlx_cxx_bridge.{cpp,h}`, `src/lib.rs` | `GeluActivation` enum with `from_hf_name`, `in_f32`, `in_f32_compiled`, `apply`; new `gelu_tanh`, `gelu_tanh_f32` and `compiled_gelu_tanh_f32` on the C++ `gelu_tanh_approx` helper (cube as `x*x*x`); `gelu_approx` stays erf with corrected docs |
| `src/lib/mlxcel-core/src/utils_gelu_tests.rs` (new) | Fold tests (bit-equal for f32/bf16/f16), saturation over large \|x\|, compiled-vs-eager bounds |
| `src/models/bert.rs`, `src/models/kokoro/ops.rs`, `src/vision/encoders/{molmo2,siglip,kimi_vl,qwen3_vl}.rs` | Private tanh copies folded onto `TanhF32` or `Tanh`, keeping each copy's precision |
| `src/vision/config.rs`, `src/loading/vlm_gemma.rs`, `src/loading/vlm_siglip.rs` | `VisionHiddenActivation::SiglipDefaultTanh` and `default_missing_hidden_act_to_siglip`, called by the Gemma 3 VLM and PaliGemma loaders only |
| `src/vision/encoders/minicpmv4_6.rs`, `minicpmo.rs` | 4.6 encoder MLP from `hidden_act`, window merger tanh, downsample merger erf; MiniCPM-o stays erf |
| `src/vision/encoders/gemma4.rs` | Vision GeGLU from `hidden_activation` (default `gelu_pytorch_tanh`) |
| `src/lib/mlxcel-xla/src/emitter/vision_config.rs` | XLA Gemma 3 vision emitter resolves the missing key to tanh the same way |
| `src/vision/encoders/internvl.rs`, `src/models/gemma3n.rs`, `deepseek_vl2.rs` | Erf kept, made explicit or commented |
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`, `tests/cuda_custom_kernel_disk_cache.rs` | PTX cache temporary named from suffix and extension; second-process test on the compiled tanh kernel |
| `docs/gelu-call-sites.md`, `docs/benchmark_results/gelu-erf-vs-tanh-2026-10-23.md` (new) | Inventory of about 110 call sites with pinned upstream links; measurement record |

## 2. Technical Decisions

### 2.1 Name the formula, not the approximation

The trap is the name: this crate's `gelu_approx` is erf, MLX's `nn.gelu_approx` is tanh, and MLX maps both `"precise"` and `"tanh"` to tanh. A port that copies the call name gets the wrong formula without any compile-time signal. `GeluActivation` makes each call site state which of erf, tanh or fast it computes, and `from_hf_name` maps Hugging Face names onto `ACT2FN` semantics (`gelu` to erf; `gelu_pytorch_tanh`, `gelu_new`, `gelu_fast` to tanh; `quick_gelu` to fast), returning `None` for anything else. `gelu_approx` keeps its name because most of its call sites genuinely port an erf GELU; its doc comment now says so.

### 2.2 The gate was blind, so the checkpoint decided

Inside mlxcel, swapping erf for tanh on the same build and token stream changed no decided token in any family, with the logit delta on the reference choice at p50 0.000 to 0.250 and max 0.031 to 1.750:

| Checkpoint | Decided positions | Decided disagreements | Logit delta p50 / max |
|---|---|---|---|
| internvl3-1b | 38 | 0 | 0.000 / 0.031 |
| minicpm-v-4.6-bf16 | 47 | 0 | 0.125 / 0.500 |
| gemma-3-4b-it-4bit | 50 | 0 | 0.250 / 1.750 |
| paligemma2-3b-6bit | 25 | 0 | 0.008 / 0.188 |
| gemma-4-e2b-it-4bit | 45 | 0 | 0.125 / 0.500 |
| gemma3n-e2b-4bit (text, sparse layers) | 244 | 0 | 0.125 / 1.000 |

The float32 mlx-vlm reference shows the same: flipping its own vision GELUs to erf changed no decided token. Against that reference, MiniCPM-V 4.6 (2 of 39) and Gemma 4 (1 of 46) show the same decided disagreements in the erf, tanh-f32 and tanh-input-dtype arms, so those come from the 4-bit weights, bf16 activations or prompt, not from GELU; the mlxcel-vs-reference gap is 1 to 7 logits against a GELU swap at the bf16 step. Perplexity moved in both directions and differed between the two tanh precisions as much as tanh differed from erf, so the record calls it bf16 noise and it is not evidence for either arm.

With no measurement able to certify either formula, the PR switched a family only where the checkpoint's `config.json` or Hugging Face code and the mlx-vlm port both say tanh. Where mlx-vlm departs from Hugging Face, the checkpoint wins: InternViT declares `hidden_act = gelu` (erf) and the MiniCPM-V 4.6 downsample merger is `nn.GELU()` (erf), so both keep erf although mlx-vlm computes tanh. Gemma3n's sparse layers are tanh upstream but stay erf here, because the compiled `gelu_topk` kernel, the fused `gemma3n_mlp_forward` sparse branch and the XLA Gemma3n host oracle have to change together.

### 2.3 The missing key is resolved per loader

Gemma 3 and PaliGemma configs omit `vision_config.hidden_act`. Hugging Face drops a key whose value equals the class default, and `SiglipVisionConfig` defaults to `gelu_pytorch_tanh`, so those towers were trained with tanh. The serde default stays erf for every other loader because the CLIP class default is `quick_gelu`, not tanh; `default_missing_hidden_act_to_siglip` is called by exactly the Gemma 3 VLM and PaliGemma loaders. Aya Vision and Granite Vision 3.x also omit the key and are left for #2548. The XLA emitter flips only `model_type == "gemma3"`. Resolution yields a distinct variant, `SiglipDefaultTanh`, instead of `GeluPytorchTanh`, so a declared `gelu_pytorch_tanh` keeps the eager bit-exact path while the resolved one runs the compiled kernel (2.5).

A `hidden_act` or `hidden_activation` that is not a GELU name now fails the load for the MiniCPM-V 4.6 encoder and the Gemma 4 vision tower instead of silently running erf. Every local config for those families declares `gelu_pytorch_tanh` or omits the key.

### 2.4 bf16 stability: the NaN was `power`, not tanh

The old `gelu_approx` comment said tanh produced NaN in the bf16 SigLIP MLP at 4096 patches. A probe on every vision MLP activation found zero NaN and zero Inf in all seven arms, including gemma-3-4b-it-4bit's bf16 tower at 27 x [1, 4096, 4304] (max |input| 161, 177 on a page image), minicpm-v-4.6-bf16 (max 214) and paligemma2-3b-6bit in f16 (max 70.4), with tanh in input dtype and in f32. The NaN came from `power(x, 3.0f)`, which MLX evaluates as `exp(3*log(x))`, undefined for negative x; `gelu_tanh_approx` cubes as `x*x*x`. bf16 has f32's exponent range, so a cube near 1e7 at |x| = 214 is far from overflow. In f16 the cube overflows above |x| of about 40, but tanh saturates to +-1 and the output is x or 0, not NaN.

### 2.5 f32 tanh, compiled

Both precisions were finite. `TanhF32` was chosen because it is what the SigLIP `gelu_pytorch_tanh` path and its StableHLO twin were qualified with, and because PyTorch evaluates `gelu(approximate="tanh")` for half inputs in f32 (on torch 2.14.1 CPU, bit-equal to f32-then-cast over 120k inputs, and different from op-by-op half arithmetic on about 15 percent). mlx-vlm evaluates in the input dtype.

Evaluated eagerly, `TanhF32` is about 11 elementwise passes against about 6 for the f32 erf, and Gemma 3's SigLIP runs it on 27 x [1, 4096, 4304]. The review measured erf 872 ms against `TanhF32` 1025 ms median (paired +151 to +172 ms) and input-dtype tanh 916 ms (+42 ms). Rather than drop to input-dtype tanh, the PR wraps the same helper with the upcast and the cast back in one shapeless `mlx::core::compile` region, `compiled_gelu_tanh_f32`, exposed as `TanhF32Compiled` and used only by the switched families. The folded sites stay on eager `TanhF32`, which is pinned bit-for-bit to the copies it replaced. The compiled kernel may contract a multiply-add into an FMA, so it is held to 4 f32 ulps for f32 output and one ulp of the narrow dtype for bf16/f16 output; on a tower-shaped bf16 [64, 4304] input with outliers to +-170 it was bit-equal to the eager helper (0 of 275456 lanes differ), and greedy 32-token replies were identical between the eager and compiled builds on all five smoke checkpoints.

Image prefill, one binary, erf and null arms through symlinked checkpoint copies declaring `gelu`, arm order rotated, `MLX_ENABLE_TF32=0`, GB10:

| Family | n | erf median | compiled tanh median | Paired tanh - erf | Null - erf | Outcome |
|---|---|---|---|---|---|---|
| Gemma 3 SigLIP | 16 | 859.5 ms | 878.7 ms | +23.8 ms, +2.8% (-11.8..+53.8); blocks +2.4% and +3.7% | -0.5 ms (-33.8..+34.8) | tanh kept, residual unresolved |
| Gemma 3, eager `TanhF32` (previous state) | 8 | 869.3 ms | 1013.4 ms | +148.0 ms, +17.0% (+116.5..+174.7) | | reference |
| MiniCPM-V 4.6 | 6 | 910.8 ms | 897.1 ms | -14.0 ms, -1.5% (-36.1..+42.4) | +2.4 ms (-34.5..+31.6) | within null floor |
| Gemma 4 vision | 12 | 661.7 ms | 677.9 ms | +13.2 ms, +2.0% (-6.0..+36.3) | +1.0 ms (-27.9..+24.8) | tanh kept, residual unresolved |
| PaliGemma SigLIP | 6 | 995.5 ms | 987.6 ms | -15.1 ms, -1.5% (-37.5..+12.8) | +8.1 ms (-41.6..+20.3) | within null floor |

The Gemma 3 and Gemma 4 residual has the same sign in every block while the null arm is centered on zero, so it is a small real cost, not noise. Pooled it is under the 3% acceptance line, but one Gemma 3 block alone (+3.7%) was above it. The cached modules show the cube and the tanh tail as separate kernels, so on CUDA the compiled region is still more than one fused kernel; a fully fused kernel is a #2548 item.

### 2.6 PTX cache temporaries under NAME_MAX

The first compiled measurement showed a fixed cost unrelated to the kernel: Gemma 3 prefill 2505 ms uncached against 869 ms cached, because every process NVRTC-compiled the fused kernel again. Since #2538 the `jit_module.cpp` overlay wrote each cache file through `<module name>.txt.tmp<pid>_<tick>`. The compiled tanh kernel's module name is 234 characters, which `get_ptx_path` keeps as one path component, so the temporary passed the 255-byte NAME_MAX, the open failed with ENAMETOOLONG, and the entry was never written. Any kernel with a module name of roughly 229 to 245 characters was affected. The temporary is now named from the unique suffix and the extension only, in the final file's directory, so `rename(2)` stays atomic. This affects any kernel in that length band, not only GELU. `tests/cuda_custom_kernel_disk_cache.rs` runs this kernel in a second process, requires it to compile nothing, and fails on the previous overlay.

### 2.7 Changes during review

- `715ae4a8`: the implementation review found that the PR did not state the eager `TanhF32` prefill cost (HIGH) or that a non-GELU activation name now fails the load (MEDIUM), and that `gemma4_activation_tests.rs` cited gemma-4-12b-it-4bit, which is encoder-free (LOW). The cost section, the body text and the test note were fixed.
- `ecabb220`: the compiled f32 tanh kernel and `TanhF32Compiled`, replacing the eager path on the switched families.
- `a78ffcf0`: the NAME_MAX fix, found while measuring `ecabb220`.
- `f2617730`, `3734c1b4`: the per-family compiled A/B with null arms, including both Gemma 3 blocks. This resolved the review's open item that the cost on PaliGemma, MiniCPM-V 4.6 and Gemma 4 had not been timed.
- The security and performance review at `316fea12` had no CRITICAL, HIGH or MEDIUM findings and produced no fix commits.

## 3. Verification

From the PR test plan and the security review on GB10 (kernel 7.0.0-1019-nvidia, driver 580.178.04, `MLX_CUDA_ARCHITECTURES=121`, `test-fast` + `cuda`, `MLX_ENABLE_TF32=0`, `--test-threads=1`, under `gpu-lock`), at `316fea12`:

| Check | Result |
|---|---|
| `-p mlxcel-core --lib gelu` | 24 passed, including the compiled-vs-eager bounds |
| `ffi_tests::shapeless_compile_audit_harness` (`MLXCEL_SHAPELESS_COMPILE_AUDIT=1`) | `compiled_gelu_tanh_f32` PASS, 3 signatures warmed |
| `--test cuda_custom_kernel_disk_cache` | Fails on the previous overlay, 1 passed with the fix (child compiled nothing, no `.tmp` leftovers) |
| `-p mlxcel --lib` filters | siglip 39, minicpmv4_6 14, gemma4::activation 2, vision::config 2 passed; earlier on the branch also internvl, molmo2, qwen3_vl, bert, kimi, kokoro, moonvit, and `-p mlxcel-xla --lib vision_config emitter::vision` |
| clippy `-D warnings` (mlxcel-core and mlxcel lib+tests), `cargo fmt --all --check` | Clean |
| CLI `--image` smoke | minicpm-v-4.6-bf16, gemma-3-4b-it-4bit, paligemma2-3b-6bit, gemma-4-e2b-it-4bit, internvl3-1b describe the three shapes; eager and compiled builds give identical text |
| Server smoke (`/v1/chat/completions`) | Same checkpoints on the earlier commit; DiffusionGemma through the CLI only |

The fold tests reproduce the SigLIP, Molmo2, Qwen3-VL and Kimi copies op for op and assert bit equality for f32, bf16 and f16. BERT's copy equals the f32 oracle for f32 input; for half input the old copy left the result in f32 and the selector now casts it back, and no local BERT config takes the tanh branch.

## 4. Unverified and Open Items

- **Issue stays open:** criteria 3, 4 and 6 of #2423 are met only for the measured families. About 20 remaining mismatched families (Gemma3n sparse, Gemma 4 bf16 dense MLP and drafter, Gemma 4 TP gate, RecurrentGemma, MiniCPM-o, Aya/Granite SigLIP, CLIP `quick_gelu` defaults, Phi, GPT-2, Moondream, LFM2-VL, DeepSeek-VL2 and others) are in #2548 with one acceptance item each.
- **Cross-runtime comparisons invalid:** `# prompt_tokens` disagree with mlx-vlm for InternVL (262 vs 268), MiniCPM-V 4.6 (78 vs 76) and Gemma 3 (273 vs 274), so those mlxcel-vs-mlx-vlm rows cannot be read; tracked in #2547. Gemma 4 matches (275).
- **Residual prefill cost:** Gemma 3 +2.8% pooled (one block +3.7%) and Gemma 4 +2.0% were not investigated further. The MiniCPM-V 4.6 erf arm still ran the hard-coded tanh window merger, so that row slightly understates the difference.
- **Not run:** Metal (compiled path and overlay untouched there), EmbeddingGemma2 (no local checkpoint), DiffusionGemma through the server (rejects image input, pre-existing), whole-workspace test and clippy.
- **Security review LOW items, not changed:** a crash orphan is now a dotfile without the module name and nothing sweeps the cache; `std::ofstream` follows a symlink at the temporary name (as upstream did; `O_CREAT | O_EXCL | O_NOFOLLOW` would close it); a JSON `null` `hidden_act` (MiniCPM-V 4.6) or `hidden_activation` (Gemma 4 vision) fails the load instead of falling back to the default, unlike `VisionConfig`.
- **Review LOW item, kept:** BERT's `Activation` parses its own GELU names instead of `GeluActivation::from_hf_name`, because it also maps `relu`.
- **Not folded:** `gemma3_backbone.rs` (Nemotron VoiceChat TTS) still runs the tanh form with `power(x, 3)` in production, and `nemotron_codec` keeps its own erf; folding would change numerics.

## 5. Learning Points

- A teacher-forced gate on decided positions bounds how far an arm can drift; it does not certify which of two arms is right when both sit inside the bf16 step. When the gate is blind, record that as the finding and decide from the trained configuration, instead of reading perplexity noise as a signal.
- A reference implementation is not ground truth for activations: mlx-vlm departs from the checkpoint's Hugging Face activation for InternViT and the MiniCPM-V 4.6 downsample merger. Check the checkpoint's `config.json` and the HF class default before treating a mismatch as a bug.
- Hugging Face omits config keys equal to the class default, so a missing `hidden_act` means "class default", and the default differs by class (SigLIP tanh, CLIP `quick_gelu`). Resolve it per loader, not in a shared serde default.
- When a numerically pinned helper also needs a faster path, add a separate variant for the new users and leave the pinned one alone; the fold tests stay bit-exact and the new path is held to a stated ulp bound.
- A cache bug can sit unnoticed until a kernel lands in the length band that triggers it. Name temporaries independently of the final file name, and test cache hits in a second process.
