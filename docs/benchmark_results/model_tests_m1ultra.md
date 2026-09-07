# Model Compatibility and Performance Tests (M1 Ultra)

Compatibility and single-stream performance for mlxcel on **Mac Studio M1 Ultra 128GB**, measured against the Python mlx-lm and mlx-vlm baselines on the same host, the same day, and the same prompt shape.

Every number here comes from the 2026-09-06 and 2026-09-07 sweep. Earlier sweeps used a different measurement shape and are not comparable, so they are not carried forward; the CSVs under `benchmarks/` remain the record of what was measured when.

## Test environment

| Item | Value |
|------|-------|
| Hardware | Mac Studio M1 Ultra, 128GB unified memory |
| OS | macOS 26.6.2 |
| mlxcel version | 0.7.0-beta.1 |
| mlxcel commit | `30ab5a39` (both sweeps; later commits changed only the harness) |
| MLX C++ pin | `9a795735` |
| Build | `cargo build --release --features metal,accelerate` |
| mlxcel harness | `mlxcel-bench-decode` (load, warmup and measured pass in one process) |
| mlx-lm baseline | 0.31.3 |
| mlx-vlm baseline | 0.6.17 |
| Baseline stack | mlx 0.32.2, transformers 5.16.1, torch 2.14.0, torchvision 0.29.0, timm 1.0.29, numba 0.67.0 |
| CSVs | `metal_m1ultra_2026-09-06.csv`, `pylm_m1ultra_2026-09-06.csv`, `metal_m1ultra_vlm_2026-09-07.csv`, `pylm_m1ultra_vlm_2026-09-07.csv` |

## Measurement shape

Text rows follow the llama-bench `pp512/tg128` convention: a 512-token synthetic prompt, exactly 128 generated tokens, EOS suppressed on both sides so a model that would answer in twelve words is still timed over 128 tokens. Without that suppression the decode figure is an average over whatever length the model chose, which is a property of the model's verbosity rather than of the runtime, and it is not comparable across models.

VLM rows keep the 128-token generation but take their prompt length from the image, since the image fixes it. Prompt lengths there run from 56 to 1032 tokens depending on the tower's patch count.

Time Machine is stopped for the duration (`tmutil stopbackup`) and models run with a 30 second cooldown between them.

### What the harness refuses to record

Three classes of row look like measurements but are not, and each is now rejected by the sweep rather than left for a reader to notice:

- **A checkpoint with no vision tower in a `--vlm` sweep.** mlx-vlm loads one without complaint and silently drops the image, so the row is a text generation wearing a VLM label. Both harnesses now run `scripts/vlm_detect.py` and cover the same model set.
- **A VLM whose image never reached the prompt.** The child reports the token count of the formatted prompt as plain text; an image that arrived always expands the prompt past it, so `prompt_tokens <= text_only_prompt_tokens` is recorded as `FAIL:image_not_applied`. Three checkpoints here were affected, and `llava-next-mistral-7b-4bit` reported the same 7 tokens on both M1 Ultra and M5 Max across three runs.
- **A duplicate checkpoint.** Identity is `sha256(config.json)` plus the sorted shard names and sizes, not the directory name. Sixteen directories on this host are byte-identical copies of another, several of them named as though they were unquantized when they hold 4-bit weights. They are recorded as aliases and measured once.

### A stale shard index is an upstream property, not local damage

Twelve checkpoints here carry a `model.safetensors.index.json` that names shards which do not exist, and the loss is total rather than partial: none of the shards the index lists is present. `qwen3-vl-32b-4bit` is the widest case, with the index declaring 14 shards and 62 GB against 4 shards and 18 GB on disk.

They are not damaged downloads. The local copies match their repositories file for file, M1 Ultra and M5 Max independently hold the same twelve in the same state, and the index is the *pre-quantization* original's index carried through unchanged: `gemma-3-4b-it-4bit` declares the 2 shards and 8.60 GB of `google/gemma-3-4b-it`, `qwen3-vl-32b-4bit` the 14 shards and 66.71 GB of `Qwen/Qwen3-VL-32B-Instruct`. The conversion wrote new weights and copied the old index.

The scope is VLM conversions, at roughly 12% of the most-downloaded image-text-to-text repositories, concentrated in the gemma-3 and Qwen3-VL families. mlx-lm text conversions are unaffected. The mlx-vlm version recorded in each repository does not predict it: 0.3.2 appears on both the healthy and the stale side.

These load and measure correctly because **mlxcel globs `*.safetensors` and does not read the index**, which in this ecosystem is tolerance rather than a shortcut: a loader that trusted the index would refuse all twelve outright. `scripts/checkpoint_fingerprint.py` reports which path a checkpoint took in its `shard_source` field.

## Coverage

The sweep walks every checkpoint directory under `models/mlx`. What it does with each one:

| Outcome | Text | Notes |
|---|--:|---|
| Measured | 165 | |
| Alias of another checkpoint | 16 | Same `config.json` hash and shard set; measured once |
| Not a checkpoint | 2 | `models` and `large_models`, container directories |
| Over the memory limit | 1 | `mimo-v2-flash-4bit` |
| Not a single-stream text model | 14 | See below |
| Unsupported architecture | 1 | `afm-4.5b` |

### Nothing in the failure column is a defect

All 15 text-sweep failures resolve to something other than a runtime bug, so the failure count is a statement about what is on disk rather than about mlxcel:

| Kind | Count | Checkpoints |
|---|--:|---|
| Embedding and rerank models | 3 | `all-minilm-l6-v2`, `bge-small-en-v1.5`, `ms-marco-minilm-l6-v2` |
| Speculative-decoding variants | 4 | `qwen3.5-4b-dflash`, `qwen3.5-27b-dflash`, `qwen3.8-27b-mtp-4bit`, `qwen3.8-27b-mtp-bf16` |
| MTP drafters | 2 | `gemma-4-12b-it-assistant-4bit`, `gemma-4-31b-it-assistant-bf16` |
| Audio models | 2 | `whisper-tiny`, `granite-speech-4.1-2b-nar-mlx` |
| Source repositories, not release weights | 2 | `klear-46b-src`, `phixtral-4x2_8-src` |
| Object detection | 1 | `docling-layout-heron-mlx-bf16` (RT-DETRv2) |
| Unsupported architecture | 1 | `afm-4.5b` |

The first six rows are not text-generation models and cannot produce a decode figure; the drafters and speculative variants are components of a pairing rather than standalone targets, and belong in the speculative sweep instead. `afm-4.5b` is the only real coverage gap: `mlxcel generate` reports `Unsupported model type: arcee` and `arcee` is absent from `mlxcel list`.

The Python baseline measured 124 of the same set, so parity is computed over the 107 models both sides measured. One of those, `plamo-2-1b`, was added on 2026-09-07 after `numba` was installed; its earlier `FAIL:warmup` was a missing dependency of the checkpoint's remote code, not a property of the model. Its own failures are not analysed here; they say what mlx-lm loads, not what mlxcel does.

## Performance against the Python baselines

Parity is `mlxcel decode tok/s / baseline decode tok/s`, over the 107 text models both sides measured at prompt lengths agreeing within 10%, the same rule the VLM section uses. Values above 100% mean mlxcel is faster.

| Population | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| All text models | 107 | 100% | 98 / 105 | 26-140% |
| MoE | 24 | 106% | 101 / 110 | 78-140% |
| Dense | 83 | 100% | 97 / 102 | 26-115% |

The overall median sits at parity, which is the expected result for two runtimes calling the same MLX kernels on the same weights.

This ratio is the runtime claim and it is the one to read first. It is measured on one machine against a baseline run on that same machine, so it is independent of the hardware. The hardware question is separate and lives in the cross-hardware section of [model_tests.md](model_tests.md); a ratio taken between two machines on mlxcel alone cannot distinguish a hardware gap from a place where mlxcel fails to exploit the hardware. On M5 Max three checkpoints turn out to be exactly that, and they are only visible when the same-machine ratio is computed on both machines and the two are compared.

MoE is the one population that separates. Its lower quartile (101%) is above the dense median, so three quarters of MoE checkpoints are ahead rather than a few large wins pulling an average. The direction matches the fused decode-MoE kernel, which replaces `gather_qmm` on small-expert families and gains in proportion to how much of the model is MoE. `trinity-nano-preview-4bit` at 140%, `qwen3-30b-a3b-4bit` at 125% and `klear-46b-a2.5b-instruct-4bit` at 123% are the largest.

Quantization is not a factor. Non-quantized checkpoints have a median of 98% against 100% for quantized, and 11 of the 14 sit between 89% and 109%. An earlier reading that the slowest three models were all non-quantized was a coincidence of a three-model sample, not a property of the non-quantized path.

### Open performance gaps

Three checkpoints fall far outside the distribution, and prefill is down with decode on all three, which points at something earlier than the decode loop. They have no architecture, quantization or family in common, so each needs its own investigation.

| Model | Decode | Prefill | mlxcel | Baseline |
|---|--:|--:|--:|--:|
| `qwen2.5-vl-3b-hf` | 26% | 81% | 18.5 | 71.0 |
| `gpt_bigcode-santacoder` | 28% | 50% | 51.2 | 183.4 |
| `pythia-1b` | 31% | 53% | 61.3 | 195.8 |

The Qwen VL family forms a second, milder band: `qwen2.5-vl-3b-4bit` at 59% and `qwen2-vl-2b-4bit` at 60%, with the qwen3-vl checkpoints at 83-92%. The 4-bit Qwen 2.5 VL checkpoint is more than twice as fast relative to baseline as its bf16 sibling above, so whatever the bf16 row hits is not what the 4-bit rows hit.

A single ratio understates this one. Measured on M5 Max across prompt lengths, `qwen2.5-vl-3b-4bit` loses throughput as context grows and the baseline does not:

| Prompt tokens | mlxcel | mlx-lm | Ratio |
|--:|--:|--:|--:|
| 64 | 160.5 | 225.5 | 71% |
| 128 | 156.2 | 224.4 | 70% |
| 512 | 137.8 | 218.0 | 63% |
| 2048 | 95.8 | 206.1 | 46% |

Over a 32x increase in prompt length mlxcel gives up 40% of its decode rate and mlx-lm gives up 9%. The sweep figure is one point on that curve at 512, and production contexts are longer, so this is a structural gap that widens rather than a fixed deficit.

## Vision-language models

87 VLM checkpoints, 70 measured by mlxcel and 67 by mlx-vlm, 60 by both. Prompt length comes from the image rather than being fixed, so the comparison below is restricted to the 41 models whose two prompt lengths agree within 10%; the other 19 are listed under the shape mismatch above.

| Population | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| Comparable VLM rows | 41 | 109% | 101 / 128 | 27-206% |

mlxcel is ahead of the mlx-vlm baseline on most of this set, further ahead than on text. The widest margins are `jina-vlm-mlx` at 206%, `phi-3.5-vision-4bit` at 171% and `qwen3-omni-30b-a3b-instruct-4bit` at 166%.

The bottom of the range repeats a name from the text table: `qwen2.5-vl-3b-hf` is at 27% here and 26% there. A gap that survives both modes belongs to that checkpoint rather than to either path. `mistral-small-4-119b-2603-4bit` at 36% and `paddleocr-vl-bfloat16` at 39% are the other two below half, and the 4-bit Qwen 2.5 VL sibling sits at 70% against the bf16 checkpoint's 27%.

### Failures

20 of the 87 produced no row. Three are `FAIL:image_not_applied`, where the guard rejected a measurement whose prompt never grew past its plain-text template: `llava-next-mistral-7b-4bit` (7 tokens), `bunny-llama3-8b-4bit` (20) and `fastvlm-0.5b-bf16` (27). All three are mlx-vlm side failures, and mlxcel expands the image correctly on the first of them, which is why neither runtime can be treated as the reference.

The remaining 17 are load or warmup failures. Causes established for six of them, on both hosts:

| Checkpoint | Cause |
|---|---|
| `deepseek-ocr-4bit`, `-2-4bit` | Bundled remote code targets an older transformers (`LlamaFlashAttention2`, `is_torch_fx_available`) |
| `deepseek-vl2-small-4bit` | mlx-vlm passes an `mx.array` where an int is required; strict since nanobind 2.15 (Blaizzy/mlx-vlm#2177) |
| `llama-4-scout-17b-4bit` | `attn_temperature_tuning` is `4` where transformers 5.16 requires a bool; upstream config is the same |
| `llava-1.5-7b-4bit` | `LlavaProcessor` has `patch_size=None` |
| `youtu-vl-4b-instruct` | Bundled `image_processing_siglip2_fast.py` imports a removed symbol |

### The baseline environment is not a free upgrade

Installing torch, torchvision and timm into `.venv-mlxlm` recovered `idefics2-8b-4bit` and `idefics3-8b-llama3-4bit`, which had failed on a missing torchvision image-processor backend. It also broke `llava-interleave-qwen-0.5b-bf16`, which measured cleanly before and now fails on all three attempts with `ImagesKwargs.__init__() got an unexpected keyword argument`.

It also changed preprocessing for a model that was already working: `granite-vision-3.2-2b-4bit` went from 56 prompt tokens to 1540 across the install. Rows measured on either side of a dependency change are not comparable, so the baseline CSV here is a single-environment run from an empty file rather than a resumed one.

## What these numbers do not say

### The VLM fixture measures per-request overhead, not throughput

`tests/fixtures/test_image.png` is 679 bytes, 224x224, a solid colour. Its size, not the model, sets the visual token count: 71 visual tokens on Qwen3-VL-Embedding and 264 on Llama-Nemotron-VL-Embed, against 1000-4000 for a real page scan in the same families. Read a VLM prefill figure here as the cost of a request, not as a page-processing rate.

The fixture has already moved once by a factor of 21. PR #792 (2026-07-13) changed Pixtral and Mistral3 from forced-square upscaling to aspect-preserving resize, and this image's prompt length fell from 4099 tokens to 213. The arithmetic is exact: the old path stretched 224 to a 1024 square for `(1024/16)^2 = 4096` patches, the new path leaves it at `(224/16)^2 = 196`, and the remaining 17 tokens are the chat template. `pixtral-12b-4bit` measures 213 here, confirming the post-#792 path. Any VLM comparison that spans #792 reads a preprocessing change as a performance change.

Replacing the fixture with a representative image would make every prior VLM number incomparable. If it is replaced, measure both fixtures side by side once at the switch to leave a conversion basis, and switch every host together.

### VLM parity covers 41 of the 61 models both sides measured

Decode throughput depends on context length, so a decode ratio is only meaningful when both sides ran the same prompt length. On 20 of the 61 common models they did not:

| Models | mlxcel | mlx-vlm | Ratio |
|---|--:|--:|--:|
| `granite-vision-3.2-2b-4bit` | 1543 | 56 | 27.6x |
| `idefics3-8b-llama3-4bit` | 189 | 3041 | 16.1x |
| `idefics2-8b-4bit` | 81 | 340 | 4.2x |
| `minicpm-v-4.6-bf16`, `-mxfp4` | 32 | 78 | 2.4x |
| 5 qwen3-vl checkpoints | 65 | 80 | 1.2x |
| 10 qwen3.5 / 3.6 / 3.8 checkpoints | 69 | 84 | 1.2x |

The 1.2x group is a fixed 15-token offset, which is a chat-template difference. The larger five tokenize the image differently. The parity figure below is computed over the 41 models whose prompt lengths agree within 10%; the other 20 are reported as a shape mismatch rather than a number.

### A prompt-token count cannot prove the model saw the image

The sweep rejects a row whose prompt never grew past its plain-text template (`FAIL:image_not_applied`), which catches an image that was dropped before tokenization. It cannot catch an image that was tokenized and then ignored. `granite-vision-3.2-2b-4bit` under mlxcel logs 1485 image tokens, measures a 1543-token prompt, and answers "I can't provide a description of the image as I can't see it" (lablup/mlxcel#1683, reproduced on M1 Ultra and M5 Max).

Which side is blind varies by model, so neither runtime can be assumed correct: `llava-next-mistral-7b-4bit` is the mirror case, with mlxcel expanding the image to 590 tokens while mlx-vlm reports 7 and is caught by the guard.

The check that separates these is differential, not length-based: run the model twice with two visually different images and compare the generated text. Identical output means the image contributed nothing, and the test does not depend on knowing what the fixture depicts. It costs two runs per model, so it is a post-hoc check rather than part of the sweep, and it applies in two branches. Rows both harnesses measured are selected by the prompt-length disagreement above; rows only one harness measured have no cross-comparison at all and need the differential on their own.

### The two hosts are on different mlxcel commits

M5 Max measured at `a50ff440`, M1 Ultra at `30ab5a39`, eight commits later on the same branch, both at mlxcel 0.7.0-beta.1 and MLX pin `9a795735`, both on the pp512/tg128 shape. Of those eight, one touches shared inference code: #1656, which converts bf16 weights to f16 at load on pre-Ampere CUDA and rewrites 297 lines of `src/models/sanitize.rs`. Its Metal behavior is unchanged: `cuda_f16_normalize_for_config` returns false as soon as `cuda_is_available()` is false, and the BitNet exclusion that keeps `bitnet-b1.58-2b-4t` on native bf16 moved from inside the decision function to its call site at `src/models/sanitize.rs:1726` rather than being dropped. The remaining seven are CUDA JIT serialization, harness fixes and documentation. A cross-host gap in the tables below is therefore attributable to hardware rather than to the commit difference.

## Open items

| Item | State |
|---|---|
| `qwen2.5-vl-3b-hf` at 26% text and 27% VLM | Unexplained. The gap survives both modes, so it belongs to the checkpoint rather than to either path. Its 4-bit sibling is at 70% VLM |
| `gpt_bigcode-santacoder` 28%, `pythia-1b` 31% | Unexplained. Prefill is down with decode on both (50% and 53%), so the cost is earlier than the decode loop |
| Qwen VL band, 59-92% | `qwen2.5-vl-3b-4bit` 59% and `qwen2-vl-2b-4bit` 60% on text, qwen3-vl 83-92% |
| `granite-vision-3.2-2b-4bit` refuses descriptive prompts | lablup/mlxcel#1683. The image does reach the model: it answers colour questions correctly on three different solid images. Only descriptive prompts draw a refusal, and mlx-vlm answers those |
| `minicpm-v-4.6-bf16` names the wrong colour under mlxcel | Under investigation. mlx-vlm gets it right, and the two prompt lengths differ (32 against 78) |
| `afm-4.5b` | `arcee` is not in `mlxcel list`. A coverage gap, not a defect |
| Speculative and batched-serving sweeps | Not re-run on this shape yet |

### A note on reading the low end of these tables

Neither runtime is the reference. `granite-vision` looked blind under mlxcel until a second prompt showed it was not, and `llava-next-mistral-7b-4bit` is blind under mlx-vlm while mlxcel handles it. Before attributing a low ratio to mlxcel, check that the baseline actually did the work: ask the model a question whose answer the image determines, on two images, on both sides.

## Text results

165 models, 512-token prompt, 128 generated tokens. `vs baseline` is mlx-lm 0.31.3 on the same host and day; `-` means mlx-lm did not measure that model.

| Model | Architecture | Prompt | Prefill tok/s | Decode tok/s | vs baseline |
|---|---|--:|--:|--:|--:|
| `trinity-nano-preview-4bit` | AfmoeForCausalLM | 512 | 2491.4 | 94.4 | 140% |
| `apertus-8b-instruct-2509-4bit` | ApertusForCausalLM | 512 | 529.1 | 81.8 | 100% |
| `aya-vision-8b` | AyaVisionForConditionalGeneration | 512 | 701.2 | 106.8 | - |
| `baichuan-m1-14b-4bit` | BaichuanM1ForCausalLM | 512 | 287.8 | 45.5 | 99% |
| `ling-lite-1.5` | BailingMoeForCausalLM | 512 | 1529.0 | 68.8 | 100% |
| `ring-mini-linear-2.0-4bit` | BailingMoeLinearV2ForCausalLM | 512 | 1864.1 | 151.2 | 103% |
| `bitnet-b1.58-2b-4t` | BitNetForCausalLM | 512 | 465.1 | 137.3 | 101% |
| `bitnet-b1.58-2b-4t-4bit` | BitNetForCausalLM | 512 | 460.4 | 149.3 | 102% |
| `bunny-llama3-8b-4bit` | BunnyLlamaForCausalLM | 512 | 747.1 | 103.2 | - |
| `aya-expanse-8b-4bit` | CohereForCausalLM | 512 | 696.8 | 104.9 | 95% |
| `command-r7b-4bit` | Cohere2ForCausalLM | 512 | 694.9 | 107.1 | 111% |
| `dbrx-instruct-4bit` | DbrxForCausalLM | 512 | 90.3 | 23.9 | - |
| `deepseek-vl2-small-4bit` | deepseek_vl_v2 | 512 | 532.1 | 111.7 | - |
| `deepseek-ocr-2-4bit` | DeepseekOCR2ForCausalLM | 512 | 4469.7 | 284.5 | - |
| `deepseek-ocr-4bit` | DeepseekOCRForCausalLM | 512 | 4430.4 | 278.5 | - |
| `deepseek-v2-lite-4bit` | DeepseekV2ForCausalLM | 512 | 537.1 | 110.8 | 100% |
| `diffusiongemma-26b-a4b-it-4bit` | DiffusionGemmaForBlockDiffusion | 512 | 785.4 | 66.7 | - |
| `dots.ocr-4bit` | DotsOCRForCausalLM | 512 | 2245.1 | 193.4 | - |
| `ernie-4.5-0.3b-4bit` | Ernie4_5_ForCausalLM | 512 | 7224.0 | 464.2 | - |
| `ernie-4.5-vl-28b-a3b-thinking-4bit` | Ernie4_5_VLMoeForConditionalGeneration | 512 | 840.6 | 82.9 | - |
| `exaone-3.5-2.4b-4bit` | ExaoneForCausalLM | 512 | 2097.2 | 181.5 | 98% |
| `exaone4-1.2b-4bit` | Exaone4ForCausalLM | 512 | 2591.1 | 234.5 | - |
| `falcon-h1-tiny-90m-instruct-4bit` | FalconH1ForCausalLM | 512 | 5679.1 | 330.7 | 108% |
| `falcon-mamba-7b-4bit` | FalconMambaForCausalLM | 512 | 198.5 | 71.8 | 113% |
| `falcon-ocr` | FalconOCRForCausalLM | 512 | 9792.6 | 241.2 | - |
| `florence-2-base-ft-4bit` | Florence2ForConditionalGeneration | 512 | 10215.5 | 416.8 | - |
| `florence-2-large-ft-4bit` | Florence2ForConditionalGeneration | 512 | 6275.9 | 230.0 | - |
| `gemma-2b-4bit` | GemmaForCausalLM | 512 | 2095.2 | 186.3 | 99% |
| `gemma-3-1b-it-4bit` | Gemma3ForCausalLM | 512 | 4137.0 | 222.9 | 113% |
| `gemma-3-4b-it-4bit` | Gemma3ForConditionalGeneration | 512 | 968.8 | 100.5 | 107% |
| `gemma-4-12b-it-4bit` | Gemma4UnifiedForConditionalGeneration | 512 | 331.3 | 36.5 | - |
| `gemma-4-26b-a4b-it-4bit` | Gemma4ForConditionalGeneration | 512 | 739.2 | 74.5 | 111% |
| `gemma-4-26b-a4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 512 | 749.1 | 72.9 | 108% |
| `gemma-4-31b-4bit` | Gemma4ForConditionalGeneration | 512 | 129.8 | 19.3 | 99% |
| `gemma-4-31b-it-4bit` | Gemma4ForConditionalGeneration | 512 | 130.1 | 19.2 | 99% |
| `gemma-4-31b-it-nvfp4` | Gemma4ForConditionalGeneration | 512 | 131.7 | 13.2 | - |
| `gemma-4-31b-it-qat-4bit` | Gemma4ForConditionalGeneration | 512 | 130.7 | 16.0 | 98% |
| `gemma-4-e2b-it-4bit` | Gemma4ForConditionalGeneration | 512 | 1409.8 | 112.9 | - |
| `gemma-4-e2b-it-8bit` | Gemma4ForConditionalGeneration | 512 | 1413.2 | 97.0 | - |
| `gemma-4-e2b-it-qat-4bit` | Gemma4ForConditionalGeneration | 512 | 1391.9 | 104.0 | 102% |
| `gemma-4-e4b-it-4bit` | Gemma4ForConditionalGeneration | 512 | 768.6 | 76.3 | - |
| `gemma-4-e4b-it-8bit` | Gemma4ForConditionalGeneration | 512 | 757.4 | 62.8 | - |
| `gemma-4-e4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 512 | 745.6 | 68.6 | 101% |
| `gemma2-2b-4bit` | Gemma2ForCausalLM | 512 | 1912.6 | 144.9 | 99% |
| `gemma3n-e2b-4bit` | Gemma3nForConditionalGeneration | 512 | 1359.8 | 82.2 | - |
| `gemma3n-e4b-4bit` | Gemma3nForConditionalGeneration | 512 | 761.7 | 62.3 | - |
| `gemma3n-e4b-bf16` | Gemma3nForConditionalGeneration | 512 | 928.1 | 33.6 | 89% |
| `glm4-flash-4bit` | Glm4MoeLiteForCausalLM | 512 | 718.2 | 50.3 | 105% |
| `glm-ocr-4bit` | GlmOcrForConditionalGeneration | 512 | 5250.5 | 206.1 | - |
| `gpt2` | GPT2LMHeadModel | 512 | 13868.3 | 208.7 | 96% |
| `gpt_bigcode-santacoder` | GPTBigCodeForCausalLM | 512 | 2360.1 | 51.2 | 28% |
| `pythia-1b` | GPTNeoXForCausalLM | 512 | 3102.3 | 61.3 | 31% |
| `gpt-oss-120b-4bit` | GptOssForCausalLM | 512 | 452.6 | 61.2 | 105% |
| `gpt-oss-20b-mxfp4` | GptOssForCausalLM | 512 | 787.2 | 91.7 | 102% |
| `granite-3.3-2b-instruct-4bit` | GraniteForCausalLM | 512 | 1995.7 | 172.3 | 98% |
| `granite-4.0-3b-vision-4bit` | Granite4VisionForConditionalGeneration | 512 | 1103.1 | 128.1 | - |
| `granite-4.1-3b-4bit` | GraniteForCausalLM | 512 | 1089.6 | 128.6 | 100% |
| `granite-4.1-8b-4bit` | GraniteForCausalLM | 512 | 469.2 | 71.9 | 100% |
| `granite-4.0-h-350m-4bit` | GraniteMoeHybridForCausalLM | 512 | 3511.7 | 266.2 | 100% |
| `granite-4.0-h-tiny-4bit` | GraniteMoeHybridForCausalLM | 512 | 1402.5 | 109.1 | 93% |
| `helium-1-preview-2b-4bit` | HeliumForCausalLM | 512 | 2471.0 | 196.7 | 96% |
| `hunyuan-1.8b-4bit` | HunYuanDenseV1ForCausalLM | 512 | 1965.2 | 173.1 | 94% |
| `hunyuan-13b` | HunYuanMoEV1ForCausalLM | 512 | 244.0 | 44.1 | - |
| `hunyuanocr-mlx-4bit` | HunYuanVLForConditionalGeneration | 512 | 4872.7 | 199.3 | - |
| `idefics2-8b-4bit` | Idefics2ForConditionalGeneration | 512 | 786.9 | 109.0 | - |
| `idefics3-8b-llama3-4bit` | Idefics3ForConditionalGeneration | 512 | 749.4 | 104.5 | - |
| `internlm2-7b-4bit` | InternLM2ForCausalLM | 512 | 764.2 | 105.4 | 99% |
| `internlm3-8b-4bit` | InternLM3ForCausalLM | 512 | 671.4 | 84.2 | - |
| `internvl3-1b` | InternVLChatModel | 512 | 6982.6 | 331.9 | - |
| `iquest-coder-v1-7b-instruct-8bit` | IQuestCoderForCausalLM | 512 | 563.6 | 70.7 | 100% |
| `jamba-v0.1-4bit` | JambaForCausalLM | 512 | 213.9 | 131.0 | 100% |
| `jina-vlm-mlx` | JinaVLMForConditionalGeneration | 512 | 2478.9 | 168.4 | - |
| `klear-46b-a2.5b-instruct-4bit` | KlearMoeForCausalLM | 512 | 963.7 | 93.1 | 123% |
| `lfm2-350m-8bit` | Lfm2ForCausalLM | 512 | 7711.8 | 568.1 | 105% |
| `lfm2-8b-a1b-4bit` | Lfm2MoeForCausalLM | 512 | 2128.4 | 195.2 | 106% |
| `lfm2-vl-450m-4bit` | Lfm2VlForConditionalGeneration | 512 | 8722.9 | 574.8 | - |
| `llada2.0-mini-preview-4bit` | LLaDA2MoeModelLM | 512 | 2120.1 | 147.4 | - |
| `deepseek-coder-1.3b-4bit` | LlamaForCausalLM | 512 | 3739.6 | 146.1 | - |
| `llama-3.1-8b-4bit` | LlamaForCausalLM | 512 | 750.9 | 105.0 | 99% |
| `llama-3.1-8b-bf16` | LlamaForCausalLM | 512 | 810.5 | 36.1 | 102% |
| `llama-3.2-1b-4bit` | LlamaForCausalLM | 512 | 4140.0 | 402.9 | 101% |
| `llama-3.2-1b-instruct` | LlamaForCausalLM | 512 | 4585.6 | 189.1 | 105% |
| `llama-4-scout-17b-4bit` | Llama4ForConditionalGeneration | 512 | 281.6 | 35.2 | - |
| `minicpm-2b-4bit` | LlamaForCausalLM | 512 | 1906.4 | 148.3 | 100% |
| `smollm-135m-4bit` | LlamaForCausalLM | 512 | 12605.1 | 370.7 | 114% |
| `llava-1.5-7b-4bit` | LlavaForConditionalGeneration | 512 | 848.9 | 106.9 | - |
| `llava-interleave-qwen-0.5b-bf16` | LlavaForConditionalGeneration | 512 | 7988.8 | 282.0 | - |
| `pixtral-12b-4bit` | LlavaForConditionalGeneration | 512 | 490.6 | 67.8 | 101% |
| `granite-vision-3.2-2b-4bit` | LlavaNextForConditionalGeneration | 512 | 1448.1 | 150.2 | - |
| `llava-next-mistral-7b-4bit` | LlavaNextForConditionalGeneration | 512 | 789.3 | 109.4 | - |
| `fastvlm-0.5b-bf16` | LlavaQwen2ForCausalLM | 512 | 7681.3 | 266.2 | - |
| `mamba2-1.3b-4bit` | mamba2 | 512 | 2405.5 | 101.0 | - |
| `mamba2-130m` | mamba2 | 512 | 9828.6 | 213.4 | - |
| `mellum2-12b-a2.5b-base` | MellumForCausalLM | 512 | 1675.1 | 76.0 | - |
| `mimo-7b-4bit` | MiMoForCausalLM | 512 | 569.8 | 82.8 | 100% |
| `minicpm3-4b-4bit` | MiniCPM3ForCausalLM | 512 | 1191.8 | 77.5 | 109% |
| `minicpm-v-4.6-bf16` | MiniCPMV4_6ForConditionalGeneration | 512 | 4972.9 | 206.5 | - |
| `minicpm-v-4.6-mxfp4` | MiniCPMV4_6ForConditionalGeneration | 512 | 3919.0 | 217.3 | - |
| `ministral-3b-4bit` | Mistral3ForConditionalGeneration | 512 | 1120.9 | 153.0 | 102% |
| `mistral-small-3.1-24b-4bit` | Mistral3ForConditionalGeneration | 512 | 179.0 | 31.4 | 100% |
| `mistral-small-4-119b-2603-4bit` | Mistral3ForConditionalGeneration | 512 | 382.2 | 18.9 | - |
| `mixtral-8x7b-4bit` | MixtralForCausalLM | 512 | 333.4 | 54.5 | 100% |
| `molmo-7b` | MolmoForCausalLM | 512 | 789.8 | 108.9 | - |
| `molmo2-4b` | Molmo2ForConditionalGeneration | 512 | 1072.9 | 91.1 | - |
| `nemotron-3-nano-omni-30b-a3b-reasoning-4bit` | NemotronH_Nano_Omni_Reasoning_V3 | 512 | 349.9 | 96.0 | - |
| `nemotron-h-30b-4bit` | NemotronHForCausalLM | 512 | 349.2 | 96.1 | 103% |
| `olmo2-7b-4bit` | Olmo2ForCausalLM | 512 | 814.6 | 102.3 | 100% |
| `olmo3-32b-4bit` | Olmo3ForCausalLM | 512 | 129.0 | 21.6 | 101% |
| `olmo-1b-4bit` | OlmoModelForCausalLM | 512 | 3505.6 | 184.5 | - |
| `openelm-1_1b-instruct-4bit` | OpenELMForCausalLM | 512 | 4150.9 | 278.9 | - |
| `paddleocr-vl-bfloat16` | PaddleOCRVLForConditionalGeneration | 512 | 6875.6 | 118.4 | - |
| `paligemma2-3b-6bit` | PaliGemmaForConditionalGeneration | 512 | 1814.3 | 129.9 | - |
| `phi-2-4bit` | PhiForCausalLM | 512 | 1656.9 | 128.6 | - |
| `phi-3-mini-4bit` | Phi3ForCausalLM | 512 | 1446.2 | 151.4 | 99% |
| `phi-3-small-8k-instruct-aq4_64` | Phi3SmallForCausalLM | 512 | 747.9 | 98.6 | - |
| `phi-3.5-mini-4bit` | Phi3ForCausalLM | 512 | 1435.7 | 145.7 | 98% |
| `phi-3.5-mini-bf16` | Phi3ForCausalLM | 512 | 1553.8 | 61.9 | 99% |
| `phi-3.5-mini-instruct-hf` | Phi3ForCausalLM | 512 | 1556.3 | 61.9 | 99% |
| `phi-3.5-vision-4bit` | Phi3VForCausalLM | 512 | 1435.0 | 144.2 | - |
| `phi-4-4bit` | Phi3ForCausalLM | 512 | 405.6 | 57.6 | 100% |
| `phixtral-4x2_8-4bit` | PhiForCausalLM | 512 | 926.7 | 73.1 | 78% |
| `phi-3.5-moe-4bit` | PhiMoEForCausalLM | 512 | 602.1 | 75.1 | 110% |
| `plamo-2-1b` | PlamoForCausalLM | 512 | 2961.3 | 107.5 | - |
| `deepseek-r1-distill-7b-4bit` | Qwen2ForCausalLM | 512 | 788.5 | 107.1 | 100% |
| `qwen1.5-moe-a2.7b-4bit` | Qwen2MoeForCausalLM | 512 | 1684.3 | 143.7 | 107% |
| `qwen2-0.5b` | Qwen2ForCausalLM | 512 | 6816.2 | 331.6 | 115% |
| `qwen2-vl-2b-4bit` | Qwen2VLForConditionalGeneration | 512 | 2537.8 | 131.1 | 60% |
| `qwen2.5-0.5b-bf16` | Qwen2ForCausalLM | 512 | 7548.1 | 274.3 | 109% |
| `qwen2.5-1.5b-4bit` | Qwen2ForCausalLM | 512 | 2887.8 | 230.2 | 105% |
| `qwen2.5-1.5b-instruct-4bit` | Qwen2ForCausalLM | 512 | 2910.1 | 223.8 | 102% |
| `qwen2.5-7b-4bit` | Qwen2ForCausalLM | 512 | 779.2 | 105.3 | 99% |
| `qwen2.5-7b-8bit` | Qwen2ForCausalLM | 512 | 783.5 | 67.5 | 97% |
| `qwen2.5-vl-3b-4bit` | Qwen2_5_VLForConditionalGeneration | 512 | 1397.1 | 89.7 | 59% |
| `qwen2.5-vl-3b-hf` | Qwen2_5_VLForConditionalGeneration | 512 | 1265.0 | 18.5 | 26% |
| `qwen3-0.6b-4bit` | Qwen3ForCausalLM | 512 | 4754.1 | 249.9 | 105% |
| `qwen3-1.7b-4bit` | Qwen3ForCausalLM | 512 | 2037.4 | 185.3 | 98% |
| `qwen3-30b-a3b-4bit` | Qwen3MoeForCausalLM | 512 | 858.4 | 82.3 | 125% |
| `qwen3-4b-4bit` | Qwen3ForCausalLM | 512 | 942.3 | 116.9 | 101% |
| `qwen3-8b-4bit` | Qwen3ForCausalLM | 512 | 518.0 | 81.2 | 101% |
| `qwen3-next-80b-a3b-instruct-4bit` | Qwen3NextForCausalLM | 512 | 615.1 | 60.0 | 118% |
| `qwen3-omni-30b-a3b-instruct-4bit` | Qwen3OmniMoeForConditionalGeneration | 512 | 845.4 | 70.3 | - |
| `qwen3-vl-2b-4bit` | Qwen3VLForConditionalGeneration | 512 | 2036.4 | 170.7 | 90% |
| `qwen3-vl-30b-a3b-4bit` | Qwen3VLMoeForConditionalGeneration | 512 | 837.9 | 70.8 | 107% |
| `qwen3-vl-32b-4bit` | Qwen3VLForConditionalGeneration | 512 | 125.6 | 19.5 | 92% |
| `qwen3-vl-4b-4bit` | Qwen3VLForConditionalGeneration | 512 | 945.3 | 95.9 | 83% |
| `qwen3-vl-8b-4bit` | Qwen3VLForConditionalGeneration | 512 | 519.1 | 70.9 | 88% |
| `qwen3.5-0.8b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 3779.6 | 279.3 | 102% |
| `qwen3.5-0.8b-optiq-4bit` | Qwen3_5ForConditionalGeneration | 512 | 3749.2 | 262.0 | 103% |
| `qwen3.5-27b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 152.7 | 25.1 | 97% |
| `qwen3.5-2b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 1831.4 | 202.9 | 99% |
| `qwen3.5-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 512 | 839.6 | 82.3 | 109% |
| `qwen3.5-4b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 851.4 | 112.0 | 98% |
| `qwen3.5-9b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 483.9 | 75.6 | 97% |
| `qwen3.5-9b-bf16` | Qwen3_5ForConditionalGeneration | 512 | 753.3 | 31.5 | 94% |
| `qwen3.6-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 512 | 842.6 | 81.5 | 109% |
| `qwen3.8-27b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 153.3 | 25.1 | 98% |
| `qwen3.8-27b-hf-bf16` | Qwen3_5ForConditionalGeneration | 512 | 219.2 | 10.3 | 97% |
| `seed-oss-36b-instruct-4bit` | SeedOssForCausalLM | 512 | 115.4 | 19.3 | 99% |
| `smollm3-3b-4bit` | SmolLM3ForCausalLM | 512 | 1210.9 | 128.4 | 97% |
| `solar-open-100b-4bit` | SolarOpenForCausalLM | 512 | 255.6 | 35.7 | 103% |
| `stablelm-1.6b-4bit` | StableLmForCausalLM | 512 | 3287.0 | 245.4 | 115% |
| `starcoder2-3b-4bit` | Starcoder2ForCausalLM | 512 | 1699.8 | 158.0 | 100% |
| `telechat3-36b-thinking-4bit` | Telechat3ForCausalLM | 512 | 114.8 | 20.1 | 101% |
| `youtu-llm-2b-4bit` | YoutuForCausalLM | 512 | 1728.8 | 136.3 | 92% |
| `youtu-vl-4b-instruct` | YoutuVLForConditionalGeneration | 512 | 1088.5 | 43.6 | - |

## VLM results

70 models, prompt length set by the image, 128 generated tokens. `vs baseline` is mlx-vlm 0.6.17; `shape` means the two harnesses used prompt lengths differing by more than 10%, which makes a decode ratio meaningless; `-` means mlx-vlm did not measure that model.

| Model | Architecture | Prompt | Prefill tok/s | Decode tok/s | vs baseline |
|---|---|--:|--:|--:|--:|
| `aya-vision-8b` | AyaVisionForConditionalGeneration | 735 | 638.1 | 110.4 | 107% |
| `bunny-llama3-8b-4bit` | BunnyLlamaForCausalLM | 746 | 672.5 | 101.0 | - |
| `deepseek-vl2-small-4bit` | deepseek_vl_v2 | 494 | 438.9 | 111.2 | - |
| `deepseek-ocr-2-4bit` | DeepseekOCR2ForCausalLM | 409 | 839.0 | 284.5 | - |
| `deepseek-ocr-4bit` | DeepseekOCRForCausalLM | 281 | 884.2 | 291.0 | - |
| `dots.ocr-4bit` | DotsOCRForCausalLM | 74 | 443.6 | 220.6 | 114% |
| `ernie-4.5-vl-28b-a3b-thinking-4bit` | Ernie4_5_VLMoeForConditionalGeneration | 108 | 290.3 | 91.4 | 124% |
| `gemma-3-4b-it-4bit` | Gemma3ForConditionalGeneration | 275 | 254.6 | 106.2 | 112% |
| `gemma-4-12b-it-4bit` | Gemma4UnifiedForConditionalGeneration | 277 | 299.5 | 38.1 | 102% |
| `gemma-4-26b-a4b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 316.7 | 79.2 | 118% |
| `gemma-4-26b-a4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 277 | 318.9 | 77.7 | 114% |
| `gemma-4-31b-4bit` | Gemma4ForConditionalGeneration | 265 | 97.8 | 19.8 | 101% |
| `gemma-4-31b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 102.4 | 19.8 | 101% |
| `gemma-4-31b-it-nvfp4` | Gemma4ForConditionalGeneration | 278 | 99.9 | 13.5 | 100% |
| `gemma-4-31b-it-qat-4bit` | Gemma4ForConditionalGeneration | 277 | 102.4 | 16.8 | 102% |
| `gemma-4-e2b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 818.4 | 114.0 | 110% |
| `gemma-4-e2b-it-8bit` | Gemma4ForConditionalGeneration | 277 | 796.9 | 98.5 | 101% |
| `gemma-4-e2b-it-qat-4bit` | Gemma4ForConditionalGeneration | 273 | 789.6 | 105.6 | 108% |
| `gemma-4-e4b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 522.5 | 79.7 | 106% |
| `gemma-4-e4b-it-8bit` | Gemma4ForConditionalGeneration | 277 | 518.4 | 65.8 | 99% |
| `gemma-4-e4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 273 | 515.1 | 70.5 | 104% |
| `gemma3n-e2b-4bit` | Gemma3nForConditionalGeneration | 273 | 818.7 | 84.5 | 138% |
| `gemma3n-e4b-4bit` | Gemma3nForConditionalGeneration | 273 | 533.0 | 65.2 | 135% |
| `gemma3n-e4b-bf16` | Gemma3nForConditionalGeneration | 273 | 640.4 | 35.2 | 98% |
| `glm-ocr-4bit` | GlmOcrForConditionalGeneration | 82 | 991.5 | 284.7 | 79% |
| `granite-4.0-3b-vision-4bit` | Granite4VisionForConditionalGeneration | 337 | 739.3 | 131.6 | 112% |
| `hunyuanocr-mlx-4bit` | HunYuanVLForConditionalGeneration | 284 | 1191.0 | 212.6 | 131% |
| `idefics2-8b-4bit` | Idefics2ForConditionalGeneration | 81 | 275.1 | 114.6 | shape |
| `idefics3-8b-llama3-4bit` | Idefics3ForConditionalGeneration | 189 | 500.0 | 107.8 | shape |
| `internvl3-1b` | InternVLChatModel | 293 | 1863.3 | 354.2 | 130% |
| `jina-vlm-mlx` | JinaVLMForConditionalGeneration | 436 | 1174.8 | 176.5 | 206% |
| `lfm2-vl-450m-4bit` | Lfm2VlForConditionalGeneration | 82 | 1579.3 | 626.2 | 127% |
| `llama-4-scout-17b-4bit` | Llama4ForConditionalGeneration | 162 | 154.6 | 36.4 | - |
| `llava-1.5-7b-4bit` | LlavaForConditionalGeneration | 594 | 745.4 | 103.7 | - |
| `llava-interleave-qwen-0.5b-bf16` | LlavaForConditionalGeneration | 744 | 4136.9 | 266.1 | - |
| `pixtral-12b-4bit` | LlavaForConditionalGeneration | 213 | 402.1 | 68.7 | 103% |
| `granite-vision-3.2-2b-4bit` | LlavaNextForConditionalGeneration | 1543 | 1253.8 | 131.1 | 112% |
| `llava-next-mistral-7b-4bit` | LlavaNextForConditionalGeneration | 590 | 700.0 | 106.8 | - |
| `fastvlm-0.5b-bf16` | LlavaQwen2ForCausalLM | 282 | 1590.8 | 283.7 | - |
| `minicpm-v-4.6-bf16` | MiniCPMV4_6ForConditionalGeneration | 32 | 355.9 | 209.7 | shape |
| `minicpm-v-4.6-mxfp4` | MiniCPMV4_6ForConditionalGeneration | 32 | 340.1 | 231.6 | shape |
| `ministral-3b-4bit` | Mistral3ForConditionalGeneration | 613 | 981.0 | 148.7 | 109% |
| `mistral-small-3.1-24b-4bit` | Mistral3ForConditionalGeneration | 253 | 165.4 | 31.7 | 101% |
| `mistral-small-4-119b-2603-4bit` | Mistral3ForConditionalGeneration | 93 | 191.3 | 19.2 | 36% |
| `molmo-7b` | MolmoForCausalLM | 327 | 578.8 | 109.5 | 140% |
| `molmo2-4b` | Molmo2ForConditionalGeneration | 438 | 690.1 | 90.7 | 150% |
| `nemotron-3-nano-omni-30b-a3b-reasoning-4bit` | NemotronH_Nano_Omni_Reasoning_V3 | 279 | 268.6 | 96.5 | 113% |
| `paddleocr-vl-bfloat16` | PaddleOCRVLForConditionalGeneration | 212 | 1206.9 | 125.6 | 39% |
| `paligemma2-3b-6bit` | PaliGemmaForConditionalGeneration | 1032 | 1432.9 | 110.2 | 153% |
| `phi-3.5-vision-4bit` | Phi3VForCausalLM | 773 | 993.5 | 136.1 | 171% |
| `qwen2-vl-2b-4bit` | Qwen2VLForConditionalGeneration | 91 | 783.0 | 155.4 | 70% |
| `qwen2.5-vl-3b-4bit` | Qwen2_5_VLForConditionalGeneration | 91 | 570.7 | 106.4 | 70% |
| `qwen2.5-vl-3b-hf` | Qwen2_5_VLForConditionalGeneration | 91 | 415.8 | 19.2 | 27% |
| `qwen3-omni-30b-a3b-instruct-4bit` | Qwen3OmniMoeForConditionalGeneration | 69 | 280.8 | 43.0 | 166% |
| `qwen3-vl-2b-4bit` | Qwen3VLForConditionalGeneration | 65 | 678.5 | 181.2 | shape |
| `qwen3-vl-30b-a3b-4bit` | Qwen3VLMoeForConditionalGeneration | 65 | 258.5 | 42.8 | shape |
| `qwen3-vl-32b-4bit` | Qwen3VLForConditionalGeneration | 65 | 88.3 | 18.5 | shape |
| `qwen3-vl-4b-4bit` | Qwen3VLForConditionalGeneration | 65 | 448.7 | 97.0 | shape |
| `qwen3-vl-8b-4bit` | Qwen3VLForConditionalGeneration | 65 | 299.4 | 68.8 | shape |
| `qwen3.5-0.8b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 878.2 | 294.6 | shape |
| `qwen3.5-27b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 87.2 | 25.1 | shape |
| `qwen3.5-2b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 553.0 | 206.2 | shape |
| `qwen3.5-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 69 | 280.5 | 85.1 | shape |
| `qwen3.5-4b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 344.5 | 111.9 | shape |
| `qwen3.5-9b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 224.9 | 75.6 | shape |
| `qwen3.5-9b-bf16` | Qwen3_5ForConditionalGeneration | 69 | 265.6 | 31.9 | shape |
| `qwen3.6-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 69 | 283.4 | 83.4 | shape |
| `qwen3.8-27b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 88.7 | 25.2 | shape |
| `qwen3.8-27b-hf-bf16` | Qwen3_5ForConditionalGeneration | 69 | 84.5 | 10.2 | shape |
| `youtu-vl-4b-instruct` | YoutuVLForConditionalGeneration | 28 | 216.7 | 46.1 | - |

## Benchmark families measured on their own conditions

The sections below come from separate sweeps and are not part of the pp512/tg128 campaign above. Each states its own date, build and reproducer; do not read them against the tables above.

## Tokenizer Support

| Format | File | Models | Crate |
|--------|------|--------|-------|
| HuggingFace | `tokenizer.json` | Most models | `tokenizers` |
| SentencePiece | `tokenizer.model` | Gemma, Llama 1/2, older models | `sentencepiece` |
| Tiktoken | `*.tiktoken` | HunYuan MoE (13B) | Custom BPE (fancy-regex) |

## TurboQuant KV cache: M1 Ultra speed gate readings

First dedicated M1 Ultra reading of the TurboQuant KV speed gate matrix.
Hardware: Apple M1 Ultra, 128 GB unified memory. Model:
`Meta-Llama-3.1-8B-Instruct-4bit`. Date: 2026-04-29. Reproducer:
`./scripts/bench_kv_cache.sh --modes fp16,int8,turbo4-asym,turbo4,turbo4-delegated --contexts 4096 --prefill-contexts 8192`.

Full CSV at `benchmarks/turbo_kv/2026-04-29_Apple_M1_Ultra_Meta-Llama-3.1-8B-Instruct-4bit.csv`.

> **CSV schema note:** Rows where `stage=prefill` record a single-token follow-up to force the KV
> cache to be populated. The resulting `decode_tok_s` value (e.g. 303766 tok/s) reflects a
> sub-millisecond single-token step and is not a meaningful decode throughput figure; ignore it
> for prefill rows. Use `prefill_tok_s` from those rows and `decode_tok_s` from `stage=decode` rows.

### Decode @ 4K context (80 generated tokens)

| Mode | Decode tok/s | × FP16 | M5 Max gate (tracking on M1U) |
|------|--------------|--------|------|
| `fp16`             | 90.36 | 1.000× | baseline |
| `int8`             | 61.24 | 0.678× | (no gate; tracking only) |
| `turbo4-asym`      |  3.92 | 0.043× | ≥0.97× → off-target on M1U |
| `turbo4`           | 16.34 | 0.181× | ≥0.93× → off-target on M1U |
| `turbo4-delegated` | 18.22 | 0.202× | ≥0.97× → off-target on M1U |

`turbo4-asym` produced only 51 tokens before early EOS (vs 80 requested);
the per-token decode rate is computed over those 51 tokens. The other Turbo
modes ran the full 80 tokens.

### Prefill @ 8K context (single-token decode follow-up)

| Mode | Prefill tok/s | × FP16 | M5 Max gate (tracking on M1U) |
|------|---------------|--------|------|
| `fp16`             | 678.28 | 1.000× | baseline |
| `int8`             | 676.82 | 0.998× | (no gate) |
| `turbo4-asym`      | 471.12 | 0.694× | ≥1.00× → off-target on M1U |
| `turbo4`           | 365.07 | 0.538× | ≥1.00× → off-target on M1U |
| `turbo4-delegated` | 678.96 | 1.001× | best-effort → meets target |

### M1 Ultra reading

 §"Cross-hardware regression", the M5 Max gates are tracking
only on pre-M3 hardware. The numbers above match that guidance: `turbo4`
and `turbo4-asym` decode are well below the M5 gates on M1 Ultra, while
`turbo4-delegated` prefill is bit-identical to FP16 because the cold pages
keep their FP16 representation and only the hot tail is packed.

The decode regression is consistent with the L2-cache wall documented in
https://github.com/TheTom/turboquant_plus. The fused Sparse-V Metal kernel that lands
 targets the per-thread skip path inside the SDPA inner loop and is
expected to recover most of the M5 decode budget; the M1/M2 ceiling stays
limited by L2 bandwidth and is documented but not gated.

### M1 Ultra hardware considerations

- **`turbo4-delegated` is the only currently shipping mode that holds the
  prefill gate on M1 Ultra.** Use it when prefill latency matters more than
  the maximum compression ratio.
- **`turbo4-asym` on M1 Ultra needs the fused kernel** to be a viable
  decode option. The graph-level path measured here pays a per-token
  dequant cost that the fused kernel folds into the SDPA inner loop.
- **Avoid `turbo3` on M1 Ultra** for decode-bound workloads. The 3-bit
  pack/unpack loop saturates L2 bandwidth on the older GPU microarchitecture;
  on M3/M4/M5 the regression is smaller or absent.
- Memory-vs-speed trade-off on M1 Ultra: `fp16+turbo4` (alias `turbo4-asym`)
  still delivers approximately 0.39× of the FP16 baseline KV footprint at the
  default boundary-v 2 on a 32-layer model, but the decode shortfall above
  makes `turbo4-delegated` the better practical choice on this generation
  until the fused kernel lands. Pick `fp16+turbo4` only if the memory
  savings outweigh the per-token decode cost for your workload.

### Deferred

- 32K decode reading (best-effort per epic; benefits fused kernel before the gate is meaningful on any hardware).
- Per-mode 16K decode (skipped for the initial run; M5 Max is the primary
  target for the 16K gate).
- Multi-model expansion (Qwen 2.5, Gemma 3, etc.).
- M5 Max readings of the same matrix: to be filled in by a manual run on
  the M5 Max dev box; the script is hardware-agnostic and writes
  `benchmarks/turbo_kv/<date>_<hw>_<model>.csv` keyed off
  `sysctl -n machdep.cpu.brand_string`.

## Batched serving (B = 1/2/4)

Source: `benchmarks/metal_m1ultra_batch_2026-09-04.csv` (main at `bf1cdb72`, the before-baseline) and `benchmarks/metal_m1ultra_batch_2026-09-04_pr1616.csv` (the #1616 code PR), both produced by `scripts/bench_serving_concurrency.py --prompt-tokens 512 --max-tokens 128 --concurrency 1,2,4` against one fresh `mlxcel-server -m models/mlx/<name> --parallel 4 --max-batch-prefill 4` per model, levels ascending, Time Machine stopped. Under continuous batching N concurrent streaming clients occupy N decode slots, so the concurrency level is the effective decode batch size. This is the M1 Ultra counterpart of the M5 Max section in [model_tests_m5max.md](model_tests_m5max.md); the attribution behind it is [moe-batched-decode-m1ultra-2026-09-04.md](moe-batched-decode-m1ultra-2026-09-04.md).

**Reading the TTFT column.** Levels run ascending and share the prompt, so the dense llama rows adopt the prompt cache from B=2 on and TTFT falls. The MoE rows do not get that help: dense (non-paged) prompt-cache entries are consumed on adoption, so only as many rows adopt as entries were donated by the previous level and the rest pay a cold 440-token prefill serialized ahead of the first decode tick (one at B=2, two at B=4). That, not the decode tick, is the MoE TTFT column. A second full before-pass reproduced every B=4 aggregate within 0.3%.

### qwen2.5-0.5b-bf16 (small dense bf16, no `forward_batched` override)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 90.5 | 90.5 | 253.2 | 216.2 | 1.00x |
| 2 | 2 / 0 | 25.0 | 31.5 | 87.9 | 174.2 | 0.81x |
| 4 | 4 / 0 | 39.8 | 59.8 | 93.9 | 367.5 | **1.70x** |

After the #1616 PR (untouched family): 209.2 / 202.8 / 369.5 aggregate at B=1/2/4.

### llama-3.1-8b-4bit (canonical dense 4-bit, real batched forward)

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 773.7 | 773.7 | 86.8 | 57.2 | 1.00x |
| 2 | 2 / 0 | 100.2 | 132.5 | 54.7 | 105.6 | 1.85x |
| 4 | 4 / 0 | 167.6 | 264.3 | 31.1 | 120.5 | **2.11x** |

After the #1616 PR (untouched family): 56.8 / 106.2 / 118.9 aggregate at B=1/2/4.

### qwen3-30b-a3b-4bit (MoE), before: per-row single-token forwards

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 799.3 | 799.3 | 81.5 | 54.3 | 1.00x |
| 2 | 2 / 0 | 611.2 | 611.4 | 55.2 | 88.0 | 1.62x |
| 4 | 4 / 0 | 1223.8 | 1224.2 | 29.8 | 93.3 | **1.72x** |

### qwen3-30b-a3b-4bit (MoE), after the #1616 PR: batched forward, experts on `gather_qmm` from B=2

| B | ok/fail | TTFT mean (ms) | TTFT p95 (ms) | decode tok/s per request | aggregate tok/s | scaling vs B=1 |
|---|---------|----------------|---------------|--------------------------|-----------------|----------------|
| 1 | 1 / 0 | 745.4 | 745.4 | 81.0 | 55.3 | 1.00x |
| 2 | 2 / 0 | 608.8 | 608.9 | 52.9 | 85.0 | 1.54x |
| 4 | 4 / 0 | 1232.6 | 1232.9 | 33.9 | 102.9 | **1.86x** |

### Reading

On this host the dense-versus-MoE gap the M5 Max numbers show is much narrower: llama-3.1-8b reaches 2.11x rather than 3.25x, and the MoE model 1.72x rather than 1.55x. Both `qwen2.5-0.5b` and, before the PR, `qwen3-30b-a3b` ran the `LanguageModel` default at B>=2 (the single-sequence `forward` once per row, evaluated together), so neither was on a batched kernel path; the MoE model's B=4 tick was four serialized single-token graphs that still took the fused MoE kernel, overlapping by about 30%. The PR gives `Qwen3MoeModel` a real batched forward, which lifts B=4 aggregate by 10.3% (93.3 to 102.9) and per-request decode by 13.8% (29.8 to 33.9) while leaving B=1 and the dense rows inside the run-to-run band; B=2 is unchanged within noise because at 16 expert slots `gather_qmm` and two fused launches cost about the same. The batched fused MoE kernel the issue proposed was prototyped at the op level and is slower than `gather_qmm` from n=4, so it was not built; details, including why expert-plane deduplication would buy nothing here, are in the attribution document.

No requests failed at any level for any model (0 fail across all cells, both passes).

