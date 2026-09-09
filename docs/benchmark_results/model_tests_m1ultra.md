# Model Compatibility and Performance Tests (M1 Ultra)

Compatibility and single-stream performance for mlxcel on **Mac Studio M1 Ultra 128GB**, measured against the Python mlx-lm and mlx-vlm baselines on the same host, the same day, and the same prompt shape.

The VLM tables come from a single 2026-09-09 pass at `f85898eb`, paired with a same-day mlx-vlm 0.6.17 baseline. The text tables come from the 2026-09-06 through 2026-09-09 sweeps, with 18 rows re-measured on 2026-09-08 after lablup/mlxcel#1709 and #1710 and six checkpoints taken singly on 2026-09-09 (see "Newly measured checkpoints"). Earlier sweeps used a different measurement shape and are not comparable, so they are not carried forward; the CSVs under `benchmarks/` remain the record of what was measured when.

## Test environment

| Item | Value |
|------|-------|
| Hardware | Mac Studio M1 Ultra, 128GB unified memory |
| OS | macOS 26.6.2 |
| mlxcel version | 0.7.0-beta.1 |
| mlxcel commit | `f85898eb` for every VLM row. Text rows carry `5287eb9a2` for the 16 refreshed entries and `255203e51` / `ec414719f` for rows retaken after it; those three differ only in `scripts/bench_decode.sh` and the Gemma3n load policy |
| MLX C++ pin | `9a795735` |
| Build | `cargo build --release --features metal,accelerate` |
| mlxcel harness | `mlxcel-bench-decode` (load, warmup and measured pass in one process) |
| mlx-lm baseline | 0.31.3 |
| mlx-vlm baseline | 0.6.17 |
| Baseline stack | mlx 0.32.2, transformers 5.16.1, torch 2.14.0, torchvision 0.29.0, timm 1.0.29, numba 0.67.0 |
| Model store | `models/mlx/` and `models/mlx-big/` |
| CSVs | VLM: `metal_m1ultra_vlm_2026-09-09.csv` and `pylm_m1ultra_vlm_2026-09-09_resweep.csv`. Text: `metal_m1ultra_2026-09-08.csv` and `pylm_m1ultra_2026-09-06.csv`, plus the six `metal_m1ultra_2026-09-09_single_*.csv` and `metal_m1ultra_vlm_2026-09-09_single_*.csv` files named under "Newly measured checkpoints". The superseded VLM pass is kept as `metal_m1ultra_vlm_2026-09-08.csv` and `pylm_m1ultra_vlm_2026-09-07.csv` |

### The model store is `models/mlx/` and `models/mlx-big/`, and only those

Checkpoints on this host live under two in-repository roots. `models/mlx/` holds 209 of them and is where every row in the tables below comes from. `models/mlx-big/` holds the five largest: `qwen3-coder-480b-a35b-instruct-4bit` (252 GB), `mimo-v2-flash-4bit` (162 GB), `deepseek-v3-0324-4bit` (100 GB), `minimax-m2-3bit` (94 GB) and `dots.llm1.inst-mixed-4-6bit` (81 GB). Setting aside the duplicate noted below, the largest entry left in `models/mlx/` is `dbrx-instruct-4bit` at 70 GB, so the split sits at roughly 80 GB. It is a placement decision, not a rule the tooling enforces. A third store exists at `~/.cache/mlxcel/models/` holding 12 more checkpoints, and that one is deliberately out of scope: its path and contents differ per machine, so a table assembled from it cannot be compared against another host's.

`dots.llm1.inst-mixed-4-6bit` is currently present in both roots as two independent copies, not links (different inodes, 81 GB each). The `models/mlx/` copy is the one the sweeps read. Reclaiming the other 81 GB is worth doing but is left alone here so that no table row changes underneath a release measurement.

The Coverage table below still counts `large_models` among the container directories it skipped, because that is the layout the 2026-09-06 and 2026-09-08 sweeps walked. No such directory exists now. The sweep record is left as it ran rather than back-dated to the current layout, so read that row as history.

That boundary has to be stated because getting it wrong is silent, and a one-root scan is the specific way it goes wrong. Reading only `models/mlx/` reports all five of the oversized checkpoints as absent, and the absence reads as a fact about the project rather than about the scan; `model_tests_m5max.md` lists `dots.llm1.inst-mixed-4-6bit` and `MiniMax-M2-3bit` as "not present on M1 Ultra" for exactly that reason, and the first of those has had a committed M1 Ultra CSV since 2026-09-08. Three related traps sit next to it: a checkpoint may instead sit in the per-machine cache, `models` is itself a symlink so a scan that does not resolve links can miss the whole tree, and a `model_type` is not always spelled the way the Rust module is (`nemotron-nas` in a config against `nemotron_nas` in `src/models/`). Each of those turned a real checkpoint into a false negative during the 2026-09-08 audit. Match checkpoints by basename across both roots rather than by a literal path: the two hosts do not share a layout, and M5 Max records `model_path` as `models/<name>` where this host records `models/mlx/<name>`.

## Measurement shape

Text rows follow the llama-bench `pp512/tg128` convention: a 512-token synthetic prompt, exactly 128 generated tokens, EOS suppressed on both sides so a model that would answer in twelve words is still timed over 128 tokens. Without that suppression the decode figure is an average over whatever length the model chose, which is a property of the model's verbosity rather than of the runtime, and it is not comparable across models.

VLM rows keep the 128-token generation but take their prompt length from the image, since the image fixes it. Prompt lengths there run from 8 to 1543 tokens depending on the tower's patch count.

Time Machine is stopped for the duration (`tmutil stopbackup`) and models run with a 30 second cooldown between them.

### What the harness refuses to record

Three classes of row look like measurements but are not, and each is now rejected by the sweep rather than left for a reader to notice:

- **A checkpoint with no vision tower in a `--vlm` sweep.** mlx-vlm loads one without complaint and silently drops the image, so the row is a text generation wearing a VLM label. Both harnesses now run `scripts/vlm_detect.py` and cover the same model set.
- **A VLM whose image never reached the prompt.** The child reports the token count of the formatted prompt as plain text; an image that arrived always expands the prompt past it, so `prompt_tokens <= text_only_prompt_tokens` is recorded as `FAIL:image_not_applied`. Three checkpoints here were affected, and `llava-next-mistral-7b-4bit` reported the same 7 tokens on both M1 Ultra and M5 Max across three runs.
- **A duplicate checkpoint.** Identity is `sha256(config.json)` plus the sorted shard names and sizes, not the directory name. Sixteen directories on this host are byte-identical copies of another, several of them named as though they were unquantized when they hold 4-bit weights. They are recorded as aliases and measured once.

### A stale shard index is an upstream property, not local damage

Twelve checkpoints here carry a `model.safetensors.index.json` that names shards which do not exist, and the loss is total rather than partial: none of the shards the index lists is present. `qwen3-vl-32b-4bit` is the widest on this host, with the index declaring 14 shards and 66.71 GB against 4 shards and 19.62 GB on disk. Sizes here are decimal GB throughout, matching the `total_size` byte counts the indexes carry.

They are not damaged downloads. The local copies match their repositories file for file, M1 Ultra and M5 Max independently hold the same twelve in the same state, and the index is the *pre-quantization* original's index carried through unchanged: `gemma-3-4b-it-4bit` declares the 2 shards and 8.60 GB of `google/gemma-3-4b-it`, `qwen3-vl-32b-4bit` the 14 shards and 66.71 GB of `Qwen/Qwen3-VL-32B-Instruct`. The conversion wrote new weights and copied the old index.

The scope is VLM conversions, at roughly 12% of the most-downloaded image-text-to-text repositories, concentrated in the gemma-3 and Qwen3-VL families. mlx-lm text conversions are unaffected. The mlx-vlm version recorded in each repository does not predict it: 0.3.2 appears on both the healthy and the stale side.

They load and measure correctly because mlxcel reads the index and falls back to a glob when the shards it names are absent, printing a warning when it does:

```
Warning: model.safetensors.index.json in models/mlx/gemma-3-4b-it-4bit references shards that don't match the on-disk files (likely a repackaged mlx-community quant). Falling back to all *.safetensors files in the directory.
```

The distinction between that and ignoring the index matters, because the index is load-bearing elsewhere: the pipeline-parallel partial loader picks each rank's shards from it, and that path does not have this fallback, so a stale-index checkpoint cannot be run pipeline-parallel. `docs/adr/0006-safetensors-shard-discovery-globs-past-a-stale-index.md` records why the fallback exists and what reverting it would break. `scripts/checkpoint_fingerprint.py` reports which path a checkpoint took in its `shard_source` field, and `scripts/audit_hf_index.py` checks a repository before it is fetched.

## Coverage

The sweep walks every checkpoint directory under `models/mlx`. What it does with each one:

| Outcome | Text | Notes |
|---|--:|---|
| Measured | 173 | Includes 8 checkpoints fetched on 2026-09-08 and 12 re-measured after #1686 |
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
| Object detection | 1 | `docling-layout-heron-mlx-bf16` (RT-DETRv2) |
| Unsupported architecture | 1 | `afm-4.5b` |

Two more checkpoints used to sit in this table as source repositories rather than release weights, `klear-46b-src` and `phixtral-4x2_8-src`. Both were removed from the store on 2026-09-08: they are pre-conversion originals kept for reference, not something the runtime serves, and their rows were dropped with them.

The first five rows are not text-generation models and cannot produce a decode figure; the drafters and speculative variants are components of a pairing rather than standalone targets, and belong in the speculative sweep instead. `afm-4.5b` is the only real coverage gap: `mlxcel generate` reports `Unsupported model type: arcee`. Checking that against `mlxcel arch`, which is the command that lists architectures, takes a second look: an "Arcee" heading is present, but the entry under it is the AFMoE / Trinity MoE variant, and this checkpoint is the dense `ArceeForCausalLM`. The vendor heading is not the answer; the entry under it is.

The Python baseline measured 124 of the same set, so parity is computed over the 107 models both sides measured. The eight checkpoints added on 2026-09-08 have no baseline row yet and are not in that figure. One of those, `plamo-2-1b`, was added on 2026-09-07 after `numba` was installed; its earlier `FAIL:warmup` was a missing dependency of the checkpoint's remote code, not a property of the model. Its own failures are not analysed here; they say what mlx-lm loads, not what mlxcel does.

## Performance against the Python baselines

Parity is `mlxcel decode tok/s / baseline decode tok/s`, over the 107 text models both sides measured at prompt lengths agreeing within 10%, the same rule the VLM section uses. Values above 100% mean mlxcel is faster.

| Population | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| All text models | 107 | 100% | 99 / 105 | 89-140% |
| MoE | 23 | 105% | 100 / 110 | 90-140% |
| Dense | 84 | 100% | 98 / 102 | 89-115% |

The overall median sits at parity, which is the expected result for two runtimes calling the same MLX kernels on the same weights. What moved between the 2026-09-06 and 2026-09-08 sweeps is the bottom, not the middle: the range closed from 28-140% to 89-140% and the count below 90% went from four to one, while the median stayed where it was. lablup/mlxcel#1709 is the reason, and the shape is what a fix to a shared path looks like when most models never took it.

This ratio is the runtime claim and it is the one to read first. It is measured on one machine against a baseline run on that same machine, so it is independent of the hardware. The hardware question is separate and lives in the cross-hardware section of [model_tests.md](model_tests.md); a ratio taken between two machines on mlxcel alone cannot distinguish a hardware gap from a place where mlxcel fails to exploit the hardware. On M5 Max three checkpoints turn out to be exactly that, and they are only visible when the same-machine ratio is computed on both machines and the two are compared.

MoE is the one population that separates. Its lower quartile (101%) is above the dense median, so three quarters of MoE checkpoints are ahead rather than a few large wins pulling an average. The direction matches the fused decode-MoE kernel, which replaces `gather_qmm` on small-expert families and gains in proportion to how much of the model is MoE. `trinity-nano-preview-4bit` at 140%, `qwen3-30b-a3b-4bit` at 125% and `klear-46b-a2.5b-instruct-4bit` at 123% are the largest.

Quantization is not a factor. Non-quantized checkpoints have a median of 99% against 100% for quantized, and all 14 now sit between 89% and 109%. An earlier reading that the slowest models were all non-quantized was a coincidence of a small sample: the defect behind them (lablup/mlxcel#1709) also caught the 4-bit `phixtral-4x2_8-4bit`, through the quantized GELU MLP rather than the dense one.

### The same redundancy, found switched off

`compiled_softcap_sdpa_gqa` already carried a decode branch that keeps K and V at `[B, H_kv, S, D]` and broadcasts `n_rep` inside the matmul, which is what #1686 went on to do by hand in the VL decoders. It sat behind `MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED` and was off, so every Gemma 2 decode step fell through to `do_repeat_kv` instead. Turning it on by default moves `gemma2-2b-4bit` from 99% of mlx-lm to 107% and `gemma-2-9b-8bit` from 43.66 to 46.76 tok/s.

The gain is smaller here than on M5 Max, where the same change is worth 1.12x and 1.17x against 1.07x on both here. That is the direction bandwidth predicts: this host was already at 99% because the copy it removes weighs less against a wider memory bus, so there was less to recover.

The attribution is checked rather than assumed. `MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED=1` restores the old path and measures 146.25 and 43.45, back at the recorded 144.91 and 43.66. A background-load artefact would not respond to that flag.

### Open performance gaps

One checkpoint sits below 90%:

| Model | Decode | Prefill | mlxcel | Baseline |
|---|--:|--:|--:|--:|
| `gpt_bigcode-santacoder` | 89% | 87% | 163.9 | 183.4 |

Three others were here and are not any more. `qwen2.5-vl-3b-instruct` at 26% decode was #1686, a GQA KV expansion in the shared decode attention. `pythia-1b` at 31% and `phixtral-4x2_8-4bit` at 78%, along with santacoder's own 28%, were lablup/mlxcel#1709: five activation helpers in the cxx bridge built their constants as f32 scalars and returned f32 for a half-precision input, which widened the residual stream at the first MLP and made every later matmul promote its own weight to match.

Two things about how that one was found are worth carrying forward.

The first is that this document's own reasoning about it was wrong in a specific way. It ruled out weight dtype on the grounds that both checkpoints ship F16 rather than F32, and it noted `gpt2` at 96% as the F32 control. Both observations were correct and the conclusion drawn from them was backwards: `gpt2` was not a control showing dtype does not matter, it was the one model in the set with nothing left to promote. It took the same code path and could not be harmed by it, which is why it did not move when the defect was fixed (208.7 to 210.5) while santacoder moved 3.2x.

The second is that the two remaining candidates named here, GPT-NeoX's partial rotary and GPT-BigCode's fused `c_attn`, were both wrong, and the method that replaced them is the reusable part. Ablating santacoder's layer gave 18.83 ms/token in total against 14.21 ms for the MLP alone, 2.46 ms for attention alone and 0.40 ms for the embedding and head, which located the whole gap in the MLP before any hypothesis about attention had to be tested at all.

Santacoder's remaining 11% has not been investigated. Its prefill and decode are now within two points of each other, which no longer suggests a stage-specific cost.

### The Qwen VL gap closed, and it closed in two pieces

The 2026-09-06 sweep had `qwen2.5-vl-3b-4bit` at 59% of mlx-lm and `qwen2-vl-2b-4bit` at 60%, with the qwen3-vl checkpoints at 83-92%, and this document described the shape as a structural gap that widened with context. It did widen: measured on M5 Max, `qwen2.5-vl-3b-4bit` fell from 71% of baseline at a 64-token prompt to 46% at 2048.

The cause was a GQA KV expansion. The decoder called `repeat_kv` to widen the cache by `n_rep` before handing it to the fused SDPA, which broadcasts KV heads internally, so the copy was pure duplication of the whole live cache on every decode step. Removing it (#1686) takes the curve to 9.7% loss over the same 32x context increase, against 9 to 11% for mlx-lm, which is not a smaller gap but the absence of the term that produced it.

A second fix followed. `qwen3_vl.rs` already routed a run with no vision state to a text-only path; `qwen2_vl.rs` did not, and paid the MRoPE table build on every text token. Adding it there is where the `qwen2*` prefill numbers move.

The two fixes land in different places, which is why the gains look uneven:

| Checkpoint | Prefill | Decode | Fixes applied |
|---|--:|--:|---|
| `qwen2-vl-2b-4bit` | 1.20x | 1.71x | both |
| `qwen2.5-vl-3b-4bit` | 1.20x | 1.67x | both |
| `qwen2.5-vl-3b-instruct` | 1.43x | 4.07x | both |
| `qwen3-vl-2b-4bit` | 1.02x | 1.09x | KV only |
| `qwen3-vl-4b-instruct-4bit` | 1.00x | 1.23x | KV only |
| `qwen3-vl-8b-instruct-4bit` | 1.01x | 1.15x | KV only |
| `qwen3-vl-30b-a3b-4bit` | 1.02x | 1.15x | KV only |
| `qwen3-vl-32b-4bit` | 1.02x | 1.13x | KV only |
| `paddleocr-vl-bfloat16` | 1.03x | 1.12x | KV only |

Prefill moves only where the text-only path was added, and the KV-only rows gain modestly at this prompt length because the term removed grows with context: 512 tokens is near the flat end of the curve above.

One difference inside the `qwen2*` group is not explained. `qwen2.5-vl-3b-instruct` and `qwen2.5-vl-3b-4bit` are the same architecture with the same head counts and the same two fixes, and gain 4.07x against 1.67x. The obvious account, that a KV cache is f16 regardless of weight quantization so the removed copy is a larger share of a shorter 4-bit decode step, predicts the opposite ordering. Recorded as measured, unexplained.

Over a 32x increase in prompt length mlxcel gives up 40% of its decode rate and mlx-lm gives up 9%. The sweep figure is one point on that curve at 512, and production contexts are longer, so this is a structural gap that widens rather than a fixed deficit.

## Vision-language models

93 VLM checkpoints, 78 measured by mlxcel and 67 by mlx-vlm, 65 by both. Prompt length comes from the image rather than being fixed, so the comparison below is restricted to the 47 models whose two prompt lengths agree within 10%; the other 18 are listed under the shape mismatch above.

| Population | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| Comparable VLM rows | 47 | 108% | 101 / 124 | 94-197% |

mlxcel is ahead of the mlx-vlm baseline on most of this set, further ahead than on text, and nothing in it now reads below 90%. The widest margins are `jina-vlm-mlx` at 197%, `paligemma2-3b-ft-docci-448-6bit` at 176% and `qwen3-omni-30b-a3b-instruct-4bit` at 159%. M5 Max reads the same three at 194%, 146% and 155% on its own same-day pair, and its comparable set is also 47 rows wide with a 105% median, so the two hosts agree on both the shape and the leader.

Both sides of every ratio above were taken on 2026-09-09, and that pairing is the point. The table this replaced was a patchwork: 54 of its 76 measured rows predated `f4ecc9269`, the shared bridge fix that keeps the activation helpers in the input dtype and therefore reaches any model. Re-measuring all of them in one pass moved two materially, `qwen3-omni-30b-a3b-instruct-4bit` 47.25 to 77.16 and `llama-3.2-11b-vision-instruct-4bit` 65.28 to 87.18, while the median row moved 0.987x and 63 of 76 stayed within 3%. So the exposure was real but narrow, and it could not have been found by reasoning: `qwen3-omni` is in neither family the two model-specific rotary fixes targeted, so a scope drawn from those commits would have skipped exactly the row that moved most.

The mlx-vlm side had to be re-run as well, and skipping it would have been worse than leaving both stale. Dividing the new `qwen3-omni` decode by the 2026-09-07 baseline gives 298%, which would have entered this table as its widest margin; against the same-day baseline the row reads 159%. A runtime improvement measured against an aging reference is indistinguishable from a reference regression, and the number it produces is larger than either.

Eighteen rows read `shape` rather than a percentage, and they are not measurement noise. Three are image tiling, where mlx-vlm splits the same fixture far finer: `idefics3-8b-llama3-4bit` sees 189 tokens against 3041, `smolvlm-instruct-bf16` 102 against 1562, `idefics2-8b-4bit` 81 against 340. The other fifteen differ by exactly 15 tokens, across two different bases (65 and 69) and three model families, which is a fixed chat-template overhead rather than anything that scales. Every one of those counts matches M5 Max exactly, so both effects belong to the runtimes and not to either host.

Seven further rows differ in the other direction, with mlxcel using more tokens than the baseline: `internvl3-1b-4bit` by 23, then six by one to three. Those counts also match M5 Max exactly, so they are small but reproducible differences rather than noise, and at 0.2 to 2.5% they do not move a margin. `internvl3-1b-4bit` is the exception worth naming, because 23 tokens on a 293-token prompt is 7.8% and stays inside the 10% gate: **its margin is computed across a 23-token difference**. The gate cuts by size and not by cause, so the same effect is excluded at 13% and admitted at 8%.

The gains from the activation-dtype work are smaller here than in the text table, because the term it removed grows with context and an image prompt is short: 8 to 1543 tokens against a fixed 512. `qwen2.5-vl-3b-instruct` shows that directly, gaining 4.07x on a 512-token text prompt and 1.02x on its 91-token image prompt from the same binary and the same weights. A gap that behaves that way is a context-scaling term, not a property of the checkpoint.

The two rows that used to contradict that reading no longer do. `qwen2.5-vl-3b-instruct-4bit` and `qwen2-vl-2b-instruct-4bit` sat at 80% and 82% against the old baseline while reading 99% and 102% in the text table, which is the opposite of what a context-scaling cost predicts. On the same-day pair they read 96% and 98%. `mistral-small-4-119b-2603-4bit` was at 37% before its Llama-4 attention scale stopped being built in f32 and promoting the query across all 36 layers (lablup/mlxcel#1711), and reads 106% here.

### Failures

26 of the 93 produced no row on the mlx-vlm side. Four are `FAIL:image_not_applied`, where the guard rejected a measurement whose prompt never grew past its plain-text template: `bunny-llama-3-8b-v-4bit`, `fastvlm-0.5b-bf16`, `llama-3.2-11b-vision-instruct-4bit` and `llava-v1.6-mistral-7b-4bit`. All four are mlx-vlm side failures, and mlxcel expands the image correctly on each of them, which is why neither runtime can be treated as the reference.

The remaining 22 are load or warmup failures. Causes established for six of them, on both hosts:

| Checkpoint | Cause |
|---|---|
| `deepseek-ocr-4bit`, `-2-4bit` | Bundled remote code targets an older transformers (`LlamaFlashAttention2`, `is_torch_fx_available`) |
| `deepseek-vl2-small-4bit` | mlx-vlm passes an `mx.array` where an int is required; strict since nanobind 2.15 (Blaizzy/mlx-vlm#2177) |
| `llama-4-scout-17b-16e-instruct-4bit` | `attn_temperature_tuning` is `4` where transformers 5.16 requires a bool; upstream config is the same |
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

### VLM parity covers 47 of the 65 models both sides measured

Decode throughput depends on context length, so a decode ratio is only meaningful when both sides ran the same prompt length. On 18 of the 65 common models they did not:

| Models | mlxcel | mlx-vlm | Ratio |
|---|--:|--:|--:|
| `idefics3-8b-llama3-4bit` | 189 | 3041 | 16.1x |
| `smolvlm-instruct-bf16` | 102 | 1562 | 15.3x |
| `idefics2-8b-4bit` | 81 | 340 | 4.2x |
| 5 qwen3-vl checkpoints | 65 | 80 | 1.2x |
| 10 qwen3.5 / 3.6 / 3.8 checkpoints | 69 | 84 | 1.2x |

Two entries that used to sit in this table have left it. `granite-vision-3.2-2b-4bit` was recorded here at 1543 against 56 because the baseline column carried a figure from an older mlx-vlm run; on the same-day pair both sides read within three tokens of 1543. `minicpm-v-4.6-bf16` and `-mxfp4` were at 32 against 78 and now read 80 against 78, because lablup/mlxcel#1684 was fixed by `94323c20`, which upscales an image below the scale resolution and took the count from 32 to 80. That is an intended behavior change, not measurement drift, and it is the only case in this document where a prompt length moved because our own code moved.

The parity figure below is computed over the 47 models whose prompt lengths agree within 10%; the other 18 are reported as a shape mismatch rather than a number. The mismatch is not a setting either harness exposes: the two runtimes tokenize the same image into different numbers of visual tokens, and matching them would mean changing one of them. Two shapes are distinct. The first three rows are tiling, differing by factors of four to sixteen; the fifteen below them differ by exactly 15 tokens from two different bases, which is a fixed template overhead. M5 Max records identical counts for all eighteen.

Which one is wrong varies, and the checkpoint's own declaration is what settles it. For `qwen3-vl-2b-4bit` the 15-token gap decomposes into 14 image tokens and 1 template token, and the geometry the checkpoint declares (`patch_size` 16, `spatial_merge_size` 2) gives `(224/16)^2 = 196` patches merging 2x2 to 49, which is mlxcel's count; mlx-vlm produces 63. `minicpm-v-4.6` is the opposite case, where the checkpoint declares `image_feature_size: 64` and mlxcel produces 16 (lablup/mlxcel#1684). The `idefics2` and `idefics3` gaps are unjudged: those configs declare no merge size, so the arithmetic that settles the other two is not available.

### A prompt-token count cannot prove the model saw the image

The sweep rejects a row whose prompt never grew past its plain-text template (`FAIL:image_not_applied`), which catches an image that was dropped before tokenization. It cannot catch an image that was tokenized and then ignored. `granite-vision-3.2-2b-4bit` under mlxcel logs 1485 image tokens, measures a 1543-token prompt, and answers "I can't provide a description of the image as I can't see it" (lablup/mlxcel#1683, reproduced on M1 Ultra and M5 Max).

Which side is blind varies by model, so neither runtime can be assumed correct: `llava-next-mistral-7b-4bit` is the mirror case, with mlxcel expanding the image to 590 tokens while mlx-vlm reports 7 and is caught by the guard.

The check that separates these is differential, not length-based: run the model twice with two visually different images and compare the generated text. Identical output means the image contributed nothing, and the test does not depend on knowing what the fixture depicts. It costs two runs per model, so it is a post-hoc check rather than part of the sweep, and it applies in two branches. Rows both harnesses measured are selected by the prompt-length disagreement above; rows only one harness measured have no cross-comparison at all and need the differential on their own.

### The two hosts are on different mlxcel commits, for the text tables only

This applies to the text tables. The VLM tables no longer have the problem: both hosts re-swept VLM at `f85898eb` on 2026-09-09, each against a same-day mlx-vlm 0.6.17 run on its own machine, so a VLM cross-host ratio now spans hardware and nothing else.

For text, M5 Max measured at `a50ff440`, M1 Ultra at `30ab5a39`, eight commits later on the same branch, both at mlxcel 0.7.0-beta.1 and MLX pin `9a795735`, both on the pp512/tg128 shape. Of those eight, one touches shared inference code: #1656, which converts bf16 weights to f16 at load on pre-Ampere CUDA and rewrites 297 lines of `src/models/sanitize.rs`. Its Metal behavior is unchanged: `cuda_f16_normalize_for_config` returns false as soon as `cuda_is_available()` is false, and the BitNet exclusion that keeps `bitnet-b1.58-2b-4t` on native bf16 moved from inside the decision function to its call site at `src/models/sanitize.rs:1726` rather than being dropped. The remaining seven are CUDA JIT serialization, harness fixes and documentation. A cross-host gap in the tables below is therefore attributable to hardware rather than to the commit difference.

## Open items

| Item | State |
|---|---|
| `gpt_bigcode-santacoder` 89% | The 28% was lablup/mlxcel#1709 and is fixed. The remaining 11% is uninvestigated, and prefill and decode are now within two points of each other |
| Qwen VL band, 59-92% | `qwen2.5-vl-3b-4bit` 59% and `qwen2-vl-2b-4bit` 60% on text, qwen3-vl 83-92% |
| `granite-vision-3.2-2b-4bit` refuses descriptive prompts | lablup/mlxcel#1683. The image does reach the model: it answers colour questions correctly on three different solid images. Only descriptive prompts draw a refusal, and mlx-vlm answers those |
| `minicpm-v-4.6-bf16` names the wrong colour under mlxcel | Under investigation. mlx-vlm gets it right, and the two prompt lengths differ (32 against 78) |
| `afm-4.5b` | dense `ArceeForCausalLM` has no entry under `mlxcel arch`'s Arcee heading, which lists the AFMoE / Trinity MoE variant. A coverage gap, not a defect |
| Speculative and batched-serving sweeps | Not re-run on this shape yet |

### A note on reading the low end of these tables

Neither runtime is the reference. `granite-vision` looked blind under mlxcel until a second prompt showed it was not, and `llava-next-mistral-7b-4bit` is blind under mlx-vlm while mlxcel handles it. Before attributing a low ratio to mlxcel, check that the baseline actually did the work: ask the model a question whose answer the image determines, on two images, on both sides.

## Text results

167 models, 512-token prompt, 128 generated tokens. `vs baseline` is mlx-lm 0.31.3 on the same host; `-` means mlx-lm did not measure that model.

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
| `command-r7b-4bit` | Cohere2ForCausalLM | 512 | 694.9 | 107.1 | 111% |
| `aya-expanse-8b-4bit` | CohereForCausalLM | 512 | 696.8 | 104.9 | 95% |
| `dbrx-instruct-4bit` | DbrxForCausalLM | 512 | 90.3 | 23.9 | - |
| `llama-3_3-nemotron-super-49b-4bit` | DeciLMForCausalLM | 512 | 119.6 | 19.4 | 99% |
| `deepseek-ocr-2-4bit` | DeepseekOCR2ForCausalLM | 512 | 4469.7 | 284.5 | - |
| `deepseek-ocr-4bit` | DeepseekOCRForCausalLM | 512 | 4430.4 | 278.5 | - |
| `deepseek-v2-lite-4bit` | DeepseekV2ForCausalLM | 512 | 537.1 | 110.8 | 100% |
| `diffusiongemma-26b-a4b-it-4bit` | DiffusionGemmaForBlockDiffusion | 512 | 785.4 | 66.7 | - |
| `dots.llm1.inst-mixed-4-6bit` | Dots1ForCausalLM | 512 | 210.2 | 29.1 | - |
| `dots.ocr-4bit` | DotsOCRForCausalLM | 512 | 2245.1 | 193.4 | - |
| `ernie-4.5-0.3b-4bit` | Ernie4_5_ForCausalLM | 512 | 7224.0 | 464.2 | - |
| `ernie-4.5-vl-28b-a3b-thinking-4bit` | Ernie4_5_VLMoeForConditionalGeneration | 512 | 867.2 | 91.3 | - |
| `exaone4-1.2b-4bit` | Exaone4ForCausalLM | 512 | 2591.1 | 234.5 | - |
| `exaone-3.5-2.4b-4bit` | ExaoneForCausalLM | 512 | 2097.2 | 181.5 | 98% |
| `falcon-h1-tiny-90m-instruct-4bit` | FalconH1ForCausalLM | 512 | 9680.0 | 351.1 | 115% |
| `falcon-mamba-7b-4bit` | FalconMambaForCausalLM | 512 | 198.5 | 71.8 | 113% |
| `falcon-ocr` | FalconOCRForCausalLM | 512 | 10125.8 | 235.1 | - |
| `florence-2-base-ft-4bit` | Florence2ForConditionalGeneration | 512 | 10215.5 | 416.8 | - |
| `florence-2-large-ft-4bit` | Florence2ForConditionalGeneration | 512 | 6275.9 | 230.0 | - |
| `gpt2` | GPT2LMHeadModel | 512 | 13982.3 | 210.5 | 97% |
| `gpt_bigcode-santacoder` | GPTBigCodeForCausalLM | 512 | 4149.1 | 163.9 | 89% |
| `pythia-1b` | GPTNeoXForCausalLM | 512 | 4996.9 | 191.2 | 98% |
| `gemma-2-9b-8bit` | Gemma2ForCausalLM | 512 | 598.2 | 46.8 | - |
| `gemma2-2b-4bit` | Gemma2ForCausalLM | 512 | 1955.8 | 154.4 | 106% |
| `gemma-3-1b-it-4bit` | Gemma3ForCausalLM | 512 | 4137.0 | 222.9 | 113% |
| `gemma-3-4b-it-4bit` | Gemma3ForConditionalGeneration | 512 | 968.8 | 100.5 | 107% |
| `gemma3n-e2b-4bit` | Gemma3nForConditionalGeneration | 512 | 1380.5 | 79.6 | - |
| `gemma3n-e4b-4bit` | Gemma3nForConditionalGeneration | 512 | 804.5 | 61.8 | - |
| `gemma3n-e4b-bf16` | Gemma3nForConditionalGeneration | 512 | 908.2 | 40.1 | 107% |
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
| `gemma-4-12b-it-4bit` | Gemma4UnifiedForConditionalGeneration | 512 | 331.3 | 36.5 | - |
| `gemma-2b-4bit` | GemmaForCausalLM | 512 | 2095.2 | 186.3 | 99% |
| `glm4-flash-4bit` | Glm4MoeLiteForCausalLM | 512 | 718.2 | 50.3 | 105% |
| `glm-4.1v-9b-thinking-4bit` | Glm4vForConditionalGeneration | 512 | 454.3 | 60.7 | - |
| `glm-4.5v-4bit` | Glm4vMoeForConditionalGeneration | 512 | 229.6 | 33.3 | - |
| `glm-ocr-4bit` | GlmOcrForConditionalGeneration | 512 | 5423.4 | 309.3 | - |
| `gpt-oss-120b-4bit` | GptOssForCausalLM | 512 | 452.6 | 61.2 | 105% |
| `gpt-oss-20b-mxfp4` | GptOssForCausalLM | 512 | 787.2 | 91.7 | 102% |
| `granite-4.0-3b-vision-4bit` | Granite4VisionForConditionalGeneration | 512 | 1103.1 | 128.1 | - |
| `granite-3.3-2b-instruct-4bit` | GraniteForCausalLM | 512 | 1995.7 | 172.3 | 98% |
| `granite-4.1-3b-4bit` | GraniteForCausalLM | 512 | 1089.6 | 128.6 | 100% |
| `granite-4.1-8b-4bit` | GraniteForCausalLM | 512 | 469.2 | 71.9 | 100% |
| `granite-4.0-h-350m-4bit` | GraniteMoeHybridForCausalLM | 512 | 5173.2 | 264.8 | 100% |
| `granite-4.0-h-tiny-4bit` | GraniteMoeHybridForCausalLM | 512 | 1688.0 | 107.2 | 91% |
| `helium-1-preview-2b-4bit` | HeliumForCausalLM | 512 | 2471.0 | 196.7 | 96% |
| `moondream2` | HfMoondream | 512 | 3786.6 | 150.1 | - |
| `hunyuan-1.8b-4bit` | HunYuanDenseV1ForCausalLM | 512 | 1965.2 | 173.1 | 94% |
| `hunyuanocr-mlx-4bit` | HunYuanVLForConditionalGeneration | 512 | 5007.0 | 232.6 | - |
| `iquest-coder-v1-7b-instruct-8bit` | IQuestCoderForCausalLM | 512 | 563.6 | 70.7 | 100% |
| `idefics2-8b-4bit` | Idefics2ForConditionalGeneration | 512 | 786.9 | 109.0 | - |
| `idefics3-8b-llama3-4bit` | Idefics3ForConditionalGeneration | 512 | 749.4 | 104.5 | - |
| `smolvlm-instruct-bf16` | Idefics3ForConditionalGeneration | 512 | 3231.4 | 128.0 | - |
| `internlm2-7b-4bit` | InternLM2ForCausalLM | 512 | 764.2 | 105.4 | 99% |
| `internlm3-8b-4bit` | InternLM3ForCausalLM | 512 | 671.4 | 84.2 | - |
| `internvl3-1b` | InternVLChatModel | 512 | 6982.6 | 331.9 | - |
| `jamba-v0.1-4bit` | JambaForCausalLM | 512 | 213.9 | 131.0 | 100% |
| `jina-vlm-mlx` | JinaVLMForConditionalGeneration | 512 | 2478.9 | 168.4 | - |
| `kimi-vl-a3b-thinking-4bit` | KimiVLForConditionalGeneration | 512 | 528.2 | 98.4 | - |
| `klear-46b-a2.5b-instruct-4bit` | KlearMoeForCausalLM | 512 | 963.7 | 93.1 | 123% |
| `llada2.0-mini-preview-4bit` | LLaDA2MoeModelLM | 512 | 2120.1 | 147.4 | - |
| `lfm2-350m-8bit` | Lfm2ForCausalLM | 512 | 7711.8 | 568.1 | 105% |
| `lfm2-8b-a1b-4bit` | Lfm2MoeForCausalLM | 512 | 2128.4 | 195.2 | 106% |
| `lfm2-vl-450m-4bit` | Lfm2VlForConditionalGeneration | 512 | 8722.9 | 574.8 | - |
| `llama-4-scout-17b-16e-instruct-4bit` | Llama4ForConditionalGeneration | 512 | 281.6 | 35.2 | - |
| `deepseek-coder-1.3b-4bit` | LlamaForCausalLM | 512 | 3739.6 | 146.1 | - |
| `llama-3.1-8b-bf16` | LlamaForCausalLM | 512 | 810.5 | 36.1 | 102% |
| `llama-3.2-1b-4bit` | LlamaForCausalLM | 512 | 4140.0 | 402.9 | 101% |
| `llama-3.2-1b-instruct` | LlamaForCausalLM | 512 | 4585.6 | 189.1 | 105% |
| `minicpm-2b-4bit` | LlamaForCausalLM | 512 | 1906.4 | 148.3 | 100% |
| `smollm-135m-4bit` | LlamaForCausalLM | 512 | 12605.1 | 370.7 | 114% |
| `llava-1.5-7b-4bit` | LlavaForConditionalGeneration | 512 | 848.9 | 106.9 | - |
| `llava-interleave-qwen-0.5b-bf16` | LlavaForConditionalGeneration | 512 | 7988.8 | 282.0 | - |
| `pixtral-12b-4bit` | LlavaForConditionalGeneration | 512 | 490.6 | 67.8 | 101% |
| `granite-vision-3.2-2b-4bit` | LlavaNextForConditionalGeneration | 512 | 1448.1 | 150.2 | - |
| `llava-next-mistral-7b-4bit` | LlavaNextForConditionalGeneration | 512 | 789.3 | 109.4 | - |
| `fastvlm-0.5b-bf16` | LlavaQwen2ForCausalLM | 512 | 7681.3 | 266.2 | - |
| `mellum2-12b-a2.5b-base` | MellumForCausalLM | 512 | 1675.1 | 76.0 | - |
| `mimo-7b-4bit` | MiMoForCausalLM | 512 | 569.8 | 82.8 | 100% |
| `minicpm3-4b-4bit` | MiniCPM3ForCausalLM | 512 | 1191.8 | 77.5 | 109% |
| `minicpm-v-4.6-bf16` | MiniCPMV4_6ForConditionalGeneration | 512 | 5037.4 | 207.3 | - |
| `minicpm-v-4.6-mxfp4` | MiniCPMV4_6ForConditionalGeneration | 512 | 3931.7 | 217.2 | - |
| `ministral-3b-4bit` | Mistral3ForConditionalGeneration | 512 | 1120.9 | 153.0 | 102% |
| `mistral-small-3.1-24b-4bit` | Mistral3ForConditionalGeneration | 512 | 179.0 | 31.4 | 100% |
| `mistral-small-4-119b-2603-4bit` | Mistral3ForConditionalGeneration | 512 | 381.2 | 55.0 | - |
| `mixtral-8x7b-4bit` | MixtralForCausalLM | 512 | 333.4 | 54.5 | 100% |
| `llama-3.2-11b-vision-instruct-4bit` | MllamaForConditionalGeneration | 512 | 750.6 | 105.5 | - |
| `molmo2-4b` | Molmo2ForConditionalGeneration | 512 | 1072.9 | 91.1 | - |
| `molmo-7b` | MolmoForCausalLM | 512 | 789.8 | 108.9 | - |
| `nemotron-h-30b-4bit` | NemotronHForCausalLM | 512 | 349.2 | 96.1 | 103% |
| `nemotron-3-nano-omni-30b-a3b-reasoning-4bit` | NemotronH_Nano_Omni_Reasoning_V3 | 512 | 349.9 | 96.0 | - |
| `olmo2-7b-4bit` | Olmo2ForCausalLM | 512 | 814.6 | 102.3 | 100% |
| `olmo3-32b-4bit` | Olmo3ForCausalLM | 512 | 129.0 | 21.6 | 101% |
| `olmo-1b-4bit` | OlmoModelForCausalLM | 512 | 3505.6 | 184.5 | - |
| `openelm-1_1b-instruct-4bit` | OpenELMForCausalLM | 512 | 4150.9 | 278.9 | - |
| `paddleocr-vl-bfloat16` | PaddleOCRVLForConditionalGeneration | 512 | 8641.7 | 316.3 | - |
| `paligemma2-3b-6bit` | PaliGemmaForConditionalGeneration | 512 | 1814.3 | 129.9 | - |
| `phi-3-mini-4bit` | Phi3ForCausalLM | 512 | 1446.2 | 151.4 | 99% |
| `phi-3.5-mini-4bit` | Phi3ForCausalLM | 512 | 1435.7 | 145.7 | 98% |
| `phi-3.5-mini-bf16` | Phi3ForCausalLM | 512 | 1553.8 | 61.9 | 99% |
| `phi-3.5-mini-instruct-hf` | Phi3ForCausalLM | 512 | 1556.3 | 61.9 | 99% |
| `phi-4-4bit` | Phi3ForCausalLM | 512 | 405.6 | 57.6 | 100% |
| `phi-3-small-8k-instruct-aq4_64` | Phi3SmallForCausalLM | 512 | 747.9 | 98.6 | - |
| `phi-3.5-vision-4bit` | Phi3VForCausalLM | 512 | 1435.0 | 144.2 | - |
| `phi-2-4bit` | PhiForCausalLM | 512 | 1853.6 | 128.8 | - |
| `phixtral-4x2_8-4bit` | PhiForCausalLM | 512 | 989.7 | 85.1 | 90% |
| `phi-3.5-moe-4bit` | PhiMoEForCausalLM | 512 | 602.1 | 75.1 | 110% |
| `plamo-2-1b` | PlamoForCausalLM | 512 | 2961.3 | 107.5 | 100% |
| `deepseek-r1-distill-7b-4bit` | Qwen2ForCausalLM | 512 | 788.5 | 107.1 | 100% |
| `qwen2.5-0.5b-bf16` | Qwen2ForCausalLM | 512 | 7548.1 | 274.3 | 109% |
| `qwen2.5-1.5b-4bit` | Qwen2ForCausalLM | 512 | 2887.8 | 230.2 | 105% |
| `qwen2.5-1.5b-instruct-4bit` | Qwen2ForCausalLM | 512 | 2910.1 | 223.8 | 102% |
| `qwen2.5-7b-8bit` | Qwen2ForCausalLM | 512 | 783.5 | 67.5 | 97% |
| `qwen1.5-moe-a2.7b-4bit` | Qwen2MoeForCausalLM | 512 | 1684.3 | 143.7 | 107% |
| `qwen2-vl-2b-4bit` | Qwen2VLForConditionalGeneration | 512 | 3042.6 | 224.4 | 102% |
| `qwen2.5-vl-3b-4bit` | Qwen2_5_VLForConditionalGeneration | 512 | 1674.4 | 150.0 | 99% |
| `qwen3-0.6b-4bit` | Qwen3ForCausalLM | 512 | 4754.1 | 249.9 | 105% |
| `qwen3-1.7b-4bit` | Qwen3ForCausalLM | 512 | 2037.4 | 185.3 | 98% |
| `qwen3-4b-4bit` | Qwen3ForCausalLM | 512 | 942.3 | 116.9 | 101% |
| `qwen3-8b-4bit` | Qwen3ForCausalLM | 512 | 518.0 | 81.2 | 101% |
| `qwen3-30b-a3b-4bit` | Qwen3MoeForCausalLM | 512 | 858.4 | 82.3 | 125% |
| `qwen3-next-80b-a3b-instruct-4bit` | Qwen3NextForCausalLM | 512 | 615.1 | 60.0 | 118% |
| `qwen3-omni-30b-a3b-instruct-4bit` | Qwen3OmniMoeForConditionalGeneration | 512 | 854.4 | 81.2 | - |
| `qwen3-vl-2b-4bit` | Qwen3VLForConditionalGeneration | 512 | 2082.6 | 186.2 | 98% |
| `qwen3-vl-32b-4bit` | Qwen3VLForConditionalGeneration | 512 | 128.1 | 21.9 | 103% |
| `qwen3-vl-30b-a3b-4bit` | Qwen3VLMoeForConditionalGeneration | 512 | 851.9 | 81.2 | 123% |
| `qwen3.5-0.8b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 3779.6 | 279.3 | 102% |
| `qwen3.5-0.8b-optiq-4bit` | Qwen3_5ForConditionalGeneration | 512 | 3749.2 | 262.0 | 103% |
| `qwen3.5-27b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 152.7 | 25.1 | 97% |
| `qwen3.5-2b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 1831.4 | 202.9 | 99% |
| `qwen3.5-4b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 851.4 | 112.0 | 98% |
| `qwen3.5-9b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 483.9 | 75.6 | 97% |
| `qwen3.5-9b-bf16` | Qwen3_5ForConditionalGeneration | 512 | 753.3 | 31.5 | 94% |
| `qwen3.8-27b-4bit` | Qwen3_5ForConditionalGeneration | 512 | 153.3 | 25.1 | 98% |
| `qwen3.8-27b-hf-bf16` | Qwen3_5ForConditionalGeneration | 512 | 219.2 | 10.3 | 97% |
| `qwen3.5-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 512 | 839.6 | 82.3 | 109% |
| `qwen3.6-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 512 | 842.6 | 81.5 | 109% |
| `seed-oss-36b-instruct-4bit` | SeedOssForCausalLM | 512 | 115.4 | 19.3 | 99% |
| `smollm3-3b-4bit` | SmolLM3ForCausalLM | 512 | 1210.9 | 128.4 | 97% |
| `solar-open-100b-4bit` | SolarOpenForCausalLM | 512 | 255.6 | 35.7 | 103% |
| `stablelm-1.6b-4bit` | StableLmForCausalLM | 512 | 3298.5 | 211.1 | 99% |
| `starcoder2-3b-4bit` | Starcoder2ForCausalLM | 512 | 1699.8 | 158.0 | 100% |
| `telechat3-36b-thinking-4bit` | Telechat3ForCausalLM | 512 | 114.8 | 20.1 | 101% |
| `youtu-llm-2b-4bit` | YoutuForCausalLM | 512 | 1728.8 | 136.3 | 92% |
| `youtu-vl-4b-instruct` | YoutuVLForConditionalGeneration | 512 | 1088.5 | 43.6 | - |
| `deepseek-vl2-small-4bit` | deepseek_vl_v2 | 512 | 532.1 | 111.7 | - |
| `mamba2-1.3b-4bit` | mamba2 | 512 | 2405.5 | 101.0 | - |
| `mamba2-130m` | mamba2 | 512 | 9828.6 | 213.4 | - |

## VLM results

77 models, prompt length set by the image, 128 generated tokens, all measured in one pass at `f85898eb` against a same-day mlx-vlm 0.6.17 run. `vs baseline` compares the two; `shape` means the two harnesses used prompt lengths differing by more than 10%, which makes a decode ratio meaningless; `-` means mlx-vlm did not measure that model. The embedding and rerank checkpoints the VLM sweep also loads are not listed here, for the reason given under "Newly measured checkpoints".

| Model | Architecture | Prompt | Prefill tok/s | Decode tok/s | vs baseline |
|---|---|--:|--:|--:|--:|
| `aya-vision-8b-4bit` | AyaVisionForConditionalGeneration | 735 | 643.5 | 109.3 | 106% |
| `bunny-llama-3-8b-v-4bit` | BunnyLlamaForCausalLM | 746 | 652.0 | 99.8 | - |
| `deepseek-vl2-small-4bit` | deepseek_vl_v2 | 494 | 437.1 | 106.3 | - |
| `deepseek-ocr-2-4bit` | DeepseekOCR2ForCausalLM | 409 | 777.4 | 264.4 | - |
| `deepseek-ocr-4bit` | DeepseekOCRForCausalLM | 281 | 853.2 | 272.1 | - |
| `dots.ocr-4bit` | DotsOCRForCausalLM | 74 | 444.7 | 211.1 | 110% |
| `ernie-4.5-vl-28b-a3b-thinking-4bit` | Ernie4_5_VLMoeForConditionalGeneration | 108 | 291.3 | 91.3 | 118% |
| `gemma-3-4b-it-4bit` | Gemma3ForConditionalGeneration | 275 | 254.8 | 108.2 | 113% |
| `gemma-3n-e2b-it-4bit` | Gemma3nForConditionalGeneration | 273 | 883.1 | 84.1 | 130% |
| `gemma-3n-e4b-bf16` | Gemma3nForConditionalGeneration | 273 | 653.8 | 41.3 | 111% |
| `gemma-3n-e4b-it-4bit` | Gemma3nForConditionalGeneration | 273 | 574.7 | 64.2 | 125% |
| `gemma-4-26b-a4b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 313.7 | 77.6 | 115% |
| `gemma-4-26b-a4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 277 | 316.7 | 74.8 | 110% |
| `gemma-4-31b-4bit` | Gemma4ForConditionalGeneration | 265 | 97.7 | 19.5 | 100% |
| `gemma-4-31b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 102.4 | 19.6 | 100% |
| `gemma-4-31b-it-nvfp4` | Gemma4ForConditionalGeneration | 278 | 99.9 | 13.3 | 99% |
| `gemma-4-31b-it-qat-4bit` | Gemma4ForConditionalGeneration | 277 | 102.7 | 16.5 | 100% |
| `gemma-4-e2b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 795.0 | 111.8 | 108% |
| `gemma-4-e2b-it-8bit` | Gemma4ForConditionalGeneration | 277 | 781.0 | 96.6 | 97% |
| `gemma-4-e2b-it-qat-4bit` | Gemma4ForConditionalGeneration | 273 | 762.8 | 104.0 | 105% |
| `gemma-4-e4b-4bit` | Gemma4ForConditionalGeneration | 265 | 499.1 | 78.8 | 105% |
| `gemma-4-e4b-it-4bit` | Gemma4ForConditionalGeneration | 277 | 504.7 | 78.2 | 104% |
| `gemma-4-e4b-it-8bit` | Gemma4ForConditionalGeneration | 277 | 501.6 | 63.7 | 94% |
| `gemma-4-e4b-it-qat-4bit` | Gemma4ForConditionalGeneration | 273 | 511.0 | 69.8 | 101% |
| `gemma-4-12b-it-4bit` | Gemma4UnifiedForConditionalGeneration | 277 | 297.1 | 37.7 | 101% |
| `glm-4.1v-9b-thinking-4bit` | Glm4vForConditionalGeneration | 78 | 225.5 | 62.2 | 94% |
| `glm-4.5v-4bit` | Glm4vMoeForConditionalGeneration | 82 | 103.4 | 33.5 | 105% |
| `glm-ocr-4bit` | GlmOcrForConditionalGeneration | 82 | 1075.8 | 372.9 | 104% |
| `granite-4.0-3b-vision-4bit` | Granite4VisionForConditionalGeneration | 337 | 793.4 | 129.3 | 108% |
| `moondream2` | HfMoondream | 8 | 22.6 | 144.9 | - |
| `hunyuanocr-mlx-4bit` | HunYuanVLForConditionalGeneration | 284 | 1264.0 | 257.8 | 155% |
| `idefics2-8b-4bit` | Idefics2ForConditionalGeneration | 81 | 276.9 | 112.6 | shape |
| `idefics3-8b-llama3-4bit` | Idefics3ForConditionalGeneration | 189 | 502.4 | 106.8 | shape |
| `smolvlm-instruct-bf16` | Idefics3ForConditionalGeneration | 102 | 670.9 | 137.9 | shape |
| `internvl3-1b-4bit` | InternVLChatModel | 293 | 2201.7 | 342.3 | 125% |
| `jina-vlm-mlx` | JinaVLMForConditionalGeneration | 436 | 1180.2 | 174.2 | 197% |
| `kimi-vl-a3b-thinking-4bit` | KimiVLForConditionalGeneration | 90 | 351.1 | 95.0 | 138% |
| `lfm2-vl-450m-4bit` | Lfm2VlForConditionalGeneration | 82 | 1554.0 | 601.5 | 124% |
| `llama-4-scout-17b-16e-instruct-4bit` | Llama4ForConditionalGeneration | 162 | 135.4 | 36.1 | - |
| `llava-1.5-7b-4bit` | LlavaForConditionalGeneration | 594 | 763.0 | 104.4 | - |
| `llava-interleave-qwen-0.5b-bf16` | LlavaForConditionalGeneration | 744 | 4317.3 | 264.8 | - |
| `pixtral-12b-4bit` | LlavaForConditionalGeneration | 213 | 408.1 | 68.3 | 104% |
| `granite-vision-3.2-2b-4bit` | LlavaNextForConditionalGeneration | 1543 | 1266.0 | 128.1 | 110% |
| `llava-v1.6-mistral-7b-4bit` | LlavaNextForConditionalGeneration | 590 | 715.1 | 107.4 | - |
| `fastvlm-0.5b-bf16` | LlavaQwen2ForCausalLM | 282 | 1624.5 | 276.0 | - |
| `minicpm-v-4.6-bf16` | MiniCPMV4_6ForConditionalGeneration | 80 | 745.0 | 210.2 | 113% |
| `minicpm-v-4.6-mxfp4` | MiniCPMV4_6ForConditionalGeneration | 80 | 623.5 | 228.3 | 112% |
| `ministral-3-3b-instruct-2512-4bit` | Mistral3ForConditionalGeneration | 613 | 990.6 | 146.7 | 107% |
| `mistral-small-3.1-24b-instruct-2503-4bit` | Mistral3ForConditionalGeneration | 253 | 167.1 | 31.5 | 102% |
| `mistral-small-4-119b-2603-4bit` | Mistral3ForConditionalGeneration | 93 | 177.2 | 56.5 | 106% |
| `llama-3.2-11b-vision-instruct-4bit` | MllamaForConditionalGeneration | 17 | 8.6 | 87.2 | - |
| `molmo2-4b` | Molmo2ForConditionalGeneration | 438 | 679.9 | 90.4 | 140% |
| `molmo-7b-d-0924-4bit` | MolmoForCausalLM | 327 | 585.5 | 108.8 | 139% |
| `nemotron-3-nano-omni-30b-a3b-reasoning-4bit` | NemotronH_Nano_Omni_Reasoning_V3 | 279 | 267.4 | 95.6 | 108% |
| `paddleocr-vl-bfloat16` | PaddleOCRVLForConditionalGeneration | 212 | 1337.9 | 327.5 | 103% |
| `paligemma2-3b-ft-docci-448-6bit` | PaliGemmaForConditionalGeneration | 1032 | 1488.2 | 129.5 | 176% |
| `phi-3.5-vision-instruct-4bit` | Phi3VForCausalLM | 773 | 1028.7 | 135.0 | 145% |
| `qwen2.5-vl-3b-instruct` | Qwen2_5_VLForConditionalGeneration | 91 | 628.9 | 70.5 | 98% |
| `qwen2.5-vl-3b-instruct-4bit` | Qwen2_5_VLForConditionalGeneration | 91 | 695.9 | 140.8 | 96% |
| `qwen2-vl-2b-instruct-4bit` | Qwen2VLForConditionalGeneration | 91 | 950.8 | 209.7 | 98% |
| `qwen3.5-0.8b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 1013.9 | 286.4 | shape |
| `qwen3.5-27b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 90.9 | 25.2 | shape |
| `qwen3.5-2b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 638.7 | 205.7 | shape |
| `qwen3.5-4b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 384.8 | 112.2 | shape |
| `qwen3.5-9b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 237.5 | 75.5 | shape |
| `qwen3.5-9b-bf16` | Qwen3_5ForConditionalGeneration | 69 | 292.4 | 32.9 | shape |
| `qwen3.8-27b-4bit` | Qwen3_5ForConditionalGeneration | 69 | 91.2 | 25.1 | shape |
| `qwen3.8-27b-hf-bf16` | Qwen3_5ForConditionalGeneration | 69 | 87.7 | 10.5 | shape |
| `qwen3.5-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 69 | 291.2 | 82.2 | shape |
| `qwen3.6-35b-a3b-4bit` | Qwen3_5MoeForConditionalGeneration | 69 | 293.2 | 80.1 | shape |
| `qwen3-omni-30b-a3b-instruct-4bit` | Qwen3OmniMoeForConditionalGeneration | 69 | 272.6 | 77.2 | 159% |
| `qwen3-vl-2b-instruct-4bit` | Qwen3VLForConditionalGeneration | 65 | 683.0 | 195.3 | shape |
| `qwen3-vl-32b-instruct-4bit` | Qwen3VLForConditionalGeneration | 65 | 74.6 | 21.4 | shape |
| `qwen3-vl-4b-instruct-4bit` | Qwen3VLForConditionalGeneration | 65 | 394.6 | 114.4 | shape |
| `qwen3-vl-8b-instruct-4bit` | Qwen3VLForConditionalGeneration | 65 | 259.0 | 78.9 | shape |
| `qwen3-vl-30b-a3b-instruct-4bit` | Qwen3VLMoeForConditionalGeneration | 65 | 267.7 | 77.7 | shape |
| `youtu-vl-4b-instruct` | YoutuVLForConditionalGeneration | 28 | 213.3 | 46.0 | - |

## Newly measured checkpoints (2026-09-09)

Six checkpoints taken singly after the 2026-09-08 sweep closed, at `0accedd9` with MLX pin `9a795735`, on the standard pp512/tg128 condition. They are the M1 Ultra half of the pass `model_tests_m5max.md` records under the same heading, taken so that neither host is left quoting a cross-host ratio against a checkpoint only one of them measured.

One of the six belongs in the tables above:

| Model | Architecture | Prefill | Decode | CSV |
|---|---|--:|--:|---|
| `gemma-4-e4b-4bit` | Gemma4ForConditionalGeneration | 803.70 | 77.44 | `metal_m1ultra_2026-09-09_single_gemma-4-e4b-4bit.csv` |

M5 Max read 4768.35 prefill and 136.07 decode on that checkpoint at the same commit and pin, so the pair is a hardware comparison and not one spanning a version or a condition.

The other five are embedding, rerank and speech checkpoints. They stay out of the tables above because a decode rate does not describe the work any of them does; their ladder is `scripts/bench_embeddings.py`, reported in [`embeddings-rerank-m1ultra-2026-09-09.md`](embeddings-rerank-m1ultra-2026-09-09.md), where all four of the embedders and rerankers already have rows.

| Model | `bench_decode` result | Reading |
|---|---|---|
| `qwen3-reranker-0.6b-4bit` | 4817.19 / 232.60 | Causal backbone, so the harness loads it and returns a rate. Asked "The capital of France is" it answers "No relevant content."; through `/v1/rerank` the same checkpoint reads 7377 tok/s at eight documents |
| `qwen3-vl-reranker-2b` | 866.52 / 114.62 | Same shape as the row above; `/v1/rerank` reads 3176 tok/s |
| `qwen3-embedding-0.6b` | `FAIL:bench` | Loader refuses it for generation and names the endpoint that serves it |
| `qwen3-vl-embedding-2b` | `FAIL:bench` | Same refusal |
| `whisper-base` | `FAIL:bench` | ASR, no autoregressive text-decode path |

The two reranker rates are recorded here rather than dropped because the gap between them and the `/v1/rerank` figures is the argument for keeping these checkpoints out of the decode tables. A reader who finds only the 232 tok/s has no way to see that it describes the wrong task.

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

**The llama directory has since been renamed.** These rows ran against `models/mlx/llama-3.1-8b-4bit`, which a 2026-09-08 audit found byte-for-byte identical to `models/mlx/meta-llama-3.1-8b-instruct-4bit` and retired. The model names above are left as measured, because changing them would misstate what the harness was pointed at; reproduce them against the surviving name. `CLAUDE.md` was never affected, since its recommended-checkpoint table carries HuggingFace repo ids rather than local directory names.

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

