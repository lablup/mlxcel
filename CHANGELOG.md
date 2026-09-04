# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.7.0-beta.1] - 2026-09-04

### llama-server b10621 compatibility, complete

Epic #1431 pinned llama.cpp release b10621 (commit `c1d0e7a0`, published 2026-08-25), extracted its whole server surface (249 help entries, 323 long-option spellings, 136 environment variables, 53 routes, 74 native request fields) and reduced it to 376 manifest entries across 18 shards under `compat/llama-server/b10621/`. Every entry now carries a terminal state: 169 supported, 22 aliased onto an mlxcel spelling, 35 recorded as a deliberate divergence with a written rationale, 150 not applicable to a runtime that does not execute GGUF. Nothing is deferred, and a CI gate fails the build if an entry regresses. Full map in [llama-server compatibility](docs/llama-server-compat.md).

- **HTTP transport, authentication, TLS and CORS** (#1462, #1465). Socket timeouts, keep-alive, request-size limits, path prefixes, Unix domain sockets, TLS certificates, CORS origins, and multi-key authentication from a list, a file, or both combined.
- **The native completion, embedding, tokenization, template and infill routes** (#1467, #1480). `/completion`, `/embedding`, `/tokenize`, `/detokenize`, `/apply-template` and `/infill` are served separately from the OpenAI-compatible surface, with `response_fields`, `stream_options` and real prefill timings on the responses.
- **The sampler remainder and a GBNF grammar engine** (#1487, #1506). Mirostat v1 and v2, dynamic temperature, adaptive-p, `min_keep`, `logit_bias`, b10621 DRY breaker-string semantics, `n_probs` and `post_sampling_probs`, and the uint32 seed fold; `--grammar`, `--grammar-file`, `--json-schema`, `--json-schema-file`, and the native `json_schema` field with its `grammar` alias, lazy grammars, triggers and preserved tokens.
- **Prompt cache, batching, context retention and YaRN** (#1468, #1503, #1522). `--rope-scaling yarn` and the five `--yarn-*` knobs run on the shared RoPE path, `--parallel auto` resolves, and the prompt-cache and slot-scheduling shapes match the pinned binary.
- **Router mode** (#1495, #1501). `--models-dir` discovers checkpoints, a model cache is a second source, INI presets are read, and the management surface is served from an in-process model pool where upstream spawns child processes.
- **LoRA adapters served unfused with live scales** (#1497, #1520). Multiple adapters stay unfused for per-request or hot-swapped scale changes, or are fused for zero decode overhead.
- **mp3 and flac transcription, streamed per token** (#1523). The container is identified from the clip's own magic bytes rather than the client's `format` string, and the ASR route emits one delta per decoded token instead of one delta at the end.
- **Embedding and reranking mode and pooling** (#1483), **speculative flags mapped to MTP and DFlash** (#1488), and **the Vertex AI custom-container predict adapter** behind `AIP_MODE=PREDICTION` (#1493).
- **Observability and slots** (#1492, #1524). `/props`, `/slots`, `/metrics`, `/health` and slot persistence, with a slot bound on a request's first progress signal rather than at route entry, so a queued request no longer holds a slot away from one the worker is serving.
- **Idle sleep** (#1528, #1530). `--sleep-idle-seconds` drops the serving worker and its weights after the idle window; the next request reloads and is served by the rebuilt scheduler, exactly as upstream's blocking wake is.
- **Assistant prefill, echo, and streamed thinking tags** (#1526, #1527), with `reasoning_in_content` settled against the pinned binary.
- **Unsupported options fail instead of being ignored** (#1463, #1469, #1481, #1489). GGUF model-source semantics, GGML runtime flags and KV quantizers, projector flags, and control-vector options are refused at startup with a diagnostic naming the option.

### Added

- **The Inkling family, across four modalities** (#1532, #1535, #1540, #1546, #1548). The text backbone brings hybrid sliding/global NoPE attention, learned banded relative-position logits, four f32 short-convolution states per layer, and logsigmoid-normalized routed and shared experts, loading original bf16/f32 weights, native ModelOpt NVFP4 expert planes, and pre-converted affine MLX 4-bit experts. On top of it: HMLP image input with the reference 40x40 tiler, adjacent-frame video for CLI `--video` and server `video_url` under one request-wide 16-pair budget, dMel audio for `--audio` and `input_audio`, and a native chained-MTP drafter whose default verify width is `num_nextn_predict_layers + 2`.
- **IQuest-Coder (`iquestcoder`) and text-only Youtu-LLM (`youtu`, `youtu_llm`)** (#1594, #1593). Both are decoders mlxcel already runs under a label detection rejected, so the checkpoints failed with `Unsupported model type`. Each is a new `ModelType` arm on an existing loader with no new architecture code; IQuest-Coder additionally refuses the two config switches that would stop it being a Llama decoder and fixes a SentencePiece tokenizer defect the checkpoint exposed.
- **Qwen-VL video input and Responses-native image parts** (#1510, #1310), and **LFM2-VL image splitting** (#1405).
- **Locally typical sampling (`typical_p`) and top-n-sigma logit filtering** (#1482, #1479). Both are row-wise graph transformations applied through `apply_row_filters` before the single fused sampling dispatch, so requests stay on the batched path.
- **A live runtime settings endpoint** (#1516), optionally authenticated, and **per-request speculative acceptance on the response** (#1588): the drafter kind, verify rounds, and drafted and accepted token counts, which previously reached only a `tracing` line no client can read.
- **`tool_choice` accepts `required` and a named function** (#1319), and tools are declared to the template only when the request actually sends them (#1598).
- **`mlxcel inspect --json` and `mlxcel arch --json`** (#1605, #1606, #1608). The first emits the memory estimator's byte fields as a stable contract instead of a human-readable banner; the second derives a per-family registry of runtime, modality, backend, distributed, drafter and KV facts from `ALL_MODEL_TYPES`. `make recipes-registry` writes a versioned snapshot under `recipes/registry/` for downstream recipe builders.
- **Chat Completions reasoning is available under both common field names** (#1308). Streaming deltas and non-streaming assistant messages now emit an OpenRouter-compatible `reasoning` alias alongside `reasoning_content`, with identical values and joint omission when no reasoning exists; `--reasoning-alias-field none` or `MLXCEL_REASONING_ALIAS_FIELD=none` disables only the duplicate for byte-sensitive deployments, while Responses API events and Anthropic thinking blocks remain unchanged.
- **OpenAI-shaped reasoning controls now drive chat-template thinking behavior** (#1307). Chat Completions `reasoning_effort`, Responses `reasoning.effort`, and compatible extra-body `reasoning` values derive `enable_thinking`; `none`, `off`, `disabled`, `false`, and `0` disable thinking without becoming invalid level kwargs, enabled effort is forwarded verbatim through a template's `reasoning_effort` or `reasoning_strength` spelling, and explicit per-request chat-template kwargs keep per-key precedence.
- **Embeddings are available through `mlxcel embed` and the OpenAI-compatible `POST /v1/embeddings` endpoint** (#1408, #1410-#1416). The server can load an embedding checkpoint alone or next to a chat model with `--embedding-model`; supported families cover BERT/XLM-RoBERTa, ModernBERT, SigLIP text, EmbeddingGemma, Qwen3 and Qwen3-VL embeddings, bidirectional Llama/Nemotron/LFM2.5, Llama-Nemotron-VL, and ColBERT-style multimodal late-interaction models. Pooling, L2 normalization, Matryoshka dimensions, token and image inputs, bounded worker admission, and per-checkpoint length limits are shared by the CLI and server paths.
- **Reranking is available through `mlxcel rerank` and the Cohere/Jina-compatible `POST /v1/rerank` endpoint** (#1417). One-label BERT/XLM-RoBERTa/ModernBERT cross-encoders, Qwen3 generative rerankers, and Qwen3-VL multimodal rerankers return ranked relevance probabilities; `--reranker-model` can serve one beside the chat and embedding workers.
- **DeepSeek-V4 (`deepseek_v4`) text model support** (#523). A genuinely new architecture rather than a fifth member of the DeepSeek MLA family: HyperConnections replace the plain residual stream, attention runs one shared 512-wide KV head across 64 query heads inside a 128-token rotating window with per-head softmax sinks, per-layer compression (`compress_ratios`) picks local, pooled, or sparse-compressed attention, sparse layers add a HiSA indexer for top-k pooled-row selection, and MoE routing hashes the first three layers by token id (`tid2eid`) rather than argpartitioning a score row. Ported from the in-tree `references/mlx-vlm` reference after an earlier V3-wrapper attempt (PR #592) could not load a real checkpoint. Validated against `mlx-community/DeepSeek-V4-Flash-4bit` (43 layers, ~151 GB on disk): a strict weight-coverage check at load, plus three real-checkpoint generation gates covering a capital-of-France sanity decode, a decode that crosses a pooling-window boundary, and a prompt past 2100 tokens that pushes the HiSA path into its sparse selection branch. Review closed several boundaries where untrusted checkpoint or config data reached MLX unguarded (an out-of-range hash-routed expert id, a wrong-length correction bias, an indivisible `wo_a` reshape plane, and architecture scalars that could overflow an `i32` cast on their way into a kernel), each now a load-time `Result` instead of an uncatchable abort. MTP drafting, the reference's fused HyperConnection Metal kernel, and tensor/pipeline-parallel sharding are follow-ups.

### Changed

- **CUDA `qmm_naive` sizes its CTA tile against the device's real shared-memory budget, and opts into more of it when the tile needs it** (#1541). The quantized matmul behind CUDA prefill picked its N tile with `enough_smem = sm80 && itemsize <= 2 && group_size <= 64`, folding two unrelated decisions into one name. The shape terms are a shared-memory rule and are now written as one: a comparison against `cudaDevAttrMaxSharedMemoryPerBlockOptin` queried from the running device. The architecture term is not, and a Tesla V100 disproves it, offering 96 KB per block against the 24 KB the widest eligible tile needs. Three defects fall out of writing the budget down. A tile larger than the 48 KB a launch gets without `cuFuncSetAttribute` now opts in first, which f32 activations at group size 128 need and never had, so that shape stops failing inside the driver on every architecture. A tile that does not fit at all is refused at the launch site with a message naming the tile, the requirement and the budget, instead of reaching the driver; Turing's 64 KB budget is the case that hits. And the N tile joins the JIT module name, which is the sole key for both the in-process module cache and the persistent PTX cache. **The wide tile itself was measured on Volta and refused.** At bf16 and group size 64 the 128-wide instantiation needs 255 registers and spills 128 bytes per thread where the 64-wide one needs 224 and spills nothing, both reach the same 2 blocks per SM so the extra width buys no occupancy, and halving the CTA count on a grid already smaller than an 80-SM part makes `qmm_naive` 1.50x slower at a 106-token prompt, 1.06x at 516 and 1.02x at 4,106, at identical launch counts. There is no crossover: prefill is chunked at 2,048 tokens, so the M extent of the grid is capped and the wide tile's best case is bounded rather than approached. The selected tile is therefore identical to upstream's on every architecture, asserted by enumerating every `(itemsize, group_size, m)` combination against every real per-block budget in a host-side unit test that needs no GPU. `MLXCEL_QMM_NAIVE_TILE_N` pins the width and `MLXCEL_TRACE_QMM_TILE` prints tile, registers, spill and occupancy, so the sweep is one command to re-run against #1543's Volta tensor-core MMA. Full record in `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
- **CUDA `qmv` accumulates bf16 in float below Ampere, at every bit width** (#1539). The quantized GEMV that serves every single-stream decode step picked its accumulator from the weight bit width alone, float at `bits >= 8` and the element type below that, which handed a bf16 checkpoint a bf16 accumulator at 4 bits. No pre-Ampere part has a bf16 ALU, so each FMA in the k-loop ran as convert-to-float, fma, convert-back. Below Ampere the kernel now accumulates bf16 in float regardless of bit width, behind a `__CUDA_ARCH__ < 800` guard in device code; the float specializations it selects already existed for the `bits >= 8` case. Measured on a Tesla V100-PCIE-32GB: `qwen3.8-27B-4bit` decodes at 117.83 ms/token against 219.96 before (4.55 to 8.49 tok/s, 1.87x) and `gemma-4-12B-it-4bit` at 71.66 against 124.26 (8.05 to 13.95 tok/s, 1.73x). In an nsys profile at identical launch counts `qmv` itself runs 2.01x faster and accounts for 99.4% of the end-to-end gain, while `qmm_naive`, which already accumulated in float, moves by less than 0.6%. The 8-bit sibling checkpoint, the control, is unmoved at 66.73 ms/token. `qmv` roofline attainment rises from 7.27% to 14.80% of 900 GB/s, short of the 25% the issue asked for and close to the 2.14x ceiling the controlled 4-bit against 8-bit measurement had put on it. **Pre-Ampere output changes**: float accumulation is strictly more accurate than the emulated bf16 accumulation it replaces, so quality improves, but greedy decode on a Volta host is no longer token-identical to a build from before this change. No published Volta baseline exists for it to regress against. Determinism is unaffected and still tested. sm_80 and later are untouched: the emitted SASS for the sm_80 and sm_121 device passes is byte-identical before and after. Full record in `docs/benchmark_results/qmv-float-accum-v100-2026-08-31.md`.
- **`--models-dir` changes meaning, and the old meaning moves to `--model-store-root`** (#1495). On `mlxcel serve` and `mlxcel-server`, `--models-dir` and `LLAMA_ARG_MODELS_DIR` now select b10621 router-mode discovery; the mlxcel store root that spelling used to name is `--model-store-root`. A command line carrying both logs a migration diagnostic at startup.
- **`--timeout` and `LLAMA_ARG_TIMEOUT` change meaning** (#1432). They are now the HTTP socket read/write timeout with b10621's 3600-second default. The per-request decode watchdog they used to configure is `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT`, with its 600-second default unchanged; setting the old spelling without the new one logs a migration warning.
- **Constrained decoding keeps the grammar mask packed** (#1578). The token bitmask stays in `u32` form from the matcher to the GPU, where four graph ops expand it and `where_cond` applies it. On the Qwen3.8-27B geometry that removes two 248k-element host loops and 97% of the per-step host-to-device copy from the scheduler thread, on every tick of every constrained sequence.
- **Chat templates are compiled once and cached** (#1518), **response stores are bounded by bytes rather than entry count** (#1519), and **the JSON body limit is derived from the image budget** (#1515) instead of being a fixed number that either truncated valid image requests or admitted oversized ones.
- **CUDA compute capability is visible to the runtime** (#1551), and **the Volta sm_70 baseline is recorded and gated in CI** (#1556) so the pre-Ampere arms above cannot regress unmeasured. The Volta first-token fixed cost is instrumented rather than guessed (#1545).
- **OpenXLA compiles a set of context capacities and routes each request to the smallest that fits** (#1302), its warning backlog is cleared with warnings denied in CI (#1381), and CI links an OpenXLA binary so a link-only regression cannot reach `main` (#1305).
- **The Python client is tested across CPython 3.9 through 3.13** (#1572).

### Fixed

- **A malformed `MLXCEL_PAGED_SLAB_BLOCKS` now falls back to the derived slab size, as its warning always claimed** (#1137). `resolve_paged_slab_blocks` returned early on a parse error, which is the `0` pin: the pool kept the historical 32-block slab while the log said the derived size was in use, so a typo in the variable quietly made the fused paged decode path unreachable past one slab. The parse-error arm now falls through to the derivation, `0` still pins the pool default, and a unit test keeps the three arms distinct.
- **Speculative-burst responses report `prompt_ms` and `predicted_ms` the way classic decode does** (#1592). The B=1 and batched bursts run every verify round before streaming, and the first-token stamp was taken when the finished token vector was replayed, so `prompt_ms` absorbed drafter load, prefill and all verify rounds while `predicted_ms` reported the sub-millisecond replay (`predicted_per_second` read 95000 on a 96-token DFlash request). The burst now carries the instant its target prefill finished and stamps the sequence with it, so `predicted_ms` covers the verify rounds; the drafter's once-per-process disk load is excluded from every `timings` field and stays on the `Drafter loaded` log line. Classic decode is unchanged.
- **Youtu-VL restores raster order after the windowed vision tower** (#1600). `reverse_window_indices` sorted `(value, index)` pairs and wrote the rank back by index, which for a permutation of `0..N` reproduces `window_index` itself rather than its inverse, so the merged vision tokens were un-reordered by applying the window permutation a second time. That only restores raster order when the permutation is an involution, and with Youtu-VL's 8x8 merged-token window even a 512x512 image (a 16x16 merged grid) is not one: 192 of its 256 merged tokens reached the language model out of place. The helper now builds `argsort(window_index)` directly, bounds-checked like the Qwen2.5-VL fix in #1601, with checkpoint-free tests pinning the non-involution, single-window, and multi-image cases. This corrects the token order at that stage only, and does not on its own make the family describe images correctly: the processor emits patches in raster order where the tower's rotary positions and the patch merger both expect merge-block-major, which is filed separately as #1610.
- **`mlxcel_core::is_gpu_available()` is renamed `default_device_is_gpu()`, and `gpu_backend_available()` answers whether a GPU backend exists** (#1421). The old name read as a hardware query but reported whether the MLX default device was currently the GPU, so it turned `false` the moment anything called `set_default_device(false)` on a machine with a GPU. `gpu_backend_available()` is the unclamped `device_count(Device::gpu) > 0`: `true` on Metal and on a CUDA build with a device, `false` on a CPU-only build and on a CUDA build without a driver, and unchanged by the default device. `is_gpu_available()` remains as a deprecated shim for one release. `initialize_runtime()` now resolves `RuntimeSetup.device` from backend availability and reports `MLXCEL_DEVICE=cpu` separately in the new `RuntimeSetup.cpu_override` field, which the server and CLI startup lines print, so a CPU device is distinguishable as requested or as the only option. Every CPU resolution is applied to MLX's default device, not only the operator's override: the CUDA backend answers `gpu::is_available()` unconditionally while `device_count(Device::gpu)` reports what the driver found, so a CUDA build on a driverless host would otherwise report `Cpu` while MLX kept dispatching to the unusable GPU. Test modules that moved the default device to the CPU inside a `Once` and never moved it back, which made every real-checkpoint gate sorting after them measure the CPU backend under `--test-threads=1`, now hold the new `mlxcel_core::streams::DefaultDeviceGuard` per test, and the unconditional GPU pin PR #1420 added to `mlx_test_guard` is reduced to an assertion so a future leak fails at the first gate after it instead of being silently repaired. Restoring the device is not the same as excluding other movers, so there is also one process-wide lock for it, `mlxcel_core::streams::lock_default_device`, held by every test that moves the default device and by `mlx_test_guard` for the duration of the tests it serializes: a lock private to each module leaves the modules interleaving, which under the parallel `cargo test --lib` in `scripts/run_quality_gate.sh` reported another module's live guard as a leak in 4 of 300 runs and would have let a real-checkpoint gate score on the CPU backend without saying so.
- **CUDA grouped GEMM no longer tells CUTLASS a Volta part is a Turing part** (#1544). `dispatch_cutlass_arch` mapped every device below compute capability 8.0 to `cutlass::arch::Sm75`, which names the `m16n8k8` MMA that Turing introduced and an sm_70 part does not have, and `get_grouped_mm_funcion` opened with a matching `Sm75` placeholder. The pre-Ampere arm now selects `cutlass::arch::Sm70` and the placeholder is gone. **The branch is live, and on the checkpoint the epic benchmarks.** #629's sorted-MoE prefill fast path routes a quantized `GatherQMM` into `cutlass_grouped_gemm_unaligned` once the batch clears `B >= 8 * num_experts`, which for `gemma-4-26b-a4b-it-4bit` means a prompt of 128 tokens or more; an nsys profile at 573 prompt tokens shows 180 `cutlass::Kernel<GemmGrouped>` launches taking 3.8% of GPU time, and the same profile at the 46-token prompt the Volta baseline used shows none, which is why nobody had seen it. **Nothing was computing wrong.** The pre-Ampere arm resolves to `GemmConfiguration`'s primary template, which is `OpClassSimt` with `InstructionShape<1, 1, 1>`, so no arch tag can reach an MMA atom there and CUTLASS erases the tag: the running kernel is an `MmaSimt` / `OpMultiplyAdd` / `MmaPipelined` instantiation whose mangled name carries no architecture token at all. Correctness was measured rather than assumed, because a mismatched CUTLASS path can return plausible wrong numbers that greedy decoding hides: the grouped path and the legacy `qmm_naive` path produce a byte-identical 64-token greedy continuation, and new unit tests compare `gather_mm` against an `f64` dense per-expert reference on the model's real expert dims across both entry points, both output-alignment arms, both operand layouts and f32, bf16 and f16. The retag moves no device code on any architecture, which was checked rather than argued: compiling the translation unit before and after at `compute_70`, `compute_80` and `compute_121` yields the same 51 device symbols with byte-identical SASS bodies at all three, over 144 MB of dump compared per symbol. The architecture decision now lives in `gemms/grouped_gemm_arch.h` as a pure function of the compute capability major version, enumerated over every architecture by a host-side test through a C shim with no GPU involved, which is what closes this issue's "zero change on sm_80+" criterion locally instead of deferring it to GB10. Two `static_assert`s pin the preconditions: that the pre-Ampere configuration is still SIMT, without which one arm could not cover Volta and Turing together, and that its stage count is 2, since the 3-stage `SM80_CP_ASYNC` pipeline is bound to an explicit `cutlass::arch::Sm80` specialization and `cp.async` does not exist before Ampere. MoE decode and prefill are unmoved on a V100, as byte-identical device code requires. Full record in `docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`.
- **`mlxcel serve` and `mlxcel-server` match the low-risk llama-server b10621 deployment surface** (#1430). Canonical `LLAMA_ARG_*` variables and `--temperature` now work on both entry points, server sampling defaults match the nightly, OpenAI-shaped requests accept llama aliases, scalar stops and seed `-1`, safe health/reranking route aliases reach the existing handlers, `f16` maps exactly to FP16 split-cache storage, and unsupported cache-reuse chunk sizes or GGML cache quantizers fail instead of changing unrelated behavior.
- **Qwen2.5-VL runs its vision tower norm in f32 and restores raster order correctly** (#1601), and the raw Hugging Face patch-embed layout is normalized at load rather than assumed (#1582).
- **Four RoPE scaling defects across shared decoder paths.** Qwen3 ignored a configured `rope_scaling` (#1398); the shared Llama 3 path dropped the scaled frequencies (#1385); Gemma 3 skipped the linear factor on global-attention layers (#1386); InternLM doubled RoPE positions under dynamic NTK scaling (#1389). Each produced degraded long-context output with nothing failing.
- **Phi-3 selects its LongRoPE table by whole-prompt position** (#1580) rather than by the current chunk, so a prompt that crosses the short/long boundary mid-chunk no longer switches tables partway through.
- **Falcon-Mamba applies the B/C/dt RMS norm once** (#1574), not once per chunk.
- **Recurrent decode state is persisted across requests** (#1513), and **KV modes are applied to model-owned caches** (#1400) instead of only to the scheduler's.
- **Symmetric Turbo4 stops reading past its sign vectors on MLA latent caches** (#1395).
- **Sampling filters on the untempered distribution and scales by temperature last** (#1391), which is the order every filter's threshold was defined against.
- **The prompt cache no longer donates or adopts KV-less shadow paged entries** (#1390), and **preemption victim selection has a total order** (#1301), so two equally-ranked sequences no longer produce a nondeterministic eviction.
- **Gemma 4 Unified keeps its vision overlay when audio is present** (#1403), and **SmolVLM split-image framing is aligned with the reference** (#1402).
- **LoRA adapters that do not map onto the loaded model are refused** (#1576) instead of loading and silently contributing nothing.
- **XML tool-call arguments are typed by the request's own schema** (#1575), and Python `repr` values in tool calls are parsed (#1404).
- **Thinking state is derived from the checkpoint rather than guessed.** The primed-open case reads the tokenizer's markers (#1554) and the `thinking_mode` sentinel comes from the chat template (#1547).
- **Chat-template failures surface.** A template rejection is no longer swallowed (#1511), a missing map key returns `None` instead of raising (#1394), and the `tojson` filter matches `json.dumps` (#1382).
- **Boundary snapshots are gated by model capability and sized by model** (#1507, #1509), so a model that cannot produce one no longer allocates for it.
- **The audio frontend shares one bounded real-FFT implementation** (#1517), and an Inkling sub-config that carries both a field and its alias is accepted (#1561).
- **`bench` reports KV sizes in binary byte units** (#1514) and refuses a `--target` path with no file name (#1569).

## [v0.6.0] - 2026-08-22

### Changed

- **Gemma 4 MTP is gated on a measured exactness probe instead of an unconditional yes** (#1188, #1258). The three Gemma 4 arms of `mtp_capable_target` returned `true` without checking anything, which advertised temperature-0 byte-identity the hardware does not always provide: on Apple GPU generation 15 and later, MLX routes `M >= 2` affine-quantized matmuls to `qmv_wide`, whose K-reduction order differs from the single-token `qmv`, and the result is a systematic token divergence rather than f16 jitter. The Gemma 4 wrapper now runs the same block-vs-chain probe the Qwen 3.5 family has run since #1186, on throwaway caches built from the inner model so no scheduler-owned sequence slot is touched, with both arms projected through the tied LM head at their own width so the head's dispatch is covered too. On a failing probe the gate first retries with `qmv_wide` disabled and keeps the narrow kernel when that restores exactness, which is what generation 15+ hosts now do by default: measured on M5 Max, `gemma-4-12b-it-4bit` plus its 4-bit assistant serves 93.2 tok/s against 43.8 classic (2.13x) with byte-identity kept, where the fast kernel would serve 120.4 (2.76x) without it. **Upgrade note:** a Gemma 4 MTP deployment on M3, M4 or M5 will see roughly 23% lower speculative throughput than v0.5.2 and byte-identical temperature-0 output in exchange. `MLXCEL_MTP_ALLOW_INEXACT=1` together with `MLXCEL_QMV_WIDE=1` restores the previous behavior and forfeits the contract.
- **The B=1 MTP burst is gated on GPU generation rather than the Neural Accelerator** (#1217). The static default for batch-capable targets keyed on `has_neural_accelerator`, an M5-only proxy that lumped M3 Ultra with M1 Ultra although every recent measurement places it much closer to M5. The discriminator is the `use_qmv_wide` split, so the gate now enables from generation 15 (M3, M4, M5). M3 Ultra, never measured on this pairing before, reads 1.95x (prose), 2.41x (source code) and 2.65x (enumeration) at a verify-round cost of 1.51 classic steps against 2.71 on M1 Ultra, which keeps declining. See `docs/benchmark_results/mtp-b1-gate-m3ultra-2026-08-20.md`.
- **The MTP verify width is decided by measured throughput, not by an acceptance proxy** (#1207). The adaptive controller held a drafter at its configured depth until the configured prefix was usually fully accepted, a proxy that cannot pass on the Gemma 4 12B pairing (acceptance 0.585 over a 3-proposal prefix) no matter how profitable widening would be. The B=1 round loop now alternates 32-round measurement windows between the configured depth and the requested ceiling and holds whichever measures more emitted tokens per millisecond, with a 2% adoption margin, a 4/16/64-window re-challenge backoff, and an early abort that drops a collapsing challenger after 4 rounds. Measured on M5 Max at pinned widths: 89.15 tok/s at 4 against 94.91 at 5, and the controller adopts 5; at a requested 8 (85.96 tok/s) it refuses. `MLXCEL_MTP_BLOCK_CONTROLLER=proxy` restores the previous gate and `=requested` pins the requested width for width sweeps. The batched (B>1) loop keeps the proxy, because the row-averaged accept length is the only per-round signal it measures.
- **`mlxcel-xla` tracks the root version** (this release). The crate was exempted from the workspace version check as a default-off backend outside the release contract. It now carries model support users select (Molmo2 indexed attention pooling, LLaVA and Qwen2-VL image context floors), its worker sits on the server's prompt-cache path, and CI compiles its feature combinations, so a version standing still at 0.4.0 while the code moved was telling readers the backend was dormant. It moves 0.4.0 to 0.6.0 with the rest of the workspace, and `scripts/ci/check_crate_versions.py` now has no exemptions at all.

### OpenXLA backend, now alpha

The OpenXLA / StableHLO backend leaves the experimental shelf and ships as alpha. It stays off by default behind the `xla-backend` and `xla-iree` features, and alpha means what it says: the surface is real enough to run and to report bugs against, not to deploy.

Its crate was exempted from the workspace version check on the reasoning that a default-off backend sits outside the release contract, so `mlxcel-xla` sat at 0.4.0 while 23 pull requests landed in it and none of them reached a release note. That was the wrong trade. The work below shipped between v0.4.0 and now, and is collected here because this is the release that folds the crate back into the workspace version line.

- **Multimodal execution through IREE.** LLaVA vision (#913) against a reference architecture validated end to end (#897), Qwen2-VL vision (#915), Phi4MM audio with per-slot adapters (#914), Gemma3n text runtime with dense PLE prefill (#892), sparse DeepStack prefill (#893), multimodal RoPE position state (#894), Molmo2 indexed attention pooling (#916), and image requests admitted through both the CLI and continuous-batch serving (#895). Image context floors are derived for LLaVA and Qwen2-VL (#1280), so a graph too small to admit an image fails at startup instead of dropping it silently.
- **Session and prefill plumbing.** A prefill-embeddings entry point (#879), IREE sessions seeded from prepared embeddings (#888), and static context capacity as a parameter rather than a constant (#880).
- **Operator numeric contracts.** Auxiliary dtype contracts are versioned (#934) and operator contracts are bound to them (#937), with a bounded numeric oracle harness (#938) and probes for dense matmul (#940), the core numeric set (#941), CUDA prefix scan (#943) and affine Q4 dequant (#944), sharing one emitter helper layer (#942). Issue #932, which defines the contracts these probes check, stays open.
- **Fixes.** The terminating EOS token is no longer emitted as output (#968), diagnostic local-task threads are bounded (#945), backend-seam tests take the env lock on `xla-backend` builds (#1169), the OpenXLA worker handles `PromptCacheWarmup` instead of falling through (#1273), and integration tests link again with `libc` after the IREE archives (#1275). CI compiles the feature combinations (#1282), which is what keeps this list from going stale again.

### Added

- **`GET /v1/internal/mtp-policy` reports the adaptive MTP verdict** (#1257, #1268). A versioned read interface for what the policy settled on and why, so an operator does not have to parse the private hint files, whose empty directory used to mean "still profiling", "no MTP configured" and "the cache root resolved elsewhere" all at once. States are `settled`, `profiling`, `forced`, `unavailable` (with a reason) and, since #1298, `exactness_declined`. The same state rides the `/health` snapshot under `observability.mtp_policy`. See `docs/mtp-policy-api.md`.
- **MTP can draft and verify a token tree instead of a chain, behind `MLXCEL_MTP_TREE`, default off** (#1204, #1212, #1214). The tree path carries per-node RoPE positions (`DraftTree::depths`), because a flattened tree handed to the target as a contiguous span puts every node after a sibling one place too far along, which is a correctness fix for any branching tree and was invisible while only linear trees ran. **The measured verdict is negative and the flag stays off:** a linear tree costs 1.9% to 4.1% against the chain and branching costs 8.1% to 10.1%, with an oracle upper bound of about 6.5%, because branching rescues only the rounds whose chain breaks early while paying for the leaf on every round. The drafter's own top-two confidence does not predict which rounds those are.
- **`scripts/bench_speculative.sh` measures speculative decoding with the protocol the published numbers were taken under** (#1215): baked-in prompts, a quiet-host gate, ABBA alternation with discarded warm-ups, and a spread limit above which a run is reported as untrustworthy rather than averaged. Three-host records for M5 Max, M3 Ultra and M1 Ultra are in `docs/benchmark_results/`.

### Removed

- **`MLXCEL_ENABLE_MTP_DEFERRED`** (#1179). The flag gated a "deferred" verify path that deferred nothing: it split the verify forward from the LM-head projection, producing identical compute with one extra bridge crossing and one extra intermediate tensor. The comment above the gate still described a per-position projection loop that had been rewritten into a single batched graph long before. The flag, the path, and its four orphaned helpers are gone.

### Fixed

- **`/v1/internal/mtp-policy` no longer reports `profiling` forever when the exactness probe vetoes MTP** (#1298). A vetoed pairing never dispatches a burst, so the adaptive policy never accumulates a sample and the endpoint invited the operator to wait for a verdict that structurally could not arrive. The veto is now the state `exactness_declined`, carrying the probe's own one-line reason in a new `decline_detail` field, with `mtp_enabled: false` and no verdict. The label addition stays inside `schema_version` 1 under the documented growth rule. This is what an operator sees today on the Gemma 4 31B plus bf16 assistant pairing, which fails the probe under both kernel selections on every generation 15+ host measured (#1279).
- **A `--lang-bias-config` YAML file with two or more languages now steers deterministically** (#1267). The `bias:` block deserialized into a `HashMap`, whose per-instance iteration order became the first-language-wins priority order. Han is shared by `ja`, `zh` and `ko`, so a config naming two or more CJK languages assigned a different bias to every shared Han token on every run, with no error and no warning, and the shipped schema example is itself a three-CJK config. The block is now collected through `MapAccess` into an ordered `Vec`, so index 0 is the first language written in the file, matching what `--lang-bias` and `LLAMA_ARG_LANG_BIAS` have always done. The accepted YAML syntax is unchanged.
- **A language code repeated inside one YAML `bias:` block is rejected** (#1267). `serde_yaml` resolved a repeated key last-wins with no diagnostic, which made the duplicate check unreachable and let the YAML path accept input the flag parser has always rejected.
- **Four more `HashMap` iteration-order defects, all silent** (#1266, #1281, #1284, #1288, #1299). RT-DETRv2's `needs_sanitize` decided on the first key a `HashMap` walk produced; the distributed registry's node accessors returned an arbitrary order; the pipeline cache manager's eviction sort and the server's four LRU `min_by_key` calls broke ties by whatever the map yielded, so two runs could preempt different sequences from identical state; and the synthetic model fixtures seeded from an unordered key walk. Each is now ordered by a stable key. The defect class is recorded in `docs/code-guidelines.md`.
- **The MTP round-loop diagnostics report the width the loop actually drafted at** (#1206, #1208). The log line carried the requested `block_size`, so a run overridden to the configured depth was indistinguishable in the record from one that ran at the requested width, which made any width sweep unreadable. `effective_block_min` and `effective_block_max` are reported alongside the request, and a controller override that the token budget did not force now warns once.
- **macOS release signing is gated on a Developer ID certificate** (#1216). The workflow signed with whatever identity the keychain offered, which is how v0.0.26 and v0.0.27 shipped assets signed with a since-revoked Apple Distribution certificate that Gatekeeper blocks on launch. The job now gates on the certificate type, pins the identifier, asserts the signature after the fact, and fails the build rather than shipping an asset that will be blocked.
- **The Metal test gate runs single-threaded, like the CUDA one** (#1210), removing the workspace-parallel SIGSEGV that made the nightly red without any code being wrong.
- **The test harness pins f32 GEMMs to full precision** (#1259, #1260). MLX dispatches f32 matmuls to a TF32-class Neural Accelerator kernel on generation 17 when `MLX_ENABLE_TF32` is unset, which defaults on, so 17 algorithmic-equivalence tests (chunked GLA against the sequential recurrence, prefill against the decode chain, absorbed MLA against the decompressed block) failed their full-precision tolerances on M5 hardware. Both lib test binaries now pin `MLX_ENABLE_TF32=0` at startup unless the environment sets it explicitly. Unit tests assert full-precision equivalence; shipping numerics are decided by the runtime probes. The same commit accepts minijinja 2.24's Python-style boolean rendering, which matches HF transformers.

### Performance

- **The dense KV tail trim is a logical rewind instead of a buffer rewrite** (#1209). Every speculative round ended by re-slicing the entire live window to discard a few rejected tail tokens, an O(context) copy paid twice: the re-slice also threw away the step-aligned capacity, forcing the next append through its regrow path. Fp16 and Int8 trims now move only the offset and keep the allocation, the contract `RotatingKVCache::trim` has always had. Measured on M5 Max with `qwen2.5-7b-instruct-4bit` plus a 0.5B drafter: +0.9% at 1k context, +2.5% at 8k, +5.3% at 24k, with the saved term scaling linearly in context. Output is byte-identical. Turbo modes keep the physical trim, because their packed sidecars feed fused kernels that have not been audited for a stale tail.
- **The MTP accept hook builds its paired-hidden block with one slice** (#1185), replacing a chain of per-position `concatenate` calls, and multi-token drafter forwards route through the shared causal wrapper instead of materializing an `[s, s+offset]` mask on every call.
- **The early-exit verify walk was decided by measurement and not built** (#1179). The LM head is weight-read-bound, so a sequential walk re-reads the full 566 to 715 MB weight per walked position and costs 1.2 to 1.6x the batched projection at the production widths (K = 3 to 4, acceptance 0.80 to 0.88), against a ceiling of 2 to 4% of the verify forward. Where a wide block would make the ceiling larger, padding the projection past the qmv batch limit into the matrix-matrix kernel collects it more cheaply (2.56 ms against 4.51 at width 8). Recorded in `docs/benchmark_results/mtp-verify-early-exit-decision-m5max-2026-08-22.md` with the new `examples/mtp_projection_width_bench` harness.
- **The qmv_wide narrow pin's collateral is priced** (#1261, #1278). The exactness gate's retry pins the narrow kernel process-wide, so batched decode that never asked for byte-identity pays for it too. Measured on M3 Ultra and recorded in `docs/benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md`; #1289 tracks the order-preserving streamed kernel that would remove the tradeoff.

### Dependencies

- h2 to 0.4.16 for RUSTSEC-2026-0258 (#1213).
- The minor-and-patch group across 11 updates (#1162), plus a cargo package refresh.

## [v0.5.2] - 2026-08-18

### Changed

- **The pinned MLX C++ commit moves to `9a795735`** (2026-08-17), 168 commits on from `2c46b953`. Four in-tree overlays were three-way merged against the new base and the rest were left alone, which is the whole reconciliation: upstream touched only `mlx/backend/cuda/matmul.cpp`, `mlx/backend/cuda/jit_module.cpp`, `mlx/backend/cuda/reduce/init_reduce.cu` and `mlx/ops.cpp` (the last through 23 commits and 286/151 lines) in that range, and the two Metal overlays (`compiled.cpp`, `kernels/utils.h`) are byte-identical across it. Three merged cleanly; `matmul.cpp` conflicted once, where upstream's `#3929` added `out.size() == 0` to `GatherMM::eval_gpu`'s empty-input guard on the same lines our overlay renamed `a_pre`/`b_pre` to `a`/`b`, and the resolution takes upstream's added condition under our names. Every merged overlay's delta against the new base is the same +/- count it had against the old one, so upstream's changes landed and ours survived. `mlx/backend/cuda/quantized/` is untouched upstream in this range and `qmm/qmm.h` is byte-identical, so the dispatch call sites the `quantized.cpp` overlay warns about needed no adjustment; its sync comment records that rather than repeating the previous bump's reasoning. The `astype` overload taking `std::optional<bool> copy` is gone (`#4207`) and `linspace` gained a required `endpoint` parameter on its primary overload (`#4184`), neither of which reaches this tree: every `astype` call site uses the two- or three-argument form and `linspace` resolves to the defaulted inline overload. **Validated on Apple M5 Max (Metal) only.** The three in-tree fused Metal kernel launchers re-validate well inside their RMS < 5e-3 contract (`sparse_v_kernel_threshold_zero_matches_graph` passes; `delegated_fused_kernel_matches_reference_over_200_steps` at 1.7263e-4; `delegated_steel_envelope_matches_cold_only_fused_over_200_steps` at 1.5259e-4), and the isolated failure sets are **identical** at both pins and deterministic over three runs each: `-p mlxcel --lib` fails the same 13 tests and `-p mlxcel-core --lib` reports 1472 passed / 4 failed. All 17 pass under `MLX_ENABLE_TF32=0` and are the pre-existing M5 reduced-precision class tracked by #1065, not anything this bump introduced. The four merged overlays are CUDA-only and were not compiled or run here; per `CONTRIBUTING.md` the CUDA backend has no CI test gate at all, so `make verify-test-cuda` on NVIDIA hardware remains the only gate they get.
- **The MTP drafter's projections are quantized at load, so a bf16 drafter costs what a 4-bit one costs** (#1185). The drafter is read once per drafted token, so its cost is weight traffic: `qwen3.8-27b-mtp-bf16` is 810 MiB for one decoder layer plus an `fc` projection, and quantizing its eight 2-D projections to the scheme its config declares (affine, group 64, 4-bit) takes that to 228 MiB. Measured on M5 Max against `qwen3.8-27b-4bit`, two reps per arm, alternated: `draft_block` goes from 10.40 and 10.71 to 2.70 and 2.69 ms per round, the accept hook from 10.35 and 10.60 to 2.74 and 2.71, and the verify forward is unchanged. Throughput goes from 37.90 and 35.57 to 49.71 and 50.33 tok/s at 120 generated tokens, which is about 1.19x to 1.5x against classic decode. Acceptance does not move: 0.6831 to 0.6601 at 120 tokens and 0.6500 to 0.6589 at 300, disagreeing in sign, which is noise rather than degradation. Output stays byte-identical to classic decode, structurally rather than by tolerance: the target verifies every proposal, so drafter numerics cannot reach the output and the only exposure is acceptance. Done at load rather than as a second checkpoint, so every existing bf16 drafter gets it without being re-downloaded; a tensor whose `.scales` sibling already exists is left alone and one whose contraction axis is not a multiple of the group size stays dense. `MLXCEL_MTP_QUANTIZE_DRAFTER=0` keeps the checkpoint's precision, and `scripts/tools/quantize_mtp_drafter.py` converts offline.
- **MTP engages on Apple GPU generation 15 and later instead of always declining** (#1187). `use_qmv_wide` in MLX sends `M >= 2` affine quantized matmuls to a kernel that reduces along K differently from the `qmv` that `M == 1` takes, so a verify block was never byte-equal to the single-token chain there and the #1189 gate, which fails closed, declined every time. The gate now re-probes with `qmv_wide` disabled and keeps it off when that is what makes the block exact, which is the path generations 13 and 14 already take. Upstream exposes no knob for this, so `mlx/backend/metal/quantized.cpp` is overlaid with one; `MLXCEL_QMV_WIDE=1` pins the faster kernel and declines MTP instead. The switch costs 17 to 20 percent on the verify forward, and measured on M5 Max over four alternating reps the result is 1.04x classic decode against 1.00x for declining and 1.16x for forfeiting the contract.

### Added

- **`MLXCEL_METAL4_ATTENTION=0` turns off the M5 neural-accelerator fused attention route** (#1065). An off-switch only: the hardware test still gates the route, so a stray value on an M1 is inert. Both copies of that hardware test, `layers::should_use_metal4_attention` and the inline one in `lib.rs`'s causal-attention dispatch, now go through the shared predicate, because a half-applied switch produces a run that looks like the route is off and is not. Measured on M5 Max across `llama-3.2-1b-4bit`, `gemma-3-4b-it-4bit` and `qwen2.5-7b-instruct-4bit`: the route engages on all three (528, 1122 and 924 dispatches over a 32-token generation) and disabling it changes neither the generated text nor throughput beyond noise, so it is a diagnostic rather than a tuning knob.
- **`mlx-community/Qwen3.8-27B-4bit` is qualified on the existing `qwen3_5` path** (#1163). Qwen3.8 declares `model_type: "qwen3_5"` / `Qwen3_5ForConditionalGeneration` and is architecturally identical to Qwen3.5-27B down to a byte-identical weight-map key set, so it already loaded and ran on `main` with zero code changes; that was accidental rather than guaranteed, since `Qwen35Config` carries no `deny_unknown_fields` and silently dropped three keys the generation added that are load-bearing upstream (`output_gate_type`, `rope_parameters.mrope_interleaved`, and the top-level `language_model_only`). Those three are now read at load time and rejected with a named error when they ask for behavior mlxcel does not implement (`output_gate_type: "sigmoid"`, `mrope_interleaved: false`, `language_model_only: true`), across every Qwen3.5-family parse site including the MiniCPM-V 4.6 text backbone, instead of being dropped silently and producing wrong output. **Upgrade note:** `vision_start_token_id` is now mandatory for the Qwen3.5-family VLM path; the stale `248045` default this replaces was wrong for every shipped checkpoint in the family (all use `248053`), and loading with the wrong id used to mis-segment MRoPE vision spans without failing. Every known checkpoint in the family supplies the key, and the new failure is fail-closed at startup with a message naming the missing key, but a deployment running a checkpoint whose `config.json` happens to omit it will see a new startup failure on upgrade rather than the previous silent misbehavior.
- **Qwen 3.5 MTP speculative decoding, driven by the `qwen3_5_mtp` drafter family (`mlx-community/Qwen3.8-27B-MTP-bf16` and siblings)** (#1165). The Qwen 3.5 (dense and MoE, text and VLM) targets now implement `MtpTarget` alongside Gemma 4, so the same `MtpGenerator` round loop, the tick-cooperative slice path, and the run-to-completion burst all drive this pairing on both the offline CLI and the server. Temperature-0 output is byte-identical to classic decode on this pairing only when the Metal chain-parity gated-delta kernel (`gated_delta_step_seqpar`, `MLXCEL_GDN_CHAIN_PARITY`) is taken: the standard kernel carries float32 recurrent state across a verify block and rounds it to the storage dtype once at the end, while the classic single-token decode chain rounds after every token, so without the parity kernel a near-tie argmax can flip about once every 100-250 generated tokens. `mtp_capable_target` gates this pairing on both Metal availability and the checkpoint's GDN shape satisfying the parity kernel's contract, declining to classic decode rather than silently forfeiting exactness when either does not hold. **Measured verdict (M1 Ultra, Mac Studio):** MTP decode is 0.70x classic at the drafter's own configured block size (3) and 0.59x at the historical block-4 default, both below the 1.0x break-even, because the target's multi-token verify forward does not amortize across the block on this GPU generation (about 29 ms per extra block position, roughly two thirds of a full decode step); the adaptive MTP policy (issue #333) profiles and declines this pairing by default on pre-M5 hardware (`mtp_b1_default`), matching the precedent already recorded for Gemma 4 MTP. M5-class hardware is not measured; that cell is open. See `docs/benchmark_results/qwen38-mtp-m1ultra-2026-08-16.md` for the full record, including the DFlash blast-radius A/B (`forward_speculative` is shared verify machinery) and the explicitly unmeasured acceptance-under-concurrent-slice-rotation cell. The drafter-configured `block_size` (3) is now the default when `--draft-block-size` is not passed, instead of the flat Gemma-4-derived constant (4); the PR's own measurement found the configured value faster (16.48 vs 13.81 tok/s, 0.591 vs 0.465 acceptance).

### Fixed

- **`mlxcel generate --draft-model <dflash-drafter>` now fails with a named error instead of a misleading `Weight not found: model.embed_tokens.weight`** (#1168). A DFlash drafter borrows `embed_tokens` and `lm_head` from its target when it binds, so it ships neither, but its `config.json` still declares an ordinary `model_type` (`qwen3`, say), so the offline path classified it as a standalone model and drove it through the full `LoadedModel` loader, which failed on the first missing tensor with a message that named a tensor rather than the problem. Detection now rejects a directory that is structurally a DFlash drafter (a nested `dflash_config` object and/or `architectures: ["DFlashDraftModel"]`) before dispatch, both for `-m/--model` (`get_model_type`, which also covers server startup and the distributed stage loaders) and for offline `--draft-model` (a new pre-load check in `mlxcel generate`), and points at `mlxcel-server --draft-kind dflash` instead. The discriminator is the checkpoint's own markers, not the resolved `DrafterKind`: an ordinary small full model still auto-resolves to `DrafterKind::Dflash` by default (`DEFAULT_DRAFTER_KIND`) and keeps loading through the classic `SpeculativeGenerator` path unaffected, confirmed by a non-regression run (`qwen3-0.6b-4bit` as both target and drafter, acceptance_rate 1.0000, 137.54 tok/s). `docs/supported-models.md` and `docs/speculative-acceptance.md` are corrected to match: DFlash's offline entry is `mlxcel-server` only.
- **A chat template that refuses a caller-supplied value now fails the request with `400` instead of answering `200` from a stripped prompt, and the OpenAI-standard top-level `reasoning_effort` field reaches the template instead of being dropped** (#1164). A refusal raised through Jinja's `raise_exception` used to be swallowed into `render_simple_fallback` and served as `200` from a prompt with no chat framing, no system message, and no tool declarations, with only a server-side `WARN` recording it; the request now fails with `400` carrying the template's own message. The discriminator is type-level rather than a heuristic: `raise_exception` attaches a `TemplateRejection` sentinel as the `minijinja::Error` source and `template_rejection_message` recovers it by walking the error chain, because `ErrorKind::InvalidOperation` is the same kind minijinja raises for genuine engine problems and matching the message text would depend on wording the template author picks. A template mlxcel genuinely cannot render (an unimplemented filter, a malformed template, a fuel-budget blowout) still degrades to the plain prompt exactly as before, pinned by test rather than asserted. Both fallback sites are covered, the typed path and the raw/multimodal path that any request carrying tool calls, a prior-turn `reasoning` field, or typed media parts routes through. The error propagates out of `prepare_chat_request_with_cache`, which the chat, Responses, and Anthropic routes already map to a `400` before generation starts, so the streaming routes return it in place of the SSE stream rather than mid-stream; the disaggregated router front (`src/server/router_front.rs`) is the one exception and reports it as a `500` like every other request-preparation error on that surface. `reasoning_effort` was not on `ChatCompletionRequest` at all, so serde dropped it before anything could act; it is now resolved through the same three-tier chain as `prompt_cache_key` and `user` (top-level field, flattened OpenAI-SDK `extra_body`, nested `extra_body`) and mapped onto the `reasoning_effort` chat-template kwarg behind two guards: an explicit `chat_template_kwargs.reasoning_effort` wins, and the loaded template must actually mention the name, so a checkpoint that ignores the field does not silently acquire a kwarg. Values are not translated: OpenAI's vocabulary is `minimal` / `low` / `medium` / `high` and Qwen3.8's is `xhigh` / `medium` / `low`, so `high` is valid OpenAI and invalid there while `xhigh` is the reverse, and remapping would silently set a reasoning budget the caller did not ask for and could not detect, so `high` surfaces the template's `400` naming the accepted set instead. The mapped value participates in the prompt cache's `template_sig`, so two requests differing only in effort do not share a bucket, and the next-turn warm-up render resolves its kwargs through the same helper so the warmed vector matches the bucket it is filed under; the prompt-cache `preserve_thinking=true` default is deliberately kept out of that shared resolver, so no existing deployment's `template_sig` changes on upgrade. Measured on `models/qwen3.8-27b-4bit` with `max_tokens: 1` and the message `"hi"`, where `prompt_tokens` is the discriminator because the template injects a reasoning-instruction system message whose length is a direct function of the resolved effort: `chat_template_kwargs` `xhigh` and unset both render 53 prompt tokens, `low` 41, and `medium` 11, while `high` and `HIGH` used to answer `200` from a 7-token stripped prompt and now return `400`; top-level `reasoning_effort: low` went from an ignored 200/53 to 200/41 and `medium` to 200/11, top-level `high` returns `400`, and the streaming path returns the `400` with no stream opened. The control `models/qwen3-0.6b-4bit`, whose template never mentions the name, is byte-identical before and after across every case. **Upgrade note:** a deployment serving a checkpoint whose template validates caller input will see that traffic turn from `200` with a silently degraded prompt into `400`. On the four Qwen3.8/Qwen3.5-27B checkpoints in this tree the two highest-traffic new 400s are neither an unsupported role nor an unknown kwarg: a `system` message at any index other than 0 (`'System message must be at the beginning.'`), which fires on the common mid-conversation system-reminder injection and on any second system message, and a conversation with no `user` message at all (`'No user query found in messages.'`), which covers assistant-prefill and system-only requests and also fires when every user turn is `<tool_response>`-wrapped; both are sticky per conversation once triggered. `qwen3.5-27b-4bit` carries eight of the nine `raise_exception` sites in the Qwen3.8 template but not its `reasoning_effort` guard, so it gains the role/ordering 400s above without gaining the effort one. That is the defect being fixed, but it is visible as new 4xx on upgrade rather than as a quiet improvement.

- **A VLM wrapper no longer re-enables a padded prefill its own text backbone disabled** (#1201). `LanguageModel::supports_padded_prefill` defaults to `true` and every hybrid and recurrent family overrides it to `false`, because a tile-aligned prefill appends up to 31 pad positions and, while the causal mask and cache trimming undo their effect on the KV caches, a Mamba / GatedDeltaNet / RWKV / DeltaCache state that has already absorbed them cannot be rewound. `Qwen35VLModel` and `MiniCPMV46VLModel` both hold a `Qwen35Model`, which answers `false`, and both forwarded four other capability predicates while leaving this one defaulted. On Neural Accelerator hardware a text-only run through either wrapper therefore padded the prompt and corrupted the backbone's recurrent state, silently: it compiled, it ran, and greedy output changed whenever the prompt length was not already a multiple of 32. `qwen3.8-27b-4bit` is affected because its `architectures` field routes even a text-only run through the VLM wrapper. A source-level test now fails when a wrapper holds a backbone that refuses padded prefill and does not forward the predicate, and it derives the refusing set by scanning `src/models` rather than hard-coding it.
- **The server gates tile-aligned prefill on the model, not on the hardware alone** (#1201). `execute_full_prefill` and both chunked-prefill paths padded to a 32-token tile whenever the host had a neural accelerator, without asking whether the model tolerates it, so every hybrid family served on an M5-class host had its recurrent state corrupted whenever a prompt or chunk was not already tile-aligned. Unlike the wrapper case this needed no wrapper: it reached Mamba, Mamba2, Jamba, RWKV, Nemotron-H, Falcon-H1, Kimi Linear, LFM2, Plamo2, GraniteMoeHybrid and Qwen 3.5 directly. The same scheduler already declined to pad on its boundary-snapshot path and explained why in a comment, and its batched path read the predicate into a local; these three sites simply never asked. Confirmed by A/B on one endpoint: the same `/v1/completions` request at temperature 0 returns different text before and after the guard. The guard test grows a second case for the second way to lose a guard, a call site that never consults the predicate.
- **The MTP exactness probe compares three inputs before reporting equality** (#1186). One input is not enough to conclude that two arms took the same kernels: measured at op level, mxfp4 group 32 at 5120 to 5120 moves 1 to 11 bytes of 10240 depending only on the operand draw, while the affine row at the same shape moves about 39%, and in that low-amplitude regime a single input is a coin toss. The probe now short-circuits on the first divergence, so the added cost lands only on the passing case at worker startup.
- **MTP is gated on a measured block-vs-chain exactness probe rather than a static check** (#1186). Whether a `T = K` verify block is byte-identical to `K` single-token steps depends on which MLX kernel each quantized projection dispatches at `M = K` versus `M = 1`, which varies by GPU generation, quantization mode, operand size and block width, and cannot be predicted from the checkpoint. The gate measures it on the loaded model and fails closed; `MLXCEL_MTP_ALLOW_INEXACT=1` overrides it loudly.
- **`mlxcel video` uses `-fps_mode` instead of the removed `-vsync`, and the tests no longer hide that** (#1183).
- **A partial or unpinned Muse Glimmer checkpoint skips its tests instead of panicking** (#1172, #1179), and the loading-side pinned guard is gated on `MLXCEL_REQUIRE_PINNED_CHECKPOINTS`.

## [v0.5.1] - 2026-08-15

### Added

- **A history-boundary snapshot taken during prefill makes multi-turn prompt-cache reuse work on snapshot-only families** (#1143). Every model reporting `supports_snapshot_reuse()` could reuse a prompt cache only through an exact-prefix match against a stored token vector, and the vector donated at end of generation is `prompt + generated`, which fails to prefix the next turn for three independent reasons that all live past the history boundary: templates append generation-prompt-only scaffolds, templates drop the `<think>` block when re-rendering an assistant turn as history, and a sampled token sequence is not the canonical tokenization of its own text. The server now takes a second snapshot during prefill, keyed by the tokenization of the `add_generation_prompt = false` render clipped to the longest common prefix it shares with the live prompt, which is a prefix of every follow-up turn by construction. On `qwen3.5-0.8b-4bit`, turn 2 goes from 0 to 150 cached tokens of 189 and turn 3 from 0 to 184 of 214; a `llama-3.2-1b-instruct` dense-KV control is identical in every cell. Total tokens forwarded are unchanged; the cost is one extra graph launch plus a model-state copy on the foreground prefill of every qualifying request, so `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` restores the previous behavior for a deployment serving only single-turn traffic. Greedy output is not bit-exact across the split, for the same near-tie reason `--prefill-chunk-size` and any prompt-cache hit already move it.
- **The next turn's history prefix is warmed in the background** (#1144). The previous assistant reply still landed on the foreground prefill of every turn. After a healthy completion the server now renders the expected history prefix and, when it has nothing else to do, restores the conversation's snapshot and prefills only the delta. On `qwen3.5-0.8b-4bit` this takes turn 2 from 150 to 194 cached tokens of 227 and drops uncached tokens 57%. Warm-ups are dispatched only from the idle tick with an empty active batch, an empty prefill queue, and no parked chunked prefill, verified by counter: 12 requests over 2.6s of continuous load ran 0 warm-ups and skipped 10, and the 2 runs landed after the load stopped. The target vector is the head two probe renders agree on, which is what keeps a warm-up from superseding a working snapshot with one the next turn cannot match. Tool-calling turns are skipped. `MLXCEL_DISABLE_CACHE_WARMUP=1` turns it off.
- **A newer snapshot supersedes its own ancestor within a conversation, and the snapshot budget is operator-controlled** (#1146). A single 31B-class conversation snapshot runs 300-370 MB against a fixed 512 MiB budget with no override, so the second turn's insert LRU-evicted the first turn's snapshot instead of replacing it and two concurrent conversations evicted each other. `insert_snapshot` now removes every stored snapshot whose token vector is a strict prefix of the incoming one, within the same session and the same model/lora/template/multimodal bucket, and it runs before the new entry's bytes are accounted so an extension can be admitted where its ancestor's bytes were the obstacle. A `None` session key never triggers it and two session keys never touch each other. `--prompt-cache-snapshot-capacity-bytes`, `--prompt-cache-snapshot-max-entries`, and `--prompt-cache-snapshot-ttl` are on both binaries with `MLXCEL_*` fallbacks, and `/v1/cache/stats` reports `snapshot_supersedes` separately from `snapshot_evictions_lru` so deterministic in-session replacement is distinguishable from real budget pressure. Snapshot cost tracks model width rather than prompt length, so the budget scales with concurrent conversations, not with turns. The default stays at 512 MiB: deriving it from a per-family state-size formula for the thirteen `supports_snapshot_reuse()` families is left as follow-up work, because getting it wrong silently mis-sizes the store.
- **Gemma 4 restores a snapshot at the longest common prefix instead of requiring an exact one** (#1145). Exact-prefix matching is stricter than rotating-attention hardware demands: the "cannot truncate" constraint holds only after a sliding layer's ring has wrapped. Following the upstream `mlx-lm` `can_trim_prompt_cache` precedent, a Gemma 4 snapshot that shares a long prefix but diverges before its own end is now adopted at that prefix, with both linearity failure modes excluded (a wrapped ring, and the oversized temporary buffer an over-window prefill leaves for one step). An exact-prefix candidate always wins, a refusal stays the classified `snapshot_diverged` reject, and every recurrent family is unchanged, which is pinned by a negative control rather than assumed. Validated on `gemma-4-e2b-it-4bit`, where a diverging turn adopted 66 tokens of a 168-token stored entry.
- **`/v1/cache/stats` and `/metrics` classify a snapshot that diverges from the request it was looked up for** (#1147). A candidate in the request's own session bucket that was not a prefix of it returned a bare `None`, so a structural multi-turn miss was indistinguishable from an empty store and diagnosing one meant detokenizing by hand. The `snapshot_diverged` reject reason now carries the divergence geometry as `last_reject_context_len` and `last_reject_entry_len`, with a `reason="snapshot_diverged"` series on `/metrics`. A cold store and a foreign session bucket emit nothing.
- **`-m/--model` accepts `--revision <REV>`** on `generate`, `run`, `serve`, `inspect` and `mlxcel-server` (#1113), matching `mlxcel download --revision`. Previously a pinned revision could be fetched but not then run by repo-id: the resolver always resolved against `main`. The flag is honoured only where it can be honoured correctly, which is a deliberate limit rather than an omission. The HuggingFace cache probe is revision-aware and answers normally, and a miss fetches the requested revision. The legacy `./models/<name>` directory and the mlxcel store are keyed on `<owner>/<name>` with no revision component, so for a revision-qualified request they are skipped rather than allowed to answer with an unknown revision, and a request whose store directory is already occupied is **refused with an explanation** instead of being silently answered with whatever is on disk. That last case is not hypothetical: the downloader treats same-named non-zero files as "already present" and skips the fetch, which is also why `mlxcel download --revision` can silently return the wrong revision today. Use `--models-dir` to give each revision its own root. `--revision` alongside an existing local path is an error, since a local directory is used exactly as given. Revision-namespacing the store would lift these restrictions but changes an on-disk layout shared with `list`, `rm` and `download`, so it is left as follow-up work.
- **Checkpoints that quantize `q_proj` / `k_proj` / `v_proj` at different bit widths now load on every family that shares the fused QKV projection** (#1090). `mlx_lm`'s `mixed_4_8` predicate raises selected tensors to 8 bits while the rest of a model stays at 4, and the loader concatenated the three packed planes along one axis and inferred a single width from `q_proj`, so such a layer died inside MLX's `concatenate` instead of loading. A layer whose planes cannot be concatenated now keeps them separate, each in exactly the layout the checkpoint stored it in: nothing is dequantized and nothing is requantized, so the values are the checkpoint's and the extra memory cost is zero. This reached all 16 families using this loader, including Llama, Mistral, Qwen2/3, Gemma v1 through v4, Cohere2, StarCoder2, InternLM3 and Jamba. Validated on a `mixed_4_8` Llama-3.2-1B checkpoint, which greedy-decodes byte-identically to the uniform 4-bit baseline. The decision is made on the packed shapes, which is what `concatenate` actually constrains, rather than on the reconciled bit width and group size, which can alias two different packings onto one pair.

### Changed

- **LocateAnything no longer dequantizes its mixed-precision attention layers at load** (#1090). The per-family workaround added in #1070 turned 18 of the released 3B checkpoint's 36 layers' q/k/v planes into dense bf16, about 190 MB, so the fused projection could concatenate them. Those planes now stay packed and the model loads through the shared path. Helium's pre-flight weight validator no longer rejects a `mixed_4_8` attention block either; it stopped comparing the packed width across q/k/v, which scales with the bit width, and still checks each plane's logical input width against `hidden_size`.

### Fixed

- **`mlxcel detect` returned boxes that matched no page content** (#1089). The cause was the host readback in `predictor`, not the coordinate mapping the issue suspected: `pixel_values` is cast to f32 but the first conv against a bf16 weight settles the graph back into bf16, so RT-DETRv2's `pred_logits` and `pred_boxes` come out bf16 for the shipped checkpoints while the readback parsed the raw buffer as 4-byte f32. That fused each adjacent pair of bf16 values into one bogus float and returned half the elements, and the decode then indexed `query * num_labels` and `query * 4` into buffers half the advertised size, misaligning every query, label, and box association without erroring. Outputs are now read at their actual dtype. The aspect-ratio dependence the issue reported was a coincidence of which garbage indices survived; the same collapse reproduces on a square page. Three module doc comments that asserted the whole graph ran in f32, which is what let the mismatch sit unnoticed, now describe what actually happens.
- **An explicit `--kv-cache-budget` that resolves to zero KV blocks now names what consumed it** (#1091). The paged decode v2 workspace reserve is charged to the requested budget before the remainder is divided into blocks (#899), and that reserve is device-derived rather than bounded by what the operator asked for: `device_target_ctas()` is 512 on every non-Metal host, which puts it at about 16 MiB for the common 8-kv-head, 128-head-dim geometry. A byte budget below that resolves to zero blocks and the pool is left unbounded, which is the intended behavior and is unchanged, but the warning blamed model size and available memory, so an operator whose `--kv-cache-budget 8MiB` silently did nothing had no way to see why. The warning now reports the requested budget, the reserve, and the smallest budget that would mint one block. `--kv-cache-budget auto`, the default, keeps its previous message, since reaching zero there really does mean the model leaves no room for KV and pointing at a 16 MiB reserve would send the operator after the wrong knob. No budget resolves to a different block count than before.
- **Server token streams change at `temperature <= 0` when DRY is enabled with sequence breakers.** A per-request `dry_sequence_breakers` value was dropped by the greedy branch of `build_sampling_config` and replaced with an empty vector (#1102). DRY is not gated on temperature, and the breakers are the backward match's termination condition, so the match ran past the intended boundary and the penalty came out at or above what the request asked for. Output for those requests changes, toward the requested configuration. Requests that leave `dry_multiplier` at its `0.0` default, or that set no breakers, are byte-identical to before. The CLI is unaffected: it has no way to set breakers (#1108).
- **`--dry-sequence-breaker` reaches the sampler instead of being parsed and discarded** (#1103). The value flowed from the CLI into `ServerStartupConfig` and stopped there: there was no tokenization step, no server-side default for `build_server_generate_options` to fall back on, and no `/props` entry, so an operator got a clean startup and DRY matching that ran straight through the breakers they configured, with nothing to reveal the flag was inert. The failure was silent in both directions, since running DRY without breakers makes the penalty stronger than configured rather than weaker. Breaker strings are now tokenized at startup and used as the default for any request that does not send its own `dry_sequence_breakers`; `/props` reports the resolved token IDs. Two behavior changes for existing deployments that already pass the flag: generation output changes for requests that omit the field (the flag now does what it says), and a breaker that does not encode to exactly one token for the loaded model **fails startup** with a message naming it, rather than being dropped. The escapes `\n`, `\t`, `\r` and `\\` are interpreted, so the `"\n"` in the flag's own help text works as written, and the tokenizer's prepend normalizer is discounted, so a breaker resolves to the token the operator meant on SentencePiece-derived checkpoints (Mixtral, Phi-3, MiniCPM, LLaVA and others) rather than failing startup or silently resolving a neighbouring token.
- **Four server flag spellings that worked on only one of the two binaries now work on both** (#1109). `mlxcel serve` accepts `--parallel` and `--predict`; `mlxcel-server` accepts `--n-parallel` and `--adapter`. The parallel-slots pair was the worst case: `--n-parallel` worked only on `mlxcel serve` and `--parallel` only on `mlxcel-server`, so a command line copied between them failed to parse even though both flags read the same `LLAMA_ARG_N_PARALLEL` environment variable. The DRY sequence breakers now use the llama-server spelling `--dry-sequence-breaker` as the primary name on both binaries; the plural `--dry-sequence-breakers` that `mlxcel serve` previously required is kept as an alias on both, so no command line that worked before stops working.
- **The "not a model" error names all three forms `-m/--model` accepts, not two** (#1114). It described a local path and an `owner/name` repo-id, omitting the bare name that resolves against `$MLXCEL_DEFAULT_ORG` (default `mlx-community`), which the README puts in the quick start. That omission was worst exactly where the error fires: a bare name containing a character outside `[A-Za-z0-9._-]` is what falls through to it, so a user who typed `mlxcel run "Qwen3 4B"` was told to type a full `owner/name`, more work than fixing the typo and keeping the bare name. The message now names all three forms and states the character class, matching the sibling `MLXCEL_DEFAULT_ORG` error on the same path, and keeps the `mlx-community/Qwen3-4B-4bit` example.

## [v0.5.0] - 2026-08-12

### Added

- Meta Muse Glimmer (`muse_glimmer`) vision-language support (#1101). The 52-layer mixed-cache decoder and the 50-layer vision and fusion path drive single-image and multi-image generation through both the CLI and the continuous-batching server, with expanded prompt accounting and fail-closed unsupported modes. The pinned Muse chat template, reasoning strengths, and bounded ATEM parsing and replay are wired across the Chat Completions, Responses, and Anthropic-compatible routes, including streaming cleanup, so reasoning is routed to `reasoning_content` rather than leaking into the answer. Qualified on a real NVIDIA GB10 against the pinned bf16 checkpoint: coherent text, grounded single-image and multi-image output, a 2204-token prompt crossing the 2048-token sliding-window boundary, ATEM tool call and result replay, and scheduler image concurrency at `--parallel 1` and `--parallel 2`.
- `mlx-community/Muse-Glimmer-30B-4bit` loads through the same path (#1116), preserving the canonical bf16 path and the dense vision tower. mlx-vlm weight roots are normalized, the root quantization contract is inherited into the text configuration, and text plus vision-fusion projections load through quantization-aware layers. Warm text qualification on GB10 reached 12.43 prefill and 13.34 decode tok/s, about 3.1x the recorded 4.25 tok/s bf16 baseline on the same box, and the image path inserted 64 vision tokens and decoded at 13.15 tok/s. Checkpoint-format support is fail-closed to the pinned affine-Q4 layout: conflicting root and nested quantization contracts, unsupported modes, invalid group-size and bit combinations, alias collisions, orphan sidecars, missing affine biases, global-scale sidecars, and quantized vision-tower tensors are all rejected before kernel selection, identically on the VLM and text-only paths. One-shot CLI generation now reuses the server ATEM channel parser, so `to=self` reasoning and structural tokens stay hidden by default while `to=user` content remains visible.
- Florence-2 (`florence2`) encoder-decoder vision-language support, landed across the five sub-issues of epic #850: the reusable BART seq2seq engine and text core (#1060), the DaViT vision backbone (#1063), vision-language fusion and full weight loading (#1064), the processor with the fifteen task markers and `<loc_N>` coordinate parsing (#1069), and CLI integration with real-checkpoint validation (#1071). Task answers parse back into boxes, quadrilaterals, and polygons in original-image pixels. Nine upstream defects are reproduced deliberately and pinned by tests rather than silently corrected, including the `post_process_generation` docstring that documents `image_size` as height by width where every call site unpacks it as width by height.
- Quantized Florence-2 checkpoints load at 3, 4, 6, and 8 bits, both `base-ft` and `large-ft` (#1082). The port is pinned against upstream mlx-vlm running the same 4-bit checkpoint rather than against bf16, because comparing against bf16 can only measure lossiness: relative RMS agrees to 5.8e-4 and cosine to 2.6e-4 at every stage, and the greedy ids match exactly. `image_projection`, the cosine temporal buffer, the conv stack, and the normalization weights are consumed dense, and a checkpoint that packs one of them is still refused at load with the offending tensor named.
- `mlxcel-server` serves Florence-2 through a dedicated single-stream seq2seq worker (#1083), replacing the startup refusal. `message.content` is byte-identical to the CLI answer and the parsed coordinates arrive as JSON in the `message.florence2_result` extension field. Task-prompt input is validated at the request boundary for all seven input-taking modes, and images decode under the configured `ImageInputLimits`. Serving is one request at a time; no concurrent-throughput property is claimed.
- LocateAnything (`locateanything`) grounding VLM support (#1070): a MoonViT tower reused from Kimi-VL, an MLP connector, and a Qwen2 decoder that emits `<ref>` / `<box>` markers and the 1001 coordinate tokens through plain autoregressive decode. Two real-checkpoint gaps are handled that the Python reference does not show: the released conversion is `mixed_4_8`, which the shared fused-QKV loader cannot concatenate, so affected layers are dequantized rather than requantized onto a grid they do not land on; and the checkpoint ships no `tokenizer.json`, so a Qwen2 slow-tokenizer reconstruction is added as a last resort, gated on `tokenizer_class`.
- Falcon-OCR (`falcon_ocr`) early-fusion document OCR support (#1075), the first family in the tree with no vision tower at all: each 16x16 RGB patch is flattened through one linear projector straight into the token stream, and a single 22-layer decoder reads image and text together under a mask that is bidirectional inside every image block and causal everywhere else. `mlxcel generate --layout-detections <FILE>` runs the layout-aware second stage over detections in the shape `mlxcel detect --format json` prints. Layout detection itself is not included; the boxes are an input.
- Jina VLM (`jvlm`) support (#1076): a 27-layer SigLIP-so400m-class tower feeding a Molmo-style connector into a Qwen2-class decoder on an OLMo tensor layout. The MLX conversion drops the checkpoint's chat template, and under the generic fallback the model answers `17` when asked for the capital of France, so a built-in template is consulted after every declared source comes up empty.
- `make bump-version VERSION=x.y.z` rewrites every version-tracking manifest and syncs `Cargo.lock` in one step, and `make verify-versions` derives the member list from `[workspace] members` at runtime instead of from prose, so it cannot go stale (#1056). Adding a sixth member fails the check until someone records whether it tracks the root version.
- `make verify-kernel-dtype-keys` and a CI job require every `template_args` initialiser in a file that also contains a `cuda_kernel(` call to name at least one input dtype (#1059). There is deliberately no allowlist: Metal-only launchers are excluded by the absence of that call, so adding a CUDA port to one brings it under the check automatically.
- Ant Group Ling / Bailing MoE (`bailing_moe`) text model support (#838).
- Apple OpenELM (`openelm`) text model support (#839).
- TeleAI TeleChat3 (`telechat3`) text model support (#841).
- Databricks DBRX (`dbrx`) text model support (#835).
- Ling / Ring linear-attention MoE (`bailing_moe_linear`) text model support (#840).
- Phixtral (`phi-msft` sparse MoE) text model support (#844).
- Arcee AFMoE (`afmoe`) text model support (#845).
- Kuaishou Klear (`klear`) text model support (#846).
- Fused paged-attention decode v2, with a CSR page table, a cross-CTA split-KV partial kernel, and a variable-length merge kernel (#898). v1 splits the KV inside one threadgroup, so one CTA serves one `(batch, query head)` pair and a long context adds no parallelism; v2 splits across CTAs, so parallelism is `num_chunks * kv_heads * q_groups` and grows with context. The CUDA bodies are structural transliterations of the Metal ones and have never been compiled or run.
- Batched paged decode is routed through the v2 kernel in the server (#899). On M1 Ultra the switch is gated by measured floors rather than applied unconditionally: batch 1 loses at 1024 tokens of context (0.91x) and wins at 4096 (1.08x median), so single-request launches need 4096 visible tokens, while batch 4 and batch 8 win at 1024 tokens per request (1.41x and 1.47x), so batched launches need 512 per request. `MLXCEL_PAGED_V2_MIN_KV_TOKENS` and `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST` move the floors; `MLXCEL_PAGED_ATTENTION_NATIVE=0` pins the previous gather path.
- Fused sparse-attention decode by page indirection over the v2 kernel (#904). A sparse selection becomes a `page_size = 1` page table, so no gathered copy is materialized. MiniMax-M3 block-sparse decode is routed onto it above a sparsity gate (default 8x, `MLXCEL_SPARSE_PAGED_MIN_SPARSITY`), which is where the measurement turns: at MiniMax-M3 geometry the fused path runs 0.67x at 2x sparsity and 0.77x at 4x, then 1.22x at 8x, 1.17x at 16x, and 2.06x at 32x. DeepSeek Sparse Attention is not routed onto it; the reason and the kernel generalization it needs are written up in `docs/sparse-paged-decode.md`.
- Cascade (shared-prefix) decode: a whole-page prompt prefix shared by several sequences in a decode batch is attended once for the subgroup and merged into each member's suffix state instead of being re-read per sequence (#903). Default off (`MLXCEL_CASCADE_ATTENTION`); correctness is verified, throughput is not yet measured.
- Fused residual-add RMSNorm and fused RoPE + KV-append decode kernels, both default on with `MLXCEL_FUSED_ADD_RMSNORM=0` and `MLXCEL_FUSED_ROPE_APPEND=0` as kill switches (#905).
- Shape-bucketed kernel autotuner (`mlxcel_core::autotune`) and a cold-last-level-cache microbenchmark harness (`mlxcel_core::bench_rotation`) (#906). The v1 paged-decode `NumSplits` launch shape is the first consumer, validated on Apple Silicon; the two CUDA knobs are wired but unvalidated.
- Matrix-absorbed MLA decode over a compressed-latent KV cache (`mlxcel_core::mla`), wired to `deepseek_v2` behind `MLXCEL_MLA_ABSORBED` (default off) (#907).
- Sorting-free top-p sampling by dual-pivot rejection kernels on Metal and CUDA (#901). Where top-p is active, the fused sampler now replaces the `argsort` + `cumsum` nucleus filter and the trailing `random::categorical` with one softmax and one custom kernel that resolves the filtered support by rejection sampling on a shrinking probability interval, with no sort anywhere. Measured 1.28x to 2.35x on M1 Ultra across vocab {32K, 64K, 152K} and batch {1, 4, 8}. One launch covers the whole batch, and the kernel takes per-row `{top_k, top_p, min_p}` so rows with different values need no second launch. Routing is deliberately restricted to what measured faster: top-k alone, min-p alone and top-k with min-p keep the stock chain (the kernel measured 0.31x to 0.97x there, because those configurations make the chain run no sort), and top-k with top-p routes only up to vocab 32768 (1.27x to 1.64x at 32K, 0.71x to 0.83x at 152K). Declines are announced once at INFO with the numbers behind them. The routed path evaluates nothing, so it does not collapse the software pipeline that both decode drivers run; the kernel's convergence flags are checked without waiting, on a later call. `MLXCEL_SAMPLING_REJECTION=0` restores the previous chain everywhere.
- Sampling dispatch outcomes are announced at INFO, once per distinct outcome kind (#901). Every `fused_sample` branch (greedy argmax, the #900 Gumbel-max kernel, the #901 rejection kernel, the kill switch, an unsupported backend, and the convergence-cap fallback) records what it did and why, so which sampling path a run took can be read off a normal server log instead of inferred from a benchmark. Cap-overflow events are also a reachable counter, `mlxcel_core::rejection_cap_overflow_rows()`.
- Softmax-free Gumbel-max categorical sampling kernel on Metal and CUDA (#900). On the no-filter stochastic sampling path (`temperature > 0` with no top-k, top-p, or min-p) the fused sampler now adds i.i.d. Gumbel noise to `logits / temperature` and takes the argmax, which draws exactly from the same softmax categorical distribution without the normalization pass over the 32K-152K-entry vocabulary. One launch covers the whole batch. `MLXCEL_SAMPLING_GUMBEL=0` restores the previous `random::categorical` path.
- Opt-in acceptance-optimal rejection sampling for the classic speculative path, with the proof, regression guards, and closed-form acceptance diagnostic (`MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1`, default off) (#902).
- `MLXCEL_EXTRA_CA_CERTS` accepts a path to a PEM bundle, added to the default TLS trust roots of both HTTP clients, so mlxcel can fetch checkpoints from behind a TLS-inspecting corporate proxy (#912).
- `MLXCEL_MIXED_STEP` prototype and ADR 0005, which records why model-level ragged mixing and kernel-level mixing on MLX were rejected (#908). The issue's premise was inverted: tick alternation never stalls decode, because a chunked prefill does not interleave at all, it waits for the batch to drain.
- The `modelsize` / `modeltype` / `arch` label taxonomy is documented (#1038).

### Changed

- Chunked GLA prefill is the `bailing_moe_linear` default (#1062). #1039 shipped both evaluations of the recurrence and left the promotion to a measurement; the measurement lands the opposite way from what the opt-in assumed. On `mlx-community/Ring-mini-linear-2.0-4bit` and an M1 Ultra, chunked wins perplexity in every configuration measured, by 1.50% at a 128-token window rising to 13.75% at 512 across 32 windows, because the intra-chunk sum lands in a matmul accumulator instead of compounding in a bf16 running state. Prefill is 2.06x to 2.39x faster from about 512 to about 8192 prompt tokens, and decode is unchanged by construction. The cost is real and is stated rather than buried: a checkpoint decoded here no longer matches mlx-lm token for token, since upstream implements the sequential path. `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=0` restores it.
- The pinned Rust toolchain moves from 1.93.1 (2026-02-11) to 1.97.1 (2026-07-14), and the `dtolnay/rust-toolchain` tag in `ci.yml` tracks it as the comment there requires (#1066). The workflows that install `@stable` are unaffected and were never building at a different version: that action runs `rustup default` and never exports `RUSTUP_TOOLCHAIN`, so `rust-toolchain.toml` overrode it per directory and every cargo invocation in the tree already resolved to the pin. `cargo fmt` produces no diff at the new version, so the bump reformats nothing, but six new clippy lints fire under `-D warnings` and are fixed here: `question_mark` in `memory_estimate.rs` and `sanitize.rs`, `collapsible_match` in `chat_request.rs`, `for_kv_map` and `unnecessary_cast` in two test modules, and `unneeded_wildcard_pattern` in `pipeline_remote_real_models.rs`. All six are mechanical and behavior-preserving.
- The Debian and Launchpad packaging tree is removed (#1095). It was unshippable, and release preparation carried it as a dead path.
- Fused decode-MoE byte-identity is documented as checkpoint-dependent rather than as a general property, which it never was (#1061). Measured on an M1 Ultra over 96 calls per checkpoint, the fused kernel sits at 1.65e-3 normalized RMS from an all-f32 ground truth while the `gather_qmm` fallback sits at 1.02e-2, so the fused kernel is closer to truth in 96 of 96 calls on both `Klear-46B-A2.5B-Instruct` 4-bit and `qwen3-30b-a3b` 4-bit, median 6.2x. The disagreement between the two paths is mostly the reference's own distance from truth. `docs/adding-models.md` gains the guidance to set `MLXCEL_FUSED_MOE=0` when reference-diffing a MoE port, since the kernel engages only at decode and makes prefill look exact.
- Florence-2 `large-ft` degenerate output is recorded as an upstream checkpoint and conversion-family issue with a tracker link (`Blaizzy/mlx-vlm#1840`), not an mlxcel loader defect (#1097). The same run of BOS reproduces in mlx-vlm at both bf16 and 4-bit; `base-ft` stays the documented working baseline.
- `TECHNICAL_REPORTS/` opts into git tracking through the `.keep-reports` marker the report workflow already defines (#1080), after the reports for the #1053 / #1054 / #1045 / #1040 chain (#1077) and the #847 / #848 / #849 VLM port chain (#1081) landed force-added by hand. The sixty-nine reports predating that are deliberately left untracked.
- **Top-p sampling streams differ at equal seeds, and top-k + top-p changes support slightly.** The rejection sampler (#901) consumes the shared MLX random key sequence differently from `random::categorical`, so a fixed seed no longer reproduces a token stream recorded before this release on the paths it is routed to (top-p active). Configurations that stay on the stock chain are bit-identical to before. The truncated support is unchanged for top-k alone, top-p alone, min-p alone, top-k+min-p and top-p+min-p (support equality against the `argpartition` mask including ties is a committed test), and frequencies inside the support pass chi-square against the renormalised truncated distribution. When top-k AND top-p are both active the support does change: the stock chain renormalises over the top-k set before applying top-p, so its mass target is `top_p * Z_k` with `Z_k` the top-k mass, while the kernel applies both tests to the untruncated distribution and therefore keeps a superset. Greedy decoding (`temperature == 0` or `top_k == 1`) is byte-identical. To reproduce a stream recorded before this release, or to restore the renormalised top-k+top-p semantics exactly, set `MLXCEL_SAMPLING_REJECTION=0`.
- **Sampled token streams differ at equal seeds.** The Gumbel-max sampler (#900) consumes the shared MLX random key sequence differently from `random::categorical`, so a fixed seed no longer reproduces a token stream recorded before this release on the no-filter stochastic path. The sampling *distribution* is unchanged (verified by chi-square goodness of fit against exact softmax probabilities on peaked, flat, bimodal, and `-inf`-masked logits at temperatures 0.5, 1.0, and 1.5), and greedy decoding (`temperature == 0`) is byte-identical. Runs are still fully reproducible going forward: the same seed on the same backend gives the same stream. To reproduce a stream recorded before this release, set `MLXCEL_SAMPLING_GUMBEL=0`.
- The pinned MLX C++ commit moves to `2c46b953`, and three in-tree overlays are re-derived against it (#1042).
- The MLX pin has one source of truth. It was duplicated across `CMakeLists.txt`, `build.rs`, and `release.yml`, where a mismatch produced stale build artifacts and CI breakage; the three now read one value (#1047).
- Loop detection is less trigger-happy. `LOOP_DETECTION_RECOMMENDED` moves from `min_count = 4` to `12`, and the Gemma 4 family default-on is narrowed to the requests where the collapse actually happens (#967). At the old threshold a markdown table alignment row (`| :--- | :--- | :--- |`) tripped the detector at its fourth column, so a correct answer stopped mid-table and the client saw a normal stop with no error.
- The paged block-pool slab size is a per-pool value instead of a constant 32. The server sizes it from the configuration it already reserved KV memory for, clamped by the per-layer share of the paged block budget; `MLXCEL_PAGED_SLAB_BLOCKS=0` pins the old default (#899).
- The quality gate covers all workspace members (#1007). The workspace root is itself the `mlxcel` package, so the previous bare `cargo test` and `cargo clippy` resolved to `-p mlxcel` and never built `mlxcel-core`, `mlxcel-surgery`, or `mlxcel-xla`: 1754 tests, 1354 of them `mlxcel-core`'s, plus the whole test-target lint surface of those members. `make verify-test` and `make verify-clippy` now pass `--workspace`.
- Tests build under a new `[profile.test-fast]` rather than `[profile.release]` (#1000). It inherits from release and keeps `opt-level = 3`, so the optimized MLX numerics the suite depends on are unchanged, but drops fat LTO and `codegen-units = 1`, which were making the nightly spend its entire 180-minute budget in codegen and linking without ever starting a test. `release.yml` still builds and links under `[profile.release]`. An unfinished nightly run is also reported instead of silently skipped: a timeout was recorded as `cancelled`, which skipped the "Report a red main" step.
- The CUDA merge gate is runnable and serialized (#1048).
- The `rust-toolchain` pin is restored and excluded from dependabot bumps (#952).
- Dependency updates: `dtolnay/rust-toolchain` 1.93.1 to 1.100.0, `actions/setup-python` 6 to 7, and minor and patch bumps across 10 packages (#936).

### Fixed

- **CUDA JIT launches read their buffers through the wrong pointer type when one process ran a kernel at two input dtypes.** MLX's Metal backend folds input dtypes into the JIT cache key; its CUDA backend names a kernel `"custom_kernel_" + name + template_arguments_hash(template_args)` and stops there, while generating every buffer parameter's type from the runtime dtype. A launch whose template args are all ints therefore hashed to one name for every dtype, so the first dtype to compile won for the life of the process and every later call returned numbers unrelated to its inputs, with nothing thrown. Fixed for the paged decode v2 partial and merge kernels (#1058) and then for the rest of the class: the v1 paged attention kernel, `sampling.cpp`, and `sampling_rejection.cpp` (#1059). The sampler was affected in production and not only in tests, since it admits f32, f16, and bf16 logits and its launch shape depends only on batch and vocabulary. Metal was never affected.
- `mlx::core::dequantize` returned wrong values whenever its `biases` input was not row-contiguous, so the FFI shim now forces contiguity (#1078). MLX binds `w` and `scales` on the shared compute encoder and only then calls `ensure_row_contiguous` on `biases`; for a strided input that enqueues a copy kernel on the same encoder, which rebinds the first two buffers to its own operands, and the dequantize dispatch that follows decodes the raw bias bytes as quantized weights. The window opened with the MLX pin bump in #1046 and was two days wide. `gemma4_unified`, `mistral4`, and `qwen3_5_moe` all hand strided quantized triplets downstream; the guard sits at the boundary, so it covers every caller.
- Two Molmo v1 seams that corrupt image-conditioned generation (#1099). A flat `Molmo-7B-D-0924` config with no `rope_impl` now defaults to the checkpoint's effective LLaMA rotate-half layout instead of MLX traditional RoPE, and image preprocessing pads raw black before normalization and preserves fractional coverage in `image_masks` instead of thresholding them to booleans. The pre-fix build reproduced the reported garbage byte for byte on a 640x480 COCO image; the fixed build describes the scene correctly through both the CLI and the server.
- Queue admission, image cardinality, and usage accounting on the dedicated single-stream workers (#1098). DiffusionGemma, LLaDA-2 MoE, and Florence-2 now take atomic RAII queue reservations before streaming SSE opens, on the Chat, Responses, Anthropic, `/v1/completions`, and `/completion` routes alike, so a request losing the final race returns HTTP 503 at the route boundary instead of mid-stream. A declared image count that disagrees with the resolved one is rejected during shared request preparation with a named error, and the rejection provably does not poison the worker's next request. Florence-2 usage reports the actual fused BART encoder length rather than a stand-in.
- The Gemma 4 server reasoning splitter consumes the full `<|channel>thought` opener, so `reasoning_content` no longer starts with a leaked `thought` channel argument (#1094). The visible answer path is unchanged.
- `mllama_parity::sub_max_real_tiles_keep_the_legacy_real_rows_byte_identical` no longer demands exact f32 equality (#1065). Selecting 1 real tile of 4 makes the vision encoder reduce over a different extent than the all-tiles path, and f32 addition is not associative, so the equivalence that holds in exact arithmetic never implied bitwise equality. On Apple M5 Max the difference is 5.9604645e-8 (2^-24) against a largest output element of 1.1521907, which is 0.5 ULP; a real row-selection error would surface seven orders larger. The assertion moves to a named 1e-6 bound. This is the same defect #953 fixed for the sibling assertion in `src/models/mllama/text.rs`, which missed this file. The two chunked-SDPA assertions in `layers.rs` now also report the measured divergence instead of only naming the chunk size.
- Two test suites no longer flake on a loaded runner. The autotune profiler ordering tests run on an internal deterministic timer seam instead of host `thread::sleep` accuracy, leaving production profiling on real `Instant` timing (#1096), and the `tp_e2e` scaling-efficiency assertions are recomputed from the throughputs the analysis recorded and bounded by an analytical ceiling rather than by a hand-picked 1.1 (#1057).
- **Four model families ran a fully bidirectional prefill.** Generation calls `forward` with `mask == None` and expects the model to build its own causal mask; `deepseek_v2` (#991) and then `internlm3`, `hunyuan`, and `gemma2` (#999) passed `None` through to attention instead, so every prompt token attended to every later one. Output stayed fluent, which is why it survived: a short prompt cannot expose it. `gemma2` was the mitigated case (its sliding window bounded the leak); PaliGemma 2 was checked and is unaffected.
- Speculative decoding lost about half its throughput to a draft KV cache rewind arithmetic error (#994). Aggregate decode goes from 41.8 tok/s and a 3.09x run-to-run spread to a 1.06x spread.
- Quantized Jamba MoE experts are built through the shared `SwitchGLU` (#974). The family-local expert construction was broken for every Jamba MoE checkpoint, dense ones included, not just quantized ones. The router moves to `UnifiedLinear` so a checkpoint that ships 8-bit routers under a 4-bit default loads.
- `deepseek::ModelArgs` dropped the nested `quantization` block from `config.json`, so the values reaching the expert loaders were not the ones the checkpoint declared (#975).
- `QuantizedMultiLinear` infers its quantization mode instead of assuming affine, and validates the bias plane (#1028).
- The declared `quantization.mode` string is bounded by an allowlist at load. An unparseable mode reached MLX verbatim and a block-float one was silently reinterpreted as affine, both aborting at the first forward instead of failing at load (#973).
- Quantization parameters are bounded in the family-local MoE expert loaders (#958) and rejected at load when MLX would abort on them (#929).
- The MLA `kv_b_proj` sanitizers refuse a scales-without-biases plane, in wording standardized across all five implementations (#1026), and `mamba` / `mamba2` treat a scales-without-biases embedding as non-quantized rather than misreading it (#976).
- GLM4 MoE Lite loads its own layout: `kv_b_proj` is decomposed into the per-head `embed_q` / `unembed_out` pair (#1029).
- `rope_traditional` from `config.json` is honored on the shared Llama path (#931).
- `topk_group` is bounds-checked inside `group_mask_scores` (#947).
- KV-cache estimation recognizes the `n_layer` / `n_embd` and `multi_query` config spellings (#927).
- LoRA adapter weight fusion is Conv1D-layout aware (#925).
- Per-expert tensors are stacked from borrows instead of copies, removing one `Copy` graph node per per-expert tensor: 5376 of them for `ling-lite-1.5` (#948).
- A parked chunked prefill's wait is bounded by a fairness grant, cutting worst-case admission latency from 50.5s to 23.8s at the default `--prefill-grant-interval` of 16, at 1.60x mean stream ITL during admission (#1011).
- Grammar-only requests are excluded from loop detection (#977), and generation stops when the structured matcher completes (#978).
- Disaggregated router chat guards are enforced (#979), and client token budgets are clamped (#980).
- CUDA: the broken RMSNorm overlay is dropped and DeepSeek-V2 graph capture is restored (#831). The upstream MLX CUDA small-axis attribution is corrected in the docs (#830), and the MLX CUDA graph-cache lifetime-miss abort is reported upstream (#821).
- CUDA: the fused RoPE kernel calls libdevice `cosf` / `sinf` instead of the inaccurate variants (#1049).
- XLA: the terminating EOS token is no longer emitted as output (#963), the Qwen2-VL prefill export is gated on the feature that calls it (PR #961), and the Qwen2-VL XLA loader contract is pinned with all three outcomes documented (#966).
- Test-harness fixes: a second `mlxcel-core` test binary sharing the GPU now fails loudly instead of aborting the run (#1008), the fused MoE GeGLU parity bound is derived from the reference's measured jitter (#964), the mllama ragged test tolerates f32 reassociation and gained a nightly gate (#939), the autotune rep-scaling test no longer pins an exact ceiling (PR #990), a dead-callsite race behind the drafter log assertions is removed (#1023), and the test harness honors `CARGO_TARGET_DIR` (PR #996).
- Homebrew tap formula stanzas are edited by name instead of line position (PR #949).
- The dead `AGENTS.md` links in `CONTRIBUTING.md` are replaced (PR #1014).

## [v0.4.3] - 2026-07-27

### Added

- GPT-2 text model support (#924).
- GPT-BigCode text model support (#926).
- GPT-NeoX text model support (#928).
- Kyutai Helium text model support (#930).
- Gemma3n text runtime and dense PLE prefill on the OpenXLA backend (#892).
- Gemma3n audio reference path (#883).
- Token-exact Phi4MM audio reference path (#887), and Phi4MM audio with per-slot adapters on OpenXLA (#914).
- LLaVA vision execution through IREE (#913), with an end-to-end reference-architecture test (#897).
- Qwen2-VL OpenXLA vision path (#915).
- Sparse DeepStack prefill on OpenXLA (#893).
- Multimodal RoPE position state on OpenXLA (#894).
- Image requests admitted in the CLI and continuous-batch serving (#895).
- Bounded audio preprocessing and serving plumbing on OpenXLA (#896).
- Prefill embeddings entry point (#879), and IREE sessions seeded from prepared embeddings (#888).
- Parameterized static context capacity on OpenXLA (#880).
- `--show-reasoning` CLI flag to print the reasoning channel that is otherwise hidden (#889).
- OpenXLA operator numeric contracts and a bounded numeric-oracle probe harness covering dense matmul, affine Q4 dequant, prefix scan, and core ops (#934, #937, #938, #940, #941, #942, #943, #944).
- Intra-MoE phase profiling for nemotron-h (#832).

### Changed

- The Gemma 4 reasoning channel (`<|channel>thought ... <channel|>`) is hidden from CLI output by default. Pass `--show-reasoning` to print it. Server `reasoning_content` behavior is unchanged (#889).
- The IREE compiler and runtime are pinned to 3.12.0rc20260721 on both CUDA and macOS, fetched from the official GitHub release and verified by sha256 (#882).
- VLM host prefill preprocessing was refactored to owned buffers (#881).
- OpenXLA diagnostic local-task thread count is bounded (#945).
- Dependency updates: minor and patch bumps across 11 packages (#828).

### Fixed

- 4-bit CUDA decode was non-deterministic at temperature 0 and could emit a stray token. The `qmm_sm80` quantized-matmul kernel reused shared memory still being written by in-flight `cp.async` copies; it now drains outstanding copies before the epilogue store, so greedy decode is byte-identical run to run on every 4-bit model (#910).
- The fused decode-MoE kernel rounded each of the K per-expert partials to bf16 before summing, which corrupted long multi-turn Gemma 4 output. Partials are now accumulated in f32 and rounded once (#886).
- Gemma 4 chunked-prefill continuation dropped the sliding-window attention mask when the caller mask was shorter than the returned keys, collapsing output to reserved `<unused>` tokens under concurrent load. The mask is now sized to the keys each attention family returns (#891).
- The Gemma 4 reasoning channel and its chain-of-thought no longer leak into CLI output as raw tokens (#889).

## [v0.4.2] - 2026-07-20

### Added

- MiniMax-M3 text model: a hybrid dense/MoE architecture with block-sparse attention (#799).
- MiniMax-M3-VL multimodal support (#800).
- Unlimited-OCR: long-document OCR on the DeepSeek-OCR stack with a per-layer ring sliding decode cache that keeps the full prefill and rotates only the most recent decode window (#801).
- XTC (Exclude Top Choices) sampling, read and applied end to end on the OpenAI-compatible chat, completions, and responses routes, with range validation (#802).
- `thinking_mode` chat-template kwarg is injected when the template references the identifier and thinking is enabled (#811).
- Per-reason prompt-cache reject metrics and an APC trace log (#810).
- Monotonic OOM backstop and a persistent OOM record for decode benchmark sweeps (#808).

### Fixed

- DeepSeek-V2-Lite generated repeated tokens on GB10 CUDA. An upstream MLX 0.32.1 RMSNorm kernel regression is overlaid with its last-good version, and CUDA graph capture is disabled for the DeepSeek-V2 family (as it already is for Gemma 4), so output is coherent again (#829).
- Gemma 4 audio: the CLI rendered the `<|audio|>` placeholder before the prompt text, which flipped the 12B unified model from transcription into answering the perceived content on acoustically hard clips. The placeholder now follows the prompt text (#798).
- Requests with no effective input (empty or whitespace-only prompts and messages) were dispatched to the model; they now return a 400 before dispatch on `/v1/chat/completions`, `/v1/completions`, and `/v1/messages` (#803, #813, #814).
- Long-lived speculative serving on CUDA hit a fatal MLX "Cache thrashing" abort once the graph cache filled. CUDA builds raise the `MLX_CUDA_GRAPH_CACHE_SIZE` default so the abort no longer fires (#818).
- An MLX evaluation throw in the batch decode loop aborted the whole worker and dropped every in-flight request; it now fails the affected request instead (#825).
- An empty paged-state length list from a not-yet-populated sequence returned an Err that spammed a "Failed to sync paged state" warning; it is treated as a benign no-op (#826).

### Changed

- The speculative slice slot rotates across waiting requests instead of always going to the window head (#816).
- Effective-input text check is allocation-free (#815).
- Added a fast test profile to cut edit-test iteration time (#812).

## [v0.4.1] - 2026-07-16

### Added

- Server tool-call parsers for more model families: Kimi K2 sectioned format (#783), the pythonic `[func(arg=value)]` format (#786), function-calling Gemma `start_function_call` (#791), MiniMax-M3 namespaced XML (#793), and GLM-4.7 and LongCat `arg_key` / `arg_value` grammar (#794).
- Step-3 (`step3p7`) model port (#781).
- Command MoE (Cohere2 MoE) model port (#761).
- Kimi-VL 3D MoonViT video support: image and video patch embedding (#762).
- Gemma 4 per-request image soft-token budget. Set it with `--image-soft-tokens` on the CLI or with `detail` (`low` / `high` / `auto`) or the `max_soft_tokens` extension on a server `image_url` content part; off-ladder values are rejected rather than clamped (#787).

### Fixed

- Gemma 4 E-series (`e2b` / `e4b`) audio transcription was garbled. The Conformer mel front-end (`fft_overdrive`, log floor, unfold size, frame mask) and the per-layer ClippableLinear clamp bounds now match the reference, so E-series transcription is correct again (#796).
- Pixtral and Mistral 3 force-resized every image to a square and dropped the row structure. They now preserve aspect ratio and emit `[IMG_BREAK]` between rows and `[IMG_END]` at the end, with CLIP normalization (#792).
- DiffusionGemma tool-call and channel markers were registered as special tokens, so skip-special decode stripped them and the tool-call parser never saw them. They are demoted to non-special at load (#789).
- Qwen3.5 gated the RMSNorm `+1.0` shift on the presence of MTP weights, which could shift an already-converted checkpoint a second time. The shift is now gated on the conv1d weight layout alone (#784).
- Server tool-call parsing for the bracketed Mistral format (#785).
- Dense `--max-kv-size` trim dropped the attention-sink prefix; it is now pinned (#759).
- Needless borrow on the `march` flag in the `mlxcel-core` build script (#795).

### CI/CD

- Generate and publish a CycloneDX SBOM on release (#758).
- Drop the broken `log-level` input from the cargo-deny action step (#790).

### Dependencies

- Bump the minor-and-patch group with 5 updates (#780).

## [v0.4.0] - 2026-07-12

### Added

#### Backends

- **ComputeBackend seam for the forward-execution engine.** A `ComputeBackend` abstraction at the model-load boundary lets a future non-MLX engine host `LanguageModel::forward` without routing through the MLX bridge. The existing MLX path moves behind `MlxBackend` as a behavior-preserving refactor (temp-0 output is byte-identical before and after). Under default features the selection folds to the single MLX backend at compile time with no runtime dispatch on the hot path; a default-off `experimental-backend` feature reserves the plug-in slot. No non-MLX kernels are implemented (#338).
- **Experimental OpenXLA / IREE compiler backend (issue #449, opt-in, default off).** A second forward-execution engine built on a Rust-native StableHLO emitter and the IREE runtime, selectable with `MLXCEL_BACKEND=xla` behind the `xla-backend` / `xla-iree` build features. Apple Silicon and CUDA shipping binaries compile none of it. The track landed end to end:
  - Bring-up: export-route and Rust-native StableHLO emitter spikes for Llama-3.2-1B (#453, #455, #457, #451), int4 dequant-in-graph spike (#454), GPU decode throughput on GB10 (#456), and the backend crate/seam scaffold (#458).
  - Runtime: Rust→IREE FFI proven on aarch64 via the prebuilt dist and wired into `mlxcel-xla` and the CLI (#459, #460), executing on CUDA (GB10) via a source-built IREE runtime (#461) and on macOS with a Metal HAL device defaulted on Apple Silicon (#506, #508), with device-index-aware device/stream bindings (#505).
  - Loading and precision: emit graphs from `config.json` at load (#480), load sharded and f16/f32 checkpoints (#484), dequantize MLX 4/8-bit checkpoints in the loader (#490), f16/bf16 precision modes for the emitter with a per-device default and an accuracy gate (#553, #557), int8/int4 packed weight quantization (#568), and f16-resident weights for a fusion-free bandwidth win (#577).
  - Architecture packs: Qwen2 (plain RoPE + QKV bias) (#480 line), Gemma2 sliding-window local/global attention (#555), qk-norm and Gemma-family dense pack (#558), Seed-OSS/MiMo/InternLM3/ExaOne and parallel-block/norm-variant dense packs (#560, #561), the MoE FFN graph primitive (router + top-k dispatch) with Qwen3-MoE and OLMoE (#559, #563).
  - Serving: uniform-B batched decode graph (#462), ragged continuous-batching decode graph (#466), continuous-batching scheduler (#468) productized into a serving engine (#469) and served through the batching engine (#470), plus sampling (#471), history penalties (#479), stop strings (#478), untied LM heads (#482), and `/metrics` (#485). Gemma2 serves single-sequence and batched on this path (#491).
  - Guards: fail fast on `MLXCEL_XLA_PRECISION=bf16` and `MLXCEL_XLA_QUANT=packed` for Metal targets (#615, #617).

#### New model families

- **Vision-language models.** Qwen3-Omni MoE thinker (Qwen3-VL-MoE + audio tower) and talker + code2wav speech output (#664, #677), Llama 3.2 Vision (`mllama`) cross-attention VLM (#596), GLM-4V (sectioned MRoPE) and GLM-4V MoE (half-split MRoPE) (#594, #598), Hunyuan-VL (ViT + perceive merger + XD-RoPE) (#663), ERNIE-4.5 MoE VL (DFNRope ViT + modality-split MoE + 3D MRoPE) (#662), DeepSeek-VL2 (#660), Kimi-VL / Kimi-VL 2.5 (MoonViT encoder) (#597), FastVLM (FastViTHD + Qwen2) (#661), Moondream2 (Moondream3 ViT + dense Phi) (#599), Idefics2 (SigLIP + perceiver resampler + Mistral) (#639), SmolVLM (SigLIP + pixel-shuffle + SmolLM2, also loads Idefics3) (#593, #606), LFM2-VL (packed-patch SigLIP2 + pixel-unshuffle + LFM2 hybrid) (#645), Granite Vision (SigLIP multi-tap + AnyRes) and Granite 4 Vision (window-QFormer + Granite-4 hybrid) (#647, #649).
- **OCR models.** DeepSeek-OCR and DeepSeek-OCR 2 (SAM + Qwen2 query resampler), including the SAM + CLIP encoder hooks (#651, #655, #656), dots.ocr (`dots_vit` + Qwen2) (#657), GLM-OCR (ViT + GLM-4) (#646), PaddleOCR-VL (NaViT + ERNIE-4.5 MRoPE) (#595).
- **Diffusion LLM.** LLaDA-2 MoE masked-diffusion LLM (`llada2_moe`) (#659).
- **Text architectures and attention.** DeepSeek-V3.2 / GLM-MoE DSA lightning indexer (#583), phi3-small blocksparse attention (#581), qwen3-next pipeline-parallel stage support (#580).

#### Server and CLI

- MTP speculative burst reuses an adopted APC prompt-cache prefix (#591).
- `enable_thinking` aliased to the bare `thinking` flag for `deepseek_v32` (#579).
- The default output-token cap follows llama.cpp: `-1` resolves to the context window (#477).
- Detect an incomplete download snapshot and re-fetch on load (#604), formalized below under Fixed.

### Changed

- **Serving-throughput defaults enable the batching machinery out of the box.** `mlxcel-server` and `mlxcel serve` now default to `--parallel 4` (batched decode of up to 4 concurrent sequences, clamped to 1 for SSM / hybrid / mixed-cache families that cannot batch) and `--max-batch-prefill 4` (batched prefill for families that support it), and add `--no-prompt-cache` as a clean opt-out for the already-default-on prompt-prefix cache. The batched-decode default is paired with a default `--kv-cache-budget auto` memory guard so the #122 paged block-budget admission bounds KV for the concurrent batch and returns backpressure instead of an OOM abort; the guard is inert on the dense decode backend and can be disabled with `--kv-cache-budget none` (or `0`). On Apple M1 Ultra (`meta-llama-3.1-8b-instruct-4bit`), 4 concurrent clients get 1.90x the single-client aggregate throughput and ~17x lower time-to-first-token under load, with single-client throughput unchanged. `--parallel 1`, `--no-batch`, `--max-batch-prefill 1`, `--no-prompt-cache`, and `--kv-cache-budget none` restore the previous single-client behavior. Migration note: an explicit small `--ctx-size` is now divided across 4 slots, so a value that gives fewer than 512 tokens per slot fails startup with a clear error (raise `--ctx-size` or lower `--parallel`); `--ctx-size 0`, the default, uses the model window per slot and is unaffected. GB10 numbers (including the >= 2.5x-at-4-clients target) are pending a CUDA measurement session; see `docs/benchmark_results/serving-throughput-defaults-m1u-2026-07-09.md` (#628).
- The CLI prefill path now defaults to chunked, and rotating-cache chunked prefill was unbroken in the same change (#679).
- The CLI routes through an MLX inference-session seam, a behavior-preserving refactor that gives the OpenXLA path a symmetric entry point (#450).
- Both drafter flag spellings are accepted on `serve` and `server` (#602).

### Performance

- **Cap the batched-prefill transient memory.** With `--max-batch-prefill 4` now the default (#628), the server's padded batched-prefill path engaged out of the box for `supports_batched_prefill()` families and, for mixed-length prompts, ran a single unchunked `[B, padded_len]` forward that materialized a stacked `[B, L, L]` FP32 attention mask, an `O(B*L^2)` transient that ignored `--prefill-chunk-size` (four concurrent 8k prompts built a ~1 GiB mask and could abort the server on OOM, an availability edge the `--kv-cache-budget` guard does not model). The drained batched window is now bounded by total padded tokens via the new `--max-batch-prefill-tokens` flag (and `MLXCEL_MAX_BATCH_PREFILL_TOKENS`): a cohort of `B >= 2` rows padded to `L` keeps `B*L` within the budget, so the mask stays within `~2*budget^2` bytes; rows past the budget spill to the next tick and prefill via the chunked single-sequence path, and a head prompt too long to batch skips the batched path entirely. The default budget is derived (`2 * max_batch_prefill * prefill_chunk_size`, the shipped `2 * 4 * 512 = 4096`; the 2x headroom keeps a full batch of slightly-over-chunk-sized prompts in one window), bounding the FP32 mask to about 34 MiB while keeping short-prompt concurrency batching unchanged; `0` disables the cap for the pre-#715 unbounded behavior (#715).
- **CUDA / GB10 kernel parity.** Native paged-attention decode kernel ported to CUDA (#731) with an adaptive selector for it (#709); fused SSM decode kernel ported to CUDA (#727); MoE prefill collapse fixed via sorted grouped GEMM (#726); `sm_120/121` qmm CTA tile (M=128) for Blackwell prefill (#723); single-dtype decode graph to eliminate per-token `AsType` (#732); small-M quantized matmul amortized with a multirow qmv path (#740); `sdpa_vector` decode kernels extended to head_dim 256/288 via an MLX overlay (#681); backend-aware default for the fused decode-MoE Dff threshold (#643); `MLXCEL_CACHE_LIMIT` added and the periodic decode cache-clear disabled on CUDA (#696); chunked parallel prefill for gated-delta on non-Metal backends (#590).
- **Speculative decoding.** Tick-cooperative speculative decoding removes the burst head-of-line block (#745); the adaptive MTP policy is settled from measured round cost (#742); CUDA pairing matrix and burst HOL observability added (#733).
- **Server decode loop.** Pipeline the scheduler decode with lookahead `async_eval` (#729); incremental detokenization and streaming-overhead cuts (#713).
- **NVFP4 quantization (Blackwell).** Direct-transcode ModelOpt NVFP4 triplets to MLX native (#697), default Metal NVFP4 to native transcode (#705, #706), opt-in native repack override for non-CUDA builds (#699); gemma4 folds NVFP4 global scales into the fused MLP kernel (#701) and cuts decode-path primitive count on CUDA (#680, #682).
- **Vision.** `mllama` builds cross-attention states from real tiles only (#622) and caches cross-attention K/V across decode steps (#621); PaddleOCR-VL batches vision attention segments (#700).
- Record the CUDA fused-MoE Dff cap provenance and its decline test (#711); long-prompt prefill ladder plus serving telemetry benches (#641).

### Fixed

- **Interrupted model downloads are detected and re-fetched instead of failing at load.** An interrupted `mlxcel run <repo-id>` (also `mlxcel serve` and `mlxcel-server` auto-download) previously reused the partial snapshot and died with a bare `Weight not found`. The load/resolve path now verifies the full weight set against the snapshot's own `model.safetensors.index.json` (every shard present and non-zero), not just `config.json` presence, so a partial snapshot is resumed through the shared downloader (re-fetching only the missing files, with a forced clean re-download as a fallback) before the model loads. Repackaged mlx-community quants whose stale full-precision index no longer matches the on-disk files are still reused without a re-fetch (#465).
- **CUDA.** Restore `lfm2-350m-8bit` decode via an elementwise short conv (#751); correct the Int8 KV per-token scale (#717); chunk long-prompt qmm launches to avoid `gridDim > 65535` and int overflow (#652); bound long-prompt prefill memory for flash-ineligible models (#676); repack ModelOpt NVFP4 to the native MLX layout (#692).
- **Gemma 4.** Bind the gemma-4 NVFP4 vision tower so image input works (#750); honor per-module MLP quant overrides (#690) and diagnose malformed ones (#695); disable CUDA graphs for the decode graph-race collapse (#688, #689); render the thinking-off closed channel to fix chat pad-collapse (#686, #687).
- **DeepSeek.** `deepseek_v32` causal-mask fallback so maskless prefill is not bidirectional (#667); deepseek-v3 restores the last layer, causal prefill, and f16 clip (#618) and corrects the MoE reshape for 3D forward input (#614).
- **VLM loading against real checkpoints.** moondream2 weight/contract pairing and EOS/prompt resolution (#616, #609); kimi-vl nullable text-config field (#608); paddleocr-vl `language_model.*` / `visual.*` key remap (#607); smolvlm idefics3 checkpoint detection (#606); mllama vision-tower key mapping (#610); glm4-moe fuses separate `switch_mlp` gate/up experts at load (#620); DeepStack and MRoPE-section loops capture the functional `slice_update` return (#666).
- **Qwen.** qwen3.5-moe sanitizes toward the gate/up/down expert names `SparseMoeBlock` loads and stacks per-expert weights (#671, #587); qwen3-next real checkpoints load and run (#588) with per-sequence (`SequenceId`) cache isolation (#601).
- **Tool calls.** Parse GPT-OSS Harmony tool calls into structured calls (#658); strip the leading namespace from tool-call names before filtering (#586).
- **Server / routing.** Emit usage on the disaggregated `/v1/chat/completions` responses (#707); reject model-owned paged families from the prefill handoff (#716).
- longcat n-gram raw slice uses a `-1` stop on axis 0 (#589); offload `download_repo` to a dedicated thread when a Tokio runtime is active (#668); validate the external quant scheme and surface unclosed thinking (#605); load bf16 affine quant scales in the OpenXLA weight loader (#569); repair three `mlxcel-core` tests after the MLX pin bump (#644).

### Docs

- Finalize the per-backend `MLXCEL_FUSED_QK_NORM` default decision: CUDA (GB10) was measured and is also slower than the graph path, so the fused QK-norm decode path stays opt-in (default off) on every backend; `docs/environment-variables.md` updated to drop the CUDA-pending rationale and record the determinism nuance (#355).
- Document all cargo build features and the XLA backend env vars (#747).
- OpenXLA design record: ADR 0004 compute-backend seam + StableHLO/MLIR direction (#447), Phase 1 outcome (#452), performance table and transferable-precision decision (#562), multimodal/VLM track scope (#567), SSM/hybrid/recurrent track scope (#565), low-precision validation on Metal + oracle dumper (#611), packed-int8 fusion spike findings (#578).
- Benchmarks: refresh M5 Max (#756) and M1 Ultra (#754) for 0.4.0-rc.1 (cooldown 30), M1 Ultra full refresh post VLM-port (#673), GB10 full sweep (2026-07-12), attribute the GB10 decode drops via a post-reboot re-measurement (#757), and link the GB10 findings and decode drops to issues #748, #749, #755.
- Speculative: cross-check Gemma 4 MTP acceptance on M5 Max (#737) and correct the GB10 MTP regression framing to the small-M qmv kernel gap (#739).
- CUDA/Blackwell batched-decode does not amortize (#724); re-validate the CUDA fused-MoE Dff cap on MLX 0.32.1 (#721); retire pooled paged-attention decode to a library-only API (#720); record the Metal NVFP4 native benchmark (#702); document Idefics3 support routing to the SmolVLM runtime (#640).

### Chore

- Version bumped to `0.4.0-rc.1`.
- Bump the `mlxcel-core` and `mlxcel-surgery` member crates to Rust edition 2024 and align their versions to the root crate, per the release-versioning rule (#445, #272).
- MLX pin bumped to 0.32.1 (#703, #704) and earlier to `e9463bb` with the CUDA overlays rebased (#625, #642).
- Reproducible IREE runtime toolchain via make targets (#576); requantization script and perplexity harness (#683, #685).
- Refactors: share the per-layer attention core across emitter graph kinds (#554) and the seq MLP across serve graphs (#492); share the short-conv decode helper and audit L=1 conv dispatch across SSM/hybrid decode paths (#753).
- Tests: reusable per-architecture XLA validation harness (#556); backend-conditional wired-limit expectation (#743); preserve MoE expert stack ordering.
- Microbench: qmm GEMV effective-bandwidth harness and GB10 results (#684).
- CI: bump `actions/checkout` 6 to 7 (#473), `actions/cache` 5 to 6 (#472), `actions/setup-python` 5 to 6 (#474); dependency group updates (#483, #669).

## [v0.3.3] - 2026-06-25

### Added
- **Multi-node disaggregated routing.** The server drives multi-node disaggregated prefill/decode routing with worker health checks and failover (#388), and the router serves `/v1/completions` alongside the chat and responses endpoints (#386).
- **Mellum 2 hybrid-attention MoE text model** (#397).
- **Video input for Gemma 4 Unified** (`gemma4_unified`) (#400).
- **Phase 1 Python client package over the server** (#411).
- **MTP speculative decode wired into offline `generate`** (#385).
- Env-gated sparse-V skip-rate counter to measure KV sparsity (#377, #379).
- **N-gram loop detection** that breaks degenerate repetition loops at decode, on by default for the Gemma 4 family (#433).
- **Nemotron-H Nano Omni audio input** wired into server chat audio (#443).

### Performance
- **Fused single-launch xIELU Metal kernel for Apertus**, on by default after M5 Max validation (#414, #417). Apertus and Seed-OSS decode were profiled and the xIELU op trimmed (#399).
- Wire MiniMax to the fused decode-MoE kernel (#390).
- Bound the audio request queue and add a per-request timeout (#381).

### Fixed
- **Sliding-window prefill beyond the window** corrected across models; the gemma3/gemma4 sliding-prefill mask was hoisted to a shared helper (#405, #412, #415).
- **HTTP 422 from `/v1/messages` for Claude Code >= 2.1.156** (#380). Claude Code interleaves `{"role":"system", ...}` turns inside the `messages` array as mid-conversation reminders. The missing `System` variant in `AnthropicRole` caused `serde_json` to reject those requests before any generation. A new `fold_system_messages` translator pass now relocates mid-conversation system turns into the adjacent user turn (or the head system block) so the text reaches the model under any chat template, including head-only templates (Qwen, Llama 3) that silently drop non-head system messages.
- OLMoE scores the MoE router with full softmax then gather, not top-k softmax (#391).
- Preserve the assistant `reasoning` field across turns (#394).
- Add BitNet to `FAMILY_ORDER` so `family_order_is_exhaustive` passes (#404).
- Router: harden `/router/stats` disclosure and decode_target trust (#393), and use the worker's authoritative token count for usage (#392).
- Make audio-path MLX ops fallible at the FFI boundary (#384), and make audio synthesis panic-safe in release via panic=unwind with an explicit core-thread abort (#383).
- **Prefill attention masks sized from the live window, not the monotonic offset.** Multi-token prefill causal and sliding-window masks are now sized from the cache's live length (`offset - live_start`), so a `--max-kv-size` `trim_front` cannot produce a mask wider than the K/V the cache returns. Applied across dense-cache sliding-window models (#418), the general dense path (#420), mistral4/nemotron_nas/qwen-vl (#422), and gemma3/gemma4/exaone_moe (#431, a defensive consistency fix). Byte-identical on the untrimmed path.
- **Double-transpose crash on mlx-community conv checkpoints (Gemma 4 audio, phi4mm patch-embed, nemotron audio, RT-DETRv2).** Several weight-sanitizer functions transposed conv weights from PyTorch `[out, in, kH, kW]` to MLX channel-last `[out, kH, kW, in]` unconditionally. Pre-converted mlx-community checkpoints already store these weights in channel-last order, so the unconditional transpose double-converted them and produced a corrupted shape. The confirmed crash: loading `mlx-community/gemma-4-e4b-it-qat-4bit` turned the audio subsample conv weight `[128, 3, 3, 1]` into `[128, 3, 1, 3]`, which MLX conv2d rejected because the input C_in=1 did not match the weight C_in=3. All four affected sanitizers now check the tensor shape before transposing: `conv2d_weight_is_channel_last` (already-MLX `[out, kH, kW, in]` skips; PyTorch `[out, in, kH, kW]` transposes) and `conv1d_weight_is_channel_last` (depthwise-only; MLX `[out, kW, 1]` skips; PyTorch `[out, 1, kW]` transposes). Both predicates are idempotent. Resolves #428.
- **Conv shape faults no longer abort the server.** conv1d/conv2d are fallible at the FFI boundary (#434) and the nemotron omni audio-encoder convs route through the same fallible path (#439), so a bad conv shape returns an error instead of aborting the process.
- **Gemma 4 audio placed in the user turn.** The CLI resamples audio to 16 kHz and emits the `<|audio|>` marker inside the user turn (#438), and the server emits its `<|audio|>` block inside the user turn (#440).
- **mistral4 loading and MoE routing.** Mistral3-VLM mistral4 (MLA) text backbones route to the Mistral4 loader (#423/#424), and mistral4 MoE tokens are flattened to 2D before SwitchGLU routing (#425/#426).

### Docs
- Attribute mlx-audio alongside mlx-vlm in README and NOTICE.
- Record the #370 fused-V attempt regression and keep Turbo4Asym on dequant-SDPA (#378).

### Chore
- Update dependencies to latest compatible versions (#406).
- Platform-aware release with an explicit `release-cuda` Makefile target.
- Bump actions/checkout from 6 to 7 (#395).
- Fix clippy `useless_vec`/`identity_op` lints in the nemotron audio encoder test (#441).

## [v0.3.2] - 2026-06-20

### Added
- **Whisper speech-to-text on `/v1/audio/transcriptions` and `/v1/audio/translations`** (#371), and **Kokoro-82M text-to-speech with an iSTFTNet vocoder on `/v1/audio/speech`** (#374), served through new audio request and response plumbing on the `/v1/audio/*` surface (#368).
- **`reasoning_content` on non-streaming chat completions**, splitting thinking-model output into a separate field that matches the streaming path (#359).
- Warn at startup when a CPU-only build runs on a host that has an NVIDIA GPU (#372).

### Performance
- **Hardware-gated `MLX_MAX_OPS_PER_BUFFER` decode default.** Pre-M5 Apple Silicon (M1 to M4) gets a higher command-buffer op cap, raising steady-state decode by about 8 to 12% (gemma3n e2b 82.7 to 92.5 tok/s); M5 keeps the default with no change (#360).
- **Turbo4Asym decode rerouted through dequant-then-SDPA**, lifting it from about 0.14x to 0.40x of fp16 with byte-exact output instead of the slow sparse-V path (#369).
- Fuse the batched decode sampler into a single `[B]` dispatch (#339), add incremental per-sequence penalty-state caches (#344), and split batched prefill into compatible cohorts (#346).
- Adaptive B=1 MTP enable or decline policy chosen from per-model profiling (#348).
- Generalize the fused QKV+RMSNorm+RoPE path to standard RMSNorm, opt-in behind `MLXCEL_FUSED_QK_NORM` (#341).
- `--recommend-quant` now suggests a Turbo KV-cache mode per model family and context range, advisory and opt-in only (#343).

### Fixed
- **Correct an f16/bf16 logprobs crash and corruption** where 2-byte scores were read as 4 bytes (#340).
- Suppress gemma4_unified multimodal placeholder tokens that leaked into generated output (#351).
- Reseed the RNG per row at the batched-prefill first-token sample, so a batched request's first token no longer depends on sibling rows (#356).

### Docs
- Align the Turbo KV `--recommend-quant` advisor, the `bench_kv_cache.sh` gates, and `docs/turbo-kv-cache.md` with the measured four-model decode sweep, and add ADR 0002 on why the split Turbo decode does not reproduce the upstream sparse-V speedup (#376).
- Record the GB10/CUDA fused QK-norm decode result (#357) and Gemma3n decode profiles on M5 Max (#358, #345).

## [v0.3.1] - 2026-06-17

### Performance
- **Fused decode-MoE kernel ported to CUDA.** The fused single-token MoE decode path was Metal-only in 0.3.0; this implements it on CUDA, so Linux/CUDA GPUs get the same fast path with byte-identical greedy output. Measured gains run from about 10% to 55%, up to 1.55x on qwen3-moe (#319).
- **Wired six more MoE families to the fused decode-MoE kernel**: qwen2_moe (#308), LFM2 (#309), qwen3_vl_moe (#310), Mixtral (#311), Phi-3.5-MoE (#312), and OLMoE (#314). The kernel self-gates by expert size (`MLXCEL_FUSED_MOE_MAX_DFF`, default 4096), so large-expert models such as Mixtral 8x7B and Phi-3.5-MoE keep the gather path with no regression.
- **BitNet BitLinear ternary matmul ported to CUDA**, so BitNet b1.58 models run on CUDA GPUs (#322).

### Fixed
- **Load non-affine quantized VLM weights with the correct quant mode and group size.** The loader detects the quant mode from the absence of biases and infers `group_size` from tensor shape, so non-affine VLM checkpoints such as minicpm-v mxfp4 load instead of failing (#334).
- OLMoE applies `q_norm` / `k_norm` before the head reshape, matching the reference attention order (#317).
- Report load/run out-of-memory as `SKIP:oom` rather than `FAIL:bench` (#298).

### CI
- Add an `MLXCEL_CXX_MARCH` override and pin the x86_64 CUDA release asset to `x86-64-v3`, so prebuilt CUDA binaries run on a wider range of hosts (#208).

### Docs
- First Linux/CUDA (NVIDIA GB10) full benchmark sweep for 0.3.1: 136 of 147 text models pass with no code-level failures (#320, #321, #323, #324, #335, #336).
- Record fused decode-MoE gains for the newly wired MoE families (#337).
- Refresh the README performance tables (#300).

## [v0.3.0] - 2026-06-15

### Added
- **Nine new model families.** BitNet b1.58 (1.58-bit ternary weights, #252), IBM Granite dense (#254) and GraniteMoeHybrid (Mamba2 plus attention hybrid, #259), LFM2 and LFM2-MoE (#255), Falcon-H1 (Mamba2 plus attention parallel hybrid, #256), PLaMo 2 (Mamba plus attention hybrid, #257) with PlamoTokenizer support (#264), Apertus (xIELU, QK-norm, llama3 RoPE scaling, #260), ByteDance Seed-OSS (#261), and dots.llm1 MoE (#263).
- **Linux x86_64 and aarch64 CUDA release builds** with bundled CCCL headers, so the CUDA artifacts run on nodes that do not have the build-machine CCCL path (#262).
- Configurable allowed-origins for server CORS, replacing the any-origin default when set (#253).

### Changed
- **Fused decode-MoE Metal kernel is now on by default** (`MLXCEL_FUSED_MOE`, set to `0` to disable). It speeds up single-token MoE decode across families, with the GeGLU path giving about 13% on gemma4 (#285).
- **`mlxcel run` with no model argument now defaults to `mlx-community/gemma-4-e2b-it-4bit`** (was `Llama-3.2-3B-Instruct-4bit`): a smaller checkpoint that downloads faster and runs in less memory.

### Performance
- Two-kernel fused decode-MoE that beats `gather_qmm`, staged across the kernel foundation and the expert decode kernel (#274, #275, #276). Extended to 6-bit and mixed-bit experts for dots.llm1 (#278), wired to qwen3-next / Qwen 3.5 / 3.6 (#279), and given a GeGLU variant for gemma4 (#281); the squared-ReLU kernel stays behind a dedicated flag (#280).
- Gate the Mamba2 and nemotron_h per-mixer eval to M5 Max so SSM-hybrid decode is not slowed on other Apple Silicon (#266, #271).
- CCCL header resolution at runtime now handles relative invocations and nodes without the build-machine path, and a persistent PTX kernel cache reuses JIT-compiled kernels across runs (#270).

### Fixed
- **Quantized models now stay bf16, fixing a 33-41% M1 Ultra decode regression** on bf16-scale checkpoints (qwen3, nemotron, gpt-oss, solar, and others). The blanket bf16-to-f16 quant-scale promotion added with Apertus had created a bf16-activation by f16-scale mismatch in `quantized_matmul` / `gather_qmm` (#290).
- **Infer per-tensor quantization bits for embeddings**, so mixed-precision exports that store the embedding at a different bit width than the top-level config load instead of aborting in dequant. For example diffusiongemma stores its embedding at 8-bit under a 4-bit default (#292).

### Docs
- Refreshed the M1 Ultra and M5 Max benchmark results for the 0.3.0 sweep (#295).

### Chore
- Split `mlx_cxx_bridge.cpp` into domain-specific translation units (#277).
- Bumped the minor-and-patch dependency group (#288).

## [v0.2.1] - 2026-06-13

### Added
- **Exact-prefix prompt-cache snapshots now cover model-owned recurrent and mixed-cache families.** Mamba, Mamba2, Jamba, Nemotron-H, Qwen 3.5 / 3.6 text, MoE, and VLM wrappers can donate and restore same-session whole-prefix state instead of falling back to cold prefill (#241).
- **Gemma 4 text, VLM, and Unified wrappers now donate and restore exact-prefix prompt-cache snapshots.** The snapshots preserve model-owned standard and rotating cache state; real `gemma-4-26b-a4b-it-4bit` smoke validation inserted a 10,568,520-byte snapshot with no oversized rejection (#243).

### Changed
- CLI help and user docs now describe the v0.2.x server option surface consistently across `mlxcel serve` and `mlxcel-server`, including disaggregated peer roles, VLM prefix-cache environment settings, paged KV budget settings, and Gemma 4 snapshot-cache support.

## [v0.2.0] - 2026-06-13

### Added
- **Unified paged KV cache is now live in the batching server (epic #116).** Prefix reuse and paged block storage now operate together: a concurrent shared prefix is stored once with reference counting and copy-on-write, so a second request that shares a prefix adopts the existing blocks and re-prefills only its divergent suffix. The radix prompt cache and the paged block pool were unified into one store, the scheduler backs paged sequences with the shared pool, and pool-backed decode is byte-identical to the previous dense path across qwen3 and llama3 (single, batched, and prefix-share cases) (#152, #167, #168).
- **Disaggregated serving: prefill, decode, and router roles split across processes over TCP.** `mlxcel-server --node-role {prefill,decode,router}` with `--serving-bind`, `--prefill-peers`, and `--decode-peers` runs a pipeline where a model-free router fronts HTTP, hands the prompt to a prefill node, streams continuation tokens from a decode node, and merges them back to the client. A 3-process run is byte-identical to a single hybrid node. KV block contents serialize across the node handoff (#185, #187, #188, #189, #190, #191, #192, #193).
- **DiffusionGemma block-diffusion model (#217):** text generation (#218), image input (#219), and `mlxcel-server` serving (#220). The backbone reuses the existing Gemma 4 26B-A4B path; the new pieces are the dual-mode forward, self-conditioning, and the canvas diffusion engine. Temperature-0 output is byte-identical across the MLX bump.
- **Qwen3-Coder XML tool-call parsing**, so Qwen3-Coder function calls are extracted from the model's XML emission and surfaced as OpenAI `tool_calls` (#206).
- **`--kv-cache-budget <BYTES|auto>` flag (env `MLXCEL_KV_CACHE_BUDGET`)** caps the paged KV block pool. The scheduler admits a paged prefill only when blocks are available, evicting cold cached prefixes (then preempting) to make room, and rejects or requeues otherwise. Opt-in: the pool stays unbounded by default (#174, #175, #176). Paged block-pool usage is exposed at `GET /v1/cache/stats` and on `/metrics` (#178).
- **Architecture-aware KV-cache memory estimation** for `mlxcel inspect` and the `--estimate-memory` preflight (#172). Sliding-window, MLA, hybrid, and pure-SSM models now estimate KV bytes from their real attention shape instead of a flat formula that was off by about 100x for Gemma, DeepSeek, and Mamba. A separate activation term accounts for the chunked-prefill working set on top of weights and allocator overhead (#173).
- **Opt-in VLM prompt-prefix cache sharing for multi-turn same-image conversations**, behind `--enable-vlm-prefix-cache`. A follow-up turn that keeps the same image adopts the prior turn's prefix and prefills only the new text, verified byte-identical to a cold prefill on qwen2-vl-2b (#182, #184).
- **Fused paged-attention decode Metal kernel** (split-K flash-decoding), built and numerically correct but gated off because it does not beat MLX gather-then-SDPA at long context on Apple Silicon. Enable with `MLXCEL_PAGED_ATTENTION_NATIVE` (#181).

### Changed
- **Automatic Prefix Caching is now enabled by default.** Requests that share a prompt prefix with a cached entry reuse the cached blocks, and the output is unchanged (#233).
- **The prompt-prefix KV cache now serves the Anthropic `/v1/messages` and OpenAI Responses `/v1/responses` endpoints**, not just `/v1/chat/completions` and `/v1/completions` (#240).
- **The B=1 MTP speculative-burst default is now chosen per hardware.** M1 Ultra measurements showed batch-capable MTP targets (such as Gemma 4 31B) regress at B=1 (0.75x to 0.96x), while the same targets gain on M5 (1.2x to 1.4x); the discriminator is GPU generation, not memory bandwidth. Batch-capable targets now default on only on M5-class hardware with a neural accelerator; non-batchable targets stay always-on. `MLXCEL_ENABLE_MTP_B1` overrides either way (#216).
- **Partially matched paged prefixes are now adopted instead of declined**, so a request that shares a leading block run with a cached entry but diverges later reuses the matched blocks (#230). Paged adoption is non-consuming: it clones and pins the shared blocks rather than moving them, so the donor entry stays cacheable (#232).
- **Vendored MLX bumped to upstream main (2026-06-11)** and the steel GEMM overlay retired now that the fix is upstream (#223).

### Performance
- Chunked slab storage for the paged pool, so it grows in fixed-size slabs instead of one monolithic tensor (#237).
- Presize the paged pool to the prefill span and eval grown slabs eagerly to avoid mid-decode allocation stalls (#229).
- Stream decode continuation tokens one frame at a time from the disaggregated decode role instead of buffering the full continuation (#214).
- Hardened the ragged B>1 MTP batching masks and verify tail so variable-length prompts in one burst keep greedy parity (#202).

### Fixed
- **Per-row position holes broke B>1 batched MTP greedy parity after divergent accepts.** When rows in a batched MTP burst accepted different draft-token counts, the surviving K/V is now compacted to each row's accepted end with per-row RoPE and a precise mask, so a divergent round no longer shifts later rows off their true positions (#211).
- Guard the empty-batch paged-decode fallbacks against a `drain(..1)` panic, and use absolute block indexing in append, trim, restore, and serde validation so a `logical_start > 0` write addresses the correct block (#215).
- Support chunked-prefill prompts in the disaggregated serving handoff, driving start and continue-chunked to completion with a 1M-token admission cap and pool release on extract error (#213).
- Apply the chat stream filter to disaggregated router output so reasoning-content splitting and structural-token cleanup match the single-node path (#212).
- Finish a chunked prefill when the first chunk already reaches the prompt end (#179).
- Release paged KV block pins on prompt-cache evict or decline, including a pre-existing leak that left the origin allocation pinned at reference count 1 (#170).
- Account real paged pool bytes in the prompt-cache ledger and `/v1/cache/stats` instead of a nominal placeholder (#231).
- Enforce the pack3 size contracts in release builds so a mis-sized packed buffer fails fast instead of corrupting silently (#236).
- Render assistant `tool_calls.arguments` as a JSON object rather than a string on multi-turn requests (#210).
- Render the request's `tools` into the prompt so templates that inspect the tool list receive the real definitions (#207).
- Expand bare model names to the default org in the `download` subcommand, matching the other `-m` consumers (#177).

### Security
- Hardened the paged KV handoff deserialization boundary: capped the frame size, anchored the block geometry, checked per-layer consistency, and rejected empty sequences, so a malformed handoff payload from a peer cannot drive an out-of-bounds read or an unbounded allocation. A restore that fails partway now releases the blocks it already took instead of leaking them (#186).

### Docs
- New `docs/CONTINUOUS_BATCHING.md` covering continuous batching, paged decode, and the disaggregated prefill/decode/router topology, plus an expanded unified-cache section in `docs/turbo-kv-cache.md` (#194).

### Tests
- Extended the paged KV cache scheduler and prefix-share parity suites to llama3 alongside qwen3, all byte-identical (#169).
- Added hybrid-SSM cache carve-out tests and multimodal-digest plumbing so SSM and VLM families stay correctly excluded from or included in block sharing (#182).

### Chore
- Recorded upstream attribution for ported third-party code (#238).
- Bumped the minor-and-patch dependency group with 3 updates (#180).

## [v0.1.4] - 2026-06-05

### Added
- **Gemma 4 Unified (`gemma4_unified`) multimodal architecture** (#153, closes #151).
- **Gemma 4 Unified MTP speculative drafter (`gemma4_unified_assistant`)** (#157, closes #158). The Gemma 4 Unified decode target now routes through the existing MTP speculative burst dispatch, reusing the MTP drafter and round loop unchanged. The drafter's pre/post projections load through the quantization-aware `UnifiedLinear`, so a 4-bit assistant (e.g. `gemma-4-12B-it-assistant-4bit`) no longer crashes at forward time with a matmul shape mismatch. On `gemma-4-12b-it-4bit` plus the 4-bit assistant, temperature-0 output is byte-identical to classic decode at about 1.87x decode speedup (39 to 74 tok/s).
- **Variable-length prompts in B>1 batched MTP bursts**, behind the new `MLXCEL_ENABLE_MTP_BATCH_RAGGED` opt-in (subordinate to `MLXCEL_ENABLE_MTP_BATCH`) (#162, closes #161). Rows of different prompt lengths join one burst via per-row left-padding plus a windowed left-padding causal mask; greedy parity holds because every token in a row is shifted by the same constant left-padding offset. Eligibility is limited to `max_prompt_len <= sliding_window`; out-of-regime windows fall back to per-row B=1 service. Off by default (measured 0.94x to 1.13x on the 31B), so the production path is byte-for-byte unchanged.
- **Unified paged KV cache (epic #116), Phase 0**: decode-time page-gather microbench and ADR 0001, which selects the `[num_blocks, block_size, n_kv_heads, head_dim]` pool layout (about 2.1x faster on gather-then-SDPA than the head-split layout) and the gather-then-SDPA strategy (#145, closes #117).
- **Unified paged KV cache, Phase 1**: physical block-pool K/V tensor storage in `PagedBlockPool`, lazily allocated per layer with `write_block` / `gather_visible` primitives (#148, closes #118).
- **Unified paged KV cache, Phase 2**: pooled paged-decode read path over real, possibly fragmented block tables, bit-identical to the dense fallback over 200 steps (#149, closes #119).
- **Unified paged KV cache, Phase 3**: paged prefill writer with shared-prefix copy-on-write, so a suffix write after a shared prefix allocates only the divergent blocks (#150, closes #120). These four phases are additive machinery exercised by tests; the live decode path stays byte-for-byte unchanged until the scheduler wiring lands.

### Changed
- **B=1 MTP speculative decoding now runs by default for every MTP target**, including batch-capable ones such as Gemma 4 31B (#159, closes #158). Previously batch-capable targets declined singleton MTP unless `MLXCEL_ENABLE_MTP_B1=1` was set, a calibration from an earlier "B=1 is slower" measurement. M5 Max measurement shows B=1 MTP is profitable with byte-identical output at temperature 0: about 1.2x to 1.4x on the 31B plus bf16 assistant, and about 1.87x on the 12B Unified pair. Opt out with `MLXCEL_ENABLE_MTP_B1=0`.

### Fixed
- **Quantized fused MoE experts in the `gemma4_unified` loader are now split correctly.** The fused-expert split in `sanitize_gemma4_unified_weights` only matched the bare non-quantized `.weight`, so a quantized MoE checkpoint's `.weight` / `.scales` / `.biases` legs fell through unsplit and `switch_glu` construction could not find its per-projection quantized parts. The split now matches each quantized component leg and slices it on the output (doubled-FFN) axis at the same half boundary, with a dequantize-equivalence test proving no group straddling (#156).

### Docs
- Recorded the measured Gemma 4 31B B>1 batched MTP numbers and aligned the related code comments (#160).

### Chore
- Bumped the `minor-and-patch` dependency group: `uuid` 1.23.1 to 1.23.2 and `hyper` 1.9.0 to 1.10.1 (#147).
- Added the local `/notes/` scratch directory to `.gitignore`.

## [v0.1.3] - 2026-05-30

### Changed
- **BREAKING: `mlxcel list` now lists local downloaded models by default; the supported-architecture catalog moved to the new `mlxcel arch` verb.** Previously bare `mlxcel list` printed the architecture catalog and the local store inventory was gated behind `mlxcel list --local`, which inverted the bare-verb convention of every comparable tool (`ollama list`, `docker images`, `pip list`, `brew list` all show the *local* inventory with no flag) and contradicted mlxcel's own store-centric `run` / `download` / `rm` verbs. Now `mlxcel list` (and its `ls` alias) enumerates downloaded models from the global store with repo-id, on-disk size, and path, mirroring `ollama list`; `mlxcel list --models-dir <PATH>` applies to that listing. The catalog is reachable via `mlxcel arch` (alias `mlxcel supported`), byte-identical to the prior bare-`list` output. The `mlxcel list --local` flag is **removed** outright: clap rejects `--local` as an unknown argument, and the empty-store hint now points users to `mlxcel arch` for the catalog. This is pre-1.0 (v0.1.x) with no deprecation cycle because `--local` had not seen real-world use, so carrying a hidden flag plus a deprecation shim was not warranted. Migration: use `mlxcel arch` for the catalog, and drop `--local` from any `mlxcel list --local` invocation (the bare form now does the same thing) (#138).
- **`mlxcel list` default table redesigned: columns are now NAME / SIZE / MODIFIED.** The absolute PATH column is no longer shown by default; pass `-v` / `--verbose` to restore it. A relative MODIFIED time column is derived from the snapshot directory mtime and renders as human-friendly durations ("just now", "2 days ago", "3 weeks ago", or "-" when the mtime is unavailable). The compact header contracts `$HOME` to `~` and dims secondary columns on a TTY (respecting `NO_COLOR`). New output modes: `--json` emits a stable `[{repo_id, size_bytes, path, modified}]` array (modified is Unix epoch seconds or null) suitable for scripting; `-q` / `--quiet` prints one repo-id per line for pipe-friendly use with `xargs` and `mlxcel rm`; `--sort name|size|modified` controls ordering (default: name). `--json` and `-q` are never styled and are mutually exclusive; `-v` is incompatible with both (#141).

### Fixed
- **Security: chat-template rendering is now bounded to prevent a denial-of-service from untrusted model templates.** Model-supplied chat templates render through minijinja both per request and at model-load time (the `supports_tools` probe). Rendering was previously unbounded, so a pathological template (for example deeply nested or effectively unbounded `for` loops) could consume unbounded CPU and memory. The fix enables minijinja's `fuel` feature and caps each render at 50M VM instructions in the shared `configure_environment`, covering every render path. Exhaustion surfaces as a clean `OutOfFuel` error through `Result` (the load-time probe degrades to a string heuristic) and never panics. The cap is generous: real templates run well under about 1M instructions (audited across 91 templates and 267 scenarios with 0 failures), while an unbounded loop is bounded to a fraction of a second. This is RCE-safe and matters most for multi-tenant deployments where untrusted parties can cause arbitrary models to load (#129, PR #139).
- **Base-model warning no longer presents `-it` as a universal instruction-tuned naming convention.** The warning `mlxcel run` / `mlxcel generate` prints when a model ships no chat template (added in PR #134) recommended trying a variant "named with an `-it` suffix", but `-it` is the Gemma convention. For other families the advice was wrong: Llama and Qwen2.5 instruction-tuned checkpoints use `-Instruct`, and Qwen3 / Qwen3.5 use the plain repo name (with `-Base` marking the non-instruct variant), so a user running `Qwen3.5-0.8B-Base` was pointed at a non-existent `-it` repo instead of being told to drop `-Base`. The advice now names the per-family conventions (Gemma `-it`; Llama / Qwen2.5 `-Instruct`; Qwen3 / Qwen3.5 plain name vs. `-Base`). Base-model detection is unchanged: it keys on chat-template absence, never on the model name.

## [v0.1.2] - 2026-05-29

### Fixed
- **Chat fallback for models without a `chat_template` no longer collapses into echo loops**. When `tokenizer_config.json` ships no `chat_template` field and there is no `chat_template.jinja`, `render_prompt` previously called `concat_plaintext`, which is bare content-only concatenation with no role markers. Base / non-instruction-tuned models, being completion models, then took the most natural continuation of an unstructured prompt and parroted the user's last turn indefinitely (the symptom reported in #133). The implicit "no template found" path now uses a generic `User: ... Assistant: ...` pseudo-template via `concat_userassistant_fallback`, with a trailing `Assistant:` cue (no newline) that nudges the model to produce an assistant turn next instead of completing its own prompt with another `User:` line. The `processor.is_none()` warning still fires and still names base-model behavior as the cause; the recommendation to try the `-it` Hub counterpart is unchanged. `--no-chat-template` keeps its existing raw concatenation semantics and remains the offline `mlxcel generate --no-chat-template` parallel for completion-style usage. Template-render failure inside the chat-template path now falls back to the structured form as well, rather than raw concat, since by then the user is already in chat mode. Unknown roles such as `tool` are preserved verbatim with the same `Role: ` pattern instead of silently merging into the prior turn (#133, PR #136).

## [v0.1.1] - 2026-05-28

### Fixed
- **`chat_template.jinja` is now downloaded** alongside the rest of the model snapshot. The downloader allow-list in `src/downloader/filters.rs::is_wanted_file` only accepted exact-name `chat_template` (no extension) plus the broader `*.json` / `*.safetensors` / `*.tiktoken` / `*.model` / constrained `*.txt` allowances, but the actual HuggingFace convention is `chat_template.jinja`. The file was being filtered out at download time, leaving `ChatTemplateProcessor::from_model_path`'s `chat_template.jinja` fallback dead and forcing the REPL into the raw-text path for any model that ships its template as a separate Jinja file (e.g. `mlx-community/gemma-4-e4b-it-4bit`). `is_wanted_file` now also accepts `*.jinja` files; the `is_safe_relative_path` and `is_explicitly_denied` guards still run before the allow-list so no new attack surface is opened (#132, PR #134).
- **`mlxcel run` warning for models without a chat template is now actionable**: it states that the model is likely a base / non-instruction-tuned model, that chat replies will be incoherent or repetitive, suggests trying an `-it` (instruction-tuned) variant on the Hub (e.g. for `gemma-4-e4b-4bit`, try `gemma-4-e4b-it-4bit`), and explains how to proceed silently (`--no-chat-template`) or with one-shot completion (`mlxcel generate -p <prompt>`). The explicit `--no-chat-template` path remains completely silent (no regression) (#132, PR #134).

### Docs
- **GB10 (NVIDIA Grace Blackwell) doc refreshed** to the 2026-05-28 full sweep on mlxcel 0.1.0 with MLX pin `84961223` and the warm same-process harness (`--cooldown 0`). Adds the recovered `internvl3-1b` and `molmo-7b` text rows and three VLM image-path entries (`qwen2-vl-2b`, `qwen2-vl-2b-4bit`, `qwen3-vl-30b-a3b`). The cross-hardware decode table in `model_tests.md` now reflects the canonical state of each per-hardware doc: GB10 2026-05-28, M1 Ultra 2026-05-28, M5 Max 2026-05-27 (all on mlxcel 0.1.0, same MLX pin, same same-process harness). The "vs 2026-05-19" delta framing is dropped so the doc reads as a current-state snapshot, and the `Partial (⚠️)` status is collapsed into `Pass (✅)` because the partial-token information already lives in the Notes column. Updated GB10 Overall Status counts: 101 text pass / 8 fail, 38 VLM image-path pass / 0 fail (#131).

### CI
- **macOS release binaries are now notarized.** The release workflow submits signed `mlxcel` and `mlxcel-server` to Apple's notary service via `rcodesign notary-submit --wait` so Gatekeeper no longer blocks first launch with "developer cannot be verified". Stapling is skipped because bare Mach-O executables do not support stapling, and `spctl --assess` runs as a soft warn-only check since the notary ticket may still be propagating. Paired with `rcodesign verify` after signing to catch a broken signature before shipping, `set -euo pipefail` on the prepare-cert and code-sign steps so a failure on the first binary does not silently fall through to the second, surfaced `openssl pkcs12` stderr on extraction failure, up-front validation of `APPLE_CERTIFICATE` / `APPLE_CERTIFICATE_PASSWORD` / `AC_API_*` secrets, `chmod 600` on the materialized PEM and API key files, and an always-run cleanup that scrubs `signing.pem`, `original.p12`, `AuthKey.p8`, `ac-key.json`, and the notarization zip from `$RUNNER_TEMP` so self-hosted runners no longer carry an unencrypted Developer ID private key across jobs.
- **Per-target `workflow_dispatch` filter on the release workflow** (`targets`: `all` / `macos` / `linux`). Re-uploading a single platform's artifact to an existing release (for example retrofitting notarized macOS binaries onto a release that was cut before notarization landed) no longer rebuilds and replaces the other platforms' bit-different (timestamp-driven) zips, so any sha256 pinned by a downstream consumer remains valid. Release events still build everything; the filter is dispatch-only. Modeled after the per-family `targets` filter in `all-smi`'s release workflow.
- **`actions/checkout` ref pinned to the target release tag** in both the macOS and Linux CUDA jobs. The ref is resolved as `github.event.release.tag_name` on release events, `github.event.inputs.release_tag` on `workflow_dispatch`, otherwise `github.sha`. Without an explicit ref, `actions/checkout` would grab the dispatched ref (which is `main` for `workflow_dispatch`), so re-dispatching a build for an older tag would silently use `main` HEAD's source instead of the tag's source. The workflow YAML itself still runs from the dispatched ref so a CI-only fix can be applied on `main` and replayed against an old tag without rebuilding from newer sources, matching `all-smi`'s self-healing release pattern.

## [v0.1.0] - 2026-05-28

### Added
- **`mlxcel run <repo-id-or-path>` subcommand** (#102, epic #92). Capstone of the unified download + run epic, mirroring `ollama run` / mlx-lm ergonomics. With no `-p`, `run` enters the interactive chat REPL via the shared `run_chat` entry point; with `-p`, it produces output byte-identical to the equivalent `mlxcel generate -m <model> -p <prompt>` through `run_generate_once`. With no model argument, `run` falls back to `mlx-community/Llama-3.2-3B-Instruct-4bit`, matching `mlx_lm.generate` / `mlx_lm.chat`'s `DEFAULT_MODEL`. The model is a positional argument so `mlxcel run <repo-id>` reads like `ollama run`, and the repo-id auto-downloads through the shared resolver on first use. Sampling/generation/TurboQuant KV-cache flags are shared with `generate` via clap argument groups; advanced groups not exposed by `run` (tensor/pipeline parallel, speculative, lang-bias, surgery) are lowered to clap defaults, pinned by a drift-guard test.
- **Interactive multi-turn chat REPL** (#101). `mlxcel generate` without `-p/--prompt` now enters a chat loop that streams the assistant reply token-by-token, preserves conversation context across turns by re-rendering the full transcript through the chat template each turn, and supports `/bye`, `/clear`, `/?` (alias `/help`), and ollama-style `"""` multiline input blocks. The REPL forks no generation code: it reuses `resolve_model_source`, `MlxcelTokenizer`, `ChatTemplateProcessor`, `build_sampling_config`, `CxxGenerator::generate_streaming`, and the server's byte-fallback-safe `StreamingDecodeState`. Factored as a public `run_chat(ChatOptions)` entry point so the new `mlxcel run` verb dispatches into it. The end-of-turn flush also re-emits any UTF-8 suffix the streaming detokenizer held back mid-stream, so the displayed reply is complete instead of byte-truncated.
- **Local model management.** `mlxcel list --local` enumerates downloaded snapshots under `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/` with repo-id, on-disk size, and absolute path, recognizing both `<owner>/<name>` and bare `<name>` store layouts and gating on `config.json` so partial directories are skipped. The bare `mlxcel list` (architecture summary) is unchanged. `mlxcel rm <repo-id>` deletes from the mlxcel store and reports freed size; it prompts on a TTY, refuses on a non-TTY without `--yes`, contains deletion to `store_root()/models/` (path-sanitized by `model_dir` and re-asserted before `remove_dir_all`), and treats HF-cache-only models as read-only with explicit guidance instead of silently deleting (#99).
- **Repo-id-aware `-m/--model`** across `generate`, `serve`, `inspect`, `mlxcel-server`, and `run` (#100, #92). `-m` now accepts either a local path or a HuggingFace `owner/name` repo-id with a locked resolution precedence: existing on-disk path used verbatim (byte-identical to the pre-#100 local-path behavior, even when the path looks like `owner/name`); otherwise a repo-id matching `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` is resolved as legacy per-CWD `./models/<basename>` then HuggingFace cache snapshot (read-only reuse) then mlxcel global store then auto-download into the store. The legacy and store reuse branches gate on a present `config.json` so a half-written or unrelated directory is treated as a miss rather than handed to a model loader that would then fail. Each subcommand resolves its `-m` value once at the top of the handler, leaving downstream `.model` consumers unchanged. `generate` reorders surgery YAML validation ahead of the resolver so `--surgery <bad.yaml>` still fails fast without a network download.
- **Global model store and HuggingFace cache read-reuse** (#98, epic #92 foundation). The default download destination moves from per-CWD `models/<basename>` to `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`, namespaced so two repos with the same name across owners do not collide. `download_repo()` short-circuits to an existing snapshot under `$HF_HUB_CACHE` / `$HF_HOME` / `~/.cache/huggingface/hub` when no `--local-dir` is pinned and `--force` is off; mlxcel never writes into the HF content-addressed layout. Branch / tag revisions are resolved via the `refs/<rev>` pointer to the snapshot SHA; raw commit hashes resolve directly. New `mlxcel_core::cache_root()` is the single source of truth for `MLXCEL_CACHE_DIR` / home-dir resolution, shared with the tokenizer language-analysis disk cache.
- **`MLXCEL_MODELS_DIR` environment variable and uniform `--models-dir` model-store override** (#108). Precedence: inline `--models-dir` > `MLXCEL_MODELS_DIR` > `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`. Wired through `download`, `generate`, `serve`, `inspect`, `run`, `list`, and `rm`. Closes #107.
- **Bare model names default to the `mlx-community` org** (#113). A value with no slash (e.g. `Qwen3-4B-4bit`) now resolves as `mlx-community/<name>` rather than erroring immediately. The expansion runs in `resolve_model_source_with_override` as a new step in the locked precedence (after existing-path and `owner/name` repo-id, before the error arm), so pre-existing behavior for all currently-valid inputs is byte-identical. The default org is overridable via `MLXCEL_DEFAULT_ORG`; unset or whitespace-only values fall back to `mlx-community`, and an invalid org (e.g. containing a slash) is caught up front and produces a clear error without any network request. Help text and the README quick start document the bare-name shortcut.

### Fixed
- **`mlxcel-server` legacy startup `-m`** (epic #92 hardening) now uses the same repo-id resolver and `--models-dir` store override as `mlxcel serve`, including the safetensors-only presence check. Preserves byte-identical behavior for existing local paths and aligns docs/tests with the resolver-backed path.
- **Video resource caps are now injected via a `VideoLimits` struct** resolved once at the boundary, instead of `apply_probe_caps` and `split_png_stream` reading `MLXCEL_VIDEO_MAX_PIXELS`, `MLXCEL_VIDEO_MAX_DURATION_SEC`, and `MLXCEL_VIDEO_MAX_PNG_FRAME_BYTES` from `std::env` deep in the decode path. The previous design leaked test-mutated env values into concurrently running tests under the threaded `cargo test` runner (the load-video fd-vs-path parity test intermittently saw a leaked `MLXCEL_VIDEO_MAX_DURATION_SEC=2` and failed with `DurationTooLong { seconds: 4.0, max_seconds: 2.0 }`) and was unsound because concurrent setenv/getenv is a libc data race. `load_video_source` / `load_video` / `load_videos` keep their signatures and internally call `VideoLimits::from_env()`, so production callers are unchanged; new `_with_limits` variants accept an injected `VideoLimits` for tests (#104).
- **Pipeline runtime tests** now bind `127.0.0.1:0` directly through `TcpTransport` and resolve the real port via `local_addr()`, removing the release-then-rebind window in the previous `reserve_bind_address` helper. Eliminates the intermittent "stub stage startup channel dropped" failure that happened when another concurrently-running test grabbed the freed ephemeral port between release and re-bind. Verified by 16 consecutive `cargo test -p mlxcel --lib distributed::pipeline::runtime` runs with 0 failures (#106).
- **`require_secure_endpoint_refuses_plaintext_with_token` test** now acquires `env_lock()` at the start, matching the contract its sibling opt-out tests already follow. Without the lock, when a sibling test set `MLXCEL_ALLOW_INSECURE_ENDPOINT="1"` under `env_lock`, the refusal test could observe that value mid-window and take the opt-out path, returning `Ok` instead of the expected `Err` under the threaded test runner. Removes the concurrent setenv/getenv libc data race the previous test exposed (#111).

### Docs
- README "Run a model" section now leads with the `mlxcel run` one-liner instead of the verbose explicit-download flow, and collapses the core verbs (`generate`, `serve`, `inspect`, `--estimate-memory`) into a single one-line-comment block. Store-root precedence, `-m` resolution rules, `--local-dir` / `--models-dir`, `MLXCEL_DEFAULT_ORG`, and the memory preflight env vars now live behind a single link to `docs/environment-variables.md` (which already documents them) and a compressed model-management paragraph. Net: README drops about 58 lines with no loss of documented behavior (#114).
- M5 Max detailed table, README headline version, and the benchmark report method table refreshed to the 2026-05-27 full-sweep state on mlxcel 0.1.0. `internvl3-1b` now passes in both text (661 tok/s) and VLM (601 tok/s, ahead of mlx-vlm's 529 tok/s), raising the M5 Max pass count to 94. `paligemma2-3b-6bit` text decode is 168.83 tok/s and `qwen3-vl-30b-a3b-4bit` VLM is 56.38 tok/s over a 45-token sample. Aggregate parity statistics and the M1 Ultra / GB10 columns remain on the 2026-05-19 baseline campaign that was not re-run (#115).
- M5 Max decode aggregates recomputed against the unchanged `mlx-lm` / `mlx-vlm` baselines (baselines unchanged, only mlxcel-derived ratios recomputed). Text decode average 98% to 99% (median 99%), 62 of 66 at >=90% parity; the benchmark report headline is corrected from the stale 58 of 66. VLM decode average 101% to 102%, median 100% to 101%, 22 comparable pairs (was 20), 18 of 22 at >=90% parity (coverage counts reflect internvl3). README decode tables add Gemma 2 2B (100%), Phi-3.5-mini (98%), Jamba (98%), and InternVL3 1B (114% vs mlx-vlm) (#127).
- 2026-05-28 M1 Ultra full sweep mirrored into the public benchmark docs (mlxcel 0.1.0; `mlx-lm` / `mlx-vlm` baselines unchanged). README M1 Ultra column refreshed across both representative decode tables (Phi-3.5-vision M5 VLM row corrected to its post- value, 169 tok/s, 106%); benchmark report headline updated to text 74 pairs / 99% median / 64 of 74 at >=90%, VLM 18 pairs / 98% median / 12 of 18 at >=90%; `model_tests_m1ultra.md` per-model refresh with aggregate blocks (text avg 97% / median 99%; VLM avg 101% / median 98%) and `internvl3` now passing on M1. The cross-hardware quick table stays a labeled 2026-05-19 same-version snapshot (#130).
- Reword the v0.0.31 #86 entry to match the GitHub release note. The fix landed at the batch scheduler level: a burst of concurrent VLM requests overwrote a single shared `per_layer_inputs` cell before prefill consumed it, reading the wrong sequence's tensor. The earlier wording described the symptom as a per-layer input shape issue, which understated the concurrency cause.

## [v0.0.31] - 2026-05-27

### Added
- **MiniCPM-V 4.6 VLM architecture**, including hardened image grid handling (#82, #83).
- **RT-DETRv2 object detection model** exposed through the new `mlxcel detect` subcommand (#80).
- **Anthropic-style `/v1/messages` API endpoint** on the server for Messages API clients (#74).

### Fixed
- **Chat message `content`** that is missing or explicitly `null`, such as assistant tool-call turns, is now tolerated instead of being rejected with an HTTP 422, restoring multi-turn tool loops for OpenAI-compatible clients (#91).
- **Gemma 3n VLM `per_layer_inputs`** is now keyed per sequence id, so a burst of concurrent VLM requests in the batch scheduler can no longer race on a single shared cell and read the wrong sequence's tensor (#86).
- **Qwen3.5 MTP speculative decoding** uses per-position verify attention so the draft and verify passes stay in parity (#78).
- **Batched quantized KV caches** now apply the correct mask offset (#76).

### Docs
- Document the `MLXCEL_CAPTURE_DECODE` environment variable and clarify the memory headroom wording (#72).

### CI
- Pin the Rust toolchain to 1.93.1 for reproducible builds (#87, #90).

### Chore
- Bump the `minor-and-patch` dependency group: `serde_json` 1.0.149 to 1.0.150 and `minijinja` 2.19.0 to 2.20.0 (#84).
- Exclude the root `models` symlink (#88) and AI assistant temporary directories from `.gitignore`.

## [v0.0.30] - 2026-05-23

### Added
- **Unified pre-load memory estimator** (epic #52). `mlxcel inspect` is a new read-only subcommand that prints a byte-level breakdown of model weights, KV cache, and runtime headroom against available unified memory without loading any tensors. `--estimate-memory` on `mlxcel generate` and `mlxcel serve` runs the same estimator as a preflight and aborts when the model will not fit; `--force` (alias `--no-memory-check`) overrides the abort, `MLXCEL_MEMORY_LIMIT=NGB` tightens the available figure to a soft cap, and the runtime headroom factor defaults to `1.20x` (#67).
- **Exact weight footprint from the safetensors header.** The estimator parses the safetensors header to derive real per-dtype byte counts without materializing tensors (#64).
- **KV cache memory estimator** with 256-token rounding that matches the runtime's pre-allocation steps (#65).
- **MLX runtime memory API bindings** that expose the active, peak, and limit byte counters through FFI (#66).
- **Molmo v1 (molmo-7b) VLM architecture** (#41).
- **InternVL (internvl_chat) VLM architecture** (#37).

### Changed
- **Server parallel context sizing:** `--ctx-size` is now treated as a total context budget shared across active request slots, matching llama.cpp server semantics. `--parallel N --ctx-size C` yields an effective per-slot window of `floor(C / N)`; explicit `--max-batch-size` values share the same budget, `--no-batch` keeps a single full-context slot, `/slots` reports the per-slot window, startup rejects per-slot windows below 512 tokens, and memory preflight uses the same sizing model (#57).

### Performance
- **Gemma 3n bf16 decode** reduces AltUp/MLP graph overhead (#60) and improves M5 decode bandwidth with pretransposed weights (#62).
- **Phi-3.5 SuScaledRoPE decode** speedup (#42).
- **Gemma dense GeGLU** aligned with the mlx-lm reference for faster decode (#43).
- **Jamba hybrid decode** speedup (#44).

### Fixed
- **CLI boolean cache flags** are now validated, and CLI flags correctly take precedence over their environment-variable equivalents (#70).
- **Prompt cache radix trie** is now iterative, preventing a stack overflow on deep prompt prefixes (#63).
- **Gemma 3n** gates the bf16 fused decode path off the M5 Neural Accelerator so output stays correct on that hardware (#61).
- **CUDA Hopper builds** append the `90a` architecture suffix for auto-detect and fallback builds (#51).
- **VLM server image decoding** is hardened to skip invalid entries instead of failing the whole request (#50).
- **Qwen2-VL image placeholder** is expanded to the full grid count (#39).
- Tighten memory estimator preflight coverage so the abort path is exercised across `generate` and `serve`.

### Docs
- Refresh the M1 Ultra and README decode benchmark figures for the Molmo / Phi-3.5 / Gemma / Jamba / InternVL work (#45, #46).
- Correct the M5 Max baichuan-m1-14b decode comparison and flag the qwen3-0.6b gap (#49).
- Drop change-cause notes from the result tables to keep them current-state only (#47, #48).

### Tests
- Add a qwen2.5-vl-3b-4bit warmup regression guard (#38).

## [v0.0.29] - 2026-05-20

### Added
- **`mlxcel-bench-decode` same-process benchmark harness.** Loads a model once, runs the warmup pass, resets model and cache state, then runs the measured pass in the same process. This mirrors the Python `stream_generate` timing far more closely than two cold `mlxcel generate` invocations, especially for prefill. `scripts/bench_decode.sh` now drives this binary, and `scripts/bench_mlxlm.py` provides the matching mlx-lm / mlx-vlm baseline sweep (#36).

### Fixed
- **Model-owned VLM fallback state reset.** Single-row CLI and benchmark generations, which do not carry a `SequenceId`, reused stale fallback caches between runs. A new `LanguageModel::reset_runtime_state()` hook, invoked from `CxxGenerator::reset_with_model`, clears the model-owned fallback slot in lockstep with the generator-owned cache vector for Gemma 3, Gemma 4, Llama 4, and Qwen 3.5 Next (#34).
- **Gemma 3n VLM padded prefill alignment.** Gemma 3n VLM prefill aborted with a `[broadcast_shapes]` mismatch (for example `(1,288,256)` vs `(1,273,256)`) when the projected per-layer-inputs tensor diverged from the tile-padded token stream. The per-layer tensor is now aligned to the embeddings sequence length (pad with zeros when shorter, slice when longer, leave untouched when equal) before the per-layer blend (#36).

### Docs
- Reorganize benchmark result reports and refresh the Apple Silicon and M5 Max VLM benchmark tables (#33).
- Clarify the README benchmark phases, the model surgery advantage, and the Qwen3.5-0.8B-4bit quickstart (#32).

## [v0.0.28] - 2026-05-19

### Performance
- **Gemma3n bf16 prefill path.** Materialize full-precision bf16 casts and preserve the Gemma3n language MLP bf16 path; gemma3n-e4b-bf16 decode improves from 11.16 to 38.81 tok/s on M5 Max while output stays coherent (#28).
- **Qwen3-VL text-only decode.** Fuse Q/K/V into a single `FusedQKVLinear` and add a `forward_text_only` fast path on Attention / DecoderLayer / Model that skips MRoPE position-id computation and visual-state propagation when no image is present. Qwen3-VL-30B-A3B-Instruct-4bit decode 56.00 to 146.29 tok/s (2.61x, 99% of mlx-lm parity); Qwen3-VL-32B-Instruct-4bit decode 18.79 to 27.33 tok/s (1.45x, 94% of mlx-lm parity). Image-in-prompt and DeepStack paths are unchanged (#29).
- **GatedDeltaNet fast RMSNorm.** Replace the expanded `square` / `mean_axis` / `sqrt` / `divide` / `multiply` Q/K and gated-output RMSNorm graphs in Qwen3.5, Qwen3-Next, and KimiLinear with `mlx::fast::rms_norm` kernel calls via a new shared `scaled_fast_rms_norm_no_weight` helper. qwen3.5-0.8b-4bit decode 425.16 to 535.43 tok/s on M5 Max (+25.9%, 96% of mlx-lm parity); the Qwen3.5 speculative-decoding verify pass is updated alongside prefill / decode so draft / verify dtype agreement is preserved (#30).

### Fixed
- Rewrite `mlxcel list` output to drive from the `ModelType` registry: fixes stale count, missing VLM family, missing ~30 model types, and removes broken docs link (#27).

### Docs
- Refresh README performance snapshot and `docs/benchmarks.md` to match the latest M5 Max sweep.

## [v0.0.27] - 2026-05-16

### Added
- **End-to-end speculative decoding for Gemma 4 MTP and Qwen 3.5 DFlash drafter families**. New `Drafter` trait + `DrafterKind` enum + `model_type` auto-detection. Ported drafter components: `MaskedEmbedder` for Gemma 4 E2B / E4B, drafter masks (bidirectional full + sliding-window) and `normalize_batched_shared_kv_states` helper, `Gemma4AssistantDraftModel` (4-layer drafter + pre/post projections), and `DFlashDraftModel` (5-layer drafter + `DFlashAttention` + `DFlashKVCache`). Target-side hooks: Gemma 4 `return_hidden` / `return_shared_kv` / `rollback_speculative_cache` for MTP; Qwen 3.5 `return_hidden` + `capture_layer_ids` + GDN-aware `rollback_speculative_cache` for DFlash. Round loops: DFlash single-batch, `MtpGenerator` single-batch, batched DFlash with continuous batching + GDN-aware rollback, and batched MTP with continuous batching + left-padding normalization. Greedy-parity + perf benchmark scaffolding under `src/bin/speculative_bench.rs` and real-model byte-equality end-to-end tests.
- **Server speculative dispatch.** Speculative dispatch resolution and `MtpTarget` adapters wired into the inference server; the assistant model paths now plug into the real `MaskedEmbedder` and `make_drafter_masks`; speculative dispatch is wired into the scheduler via per-request B=1 bursts and a B>1 batched path via `MtpBatchedGenerator` / `DFlashBatchedGenerator`. Per-request properties propagated through the speculative-burst path: cancellation propagation through `MtpGenerator` / `DFlashGenerator`, `token_history` threading through the speculative-burst first sample, logprobs support, thinking-budget enforcement, and prompt-cache donate symmetric with the classic path and into the B>1 batched arm.
- **CLI:** `--draft-kind {dflash,mtp}` and `--draft-block-size` flags on both `mlxcel` and `mlxcel-server`.
- **OpenAI Responses API (Phase 1)** at `/v1/responses` for both binaries. New modules `responses_store`, `responses_translator`, `conversation_store`, `streaming_responses`, `routes/responses`, and request/response/stream type modules under `types/responses_*`. Implements conversation store with shared-LRU semantics, `response.created` / `response.in_progress` / `response.completed` SSE event stream, reasoning-trace forwarding, response cancellation, and four new CLI flags. User guide at `docs/responses-api.md`.
- **APC block-level partial cache adoption in the scheduler**. When Automatic Prefix Caching is enabled, a request whose prompt shares the first N blocks with a cached entry but diverges at block N+1 now reuses blocks 0..N and re-prefills only from the divergence boundary — rather than cold-prefilling the entire prompt. Three components: `DetachedKVCache::trim_to` and `DetachedCacheSet::truncate_to` in `mlxcel-core` perform per-layer KV tensor slicing on the detached handle (mirroring `KVCache::trim` semantics, covering FP16/INT8/Turbo4/Turbo4Delegated sidecars); the `PromptCacheStore` lookup relaxes the legacy "stored prefix must be fully contained in request" gate when APC is on, routing the actual common-prefix depth through the existing `apc_consistent_prefix_len` block-hash discriminator; and `Scheduler::try_adopt_cached_prefix` calls `truncate_to(matched_len)` before adoption when the lookup returns a sub-entry-length match. APC-off retains the earlier behaviour bit-exactly. Wall-clock bench procedure on Apple Silicon documented in `docs/apc-partial-adoption-bench.md`.
- **Nemotron H Nano Omni audio modality**. Ports the Parakeet/Conformer sound encoder (`NemotronOmniSoundEncoder`), mel-spectrogram feature extractor (`NemotronOmniFeatureExtractor`), and audio projector (`NemotronOmniSoundProjection`) from the upstream `mlx-vlm` Python reference. The encoder implements depthwise/pointwise Conv2D subsampling, Transformer-XL relative positional encoding, multi-head self-attention with per-head u/v bias terms, a GLU+BatchNorm convolution module, and half-weight feed-forward blocks. Audio weights are loaded conditionally when `sound_config` is present in `config.json`; the loader applies the upstream `sanitize_audio_weights` transpose pass before population. The VLM runtime path (`generate_vlm`) accepts `--audio <wav>`, runs the feature extractor and encoder, merges the resulting token embeddings at `sound_context_token_id` slots, and interleaves them with vision tokens when both modalities are present. Bring-up procedure for Apple Silicon engineers documented in `docs/nemotron-h-nano-omni-audio-bringup.md`.
- **`mlxcel download` / `mlxcel-server download` progress bars**. New `src/downloader/progress.rs` module provides terminal-aware suppression (`should_show_progress`), a `MultiProgress` factory, and 6 suppression unit tests. The downloader streams files via `reqwest` to a `NamedTempFile` and atomically renames into place; outer `stream_file` and inner `stream_to_tempfile` are split so the progress bar covers the network read and the rename is observable.
- **Server `--max-kv-size` flag** matching llama-server, plus a tightened chat-completion response envelope.
- **Tokenizer support for multi-token think and tool-call sequences** so chat templates that emit `<think>` / `<tool_call>` across multiple BPE tokens stream and parse correctly.

### Changed
- **`StreamFilter` extended** to handle multi-token markers and reset state when a partial marker is broken by a non-marker token.
- **Speculative drafter epic follow-ups hardened** post-merge — covers misc invariants surfaced by integration testing against the real `z-lab/Qwen3.5-4B-DFlash` checkpoint and the `mlx-community/gemma-4-*` drafter variants.
- **README and speculative decoding guidance refreshed** to match the current code paths and the latest M1 Ultra / M5 Max benchmarks.

### Performance
- **Qwen 3.5 DFlash greedy-argmax decode-path optimization** that drops the per-decode-step copy and an unnecessary argmax temporary, restoring decode tok/s on Qwen 3.5 32B / 9B DFlash configurations.
- **Avoid slow Gemma 4 MTP singleton bursts** — the speculative-burst path now correctly short-circuits to the classic path when the batch size collapses to 1 with no draft tokens accepted, eliminating a per-step over-evaluation regression introduced by the initial dispatch wiring.

### Fixed
- **DFlash drafter lazy-bind** for the upstream `z-lab/Qwen3.5-4B-DFlash` checkpoint — `Drafter::bind` was previously not called on the DFlash family, causing an internal cache mis-binding on the first speculative burst. The drafter now performs lazy-bind on first use, matching the MTP path.
- **Enable DFlash for Qwen 3.5 VLM text requests** — pure-text generations against a Qwen 3.5 VLM checkpoint can now resolve a DFlash drafter when one is available, instead of silently falling back to the classic path.
- **Speculative-rollback safety:** validate trimmable cache and reserve the last token in prefill so a rolled-back speculative burst always lands on a valid sampling boundary.
- **Prompt cache RadixTrie:** `pop_prefixes` now uses correct immediate-prefix semantics.
- **MiniMax M2 parallel tool calling parser** correctly emits one `ChatToolCall` per parallel call instead of merging them into a single call.
- **Server tool-call buffering:** preserve token positions when buffering parallel tool calls; skip the tool→normal transition when `tool_call_end` is empty so streaming continues correctly for templates without an explicit close marker.
- **`video_url` allowlist TOCTOU race** closed by passing the resolved `OwnedFd` to ffmpeg via `/dev/fd/N` instead of re-opening the path inside the subprocess. Symlink swaps between `metadata` and the subsequent open now cannot mis-route the subprocess. Audit hardening.
- **Gemma 4:** skip `k_proj` / `v_proj` / `k_norm` weight load for KV-shared layers — the previous load step would error out on real Gemma 4 E2B / E4B checkpoints that omit these tensors per KV-shared design.
- **Nemotron-H:** default `time_step_limit` to `(0.0, +inf)` regardless of `time_step_min` / `time_step_max` to match the upstream mlx-lm `time_step_limit` behaviour even when only one of the bounds is supplied.
- **`gated_delta` masked Metal kernel variants:** zero-init `y[dv_idx]` when the mask is false.
- Tests: add `max_kv_size` field to the `ServeArgs` test fixture.
- Address upstream-sync review follow-ups carried over from the v0.0.26 sync cycle.

### Security
- **Downloader security hardening** post-plaintext `HF_ENDPOINT` + token warning (M1), client connect / read timeouts (M2), URL path percent-encoding for `?` / `#` / `%` in filenames (L1), `O_NOFOLLOW` tempfile creation (L2), `Result`-based token-ASCII handling instead of `expect("token must be ASCII")` (L3), stale `.mlxcel-partial.*` cleanup at download start (L5), and parallel HEAD requests bounded by `buffer_unordered(8)` (L6).
- **Closed the residual TOCTOU window in the `video_url` allowlist resolver**. The dominant canonicalise → ffmpeg-open race was already closed by passing an `OwnedFd` to ffmpeg via `/dev/fd/N`. This change hardens the narrowed metadata→open gap by opening every allowlisted video file with `O_NOFOLLOW`: a symlink swap that occurs between the `metadata` call and the `open` syscall now returns `ELOOP` instead of silently following the swapped-in link. Subprocesses continue to receive `/dev/fd/N` (never the path), so they cannot be misdirected post-open regardless. A startup warning is now also emitted on non-Unix targets when `MLXCEL_VIDEO_DIR_ALLOWLIST` is set, because `O_NOFOLLOW` and fd-passing are unavailable on those platforms.

### CI
- Bump GitHub Actions to Node 24 runtime to clear the Node 20 deprecation warning surfaced on the macOS runners.

### Chore
- Replace `.map_or(false, ...)` with `.is_some_and(...)` in tokenizer call sites (clippy 1.93 lint clean-up).

## [v0.0.26] - 2026-05-10

### Added
- **TurboQuant KV cache.** New 3–4 bit KV cache compression family built on a Walsh–Hadamard transform op and a Lloyd-Max PolarQuant codebook generator ported to Rust. Four KV cache modes wired through `KVCacheMode`: `Turbo4` symmetric with per-model allowlist, `Turbo4Asym` Fp16-K + Turbo4-V, `Turbo3Asym` 3-bit Fp16-K + Turbo3-V, and `Turbo4Delegated` with a FP16 hot tail + packed turbo cold body. TurboQuant + `RotatingKVCache` integration covers sliding-window attention (B9). Sparse-V dequant scaffolding, Boundary-V layer protection that keeps the first/last layer at FP16, and a packed-aware `PagedKvLayout` round out the runtime. Quality gates: wikitext-2 PPL + NIAH harness, full 283K-token test split fixture, per-model PPL/NIAH results committed, and VLM B3 quality gates with image-token kurtosis. Speed gate matrix runner with M1 Ultra and M5 Max readings. User guide and validated config matrix published.
- **Server flag parity for KV quantization.** `--cache-type-k` and `--cache-type-v` flags accept `f16`, `q8_0`, `q4_0`, etc. matching llama-server semantics, with TurboQuant modes exposed as `mlxcel_turbo*` variants. KV cache quantization extended to continuous batching and a unified `--kv-cache-mode` flag layout shared across `mlxcel`, `mlxcel-server`, and `mlxcel download`.
- **Automatic Prefix Caching (APC) with hash blocks**. Hash-keyed block-table prefix reuse on top of v0.0.25's cross-sequence prompt-prefix KV cache, enabling shared physical blocks across requests with the same hashed prefix without per-request token-prefix matching cost.
- **OpenAI-compatible `response_format: {"type": "json_schema", ...}`** structured-output support for `/v1/chat/completions` and `/v1/completions`. Constrained decoding via `llguidance` (the same backend used by upstream mlx-vlm PR #1047) ensures every emitted token keeps the partial output conforming to the supplied schema. Per-request schema validation enforces a 64 KiB size cap, a 32-level nesting depth limit, and a 64-entry `$ref` count limit so an adversarial schema cannot exhaust CPU or memory during grammar compilation. The tokenizer environment is cached by SHA-256 fingerprint so consecutive requests to the same model share the build cost (~1–2 s for a 150k-vocab tokenizer). Reusable per-sequence `mask_buf` and `bias_buf` allocations eliminate per-token `Vec` allocation on the hot decode path. The legacy `json_object` mode is rejected with a clean 400 in this MVP; `json_schema` with a well-formed schema is the supported path. Supported on HuggingFace BPE tokenizers; SentencePiece and Tiktoken backends return a clean `UnsupportedTokenizer` error. Verbose llguidance internals are never surfaced in public error messages — they are routed to server-side tracing only.
- **`mlxcel download` / `mlxcel-server download` subcommand** to fetch HuggingFace model repository snapshots without Python tooling. Uses `hf-hub` with an allow-list file filter (SafeTensors, tokenizer, and config files only), cache-hit detection, and formatted per-file progress output. Supports `--local-dir`, `--revision`, `--token`, and `--force`. Default destination mirrors the `models/<repo-basename>` convention from AGENTS.md.
- **`/health` endpoint** now includes `context_size` (the configured `--ctx-size` value; `0` means model default) and `tool_call_parser` (`"mlxcel"` when the chat template exposes the `tools` variable, `null` otherwise). Both fields are present once a model is loaded; `context_size` is absent while loading, `tool_call_parser` serializes as `null` during startup so monitoring clients can distinguish "template has no tool support" from "model not yet loaded". The tool-support heuristic is extracted into a shared `template_mentions_tools()` helper used by both the health route and the existing `compute_supports_tools` fallback path.
- **Paged scheduler dispatch on `PagedKvLayout::cache_mode`** so the scheduler routes batches into the matching paged decode kernel for each KV cache mode.
- **Video input infrastructure for VLMs.** Gemma 4 video support with the new VLM video input pipeline; ffmpeg-backed frame extraction with single-pass extraction and a `Drop` guard for cleanup of temporary frame files; `video_url` content blocks wired through `/v1/chat/completions`; content-preservation tests covering the frame extraction path.
- **New models.** Youtu-VL vision-language model; Nemotron H Nano Omni vision plus follow-up correctness/validation hardening.
- Multi-task M1 Ultra benchmark refresh to 2026-05-08 and full M1 Ultra column resync in `benchmarks-by-hardware.md`.

### Changed
- **MLX upstream pin bumped twice.** First from the v0.0.25 baseline (`5d7e96cd`) to v0.32.0 / `c9aa5605`, then forward to `84961223` covering 3 PRs: #3443 splits the CUDA `qmm_naive` / `qmm_sm80` kernel bodies into new `qmm_naive.cuh` / `qmm_sm80.cuh` headers without changing the public ABI consumed by mlxcel's `patches/mlx/backend/cuda/quantized/qmm/qmm.h`; #3463 routes the CPU JIT preamble through `JitCompiler::get_preamble()` and renames the prebuilt symbol from `get_kernel_preamble` to `get_prebuilt_preamble` (mlxcel does not call either directly); #3475 fixes contiguity-flag accuracy in `AsStrided` by computing `data_size` from the actually-occupied stride range. Three-location pin update applied to `src/lib/mlx-cpp/CMakeLists.txt`, `src/lib/mlxcel-core/build.rs`, and `.github/workflows/release.yml` per `CLAUDE.md`. Fused Metal kernel launchers in `src/lib/mlx-cpp/turbo/` re-validated against both bumps: `mlx::core::fast::metal_kernel`, `mlx::core::full`, `mlx::core::Shape`, `mlx::core::float32`, `mlx::core::int32`, and `metal::fast::exp` symbols unchanged.
- **Refactor:** unified TurboQuant KV-cache CLI flags across `mlxcel`, `mlxcel-server`, and `mlxcel download` so all binaries accept the same `--kv-cache-mode` / `--cache-type-{k,v}` syntax.
- mlx-lm version reference in docs bumped from 0.31.2 to 0.31.3. The `bridge-overhead-microbench` reference at v0.31.2 is preserved because it pins the MLX C++ runtime, not the mlx-lm Python package.

### Performance
- **Sparse-V kernel:** fused per-thread Metal kernel that skips the full SDPA pass when sparse-V dequant predicts zero contribution; precomputed kernel rescale to drop per-token threadgroup barriers.
- **Turbo4Delegated decode hot path:** unified K storage to drop the per-step K concat; cold-V dequant cache across decode steps followed by a cold-V dequant Metal kernel that retires the FP16 memo; steel-attention-envelope fused SDPA kernel with parallelized Pass 1 softmax; delegated FP16 predecode compaction and lazy delegated FP16 sidecars; compressed fold moved before decode.
- **Compressed dequant-SDPA paths** for TurboQuant decode.
- **Server hot-path:** thread-local generation stream and uniform-batch RoPE collapse to remove per-request allocation in the steady-state batching loop.

### Fixed
- **TurboQuant continuous batching:** correct batch cache offset merging when batches with different cache offsets are joined or split; Turbo3 split-flag, documentation alignment, and an `ENV_LOCK` race in concurrent process startup.
- **Vision / VLM mixed batching:** per-sequence MRoPE alignment for mixed VL+text batches; per-sequence `per_layer_inputs` for Gemma 4 E2B/E4B VLM; mixed-length batching support for Gemma 4; relaxed cached-position shape check in Qwen VL chunked prefill; Qwen3.5-MoE batch-size validation on cached `position_ids` reuse.
- **Streaming and sampling:** correct streamed detokenization for byte-fallback tokens that previously leaked raw byte fragments to the client; top-p filter correctness for batched logits; token queue timeout handling during long prefills so clients no longer see spurious 408s on slow first-token paths; `StreamFilter` extended to cover Hermes-style `<tool_call>` / `</tool_call>` and Mistral Nemo `[TOOL_CALLS]` markers, which previously leaked raw markup into `delta.content` during streaming. Partial-marker buffering at token boundaries correctly holds back prefixes (e.g. `<tool_`) until the full tag can be confirmed, then releases them to `delta.content` if they turn out not to be a boundary. Gemma 4 `<|tool_call>` suppression is unaffected; the delimiter table ordering ensures the Gemma 4 pipe-delimited form wins the tiebreak over the Hermes plain form.
- **Models:** Gemma3-4B attention SIGABRT from a sliding-window mask `T_k` mismatch on long-context prompts; preserve Qwen2 fused QKV bias when it is present in the checkpoint; test fixture swap to Qwen2.5-1.5B base variant for the B3 quality gate; harden post-merge review findings on the Nemotron-H Nano Omni vision PR.

### Security
- Path-traversal defense in the downloader: `is_safe_relative_path` pre-filters each sibling filename returned by the HuggingFace API (rejects absolute paths, `..` components, backslash separators, and empty components). A secondary canonicalized `starts_with` guard on the resolved destination path is applied before writing each file. Download target files are written to a temporary path and atomically renamed into place, preventing partial writes from leaving corrupt files in the output directory (fixes C1 and H1 from security review).
- Structured-output schema limits (64 KiB serialized size, 32 nesting depth, 64 `$ref` count) and tightened `llguidance` parser caps (`max_grammar_size: 100 000`, `max_lexer_states: 50 000`) applied before grammar compilation so an adversarial client cannot use the schema endpoint as a CPU/memory exhaustion vector. Schema content is never echoed in public error messages.

## [v0.0.25] - 2026-04-24

### Added
- Cross-sequence prompt-prefix KV cache. New `KVCache::trim/detach/adopt` API enables adopting a previously-cached prefix on the next request. Backed by `PromptCacheStore`, an in-process LRU keyed by tokenized prompt prefix, plus a longest common token-prefix matcher (`PrefixMatcher`) for fast lookup. Paged KV cache gains block-table prefix reuse so adopted prefixes share physical blocks. Scheduler integration prefills only the unmatched suffix on cache hits. Wired into the server via `--prompt-cache-size`, `--prompt-cache-min-tokens`, and matching `LLAMA_ARG_*` env vars; multimodal/vision-aware cache key (`MultimodalDigest`) prevents cross-modality collisions. OpenAI-compatible `cached_tokens` is reported in `/v1/chat/completions` responses, mirrored to Prometheus counters, and verified by a multi-turn E2E test plus a prefill-latency benchmark. Design rationale and operator guide added to docs.
- Language-bias steering (Axis B, Phase 1). New `lang_analyzer` module with a Unicode script classifier (B2) and `TokenLanguageIndex` builder that scans the tokenizer vocabulary, partitions tokens by script, and persists the result to disk for fast warm starts (B3, B4). Sampling primitive `TokenBiasMap` + `apply_token_bias` is wired through `LangBiasSet` with `Conservative` / `Strict` policies (B5), exposed via CLI flags and a YAML config (B6), `LLAMA_ARG_LANG_BIAS` env var in `mlxcel-server` (B7), `LangBiasConfig` injection into the generator pipelines (B8), tracing fields and Prometheus counters (B9), byte-fragment CJK classification via UTF-8 start-byte analysis so byte-level BPE tokenizers correctly attribute fragments, byte-level reverse map for token decoding, and integration tests for the steering matrix (B10). User guide and Quickstart published (B11).
- `thinking_token_budget` sampling parameter for the Qwen3 family — caps tokens emitted between `<think>` / `</think>` markers without disabling streaming.
- `preserve_thinking` chat-template hook for Qwen 3.6 so multi-turn conversations retain prior `<think>` blocks instead of stripping them on subsequent turns.
- `StreamFilter` extended to recognize Qwen-style `<think>` / `</think>` token boundaries during streaming and route the segment into `reasoning_content`.
- `thinking_budget_tokens` extended to Gemma 4.
- `feat(benchmarks)`: bridge-overhead microbench tool measuring per-op cost of the Rust cxx bridge against Python nanobind across MLX primitives, with a published baseline and reproduction steps.
- `feat(ci)`: multi-stage pipeline-parallel smoke job activated using a Qwen3-0.6B fixture so PR runs catch PP regressions.
- Per-layer + per-sub-op decode profiling for Gemma 4, plus a Gemma 4 perf harness with the 2026-04-22 baseline used to drive the parity work below.

### Fixed
- Prompt cache prefix isolation — sequences whose prompts share a non-trivial prefix no longer leak adopted KV state across each other after detach/adopt.
- `MultimodalDigest` propagated to all `PromptCacheKey` callers after the + merge so vision-aware cache lookups stay collision-free.
- Gemma 4 `enable_thinking=false` no longer triggers degenerate output, and `reasoning_content` now streams correctly when `enable_thinking=true`.
- Tool-only assistant turns now emit `content: null` instead of `""` to match the OpenAI Chat Completions schema.
- `chat-template`: support flattened `extra_body` and pseudo-user tool responses so OpenAI-style tool flows render correctly under HF-style templates.
- `lang-analyzer`: decode tokens using the byte-level reverse map instead of the textual tokenizer view, so byte-level BPE (Qwen, Llama) tokens are classified by their actual code-point payload.
- `ci`: unblock Pipeline Parallel CI on Ubuntu by installing LAPACK and treating clippy `-D warnings` consistently.
- `vision`: read Gemma 4 encoder `hidden_size` from after `input_proj` so the multimodal projector wires the correct dimension on encoders that include a learned input projection.
- Bumped `cc` to 1.2.60 to silence the BSD `ar` probe warning surfaced by recent `cc-rs` releases.

### Changed
- Gemma 4 mlx-lm decode parity pass (closes the remaining gap on 26B / 31B / e2b):
  - Router RMS norm fused with top-k-then-softmax to remove a separate normalization pass.
  - SwitchGeGLU gate / up / geglu / down fused into a single `mlx::core::compile` window.
  - Metal-trace-driven attention / RoPE / per-layer chain fusion.
  - Compiled Gemma 4 SwitchGeGLU decode path enabled.
  - Single-query causal masks skipped in decode.
  - BF16 decode graph aligned with mlx-lm.
  - Proportional RoPE aligned with mlx-lm (no rotated-only normalization).
  - SwitchGLU projection order matched to mlx-lm.
  - QKV projection shape matched to mlx-lm.
  - Router top-k aligned with mlx-lm.
  - Load and MoE decode paths tuned.
  - Redundant residual copies in the decoder layer dropped.
  - SwitchGeGLU `expand_dims` collapsed and a MoE inner profiler added.
- `Qwen 3.5`: SSM decode masks aligned with mlx-lm; benchmark artifacts cleanup.
- `MLX`: upstream pin upgraded to **v0.31.2**; in-tree SDPA and steel-attention overlays dropped now that upstream covers them. Three-location update (`src/lib/mlx-cpp/CMakeLists.txt`, `src/lib/mlxcel-core/build.rs`, `.github/workflows/release.yml`) per CLAUDE.md.
- `CUDA`: QMM patches updated for the new upstream `lhs/rhs_indices` signatures.
- `deploy`: SIGTERM the running `mlxcel-server` after binary copy so the respawned supervisor picks up the new binary.
- `style`: `cargo fmt` swept across server modules to land previously-unformatted blocks.

## [v0.0.24] - 2026-04-18

### Added
- Zero-config multi-machine pipeline-parallel bring-up: `mlxcel-server --pp-auto N` declares pipeline depth; peers register via `--cluster-peers` seeds or opt-in mDNS discovery (`--cluster-discovery=mdns`). New `src/distributed/cluster_init.rs` owns deterministic stage assignment, port allocation, and byte-identical TOML emission consumed by the existing manual-TOML runtime path.
- RDMA-aware transport backend with transparent TCP fallback. Negotiates `io_uring` registered buffers on Linux and `kqueue` batched send on macOS, emits exactly one structured log line on fallback, and preserves the `Arc<dyn Transport>` abstraction used by `activation_transfer.rs`. New `rdma_capabilities.rs`, `rdma_transport.rs`, and `bench_activation.rs` harness.
- 2D `(pp_stage, tp_rank)` mesh composing PP with TP for Llama-70B-class topologies. Adds `NodeRole::PipelineTensorParallel`, validation for exact `pp_size × tp_size` coverage with unique coordinates, registry helpers, `TrafficClass` routing (`TpCollective` / `PpActivation`), and grid-coherent KV admission (`coordinated_2d_admission`) (addresses).
- Byte-accurate pipeline auto-partition with adjacency constraints. `ModelProfile` gains per-layer byte weights plus layer-adjacency constraints so the balancer refuses to cut MoE expert layers or Gemma 4 KV-shared source/consumer pairs. Drops the hand-specified `--pp-layers` requirement for MoE and gemma-4-e2b-it-4bit. Extracted into `partition_balance`, `partition_profile`, and `partition_quality` modules.
- Elastic pipeline-parallel repartitioning behind `--enable-elastic-pp`. `RepartitionCoordinator` drives `Idle → Draining → Rebalancing → Resuming → Idle` and emits `RepartitionEvent` to a transport-agnostic sink without a full cluster restart. CLI flags: `--elastic-pp-drain-timeout`, `--elastic-pp-pressure-fraction`, `--elastic-pp-cool-down`.
- Per-stage LoRA adapter composition across pipeline ranks via the existing `--adapter` flag. Each stage loads only the adapter tensors inside its layer range through a new filtered safetensors loader (`load_safetensors_filtered`), fuses them in place with the same `fuse_lora_weights_into` primitive that backs the non-PP path, and unchanged-family guards (`ensure_no_adapter`) prevent silent drops. Llama family implements composition; parity integration test asserts bit-equality with the single-process adapter run.
- Stage-executor coverage for five new families: Mistral dense, Mixtral 8x7B MoE, DeepSeek V3 (MLA + routed MoE with MTP-trailer strip), Llama 4 Scout text-only tower, and Mamba-family hybrids Jamba and Nemotron-H. `StageFamily` enum plus `supported_families()` surfaces per-family capability on the server startup log.
- Pipeline-parallel observability: `/metrics` endpoint renders per-stage utilization, rolling bubble ratio, activation-transfer latency histograms (p50/p95/p99 per stage pair), and KV admission rejection counters labeled by stage and reason. `--metrics-port`, `--debug-pp-trace <PATH>` (chrome-tracing JSON), and `AdmissionDiagnostic` replace opaque 500s on rejection. Grafana dashboard JSON at `docs_internal/performance/pipeline-dashboard.json`.
- Multi-host pipeline-parallel regression CI harness at `.github/workflows/pipeline-parallel-ci.yml`: `two-host-logical` on GitHub runners (path-filtered, intended as required status) plus `three-host-real-model` gated by the `ci:pp-three-host` PR label or manual dispatch. Shares shell entry points with local reproduction (refs).
- `VisionFeatureCache` LRU for multi-turn VLM image feature reuse, wired through Gemma 4 VLM, Qwen2.5-VL, and Qwen3-VL via `_with_cache` variants. Cache keys are filesystem paths or SHA-256 digests of inline payloads. New `--vision-cache-size N` CLI flag (default 20, 0 disables) (matches).
- Null/empty-cache safety guards in the batch scheduler. Pure-text requests with zero tokenized prompt tokens are rejected before admission (VLM image/audio injection paths unaffected); `execute_decode_step` and `execute_batched_decode` no-op on empty `seq_ids`. Mirrors the upstream mlx-lm BatchKVCache extend/filter/merge null guards.

### Fixed
- Auto-detect per-layer quantization bit overrides in `UnifiedLinear::from_weights_with_mode` and `FusedQKVLinear::from_weights_separate_with_mode`. New `infer_quantization_bits()` verifies the MLX invariant `packed_in * 32 == bits * num_groups * group_size` and infers the actual bit width from tensor shapes when the caller-supplied bits disagree. Enables qwen3.6-35b-a3b-4bit, which stores router-gate and shared-expert-gate at 8-bit while the rest of the model is 4-bit.
- Use additive f32 attention mask (0.0 attended, f32::MIN masked) in `prepare_inputs_for_multimodal` instead of the previous multiplicative INT32 0/1 mask. `mx.fast.scaled_dot_product_attention` treats non-bool masks as additive bias on pre-softmax scores, so the old form silently leaked padding tokens into the attention distribution whenever `attention_mask` contained a zero.
- Mirror conditional `embed_scale` to `TensorParallelGemma4Model::forward_impl`. Previously, `multiply_scalar` was applied unconditionally after `embed_tokens`, double-scaling text embeddings and incorrectly scaling image/audio features from VLM callers. Moved into the `None` arm only, matching `Gemma4TextModel::forward`. Added regression test asserting TP/non-TP logits match for both `input_embeddings` and `input_ids` paths.
- Wrap every `cache.conv_state = Some(slice_axis(...))` assignment in `mlxcel_core::contiguous(&tail, false)` across mamba, mamba2, nemotron-h, and jamba, plus the two NemotronH fused-kernel paths. `slice_axis()` returns a lazy MLX `Slice` graph node that retains the source `padded_input` as a live input, causing per-step memory growth proportional to sequence length. 50-step shape-plateau regression test added per model.
- Apply RMS norm BEFORE `embedding_projection` on the encoder-side dim in `Gemma4 Multimodal Embedder` (was previously AFTER, on the text-side `hidden_size`). Mirrors upstream mlx-vlm. Renamed field `post_projection_norm` → `pre_projection_norm`. **BREAKING** for pre-fix VLM checkpoints: re-download `mlx-community/gemma-4-*-it-4bit` to obtain the post-rename weights.
- Apply `sqrt(hidden_size)` `embed_scale` to text embeddings in `Gemma4VLModel::get_input_embeddings_with_audio` BEFORE merging vision/audio features, and make the scalar multiply in `Gemma4TextModel::forward` conditional on `input_embeddings` being `None`. Vision/audio features are already in language-model embedding space; double-scaling them degraded multimodal generation quality.
- Implement proportional RoPE for Gemma 4 full-attention layers. Real Gemma 4 checkpoints declare `rope_type="proportional"` on full-attention and `rope_type="default"` on sliding-attention layers; the previous implementation silently dropped `rope_type` and normalized by the rotated-only slice instead of the full `head_dim`. New `mlxcel_core::rope_proportional` module with `compute_proportional_rope_freqs` and `apply_proportional_rope` matching the upstream slice/concat/fast_rope/re-splice pipeline. For head_dim=256, partial_rotary_factor=0.25, the two formulations differ by a factor-of-4 exponent shift.
- Gemma 4 audio feature extractor: drop `+0.5` phase shift in Hann window so it uses the periodic form `w(i) = 0.5 - 0.5·cos(2π·i/N)` matching HuggingFace Gemma 4. Prepend `frame_length/2 (160)` zero samples before frame extraction for semicausal convention (first frame centered at t=0). Use `total_len` in `num_frames` calculation and correct `frame_size_for_unfold` to use `frame_length+1` only for non-HTK preemphasis. Restores the correct 100 frames for 1s 16 kHz audio with 10 ms hop.
- Ensure `conv_input` cache slice is contiguous in GatedDeltaNet forward paths (Qwen 3.5, Kimi Linear, Qwen 3 Next). `mlxcel_core::slice()` calls `mlx::core::slice()` which creates a graph node holding source reference — without `contiguous()`, every cached entry holds the full `conv_input` buffer, preventing freeing and causing per-step memory growth proportional to sequence length. 50-step regression test added.
- Default NemotronH `time_step_limit` to `(time_step_min.unwrap_or(0.0), time_step_max.unwrap_or(+inf))` unconditionally when absent. Changed from `(f32, f32)` to `Option<(f32, f32)>` so absent configs are distinguishable from explicit `(0.0, +inf)` sentinels. Matches upstream mlx-lm behavior.

### Changed
- Replace Gemma 4 `ScaledLinear` wrapper with `UnifiedLinear` directly across both `Gemma4TextModel` and `Gemma4StageModel` (tensor-parallel path). New `per_layer_projection_scale: f32` field stores `(hidden_size as f32).powf(-0.5)` and is applied explicitly in `project_per_layer_inputs()` after the linear forward pass, preserving bit-identical math.

## [v0.0.23] - 2026-04-15

### Fixed
- Render chat templates that use Python-style dict/string methods. Extends minijinja's `unknown_method_callback` with shims for `.get`, `.items`, `.keys`, `.values`, `.strip`, `.lstrip`, `.rstrip`, `.startswith`, `.endswith`, `.split`, `.rsplit`, `.replace`, `.join`, `.upper`, `.lower`, `.title`, `.capitalize`, `.casefold`, `.swapcase`, `.find`, `.count`, `.is{digit,alpha,alnum,space,upper,lower}`. Previously rendering silently fell back to `to_prompt()`'s `User: ... Assistant:` format, and instruction-tuned models echoed `Assistant:` in a loop.
- Pass `tools` as an empty iterable (not `None`) so `{% if tools is iterable and tools | length > 0 %}` guards work under minijinja. Fixes Qwen 3 Next, Nemotron-H, and Nemotron-NAS tool-free rendering.
- Strip HuggingFace `transformers`' `{% generation %}` / `{% endgeneration %}` extension markers during template preprocessing so SmolLM 3 parses cleanly.
- Apply the Gemma 4 structural-token stream filter and non-streaming cleanup unconditionally, not only when tool parsing is enabled, so plain chat responses no longer leak `<|channel>`, `<channel|>`, `<turn|>`, `<|turn>`, `<|tool_call>`, or `<tool_call|>` markers into content.
- Extend `clean_content_markers` with `<|channel>` / `<channel|>` / `<|tool_call>` / `<tool_call|>` so stray closing tags that Gemma 4 occasionally emits in non-thinking mode are stripped even without a matching open tag.

### Added
- `test_all_local_model_templates_render` ignored-by-default audit that renders every locally-available model against three canonical scenarios (simple user, system + user, multi-turn with `<think>` blocks). Current result: 85 models checked, 249/249 scenarios pass, 0 failures, 6 intentional template `raise_exception` rejections categorized separately.

### Changed
- Clean up pre-existing `cargo clippy --release -p mlxcel --lib` warnings (7 → 0): replace `unwrap()` after `is_some()` checks in `distributed/config.rs`, bind the MoE router via `if let` chain in the Gemma 4 TP path (`distributed/tensor_parallel/llama_runtime.rs`), collapse two character-identical QKV shard branches, auto-elide `'a` lifetimes, collapse a `thunderbolt_transport.rs` nested `if`, replace a manual `% != 0` with `is_multiple_of()` in NVFP4 sanitize, and drop a now-redundant `cache.as_deref_mut()` + `mut` annotation in Qwen 3.5 `GatedDeltaNet::forward`.
- Clean up pre-existing webpage `pnpm lint` / `tsc --noEmit` errors (20 / 4 warnings → 0 / 0): replace framer-motion wrapper `any`-typed props with `HTMLMotionProps`, `let` → `const` in `downloads.tsx`, swap `<img>` for `next/image`'s `<Image />` on the local Lablup logo, and rewrite `use-os.ts` to avoid synchronously setting state inside `useEffect` with proper `NavigatorUAData` / `WebGLDebugInfoExtension` typings.

## [v0.0.22] - 2026-04-13

### Added
- Pipeline stage executor framework with per-family executors
- Gemma 3 pipeline stage executor
- Gemma 4 pipeline stage executor
- Qwen3 pipeline stage executor
- Qwen3.5 pipeline stage executor
- GLM4-family pipeline stage executors
- GLM MoE DSA pipeline stage executor
- gpt-oss pipeline stage executor
- In-process pipeline stage worker loop
- CLI pipeline generate path
- Server pipeline runtime integration
- Pipeline transport lifecycle controls
- TCP-backed remote pipeline stages
- Thunderbolt transport backend for remote pipeline parallelism
- Multi-machine validation for remote pipeline parallelism
- bench_decode `--cooldown` and `--big-cooldown` for M5 Max thermal management
- M5 Max benchmark refresh for 2026-04-13 (97 models, 88 pass; 8 multimodal models restored)

### Fixed
- Tolerate stale `model.safetensors.index.json` in mlx-community repackaged quants (gemma3-4b, gemma3n-e2b/e4b, llama-4-scout-17b, mistral-small-3.1, molmo2)
- Tolerate partial `text_config` (no `num_hidden_layers`) in single-rank tensor-parallel planning (LLaVA-1.5, LLaVA-Next-Mistral)
- Prevent Gemma 4 special tokens from leaking into streaming content deltas
- Complete remote pipeline lifecycle recovery
- bench_decode single-model runs no longer truncate the day's full-suite CSV
- Log lazy pipeline peer reconnects

### Changed
- Generalize stage executor backends and remove legacy stage executor file
- Transport-capable pipeline runtime seam

### Tests
- Pipeline server smoke validation
- Pipeline rollout real-model coverage

### Docs
- Remote pipeline usage examples
- Remote pipeline rollout workflow
- Refreshed M5 Max benchmark documentation with measurement-variance analysis
- Recorded issue execution workflow

## [v0.0.21] - 2026-04-12

### Added
- Paged KV cache substrate with batch scheduler integration
- Native paged decode kernel paths for rotating and chunked caches
- Paged compatibility for windowed caches
- Default paged decode for supported server workers
- Paged KV transfer observability
- NVFP4 load-time dequantization for Gemma 4 nvfp4 checkpoints
- F8_E4M3 / F8_E5M2 safetensors loading for nvfp4 checkpoints
- Paged decode rollout benchmark matrix and eligibility tracking

### Changed
- Unify model-owned sequence state with backend seam
- Vectorize batched decode positional metadata
- CI: auto-promote pre-release to full release after successful builds

### Fixed
- Skip Teams notification when webhook URL secret is not configured

## [v0.0.20] - 2026-04-10

### Added
- In-process tensor parallel runtime for Llama
- Tensor parallel support for Qwen2, Qwen3, and Qwen3.5 text models
- Gemma 3 tensor-parallel runtime with tp4 parity stabilization
- Gemma 4 tensor-parallel support
- Dense TP support for ERNIE 4.5 and Hunyuan v1 models
- Server batching support for tensor parallel runtimes
- Tensor-parallel config wiring into CLI and server entrypoints

### Fixed
- Qwen 3.5 tensor-parallel parity on large CUDA models

### Changed
- Expand tp4 parity coverage to larger models and server end-to-end tests

## [v0.0.19] - 2026-04-10

### Added
- Improved sharded/multi-file safetensors loading robustness
- Teams release notification via Power Automate webhook

### Fixed
- Ensure input contiguity in QuantizedMatmul for MLA models on CUDA
- Skip models exceeding system memory in bench script
- Increase CUDA warmup timeout and add JIT preheat to bench script

## [v0.0.18] - 2026-04-08

### Added
- GatherQMM CUDA implementation via upstream MLX upgrade to b98831ad
- SM80 and naive QMM dispatch paths for non-Hopper CUDA GPUs
- Gemma 4 CUDA support: all 7 variants (e2b, e4b, 26b, 31b in 4bit/8bit)
- Qwen 3.5 CUDA support: 27b-4bit, 9b-bf16, 35b-MoE-4bit

### Fixed
- Mixed-type bf16/float JIT compilation failures in CUDA binary_ops.cuh
- Remove stale NO_GPU(BlockMaskedMM) override that conflicted with upstream implementation
- Gemma 3-4b and Gemma 3n (e2b, e4b) recovered on CUDA via binary_ops fix

### Changed
- Upgrade MLX C++ upstream from 6a9a121d to b98831ad
- Replace custom gather_qmv.cu with upstream integrated qmv.cu
- Sync CUDA quantized.cpp with upstream SM80/naive dispatch paths

### Performance
- GB10 CUDA: 14 models recovered from FAIL, 24 models improved >10%
- mamba2-1.3b +180%, minicpm-2b +131%, llama-3.1-8b +130%, hunyuan-dense +125%, llama-3.2-1b +115%

## [v0.0.17] - 2026-04-06

### Fixed
- Resolve broadcast crash in Gemma 4 chunked prefill with undersized attention mask

## [v0.0.16] - 2026-04-05

### Added
- Audio input support for server chat completions endpoint
- Gemma 4 audio encoder and audio-language model support
- Metal 4 fused attention path
- OpenAI-compatible tool calling support
- M5 GPU acceleration experiments
- M5 Neural Accelerator rollout research

### Changed
- Unify attention dispatch for Metal 4 path

### Fixed
- Propagate client disconnection to BatchScheduler to prevent orphaned sequences
- Harden tool calling with input limits, parser improvements, and format handlers
- Remove eval() calls from qwen3_moe forward hot path
- Resolve Gemma SDPA crash on M1 by reducing threadgroup memory for head_dim=256
- Update compiled.cpp patch for upstream MLX API change
- Add str.split() support in chat template for Gemma 4 multi-turn

## [v0.0.15] - 2026-04-03

### Added
- Gemma 4 text and VLM model support
- User-facing warning when loading full-precision bf16 models
- Download webpage with Next.js static site (EN/KO i18n)

### Changed
- Extend bf16→f16 weight conversion to all Apple Silicon generations
- Audit f32 upcasts and optimize MoE gate sigmoid for fp16 co-issue
- Improve Metal 4 fused attention scaffolding with research documentation
- Reuse cached MLX source for faster rebuilds

## [v0.0.14] - 2026-04-03

### Added
- Logprobs support for chat completions and completions endpoints
- Runtime Apple Silicon generation detection for hardware-specific optimizations
- Prefill tile alignment for M5 Neural Accelerator
- Batched speculative decode verification for NA utilization
- Batched prefill in server mode
- Layer pipelining with strategic async_eval
- Metal 4 fused attention kernel scaffolding
- KV cache INT8 quantization for memory savings
- INT8 quantization optimization for M5 Neural Accelerator
- Multimodal chat template support for VLM image token placement
- Apple Silicon precision hardware guide documentation

### Changed
- Centralize bf16→f16 weight conversion in shared VLM loading path
- Skip bf16→f16 conversion for quantized models (restores +20% throughput)
- Add compiled gelu_topk kernel matching Python mlx-lm `@mx.compile` pattern
- Expand QKV projection fusion to GQA models
- Expand compiled MLP fusion to non-quantized models
- Fuse Q/K/V projections in Gemma v1 attention for faster decode
- Refactor AGENTS.md into focused reference docs (313→75 lines)

### Fixed
- Auto-convert bf16 weights to f16 on M5 for Metal JIT compatibility
- Skip add_special_tokens when prompt already contains BOS token (double-BOS fix)
- Prevent NemotronH all-`<unk>` output on M5 Max by avoiding mixed float32/float16 ops
- Prevent Nemotron-H/NAS GPU hang and state corruption on M5 Max
- Trim NemotronH internal caches after padded prefill to prevent GPU hang
- Fix PhiMoE expert activation from GeGLU to SwiGLU
- Fix matmul outside compile boundary in FP MLP to fix output corruption
- Replace gelu_approx power(x,3) with erf-based GELU to fix NaN in vision encoder
- Guard multimodal chat template to avoid garbled output on text-only VLMs
- Skip compiled FP MLP for bfloat16 models
- Patch MLX compiled kernel JIT to cast mixed bfloat16/float operands
- Patch MLX Metal kernels for macOS 26.4 compatibility
- Correct M5 Max benchmark results affected by GPU cascade corruption

## [v0.0.13] - 2026-03-31

### Added
- Mistral4 MLA (Multi-head Latent Attention) language model support
- Molmo-Point VLM model support
- NemotronSuper model support (upstream mlx-lm sync)
- `sync-upstream` Claude Code command for tracking mlx-lm/mlx-vlm changes

### Changed
- Fuse GatedDeltaNet decode step with `mlx::core::compile` for improved throughput
- Apply MRoPE and position ID optimizations to Qwen3-VL-MoE
- Fast-path single-token decode position IDs in Qwen3-VL
- Vectorize Qwen3-VL interleaved MRoPE with `take_along_axis`
- Optimize VLM vision encoding and sampling pipeline
- Use SDPA for NemotronH attention, boosting decode throughput 59%

### Fixed
- Improve SSM/Mamba2 numerical precision with float32 dt computation
- Improve GatedDelta numerical precision with float32 state
- Resolve Mamba/NemotronNAS output corruption with softplus overflow and fused norm grouping
- Guard Qwen3.5 GatedDeltaNet state batch dimension mismatches
- Use `h.shape` instead of `inputs.shape` for Ministral3 attn_scale
- Document scalar offset invariant for Llama4 BatchKVCache compatibility
- Correct model_tests.md table placement and dedup nemotron entries

## [v0.0.12] - 2026-03-26

### Added
- Compiled C++ operations using `mlx::core::compile(shapeless=true)` for small model throughput:
  - `compiled_gelu` / `compiled_gelu_approx`: fused GELU activation kernels
  - `compiled_geglu_activation`: fused GELU-gated activation (`gelu(gate) * x`)
  - `compiled_softcap`: fused softcap (`tanh(x/cap)*cap`) for Gemma2
  - `compiled_softcap_sdpa`: entire attention path with softcap fused into single compiled graph
  - `compiled_softcap_sdpa_gqa`: fused GQA + softcap SDPA variant
  - `compiled_clip_residual`: fused float16-safe residual addition for Gemma3
  - `compiled_gelu_mlp_forward`: full GELU MLP as single compiled graph
- `UnifiedLinear::quantized_weight()` accessor for compiled MLP kernel dispatch
- Distributed inference framework: node discovery, cluster configuration, tensor/pipeline parallelism, disaggregated serving
- Comprehensive mkdocs documentation site (EN/KO) with PDF export
- Project-specific Claude Code commands and skills

### Changed
- Gemma3: fused SDPA, pre-computed GemmaRMSNorm, skip decode masks, Gemma3 1B reaches 94% of Python mlx-lm
- Gemma2: uses `compiled_softcap_sdpa_gqa` with internal GQA head expansion
- StarCoder2: uses `compiled_gelu` activation
- Phi3: pre-compute SuScaledRoPE scale array at load time
- Hoist env var checks out of generation hot loop
- Incremental token history and cached EOS in BatchScheduler
- Use MLX native `load_safetensors()` for faster weight loading
- Optimize model loading with batched synchronization

### Fixed
- OpenAI API streaming response format compatibility
- Guard compiled MLP/MoE paths against non-standard quantization params (`group_size != 64` or `bits != 4`)

## [v0.0.11] - 2026-03-18

### Added
- Compiled kernel fusion for `relu_squared` and `silu` activation functions
- Compiled kernel fusion for MoE gate and `compute_dt` operations
- Fused SSM Metal kernel for Mamba2 single-token decode
- Compiled MoE gate function for NemotronH
- Fused MoE forward function for NemotronH
- Fused Mamba2 mixer forward for NemotronH
- NemotronH full-forward C++ decode path (experimental, disabled)
- `MLXCEL_FORCE_SYNC` debug flag for pipelining analysis
- `MLXCEL_PROFILE_PIPELINE` for precise build/wait timing
- Per-block and build/eval profiling for NemotronH

### Fixed
- Auto-cast SDPA mask to Q dtype, preventing mask type errors across models
- Load float16 weights natively on Metal (was converting to float32)
- Eliminate float32 type promotion across all models
- Prevent float32 type promotion in NemotronH hidden states
- Add affine fast-path for quantized_matmul (omit mode parameter)
- Correct mlx-lm benchmark baselines and update nemotron/mamba results

### Changed
- Optimize Mamba single-token decode path and remove unnecessary copies

## [v0.0.10] - 2026-03-17

### Fixed
- ExaOne4: Cast causal mask to bfloat16 to match model weights dtype (MLX SDPA requires mask type to promote to output type)
- StableLM: Read `eos_token_id` from config.json instead of hardcoding 0, fixing premature 1-token generation

### Changed
- Add static mode string pool for quantized ops to avoid per-call heap allocation in C++ bridge hot path

## [v0.0.9] - 2026-03-17

### Added
- GptOss MoE model with sinks SDPA support
- MXFP4/NVFP4/MXFP8 quantization mode support across FFI bridge and model layers
- GPT-OSS benchmark results to model test documentation

### Fixed
- Set wired memory limit to `gpu_max_memory_size` by default

### Changed
- Re-benchmark all models after wired limit fix

## [v0.0.8] - 2026-03-17

### Fixed
- Support explicit `head_dim` config field in Qwen3-VL, Qwen2-VL, and Qwen2-MoE models, fixes Qwen3-VL-32B crash where `head_dim(128) != hidden_size/num_heads(80)`
- Switch macOS CI runner to macos-15 for Xcode 16+ C++20 ranges support

### Changed
- Add CUDA release pipeline and refresh benchmark report with MoE results

## [v0.0.7] - 2026-03-16

### Added
- GatherMM/GatherQMM for MoE model support on CUDA (#34)
- CUDA bf16 support: type promotion table patching, mixed-precision binary kernels, normalization ops, reduce accumulation with fp32 precision, native bf16 array creation in bridge layer (#42-#46)
- CUDA bf16 validation scripts and documentation (#47)
- CUDA GB10 benchmark results for 57 models
- GB10 vs M1 Ultra benchmark comparison report
- `--batch-size` and `--ubatch-size` as llama-server compatible aliases (#32)
- Debian packaging, man pages, and optimized release profile
- CUDA build guide and build troubleshooting documentation (#33)

### Fixed
- CUDA qmv shared-memory optimization with block.sync() fix
- CUDA dtype and fp16 bridge fixes
- C++ bridge build: removed `-flto`, upgraded to C++20
- C++ bridge LTO enabled only on macOS

### Changed
- Bumped MLX to v0.31.1, GPU backend now shown in runtime display
- CUDA qmv kernel optimized with shared memory x-broadcast and `__restrict__`
- Phase 19 CUDA optimization report and final benchmarks

## [v0.0.6] - 2026-03-14

### Added
- Continuous batching with iteration-level BatchScheduler for concurrent request handling
- Request lifecycle types and sequence state machine for batch management
- Per-sequence KV cache isolation and CachePool for independent request processing
- Tensor-batched decode forward pass for efficient multi-sequence generation
- Preemptive scheduling and chunked prefill for better latency and throughput
- HTTP server integration with batch scheduler and concurrency support
- Explicit `forward_batched()` for Qwen3 with split-attention support
- Continuous batching benchmarks and observability instrumentation
- Feature gate for batching to preserve CLI single-request path

### Fixed
- Scheduling policy now admits queued requests to grow batch beyond initial size

### Changed
- Added continuous batching development guide and benchmark comparison documentation
- Benchmark results for 84 models with scheduler fix improvements

## [v0.0.5] - 2026-03-11

### Added
- Phi4-SigLIP vision-language model support with NaFlex-style patch processor and SigLIP2 vision tower
- Phi4MM vision-language model support with SigLIP + HD transform + AvgPool2d pipeline
- MiniCPM-o vision-language model support with SigLIP + Perceiver-style resampler
- Moondream3 vision-language model support with packed int4 dequantization and BOS-prefix prompting
- Runtime LoRA support on Linear layers with `Cell<bool>` active toggle for on-the-fly application
- `after_prefill()` dispatch through LoadedModel enum and LanguageModel trait
- Server support for data URIs, file URLs, bare local paths, and http(s) image fetches

### Fixed
- Phi4MM VLM: add SuScaledRoPE (longrope) to Phi3 attention for correct positional encoding
- Phi4MM VLM: fix image token placement in prompt (insert after `<|user|>` tag, not before entire prompt)
- Phi4MM VLM: use runtime LoRA instead of weight fusion, matching Python PEFT behavior
- MiniCPM-o VLM: switch text backbone from Qwen3-VL (MRoPE) to standard Qwen3 (standard RoPE)
- MiniCPM-o VLM: add automatic Qwen3-style chat template wrapping for models without chat_template
- Moondream3 VLM: fix RoPE layout (NeoX-style halves), attention mask dtype, and vision tiling
- Moondream3 VLM: use exact GELU for tau scaling and MoE GeGLU matching Python F.gelu

### Changed
- Synced mlx-vlm upstream Qwen-VL: fused-SDPA head-dim padding in shared Qwen3-VL vision encoder
- Refactored server image extraction into async edge helpers with multi-format support

## [v0.0.4] - 2026-03-10

### Added
- Tiktoken BPE tokenizer support for models using `.tiktoken` vocabulary files (HunYuan MoE 13B)
- Quality gate entry point script (`scripts/run_quality_gate.sh`) with `--include-serial-helpers` and `--full` modes
- Comprehensive model validation: 71/74 local models pass (95.9%)

### Fixed
- Solar Open 100B-4bit config parsing: add serde defaults for `n_group`/`topk_group` in GLM4 MoE config
- GatedDeltaNet `RMSNormGated`: promote SwiGLU gate path to float32 before restoring hidden-state dtype (upstream mlx-lm parity for Qwen3Next/Qwen3.5)
- Step3p5 sliding-window layers now use `RotatingKVCache` instead of plain `KVCache`
- Suppress deprecated-copy warning in mlxcel-core build for MLX v0.31.0

### Changed
- Converged model registration: centralized config-backed text model registration in `src/model_metadata.rs`
- Split mlxcel-core internals into focused modules: `cache.rs`, `ops.rs`, `dtype.rs`, `sampling.rs`, `generation_policy.rs`, `streams.rs`
- Extracted large-model helper hotspots: `gemma3n_helpers.rs`, `llama4_helpers.rs`, `qwen3_next_helpers.rs`
- Split `LoadedModel` capabilities into `loaded_model_capabilities.rs` with `VlmRuntimeRef`
- Separated model detection (`detection.rs`) and sanitization (`sanitize.rs`) helpers
- Unified model loading descriptors with `StaticModelDescriptor` and `model_load_policy()`
- Normalized server startup edge inputs into `cli_input.rs`
- Removed unsafe `Send`/`Sync` auto traits from `ModelProvider`
- Strengthened vision merge contracts with dedicated tests
- Refreshed architecture, control-plane guide, and model addition documentation

## [v0.0.3] - 2026-03-10

### Fixed
- Streaming UTF-8 corruption for multi-byte characters (e.g., Korean, CJK) caused by byte-level BPE token boundaries
- Default `max_tokens` increased from 512 to 4096 so thinking models produce complete responses
- Release archive now includes `mlx.metallib` for Metal GPU acceleration

## [v0.0.2] - 2026-03-10

### Added
- Solar Open 100B INT4 model support with GPTQ conversion
- MiniMax-M2 MoE model support

### Fixed
- GPU wired memory limit now opt-in via `MLXCEL_WIRED_LIMIT` environment variable
- Llama4 vision encoder now uses UnifiedLinear to support quantized weights
- Molmo2 VLM inherits quantization config correctly; stale examples updated
- PaliGemma2 VLM no longer produces pad/EOS tokens instead of correct output
- Qwen3.5 VLM loader variants corrected
- Resolved all clippy warnings in vision and loading modules

### Changed
- Major codebase refactoring: modularized server, CLI, loader, and multimodal paths
- Extracted loader modules into `src/loading/` directory (SigLIP, Pixtral, Gemma, LLaVA, Qwen VLM loaders)
- Moved CLI command handlers under `src/commands/`
- Grouped execution policy helpers under `src/execution/`
- Grouped multimodal helpers under `src/multimodal/`
- Split server into config, state, streaming, and media helper modules
- Centralized LoadedModel embedding dispatch and reduced accessor boilerplate
- Shared sampling config assembly across CLI and server
- Refined model detection helpers with added guide
- Refreshed architecture and vision documentation

## [v0.0.1] - 2026-03-07

Initial public release of mlxcel.

### Added
- 59+ model architectures: Transformers, MoE, SSM/RNN, and Hybrid models
- Vision-Language Model support: Gemma 3, LLaVA, Llama 4, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Pixtral, Phi-3.5 Vision, and more
- OpenAI-compatible HTTP server with SSE streaming
- `mlxcel-server` standalone binary as llama-server drop-in replacement
- LoRA adapter loading and fusion at runtime
- Speculative decoding with draft models
- Advanced sampling: Top-P, Top-K, Min-P, XTC, DRY penalty, repetition/frequency/presence penalties
- Chat template support via Jinja2 (minijinja)
- Unix domain socket support for server mode
- EOS token detection from generation_config.json
- SentencePiece tokenizer support
- Linux + CUDA backend support (CUDA 12.0+, cuDNN 9+)
- Direct MLX C++ bindings via cxx FFI (zero Python dependencies)
- Pre-allocated KV cache with slice_update for O(1) per-token performance
- Sliding window and rotating KV cache support
- UnifiedLinear layer supporting both quantized and non-quantized models
- GitHub Actions release workflow for macOS ARM64
- Profile mode for prefill/decode timing analysis

[v0.7.0-beta.1]: https://github.com/lablup/mlxcel/compare/v0.6.0...v0.7.0-beta.1
[v0.6.0]: https://github.com/lablup/mlxcel/compare/v0.5.2...v0.6.0
[v0.5.2]: https://github.com/lablup/mlxcel/compare/v0.5.1...v0.5.2
[v0.5.1]: https://github.com/lablup/mlxcel/compare/v0.5.0...v0.5.1
[v0.5.0]: https://github.com/lablup/mlxcel/compare/v0.4.3...v0.5.0
[v0.4.3]: https://github.com/lablup/mlxcel/compare/v0.4.2...v0.4.3
[v0.4.2]: https://github.com/lablup/mlxcel/compare/v0.4.1...v0.4.2
[v0.4.1]: https://github.com/lablup/mlxcel/compare/v0.4.0...v0.4.1
[v0.4.0]: https://github.com/lablup/mlxcel/compare/v0.3.3...v0.4.0
[v0.3.3]: https://github.com/lablup/mlxcel/compare/v0.3.2...v0.3.3
[v0.3.2]: https://github.com/lablup/mlxcel/compare/v0.3.1...v0.3.2
[v0.3.1]: https://github.com/lablup/mlxcel/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/lablup/mlxcel/compare/v0.2.1...v0.3.0
[v0.2.1]: https://github.com/lablup/mlxcel/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/lablup/mlxcel/compare/v0.1.4...v0.2.0
[v0.1.4]: https://github.com/lablup/mlxcel/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/lablup/mlxcel/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/lablup/mlxcel/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/lablup/mlxcel/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/lablup/mlxcel/compare/v0.0.31...v0.1.0
[v0.0.31]: https://github.com/lablup/mlxcel/compare/v0.0.30...v0.0.31
[v0.0.30]: https://github.com/lablup/mlxcel/compare/v0.0.29...v0.0.30
[v0.0.29]: https://github.com/lablup/mlxcel/compare/v0.0.28...v0.0.29
[v0.0.28]: https://github.com/lablup/mlxcel/compare/v0.0.27...v0.0.28
[v0.0.27]: https://github.com/lablup/mlxcel/compare/v0.0.26...v0.0.27
[v0.0.26]: https://github.com/lablup/mlxcel/compare/v0.0.25...v0.0.26
[v0.0.25]: https://github.com/lablup/mlxcel/compare/v0.0.24...v0.0.25
[v0.0.24]: https://github.com/lablup/mlxcel/compare/v0.0.23...v0.0.24
[v0.0.23]: https://github.com/lablup/mlxcel/compare/v0.0.22...v0.0.23
[v0.0.22]: https://github.com/lablup/mlxcel/compare/v0.0.21...v0.0.22
[v0.0.21]: https://github.com/lablup/mlxcel/compare/v0.0.20...v0.0.21
[v0.0.20]: https://github.com/lablup/mlxcel/compare/v0.0.19...v0.0.20
[v0.0.19]: https://github.com/lablup/mlxcel/compare/v0.0.18...v0.0.19
[v0.0.18]: https://github.com/lablup/mlxcel/compare/v0.0.17...v0.0.18
[v0.0.17]: https://github.com/lablup/mlxcel/compare/v0.0.16...v0.0.17
[v0.0.16]: https://github.com/lablup/mlxcel/compare/v0.0.15...v0.0.16
[v0.0.15]: https://github.com/lablup/mlxcel/compare/v0.0.14...v0.0.15
[v0.0.14]: https://github.com/lablup/mlxcel/compare/v0.0.13...v0.0.14
[v0.0.13]: https://github.com/lablup/mlxcel/compare/v0.0.12...v0.0.13
[v0.0.12]: https://github.com/lablup/mlxcel/compare/v0.0.11...v0.0.12
[v0.0.11]: https://github.com/lablup/mlxcel/compare/v0.0.10...v0.0.11
[v0.0.10]: https://github.com/lablup/mlxcel/compare/v0.0.9...v0.0.10
[v0.0.9]: https://github.com/lablup/mlxcel/compare/v0.0.8...v0.0.9
[v0.0.8]: https://github.com/lablup/mlxcel/compare/v0.0.7...v0.0.8
[v0.0.7]: https://github.com/lablup/mlxcel/compare/v0.0.6...v0.0.7
[v0.0.6]: https://github.com/lablup/mlxcel/compare/v0.0.5...v0.0.6
[v0.0.5]: https://github.com/lablup/mlxcel/compare/v0.0.4...v0.0.5
[v0.0.4]: https://github.com/lablup/mlxcel/compare/v0.0.3...v0.0.4
[v0.0.3]: https://github.com/lablup/mlxcel/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/lablup/mlxcel/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/lablup/mlxcel/releases/tag/v0.0.1
