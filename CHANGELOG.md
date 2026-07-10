# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **ComputeBackend seam for the forward-execution engine.** A `ComputeBackend` abstraction at the model-load boundary lets a future non-MLX engine host `LanguageModel::forward` without routing through the MLX bridge. The existing MLX path moves behind `MlxBackend` as a behavior-preserving refactor (temp-0 output is byte-identical before and after). Under default features the selection folds to the single MLX backend at compile time with no runtime dispatch on the hot path; a default-off `experimental-backend` feature reserves the plug-in slot. No non-MLX kernels are implemented (#338).

### Changed
- **Serving-throughput defaults enable the batching machinery out of the box.** `mlxcel-server` and `mlxcel serve` now default to `--parallel 4` (batched decode of up to 4 concurrent sequences, clamped to 1 for SSM / hybrid / mixed-cache families that cannot batch) and `--max-batch-prefill 4` (batched prefill for families that support it), and add `--no-prompt-cache` as a clean opt-out for the already-default-on prompt-prefix cache. The batched-decode default is paired with a default `--kv-cache-budget auto` memory guard so the #122 paged block-budget admission bounds KV for the concurrent batch and returns backpressure instead of an OOM abort; the guard is inert on the dense decode backend and can be disabled with `--kv-cache-budget none` (or `0`). On Apple M1 Ultra (`meta-llama-3.1-8b-instruct-4bit`), 4 concurrent clients get 1.90x the single-client aggregate throughput and ~17x lower time-to-first-token under load, with single-client throughput unchanged. `--parallel 1`, `--no-batch`, `--max-batch-prefill 1`, `--no-prompt-cache`, and `--kv-cache-budget none` restore the previous single-client behavior. Migration note: an explicit small `--ctx-size` is now divided across 4 slots, so a value that gives fewer than 512 tokens per slot fails startup with a clear error (raise `--ctx-size` or lower `--parallel`); `--ctx-size 0`, the default, uses the model window per slot and is unaffected. GB10 numbers (including the >= 2.5x-at-4-clients target) are pending a CUDA measurement session; see `docs/benchmark_results/serving-throughput-defaults-m1u-2026-07-09.md` (#628).

### Performance
- **Cap the batched-prefill transient memory.** With `--max-batch-prefill 4` now the default (#628), the server's padded batched-prefill path engaged out of the box for `supports_batched_prefill()` families and, for mixed-length prompts, ran a single unchunked `[B, padded_len]` forward that materialized a stacked `[B, L, L]` FP32 attention mask, an `O(B*L^2)` transient that ignored `--prefill-chunk-size` (four concurrent 8k prompts built a ~1 GiB mask and could abort the server on OOM, an availability edge the `--kv-cache-budget` guard does not model). The drained batched window is now bounded by total padded tokens via the new `--max-batch-prefill-tokens` flag (and `MLXCEL_MAX_BATCH_PREFILL_TOKENS`): a cohort of `B >= 2` rows padded to `L` keeps `B*L` within the budget, so the mask stays within `~2*budget^2` bytes; rows past the budget spill to the next tick and prefill via the chunked single-sequence path, and a head prompt too long to batch skips the batched path entirely. The default budget is derived (`2 * max_batch_prefill * prefill_chunk_size`, the shipped `2 * 4 * 512 = 4096`; the 2x headroom keeps a full batch of slightly-over-chunk-sized prompts in one window), bounding the FP32 mask to about 34 MiB while keeping short-prompt concurrency batching unchanged; `0` disables the cap for the pre-#715 unbounded behavior (#715).

### Fixed
- **Interrupted model downloads are detected and re-fetched instead of failing at load.** An interrupted `mlxcel run <repo-id>` (also `mlxcel serve` and `mlxcel-server` auto-download) previously reused the partial snapshot and died with a bare `Weight not found`. The load/resolve path now verifies the full weight set against the snapshot's own `model.safetensors.index.json` (every shard present and non-zero), not just `config.json` presence, so a partial snapshot is resumed through the shared downloader (re-fetching only the missing files, with a forced clean re-download as a fallback) before the model loads. Repackaged mlx-community quants whose stale full-precision index no longer matches the on-disk files are still reused without a re-fetch (#465).

### Docs
- Finalize the per-backend `MLXCEL_FUSED_QK_NORM` default decision: CUDA (GB10) was measured and is also slower than the graph path, so the fused QK-norm decode path stays opt-in (default off) on every backend; `docs/environment-variables.md` updated to drop the CUDA-pending rationale and record the determinism nuance (#355).

### Chore
- Bump the `mlxcel-core` and `mlxcel-surgery` member crates to Rust edition 2024 and align their versions to the root crate at 0.3.3, per the release-versioning rule (#272).

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
- Reword the v0.0.31 #86 entry in CHANGELOG.md and `debian/changelog` to match the GitHub release note. The fix landed at the batch scheduler level: a burst of concurrent VLM requests overwrote a single shared `per_layer_inputs` cell before prefill consumed it, reading the wrong sequence's tensor. The earlier wording described the symptom as a per-layer input shape issue, which understated the concurrency cause.

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
