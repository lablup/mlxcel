# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`-m/--model` accepts `--revision <REV>`** on `generate`, `run`, `serve`, `inspect` and `mlxcel-server` (#1113), matching `mlxcel download --revision`. Previously a pinned revision could be fetched but not then run by repo-id: the resolver always resolved against `main`. The flag is honoured only where it can be honoured correctly, which is a deliberate limit rather than an omission. The HuggingFace cache probe is revision-aware and answers normally, and a miss fetches the requested revision. The legacy `./models/<name>` directory and the mlxcel store are keyed on `<owner>/<name>` with no revision component, so for a revision-qualified request they are skipped rather than allowed to answer with an unknown revision, and a request whose store directory is already occupied is **refused with an explanation** instead of being silently answered with whatever is on disk. That last case is not hypothetical: the downloader treats same-named non-zero files as "already present" and skips the fetch, which is also why `mlxcel download --revision` can silently return the wrong revision today. Use `--models-dir` to give each revision its own root. `--revision` alongside an existing local path is an error, since a local directory is used exactly as given. Revision-namespacing the store would lift these restrictions but changes an on-disk layout shared with `list`, `rm` and `download`, so it is left as follow-up work.
- **Checkpoints that quantize `q_proj` / `k_proj` / `v_proj` at different bit widths now load on every family that shares the fused QKV projection** (#1090). `mlx_lm`'s `mixed_4_8` predicate raises selected tensors to 8 bits while the rest of a model stays at 4, and the loader concatenated the three packed planes along one axis and inferred a single width from `q_proj`, so such a layer died inside MLX's `concatenate` instead of loading. A layer whose planes cannot be concatenated now keeps them separate, each in exactly the layout the checkpoint stored it in: nothing is dequantized and nothing is requantized, so the values are the checkpoint's and the extra memory cost is zero. This reached all 16 families using this loader, including Llama, Mistral, Qwen2/3, Gemma v1 through v4, Cohere2, StarCoder2, InternLM3 and Jamba. Validated on a `mixed_4_8` Llama-3.2-1B checkpoint, which greedy-decodes byte-identically to the uniform 4-bit baseline. The decision is made on the packed shapes, which is what `concatenate` actually constrains, rather than on the reconciled bit width and group size, which can alias two different packings onto one pair.

### Changed

- **LocateAnything no longer dequantizes its mixed-precision attention layers at load** (#1090). The per-family workaround added in #1070 turned 18 of the released 3B checkpoint's 36 layers' q/k/v planes into dense bf16, about 190 MB, so the fused projection could concatenate them. Those planes now stay packed and the model loads through the shared path. Helium's pre-flight weight validator no longer rejects a `mixed_4_8` attention block either; it stopped comparing the packed width across q/k/v, which scales with the bit width, and still checks each plane's logical input width against `hidden_size`.

### Fixed

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
