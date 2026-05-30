# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **BREAKING: `mlxcel list` now lists local downloaded models by default; the supported-architecture catalog moved to the new `mlxcel arch` verb.** Previously bare `mlxcel list` printed the architecture catalog and the local store inventory was gated behind `mlxcel list --local`, which inverted the bare-verb convention of every comparable tool (`ollama list`, `docker images`, `pip list`, `brew list` all show the *local* inventory with no flag) and contradicted mlxcel's own store-centric `run` / `download` / `rm` verbs. Now `mlxcel list` (and its `ls` alias) enumerates downloaded models from the global store with repo-id, on-disk size, and path, mirroring `ollama list`; `mlxcel list --models-dir <PATH>` applies to that listing. The catalog is reachable via `mlxcel arch` (alias `mlxcel supported`), byte-identical to the prior bare-`list` output. The `mlxcel list --local` flag is **removed** outright — clap rejects `--local` as an unknown argument, and the empty-store hint now points users to `mlxcel arch` for the catalog. This is pre-1.0 (v0.1.x) with no deprecation cycle because `--local` had not seen real-world use, so carrying a hidden flag plus a deprecation shim was not warranted. Migration: use `mlxcel arch` for the catalog, and drop `--local` from any `mlxcel list --local` invocation (the bare form now does the same thing) (#138).

### Fixed
- **Base-model warning no longer presents `-it` as a universal instruction-tuned naming convention.** The warning `mlxcel run` / `mlxcel generate` prints when a model ships no chat template (added in PR #134) recommended trying a variant "named with an `-it` suffix" — but `-it` is the Gemma convention. For other families the advice was wrong: Llama and Qwen2.5 instruction-tuned checkpoints use `-Instruct`, and Qwen3 / Qwen3.5 use the plain repo name (with `-Base` marking the non-instruct variant), so a user running `Qwen3.5-0.8B-Base` was pointed at a non-existent `-it` repo instead of being told to drop `-Base`. The advice now names the per-family conventions (Gemma `-it`; Llama / Qwen2.5 `-Instruct`; Qwen3 / Qwen3.5 plain name vs. `-Base`). Base-model detection is unchanged — it keys on chat-template absence, never on the model name.

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
- 2026-05-28 M1 Ultra full sweep mirrored into the public benchmark docs (mlxcel 0.1.0; `mlx-lm` / `mlx-vlm` baselines unchanged). README M1 Ultra column refreshed across both representative decode tables (Phi-3.5-vision M5 VLM row corrected to its post-#743 value, 169 tok/s, 106%); benchmark report headline updated to text 74 pairs / 99% median / 64 of 74 at >=90%, VLM 18 pairs / 98% median / 12 of 18 at >=90%; `model_tests_m1ultra.md` per-model refresh with aggregate blocks (text avg 97% / median 99%; VLM avg 101% / median 98%) and `internvl3` now passing on M1. The cross-hardware quick table stays a labeled 2026-05-19 same-version snapshot (#130).
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
- **End-to-end speculative decoding for Gemma 4 MTP and Qwen 3.5 DFlash drafter families** (epic #633, #632). New `Drafter` trait + `DrafterKind` enum + `model_type` auto-detection (#624). Ported drafter components: `MaskedEmbedder` for Gemma 4 E2B / E4B (#627), drafter masks (bidirectional full + sliding-window) and `normalize_batched_shared_kv_states` helper (#628), `Gemma4AssistantDraftModel` (4-layer drafter + pre/post projections) (#626), and `DFlashDraftModel` (5-layer drafter + `DFlashAttention` + `DFlashKVCache`) (#635). Target-side hooks: Gemma 4 `return_hidden` / `return_shared_kv` / `rollback_speculative_cache` for MTP (#625); Qwen 3.5 `return_hidden` + `capture_layer_ids` + GDN-aware `rollback_speculative_cache` for DFlash (#634). Round loops: DFlash single-batch (#636), `MtpGenerator` single-batch (#629), batched DFlash with continuous batching + GDN-aware rollback (#637), and batched MTP with continuous batching + left-padding normalization (#631). Greedy-parity + perf benchmark scaffolding under `src/bin/speculative_bench.rs` (#632) and real-model byte-equality end-to-end tests (#685).
- **Server speculative dispatch.** Speculative dispatch resolution and `MtpTarget` adapters wired into the inference server (#666); the assistant model paths now plug into the real `MaskedEmbedder` and `make_drafter_masks`; speculative dispatch is wired into the scheduler via per-request B=1 bursts (#670) and a B>1 batched path via `MtpBatchedGenerator` / `DFlashBatchedGenerator` (#684). Per-request properties propagated through the speculative-burst path: cancellation propagation through `MtpGenerator` / `DFlashGenerator` (#681), `token_history` threading through the speculative-burst first sample (#682), logprobs support (#686), thinking-budget enforcement (#687), and prompt-cache donate symmetric with the classic path (#680) and into the B>1 batched arm (#689).
- **CLI:** `--draft-kind {dflash,mtp}` and `--draft-block-size` flags on both `mlxcel` and `mlxcel-server` (#630).
- **OpenAI Responses API (Phase 1)** at `/v1/responses` for both binaries (#622, #623). New modules `responses_store`, `responses_translator`, `conversation_store`, `streaming_responses`, `routes/responses`, and request/response/stream type modules under `types/responses_*`. Implements conversation store with shared-LRU semantics, `response.created` / `response.in_progress` / `response.completed` SSE event stream, reasoning-trace forwarding, response cancellation, and four new CLI flags. User guide at `docs/responses-api.md`.
- **APC block-level partial cache adoption in the scheduler** (issue #580). When Automatic Prefix Caching is enabled, a request whose prompt shares the first N blocks with a cached entry but diverges at block N+1 now reuses blocks 0..N and re-prefills only from the divergence boundary — rather than cold-prefilling the entire prompt. Three components: `DetachedKVCache::trim_to` and `DetachedCacheSet::truncate_to` in `mlxcel-core` perform per-layer KV tensor slicing on the detached handle (mirroring `KVCache::trim` semantics, covering FP16/INT8/Turbo4/Turbo4Delegated sidecars); the `PromptCacheStore` lookup relaxes the legacy "stored prefix must be fully contained in request" gate when APC is on, routing the actual common-prefix depth through the existing `apc_consistent_prefix_len` block-hash discriminator; and `Scheduler::try_adopt_cached_prefix` calls `truncate_to(matched_len)` before adoption when the lookup returns a sub-entry-length match. APC-off retains the pre-#580 behaviour bit-exactly. Wall-clock bench procedure on Apple Silicon documented in `docs/apc-partial-adoption-bench.md` (#580, #607).
- **Nemotron H Nano Omni audio modality** (issue #582). Ports the Parakeet/Conformer sound encoder (`NemotronOmniSoundEncoder`), mel-spectrogram feature extractor (`NemotronOmniFeatureExtractor`), and audio projector (`NemotronOmniSoundProjection`) from the upstream `mlx-vlm` Python reference. The encoder implements depthwise/pointwise Conv2D subsampling, Transformer-XL relative positional encoding, multi-head self-attention with per-head u/v bias terms, a GLU+BatchNorm convolution module, and half-weight feed-forward blocks. Audio weights are loaded conditionally when `sound_config` is present in `config.json`; the loader applies the upstream `sanitize_audio_weights` transpose pass before population. The VLM runtime path (`generate_vlm`) accepts `--audio <wav>`, runs the feature extractor and encoder, merges the resulting token embeddings at `sound_context_token_id` slots, and interleaves them with vision tokens when both modalities are present. Bring-up procedure for Apple Silicon engineers documented in `docs/nemotron-h-nano-omni-audio-bringup.md` (#582, #609).
- **`mlxcel download` / `mlxcel-server download` progress bars** (#648, #649). New `src/downloader/progress.rs` module provides terminal-aware suppression (`should_show_progress`), a `MultiProgress` factory, and 6 suppression unit tests. The downloader streams files via `reqwest` to a `NamedTempFile` and atomically renames into place; outer `stream_file` and inner `stream_to_tempfile` are split so the progress bar covers the network read and the rename is observable.
- **Server `--max-kv-size` flag** matching llama-server, plus a tightened chat-completion response envelope (#618).
- **Tokenizer support for multi-token think and tool-call sequences** so chat templates that emit `<think>` / `<tool_call>` across multiple BPE tokens stream and parse correctly (#590, #613).

### Changed
- **`StreamFilter` extended** to handle multi-token markers and reset state when a partial marker is broken by a non-marker token (#613).
- **Speculative drafter epic follow-ups hardened** post-merge — covers misc invariants surfaced by integration testing against the real `z-lab/Qwen3.5-4B-DFlash` checkpoint and the `mlx-community/gemma-4-*` drafter variants.
- **README and speculative decoding guidance refreshed** to match the current code paths and the latest M1 Ultra / M5 Max benchmarks (#700).

### Performance
- **Qwen 3.5 DFlash greedy-argmax decode-path optimization** that drops the per-decode-step copy and an unnecessary argmax temporary, restoring decode tok/s on Qwen 3.5 32B / 9B DFlash configurations.
- **Avoid slow Gemma 4 MTP singleton bursts** — the speculative-burst path now correctly short-circuits to the classic path when the batch size collapses to 1 with no draft tokens accepted, eliminating a per-step over-evaluation regression introduced by the initial dispatch wiring (#698).

### Fixed
- **DFlash drafter lazy-bind** for the upstream `z-lab/Qwen3.5-4B-DFlash` checkpoint — `Drafter::bind` was previously not called on the DFlash family, causing an internal cache mis-binding on the first speculative burst. The drafter now performs lazy-bind on first use, matching the MTP path (#683).
- **Enable DFlash for Qwen 3.5 VLM text requests** — pure-text generations against a Qwen 3.5 VLM checkpoint can now resolve a DFlash drafter when one is available, instead of silently falling back to the classic path (#694).
- **Speculative-rollback safety:** validate trimmable cache and reserve the last token in prefill so a rolled-back speculative burst always lands on a valid sampling boundary (#612).
- **Prompt cache RadixTrie:** `pop_prefixes` now uses correct immediate-prefix semantics (#617).
- **MiniMax M2 parallel tool calling parser** correctly emits one `ChatToolCall` per parallel call instead of merging them into a single call (#616).
- **Server tool-call buffering:** preserve token positions when buffering parallel tool calls (#615); skip the tool→normal transition when `tool_call_end` is empty so streaming continues correctly for templates without an explicit close marker (#614).
- **`video_url` allowlist TOCTOU race** closed by passing the resolved `OwnedFd` to ffmpeg via `/dev/fd/N` instead of re-opening the path inside the subprocess. Symlink swaps between `metadata` and the subsequent open now cannot mis-route the subprocess. Audit hardening from #601 (#611).
- **Gemma 4:** skip `k_proj` / `v_proj` / `k_norm` weight load for KV-shared layers — the previous load step would error out on real Gemma 4 E2B / E4B checkpoints that omit these tensors per KV-shared design (#608).
- **Nemotron-H:** default `time_step_limit` to `(0.0, +inf)` regardless of `time_step_min` / `time_step_max` to match the upstream mlx-lm `time_step_limit` behaviour even when only one of the bounds is supplied (#619).
- **`gated_delta` masked Metal kernel variants:** zero-init `y[dv_idx]` when the mask is false (#610).
- Tests: add `max_kv_size` field to the `ServeArgs` test fixture (#620).
- Address upstream-sync review follow-ups carried over from the v0.0.26 sync cycle.

### Security
- **Downloader security hardening** post-#649 (#650, #652): plaintext `HF_ENDPOINT` + token warning (M1), client connect / read timeouts (M2), URL path percent-encoding for `?` / `#` / `%` in filenames (L1), `O_NOFOLLOW` tempfile creation (L2), `Result`-based token-ASCII handling instead of `expect("token must be ASCII")` (L3), stale `.mlxcel-partial.*` cleanup at download start (L5), and parallel HEAD requests bounded by `buffer_unordered(8)` (L6).
- **Closed the residual TOCTOU window in the `video_url` allowlist resolver** (issue #601, PR #611). The dominant canonicalise → ffmpeg-open race was already closed by passing an `OwnedFd` to ffmpeg via `/dev/fd/N`. This change hardens the narrowed metadata→open gap by opening every allowlisted video file with `O_NOFOLLOW`: a symlink swap that occurs between the `metadata` call and the `open` syscall now returns `ELOOP` instead of silently following the swapped-in link. Subprocesses continue to receive `/dev/fd/N` (never the path), so they cannot be misdirected post-open regardless. A startup warning is now also emitted on non-Unix targets when `MLXCEL_VIDEO_DIR_ALLOWLIST` is set, because `O_NOFOLLOW` and fd-passing are unavailable on those platforms.

### CI
- Bump GitHub Actions to Node 24 runtime to clear the Node 20 deprecation warning surfaced on the macOS runners.

### Chore
- Replace `.map_or(false, ...)` with `.is_some_and(...)` in tokenizer call sites (clippy 1.93 lint clean-up) (#651).

## [v0.0.26] - 2026-05-10

### Added
- **TurboQuant KV cache.** New 3–4 bit KV cache compression family built on a Walsh–Hadamard transform op (#470) and a Lloyd-Max PolarQuant codebook generator ported to Rust (#472). Four KV cache modes wired through `KVCacheMode`: `Turbo4` symmetric with per-model allowlist (#476), `Turbo4Asym` Fp16-K + Turbo4-V (#474), `Turbo3Asym` 3-bit Fp16-K + Turbo3-V (#477), and `Turbo4Delegated` with a FP16 hot tail + packed turbo cold body (#479). TurboQuant + `RotatingKVCache` integration covers sliding-window attention (B9). Sparse-V dequant scaffolding (#480), Boundary-V layer protection that keeps the first/last layer at FP16 (#478), and a packed-aware `PagedKvLayout` (#482) round out the runtime. Quality gates: wikitext-2 PPL + NIAH harness (#475), full 283K-token test split fixture (#492), per-model PPL/NIAH results committed (#493), and VLM B3 quality gates with image-token kurtosis (#510). Speed gate matrix runner with M1 Ultra (#509) and M5 Max readings. User guide and validated config matrix published (#485).
- **Server flag parity for KV quantization.** `--cache-type-k` and `--cache-type-v` flags accept `f16`, `q8_0`, `q4_0`, etc. matching llama-server semantics, with TurboQuant modes exposed as `mlxcel_turbo*` variants (#484). KV cache quantization extended to continuous batching (#545) and a unified `--kv-cache-mode` flag layout shared across `mlxcel`, `mlxcel-server`, and `mlxcel download` (#567).
- **Automatic Prefix Caching (APC) with hash blocks** (#552). Hash-keyed block-table prefix reuse on top of v0.0.25's cross-sequence prompt-prefix KV cache, enabling shared physical blocks across requests with the same hashed prefix without per-request token-prefix matching cost.
- **OpenAI-compatible `response_format: {"type": "json_schema", ...}`** structured-output support for `/v1/chat/completions` and `/v1/completions` (#550). Constrained decoding via `llguidance` (the same backend used by upstream mlx-vlm PR #1047) ensures every emitted token keeps the partial output conforming to the supplied schema. Per-request schema validation enforces a 64 KiB size cap, a 32-level nesting depth limit, and a 64-entry `$ref` count limit so an adversarial schema cannot exhaust CPU or memory during grammar compilation. The tokenizer environment is cached by SHA-256 fingerprint so consecutive requests to the same model share the build cost (~1–2 s for a 150k-vocab tokenizer). Reusable per-sequence `mask_buf` and `bias_buf` allocations eliminate per-token `Vec` allocation on the hot decode path. The legacy `json_object` mode is rejected with a clean 400 in this MVP; `json_schema` with a well-formed schema is the supported path. Supported on HuggingFace BPE tokenizers; SentencePiece and Tiktoken backends return a clean `UnsupportedTokenizer` error. Verbose llguidance internals are never surfaced in public error messages — they are routed to server-side tracing only.
- **`mlxcel download` / `mlxcel-server download` subcommand** to fetch HuggingFace model repository snapshots without Python tooling (#457, #486). Uses `hf-hub` with an allow-list file filter (SafeTensors, tokenizer, and config files only), cache-hit detection, and formatted per-file progress output. Supports `--local-dir`, `--revision`, `--token`, and `--force`. Default destination mirrors the `models/<repo-basename>` convention from AGENTS.md.
- **`/health` endpoint** now includes `context_size` (the configured `--ctx-size` value; `0` means model default) and `tool_call_parser` (`"mlxcel"` when the chat template exposes the `tools` variable, `null` otherwise) (#549, #572). Both fields are present once a model is loaded; `context_size` is absent while loading, `tool_call_parser` serializes as `null` during startup so monitoring clients can distinguish "template has no tool support" from "model not yet loaded". The tool-support heuristic is extracted into a shared `template_mentions_tools()` helper used by both the health route and the existing `compute_supports_tools` fallback path.
- **Paged scheduler dispatch on `PagedKvLayout::cache_mode`** so the scheduler routes batches into the matching paged decode kernel for each KV cache mode (#508).
- **Video input infrastructure for VLMs.** Gemma 4 video support with the new VLM video input pipeline (#553); ffmpeg-backed frame extraction with single-pass extraction and a `Drop` guard for cleanup of temporary frame files (#597); `video_url` content blocks wired through `/v1/chat/completions` (#596); content-preservation tests covering the frame extraction path (#598).
- **New models.** Youtu-VL vision-language model (#555); Nemotron H Nano Omni vision (#554) plus follow-up correctness/validation hardening (#595).
- Multi-task M1 Ultra benchmark refresh to 2026-05-08 (#577) and full M1 Ultra column resync in `benchmarks-by-hardware.md` (#578).

### Changed
- **MLX upstream pin bumped twice.** First from the v0.0.25 baseline (`5d7e96cd`) to v0.32.0 / `c9aa5605` (#565), then forward to `84961223` covering 3 PRs: #3443 splits the CUDA `qmm_naive` / `qmm_sm80` kernel bodies into new `qmm_naive.cuh` / `qmm_sm80.cuh` headers without changing the public ABI consumed by mlxcel's `patches/mlx/backend/cuda/quantized/qmm/qmm.h`; #3463 routes the CPU JIT preamble through `JitCompiler::get_preamble()` and renames the prebuilt symbol from `get_kernel_preamble` to `get_prebuilt_preamble` (mlxcel does not call either directly); #3475 fixes contiguity-flag accuracy in `AsStrided` by computing `data_size` from the actually-occupied stride range. Three-location pin update applied to `src/lib/mlx-cpp/CMakeLists.txt`, `src/lib/mlxcel-core/build.rs`, and `.github/workflows/release.yml` per `CLAUDE.md`. Fused Metal kernel launchers in `src/lib/mlx-cpp/turbo/` re-validated against both bumps: `mlx::core::fast::metal_kernel`, `mlx::core::full`, `mlx::core::Shape`, `mlx::core::float32`, `mlx::core::int32`, and `metal::fast::exp` symbols unchanged.
- **Refactor:** unified TurboQuant KV-cache CLI flags across `mlxcel`, `mlxcel-server`, and `mlxcel download` so all binaries accept the same `--kv-cache-mode` / `--cache-type-{k,v}` syntax (#567).
- mlx-lm version reference in docs bumped from 0.31.2 to 0.31.3 (#606). The `bridge-overhead-microbench` reference at v0.31.2 is preserved because it pins the MLX C++ runtime, not the mlx-lm Python package.

### Performance
- **Sparse-V kernel:** fused per-thread Metal kernel that skips the full SDPA pass when sparse-V dequant predicts zero contribution (#505); precomputed kernel rescale to drop per-token threadgroup barriers (#520).
- **Turbo4Delegated decode hot path:** unified K storage to drop the per-step K concat (#527); cold-V dequant cache across decode steps (#525) followed by a cold-V dequant Metal kernel that retires the FP16 memo (#530); steel-attention-envelope fused SDPA kernel (#531) with parallelized Pass 1 softmax (#534); delegated FP16 predecode compaction (#536) and lazy delegated FP16 sidecars (#537); compressed fold moved before decode.
- **Compressed dequant-SDPA paths** for TurboQuant decode (#562).
- **Server hot-path:** thread-local generation stream and uniform-batch RoPE collapse to remove per-request allocation in the steady-state batching loop (#556).

### Fixed
- **TurboQuant continuous batching:** correct batch cache offset merging when batches with different cache offsets are joined or split (#564); Turbo3 split-flag, documentation alignment, and an `ENV_LOCK` race in concurrent process startup (#573).
- **Vision / VLM mixed batching:** per-sequence MRoPE alignment for mixed VL+text batches (#558); per-sequence `per_layer_inputs` for Gemma 4 E2B/E4B VLM (#561); mixed-length batching support for Gemma 4 (#560); relaxed cached-position shape check in Qwen VL chunked prefill (#557); Qwen3.5-MoE batch-size validation on cached `position_ids` reuse (#559).
- **Streaming and sampling:** correct streamed detokenization for byte-fallback tokens that previously leaked raw byte fragments to the client (#570); top-p filter correctness for batched logits (#569); token queue timeout handling during long prefills so clients no longer see spurious 408s on slow first-token paths (#571); `StreamFilter` extended to cover Hermes-style `<tool_call>` / `</tool_call>` and Mistral Nemo `[TOOL_CALLS]` markers, which previously leaked raw markup into `delta.content` during streaming (#551, #576). Partial-marker buffering at token boundaries correctly holds back prefixes (e.g. `<tool_`) until the full tag can be confirmed, then releases them to `delta.content` if they turn out not to be a boundary. Gemma 4 `<|tool_call>` suppression is unaffected; the delimiter table ordering ensures the Gemma 4 pipe-delimited form wins the tiebreak over the Hermes plain form.
- **Models:** Gemma3-4B attention SIGABRT from a sliding-window mask `T_k` mismatch on long-context prompts (#507); preserve Qwen2 fused QKV bias when it is present in the checkpoint (#517); test fixture swap to Qwen2.5-1.5B base variant for the B3 quality gate (#506); harden post-merge review findings on the Nemotron-H Nano Omni vision PR.

### Security
- Path-traversal defense in the downloader: `is_safe_relative_path` pre-filters each sibling filename returned by the HuggingFace API (rejects absolute paths, `..` components, backslash separators, and empty components). A secondary canonicalized `starts_with` guard on the resolved destination path is applied before writing each file. Download target files are written to a temporary path and atomically renamed into place, preventing partial writes from leaving corrupt files in the output directory (fixes C1 and H1 from security review of #457).
- Structured-output schema limits (64 KiB serialized size, 32 nesting depth, 64 `$ref` count) and tightened `llguidance` parser caps (`max_grammar_size: 100 000`, `max_lexer_states: 50 000`) applied before grammar compilation so an adversarial client cannot use the schema endpoint as a CPU/memory exhaustion vector. Schema content is never echoed in public error messages (#550).

## [v0.0.25] - 2026-04-24

### Added
- Cross-sequence prompt-prefix KV cache. New `KVCache::trim/detach/adopt` API enables adopting a previously-cached prefix on the next request. Backed by `PromptCacheStore`, an in-process LRU keyed by tokenized prompt prefix, plus a longest common token-prefix matcher (`PrefixMatcher`) for fast lookup. Paged KV cache gains block-table prefix reuse so adopted prefixes share physical blocks. Scheduler integration prefills only the unmatched suffix on cache hits. Wired into the server via `--prompt-cache-size`, `--prompt-cache-min-tokens`, and matching `LLAMA_ARG_*` env vars; multimodal/vision-aware cache key (`MultimodalDigest`) prevents cross-modality collisions. OpenAI-compatible `cached_tokens` is reported in `/v1/chat/completions` responses, mirrored to Prometheus counters, and verified by a multi-turn E2E test plus a prefill-latency benchmark. Design rationale and operator guide added to docs (#418, #419, #420, #421, #422, #423, #424, #425, #426, #427, #428, #429, #430, #431, #432, #433, #434, #435, #436, #437, #438).
- Language-bias steering (Axis B, Phase 1). New `lang_analyzer` module with a Unicode script classifier (B2, #391) and `TokenLanguageIndex` builder that scans the tokenizer vocabulary, partitions tokens by script, and persists the result to disk for fast warm starts (B3 #392, B4 #394). Sampling primitive `TokenBiasMap` + `apply_token_bias` (#390) is wired through `LangBiasSet` with `Conservative` / `Strict` policies (B5 #393), exposed via CLI flags and a YAML config (B6 #395), `LLAMA_ARG_LANG_BIAS` env var in `mlxcel-server` (B7 #397), `LangBiasConfig` injection into the generator pipelines (B8 #398), tracing fields and Prometheus counters (B9 #400), byte-fragment CJK classification via UTF-8 start-byte analysis so byte-level BPE tokenizers correctly attribute fragments (#408, closes #405), byte-level reverse map for token decoding (#402, fixes #401), and integration tests for the steering matrix (B10 #399, #407). User guide and Quickstart published (B11 #396, #404).
- `thinking_token_budget` sampling parameter for the Qwen3 family — caps tokens emitted between `<think>` / `</think>` markers without disabling streaming (#411).
- `preserve_thinking` chat-template hook for Qwen 3.6 so multi-turn conversations retain prior `<think>` blocks instead of stripping them on subsequent turns (#412).
- `StreamFilter` extended to recognize Qwen-style `<think>` / `</think>` token boundaries during streaming and route the segment into `reasoning_content` (#445).
- `thinking_budget_tokens` extended to Gemma 4 (#442).
- `feat(benchmarks)`: bridge-overhead microbench tool measuring per-op cost of the Rust cxx bridge against Python nanobind across MLX primitives, with a published baseline and reproduction steps (#450).
- `feat(ci)`: multi-stage pipeline-parallel smoke job activated using a Qwen3-0.6B fixture so PR runs catch PP regressions (#415).
- Per-layer + per-sub-op decode profiling for Gemma 4, plus a Gemma 4 perf harness with the 2026-04-22 baseline used to drive the parity work below.

### Fixed
- Prompt cache prefix isolation — sequences whose prompts share a non-trivial prefix no longer leak adopted KV state across each other after detach/adopt (#448).
- `MultimodalDigest` propagated to all `PromptCacheKey` callers after the issue #421 + #425 merge so vision-aware cache lookups stay collision-free.
- Gemma 4 `enable_thinking=false` no longer triggers degenerate output, and `reasoning_content` now streams correctly when `enable_thinking=true` (#440).
- Tool-only assistant turns now emit `content: null` instead of `""` to match the OpenAI Chat Completions schema (#441).
- `chat-template`: support flattened `extra_body` and pseudo-user tool responses so OpenAI-style tool flows render correctly under HF-style templates (#413).
- `lang-analyzer`: decode tokens using the byte-level reverse map (#401) instead of the textual tokenizer view, so byte-level BPE (Qwen, Llama) tokens are classified by their actual code-point payload (#402).
- `ci`: unblock Pipeline Parallel CI on Ubuntu by installing LAPACK and treating clippy `-D warnings` consistently (#414).
- `vision`: read Gemma 4 encoder `hidden_size` from after `input_proj` so the multimodal projector wires the correct dimension on encoders that include a learned input projection.
- Bumped `cc` to 1.2.60 to silence the BSD `ar` probe warning surfaced by recent `cc-rs` releases (#439).

### Changed
- Gemma 4 mlx-lm decode parity pass (closes the remaining gap on 26B / 31B / e2b, #454):
  - Router RMS norm fused with top-k-then-softmax to remove a separate normalization pass (#451).
  - SwitchGeGLU gate / up / geglu / down fused into a single `mlx::core::compile` window (#452).
  - Metal-trace-driven attention / RoPE / per-layer chain fusion (#453).
  - Compiled Gemma 4 SwitchGeGLU decode path enabled.
  - Single-query causal masks skipped in decode.
  - BF16 decode graph aligned with mlx-lm.
  - Proportional RoPE aligned with mlx-lm (no rotated-only normalization).
  - SwitchGLU projection order matched to mlx-lm.
  - QKV projection shape matched to mlx-lm.
  - Router top-k aligned with mlx-lm.
  - Load and MoE decode paths tuned.
  - Redundant residual copies in the decoder layer dropped.
  - SwitchGeGLU `expand_dims` collapsed and a MoE inner profiler added (#447).
- `Qwen 3.5`: SSM decode masks aligned with mlx-lm; benchmark artifacts cleanup (#455).
- `MLX`: upstream pin upgraded to **v0.31.2**; in-tree SDPA and steel-attention overlays dropped now that upstream covers them. Three-location update (`src/lib/mlx-cpp/CMakeLists.txt`, `src/lib/mlxcel-core/build.rs`, `.github/workflows/release.yml`) per CLAUDE.md (#449).
- `CUDA`: QMM patches updated for the new upstream `lhs/rhs_indices` signatures (#456).
- `deploy`: SIGTERM the running `mlxcel-server` after binary copy so the respawned supervisor picks up the new binary (#443).
- `style`: `cargo fmt` swept across server modules to land previously-unformatted blocks.

## [v0.0.24] - 2026-04-18

### Added
- Zero-config multi-machine pipeline-parallel bring-up: `mlxcel-server --pp-auto N` declares pipeline depth; peers register via `--cluster-peers` seeds or opt-in mDNS discovery (`--cluster-discovery=mdns`). New `src/distributed/cluster_init.rs` owns deterministic stage assignment, port allocation, and byte-identical TOML emission consumed by the existing manual-TOML runtime path (#342, #352).
- RDMA-aware transport backend with transparent TCP fallback. Negotiates `io_uring` registered buffers on Linux and `kqueue` batched send on macOS, emits exactly one structured log line on fallback, and preserves the `Arc<dyn Transport>` abstraction used by `activation_transfer.rs`. New `rdma_capabilities.rs`, `rdma_transport.rs`, and `bench_activation.rs` harness (#351, closes #343).
- 2D `(pp_stage, tp_rank)` mesh composing PP with TP for Llama-70B-class topologies. Adds `NodeRole::PipelineTensorParallel`, validation for exact `pp_size × tp_size` coverage with unique coordinates, registry helpers, `TrafficClass` routing (`TpCollective` / `PpActivation`), and grid-coherent KV admission (`coordinated_2d_admission`) (#354, addresses #346).
- Byte-accurate pipeline auto-partition with adjacency constraints. `ModelProfile` gains per-layer byte weights plus layer-adjacency constraints so the balancer refuses to cut MoE expert layers or Gemma 4 KV-shared source/consumer pairs. Drops the hand-specified `--pp-layers` requirement for MoE and gemma-4-e2b-it-4bit. Extracted into `partition_balance`, `partition_profile`, and `partition_quality` modules (#357, resolves #348).
- Elastic pipeline-parallel repartitioning behind `--enable-elastic-pp`. `RepartitionCoordinator` drives `Idle → Draining → Rebalancing → Resuming → Idle` and emits `RepartitionEvent` to a transport-agnostic sink without a full cluster restart. CLI flags: `--elastic-pp-drain-timeout`, `--elastic-pp-pressure-fraction`, `--elastic-pp-cool-down` (#349, #358).
- Per-stage LoRA adapter composition across pipeline ranks via the existing `--adapter` flag. Each stage loads only the adapter tensors inside its layer range through a new filtered safetensors loader (`load_safetensors_filtered`), fuses them in place with the same `fuse_lora_weights_into` primitive that backs the non-PP path, and unchanged-family guards (`ensure_no_adapter`) prevent silent drops. Llama family implements composition; parity integration test asserts bit-equality with the single-process adapter run (#347, #355).
- Stage-executor coverage for five new families: Mistral dense, Mixtral 8x7B MoE, DeepSeek V3 (MLA + routed MoE with MTP-trailer strip), Llama 4 Scout text-only tower, and Mamba-family hybrids Jamba and Nemotron-H. `StageFamily` enum plus `supported_families()` surfaces per-family capability on the server startup log (#345, #356).
- Pipeline-parallel observability: `/metrics` endpoint renders per-stage utilization, rolling bubble ratio, activation-transfer latency histograms (p50/p95/p99 per stage pair), and KV admission rejection counters labeled by stage and reason. `--metrics-port`, `--debug-pp-trace <PATH>` (chrome-tracing JSON), and `AdmissionDiagnostic` replace opaque 500s on rejection. Grafana dashboard JSON at `docs_internal/performance/pipeline-dashboard.json` (#350, #359).
- Multi-host pipeline-parallel regression CI harness at `.github/workflows/pipeline-parallel-ci.yml`: `two-host-logical` on GitHub runners (path-filtered, intended as required status) plus `three-host-real-model` gated by the `ci:pp-three-host` PR label or manual dispatch. Shares shell entry points with local reproduction (#353, refs #344).
- `VisionFeatureCache` LRU for multi-turn VLM image feature reuse, wired through Gemma 4 VLM, Qwen2.5-VL, and Qwen3-VL via `_with_cache` variants. Cache keys are filesystem paths or SHA-256 digests of inline payloads. New `--vision-cache-size N` CLI flag (default 20, 0 disables) (#334, matches #325).
- Null/empty-cache safety guards in the batch scheduler. Pure-text requests with zero tokenized prompt tokens are rejected before admission (VLM image/audio injection paths unaffected); `execute_decode_step` and `execute_batched_decode` no-op on empty `seq_ids`. Mirrors the upstream mlx-lm BatchKVCache extend/filter/merge null guards (#333, closes #324).

### Fixed
- Auto-detect per-layer quantization bit overrides in `UnifiedLinear::from_weights_with_mode` and `FusedQKVLinear::from_weights_separate_with_mode`. New `infer_quantization_bits()` verifies the MLX invariant `packed_in * 32 == bits * num_groups * group_size` and infers the actual bit width from tensor shapes when the caller-supplied bits disagree. Enables qwen3.6-35b-a3b-4bit, which stores router-gate and shared-expert-gate at 8-bit while the rest of the model is 4-bit (#361).
- Use additive f32 attention mask (0.0 attended, f32::MIN masked) in `prepare_inputs_for_multimodal` instead of the previous multiplicative INT32 0/1 mask. `mx.fast.scaled_dot_product_attention` treats non-bool masks as additive bias on pre-softmax scores, so the old form silently leaked padding tokens into the attention distribution whenever `attention_mask` contained a zero (#339, closes #337).
- Mirror PR #326 conditional `embed_scale` to `TensorParallelGemma4Model::forward_impl`. Previously, `multiply_scalar` was applied unconditionally after `embed_tokens`, double-scaling text embeddings and incorrectly scaling image/audio features from VLM callers. Moved into the `None` arm only, matching `Gemma4TextModel::forward`. Added regression test asserting TP/non-TP logits match for both `input_embeddings` and `input_ids` paths (#338, closes #335).
- Wrap every `cache.conv_state = Some(slice_axis(...))` assignment in `mlxcel_core::contiguous(&tail, false)` across mamba, mamba2, nemotron-h, and jamba, plus the two NemotronH fused-kernel paths. `slice_axis()` returns a lazy MLX `Slice` graph node that retains the source `padded_input` as a live input, causing per-step memory growth proportional to sequence length. 50-step shape-plateau regression test added per model (#340, closes #336).
- Apply RMS norm BEFORE `embedding_projection` on the encoder-side dim in `Gemma4 Multimodal Embedder` (was previously AFTER, on the text-side `hidden_size`). Mirrors upstream mlx-vlm. Renamed field `post_projection_norm` → `pre_projection_norm`. **BREAKING** for pre-fix VLM checkpoints: re-download `mlx-community/gemma-4-*-it-4bit` to obtain the post-rename weights (#329).
- Apply `sqrt(hidden_size)` `embed_scale` to text embeddings in `Gemma4VLModel::get_input_embeddings_with_audio` BEFORE merging vision/audio features, and make the scalar multiply in `Gemma4TextModel::forward` conditional on `input_embeddings` being `None`. Vision/audio features are already in language-model embedding space; double-scaling them degraded multimodal generation quality (#326, closes #317).
- Implement proportional RoPE for Gemma 4 full-attention layers. Real Gemma 4 checkpoints declare `rope_type="proportional"` on full-attention and `rope_type="default"` on sliding-attention layers; the previous implementation silently dropped `rope_type` and normalized by the rotated-only slice instead of the full `head_dim`. New `mlxcel_core::rope_proportional` module with `compute_proportional_rope_freqs` and `apply_proportional_rope` matching the upstream slice/concat/fast_rope/re-splice pipeline. For head_dim=256, partial_rotary_factor=0.25, the two formulations differ by a factor-of-4 exponent shift (#332, closes #321).
- Gemma 4 audio feature extractor: drop `+0.5` phase shift in Hann window so it uses the periodic form `w(i) = 0.5 - 0.5·cos(2π·i/N)` matching HuggingFace Gemma 4. Prepend `frame_length/2 (160)` zero samples before frame extraction for semicausal convention (first frame centered at t=0). Use `total_len` in `num_frames` calculation and correct `frame_size_for_unfold` to use `frame_length+1` only for non-HTK preemphasis. Restores the correct 100 frames for 1s 16 kHz audio with 10 ms hop (#327, closes #320).
- Ensure `conv_input` cache slice is contiguous in GatedDeltaNet forward paths (Qwen 3.5, Kimi Linear, Qwen 3 Next). `mlxcel_core::slice()` calls `mlx::core::slice()` which creates a graph node holding source reference — without `contiguous()`, every cached entry holds the full `conv_input` buffer, preventing freeing and causing per-step memory growth proportional to sequence length. 50-step regression test added (#328, closes #323).
- Default NemotronH `time_step_limit` to `(time_step_min.unwrap_or(0.0), time_step_max.unwrap_or(+inf))` unconditionally when absent. Changed from `(f32, f32)` to `Option<(f32, f32)>` so absent configs are distinguishable from explicit `(0.0, +inf)` sentinels. Matches upstream mlx-lm behavior (#330, closes #319).

### Changed
- Replace Gemma 4 `ScaledLinear` wrapper with `UnifiedLinear` directly across both `Gemma4TextModel` and `Gemma4StageModel` (tensor-parallel path). New `per_layer_projection_scale: f32` field stores `(hidden_size as f32).powf(-0.5)` and is applied explicitly in `project_per_layer_inputs()` after the linear forward pass, preserving bit-identical math (#331, closes #322).

## [v0.0.23] - 2026-04-15

### Fixed
- Render chat templates that use Python-style dict/string methods (#315). Extends minijinja's `unknown_method_callback` with shims for `.get`, `.items`, `.keys`, `.values`, `.strip`, `.lstrip`, `.rstrip`, `.startswith`, `.endswith`, `.split`, `.rsplit`, `.replace`, `.join`, `.upper`, `.lower`, `.title`, `.capitalize`, `.casefold`, `.swapcase`, `.find`, `.count`, `.is{digit,alpha,alnum,space,upper,lower}`. Previously rendering silently fell back to `to_prompt()`'s `User: ... Assistant:` format, and instruction-tuned models echoed `Assistant:` in a loop.
- Pass `tools` as an empty iterable (not `None`) so `{% if tools is iterable and tools | length > 0 %}` guards work under minijinja. Fixes Qwen 3 Next, Nemotron-H, and Nemotron-NAS tool-free rendering (#315).
- Strip HuggingFace `transformers`' `{% generation %}` / `{% endgeneration %}` extension markers during template preprocessing so SmolLM 3 parses cleanly (#315).
- Apply the Gemma 4 structural-token stream filter and non-streaming cleanup unconditionally, not only when tool parsing is enabled, so plain chat responses no longer leak `<|channel>`, `<channel|>`, `<turn|>`, `<|turn>`, `<|tool_call>`, or `<tool_call|>` markers into content (#315).
- Extend `clean_content_markers` with `<|channel>` / `<channel|>` / `<|tool_call>` / `<tool_call|>` so stray closing tags that Gemma 4 occasionally emits in non-thinking mode are stripped even without a matching open tag (#315).

### Added
- `test_all_local_model_templates_render` ignored-by-default audit that renders every locally-available model against three canonical scenarios (simple user, system + user, multi-turn with `<think>` blocks). Current result: 85 models checked, 249/249 scenarios pass, 0 failures, 6 intentional template `raise_exception` rejections categorized separately (#315).

### Changed
- Clean up pre-existing `cargo clippy --release -p mlxcel --lib` warnings (7 → 0): replace `unwrap()` after `is_some()` checks in `distributed/config.rs`, bind the MoE router via `if let` chain in the Gemma 4 TP path (`distributed/tensor_parallel/llama_runtime.rs`), collapse two character-identical QKV shard branches, auto-elide `'a` lifetimes, collapse a `thunderbolt_transport.rs` nested `if`, replace a manual `% != 0` with `is_multiple_of()` in NVFP4 sanitize, and drop a now-redundant `cache.as_deref_mut()` + `mut` annotation in Qwen 3.5 `GatedDeltaNet::forward` (#316).
- Clean up pre-existing webpage `pnpm lint` / `tsc --noEmit` errors (20 / 4 warnings → 0 / 0): replace framer-motion wrapper `any`-typed props with `HTMLMotionProps`, `let` → `const` in `downloads.tsx`, swap `<img>` for `next/image`'s `<Image />` on the local Lablup logo, and rewrite `use-os.ts` to avoid synchronously setting state inside `useEffect` with proper `NavigatorUAData` / `WebGLDebugInfoExtension` typings (#316).

## [v0.0.22] - 2026-04-13

### Added
- Pipeline stage executor framework with per-family executors (#272)
- Gemma 3 pipeline stage executor (#304)
- Gemma 4 pipeline stage executor (#305)
- Qwen3 pipeline stage executor (#306)
- Qwen3.5 pipeline stage executor (#307)
- GLM4-family pipeline stage executors (#308)
- GLM MoE DSA pipeline stage executor (#310)
- gpt-oss pipeline stage executor (#303)
- In-process pipeline stage worker loop (#273)
- CLI pipeline generate path (#274)
- Server pipeline runtime integration (#275)
- Pipeline transport lifecycle controls (#276)
- TCP-backed remote pipeline stages (#288)
- Thunderbolt transport backend for remote pipeline parallelism (#291)
- Multi-machine validation for remote pipeline parallelism (#301)
- bench_decode `--cooldown` and `--big-cooldown` for M5 Max thermal management
- M5 Max benchmark refresh for 2026-04-13 (97 models, 88 pass; 8 multimodal models restored)

### Fixed
- Tolerate stale `model.safetensors.index.json` in mlx-community repackaged quants (gemma3-4b, gemma3n-e2b/e4b, llama-4-scout-17b, mistral-small-3.1, molmo2)
- Tolerate partial `text_config` (no `num_hidden_layers`) in single-rank tensor-parallel planning (LLaVA-1.5, LLaVA-Next-Mistral)
- Prevent Gemma 4 special tokens from leaking into streaming content deltas (#312)
- Complete remote pipeline lifecycle recovery (#290)
- bench_decode single-model runs no longer truncate the day's full-suite CSV (#314, closes #313)
- Log lazy pipeline peer reconnects

### Changed
- Generalize stage executor backends and remove legacy stage executor file (#302)
- Transport-capable pipeline runtime seam (#287)

### Tests
- Pipeline server smoke validation (#278)
- Pipeline rollout real-model coverage (#277)

### Docs
- Remote pipeline usage examples
- Remote pipeline rollout workflow
- Refreshed M5 Max benchmark documentation with measurement-variance analysis
- Recorded issue execution workflow

## [v0.0.21] - 2026-04-12

### Added
- Paged KV cache substrate with batch scheduler integration
- Native paged decode kernel paths for rotating and chunked caches (#260)
- Paged compatibility for windowed caches (#256)
- Default paged decode for supported server workers (#246)
- Paged KV transfer observability (#258)
- NVFP4 load-time dequantization for Gemma 4 nvfp4 checkpoints
- F8_E4M3 / F8_E5M2 safetensors loading for nvfp4 checkpoints
- Paged decode rollout benchmark matrix and eligibility tracking (#262)

### Changed
- Unify model-owned sequence state with backend seam (#244)
- Vectorize batched decode positional metadata
- CI: auto-promote pre-release to full release after successful builds

### Fixed
- Skip Teams notification when webhook URL secret is not configured

## [v0.0.20] - 2026-04-10

### Added
- In-process tensor parallel runtime for Llama (#235)
- Tensor parallel support for Qwen2, Qwen3, and Qwen3.5 text models (#235)
- Gemma 3 tensor-parallel runtime with tp4 parity stabilization (#235)
- Gemma 4 tensor-parallel support (#235)
- Dense TP support for ERNIE 4.5 and Hunyuan v1 models (#235)
- Server batching support for tensor parallel runtimes (#235)
- Tensor-parallel config wiring into CLI and server entrypoints (#235)

### Fixed
- Qwen 3.5 tensor-parallel parity on large CUDA models (#235)

### Changed
- Expand tp4 parity coverage to larger models and server end-to-end tests (#235)

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
- GatherQMM CUDA implementation via upstream MLX upgrade to b98831ad (#226)
- SM80 and naive QMM dispatch paths for non-Hopper CUDA GPUs (#226)
- Gemma 4 CUDA support: all 7 variants (e2b, e4b, 26b, 31b in 4bit/8bit) (#227)
- Qwen 3.5 CUDA support: 27b-4bit, 9b-bf16, 35b-MoE-4bit (#227)

### Fixed
- Mixed-type bf16/float JIT compilation failures in CUDA binary_ops.cuh (#227)
- Remove stale NO_GPU(BlockMaskedMM) override that conflicted with upstream implementation (#226)
- Gemma 3-4b and Gemma 3n (e2b, e4b) recovered on CUDA via binary_ops fix (#227)

### Changed
- Upgrade MLX C++ upstream from 6a9a121d to b98831ad (#226)
- Replace custom gather_qmv.cu with upstream integrated qmv.cu (#226)
- Sync CUDA quantized.cpp with upstream SM80/naive dispatch paths (#226)

### Performance
- GB10 CUDA: 14 models recovered from FAIL, 24 models improved >10%
- mamba2-1.3b +180%, minicpm-2b +131%, llama-3.1-8b +130%, hunyuan-dense +125%, llama-3.2-1b +115%

## [v0.0.17] - 2026-04-06

### Fixed
- Resolve broadcast crash in Gemma 4 chunked prefill with undersized attention mask (#224)

## [v0.0.16] - 2026-04-05

### Added
- Audio input support for server chat completions endpoint (#217, #220)
- Gemma 4 audio encoder and audio-language model support (#217, #218)
- Metal 4 fused attention path (#197, #210)
- OpenAI-compatible tool calling support (#212, #213)
- M5 GPU acceleration experiments (#209)
- M5 Neural Accelerator rollout research (#203)

### Changed
- Unify attention dispatch for Metal 4 path (#201)

### Fixed
- Propagate client disconnection to BatchScheduler to prevent orphaned sequences (#219, #221)
- Harden tool calling with input limits, parser improvements, and format handlers (#215, #216)
- Remove eval() calls from qwen3_moe forward hot path (#211)
- Resolve Gemma SDPA crash on M1 by reducing threadgroup memory for head_dim=256 (#208)
- Update compiled.cpp patch for upstream MLX API change (#207)
- Add str.split() support in chat template for Gemma 4 multi-turn (#206)

## [v0.0.15] - 2026-04-03

### Added
- Gemma 4 text and VLM model support (#199)
- User-facing warning when loading full-precision bf16 models (#195)
- Download webpage with Next.js static site (EN/KO i18n)

### Changed
- Extend bf16→f16 weight conversion to all Apple Silicon generations (#193)
- Audit f32 upcasts and optimize MoE gate sigmoid for fp16 co-issue (#194)
- Improve Metal 4 fused attention scaffolding with research documentation (#196)
- Reuse cached MLX source for faster rebuilds (#200)

## [v0.0.14] - 2026-04-03

### Added
- Logprobs support for chat completions and completions endpoints (#188)
- Runtime Apple Silicon generation detection for hardware-specific optimizations (#161)
- Prefill tile alignment for M5 Neural Accelerator (#162)
- Batched speculative decode verification for NA utilization (#167)
- Batched prefill in server mode (#169)
- Layer pipelining with strategic async_eval (#170)
- Metal 4 fused attention kernel scaffolding (#171)
- KV cache INT8 quantization for memory savings (#172)
- INT8 quantization optimization for M5 Neural Accelerator (#168)
- Multimodal chat template support for VLM image token placement
- Apple Silicon precision hardware guide documentation

### Changed
- Centralize bf16→f16 weight conversion in shared VLM loading path
- Skip bf16→f16 conversion for quantized models (restores +20% throughput)
- Add compiled gelu_topk kernel matching Python mlx-lm `@mx.compile` pattern
- Expand QKV projection fusion to GQA models (#164)
- Expand compiled MLP fusion to non-quantized models (#166)
- Fuse Q/K/V projections in Gemma v1 attention for faster decode (#183)
- Refactor AGENTS.md into focused reference docs (313→75 lines)

### Fixed
- Auto-convert bf16 weights to f16 on M5 for Metal JIT compatibility
- Skip add_special_tokens when prompt already contains BOS token (double-BOS fix)
- Prevent NemotronH all-`<unk>` output on M5 Max by avoiding mixed float32/float16 ops (#187)
- Prevent Nemotron-H/NAS GPU hang and state corruption on M5 Max (#186)
- Trim NemotronH internal caches after padded prefill to prevent GPU hang (#184)
- Fix PhiMoE expert activation from GeGLU to SwiGLU (#182)
- Fix matmul outside compile boundary in FP MLP to fix output corruption (#181)
- Replace gelu_approx power(x,3) with erf-based GELU to fix NaN in vision encoder (#179)
- Guard multimodal chat template to avoid garbled output on text-only VLMs
- Skip compiled FP MLP for bfloat16 models
- Patch MLX compiled kernel JIT to cast mixed bfloat16/float operands
- Patch MLX Metal kernels for macOS 26.4 compatibility
- Correct M5 Max benchmark results affected by GPU cascade corruption

## [v0.0.13] - 2026-03-31

### Added
- Mistral4 MLA (Multi-head Latent Attention) language model support (#144)
- Molmo-Point VLM model support (#148)
- NemotronSuper model support (upstream mlx-lm sync) (#131)
- `sync-upstream` Claude Code command for tracking mlx-lm/mlx-vlm changes

### Changed
- Fuse GatedDeltaNet decode step with `mlx::core::compile` for improved throughput
- Apply MRoPE and position ID optimizations to Qwen3-VL-MoE
- Fast-path single-token decode position IDs in Qwen3-VL
- Vectorize Qwen3-VL interleaved MRoPE with `take_along_axis`
- Optimize VLM vision encoding and sampling pipeline (#149)
- Use SDPA for NemotronH attention, boosting decode throughput 59%

### Fixed
- Improve SSM/Mamba2 numerical precision with float32 dt computation (#133)
- Improve GatedDelta numerical precision with float32 state (#132)
- Resolve Mamba/NemotronNAS output corruption with softplus overflow and fused norm grouping
- Guard Qwen3.5 GatedDeltaNet state batch dimension mismatches (#145)
- Use `h.shape` instead of `inputs.shape` for Ministral3 attn_scale (#146)
- Document scalar offset invariant for Llama4 BatchKVCache compatibility (#147)
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
