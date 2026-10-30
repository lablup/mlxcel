# PR #2779: Finish the #2750 VLM prompt parity follow-ups

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Python
**Risk Level**: Low (Gemma 3 image prompts on the OpenXLA backend shrink from 273 to 272 tokens, so that backend's Gemma 3 image outputs change; the MLX path builds the same ids as before, and the rest is the trace harness, docs and tests)

## Executive Summary

#2750 aligned the InternVL, MiniCPM-V 4.6 and Gemma 3 image prompts with transformers on the MLX path and left three residuals, collected in issue #2761. This PR closes all three. Gemma 3 newline fusion moves into one helper, `expand_image_blocks_with_newline_fusion`, that both the MLX runtime and the OpenXLA host preprocessor call, so the OpenXLA path now builds transformers' 272 ids instead of 273, and the newline-run table is probed once per model instead of on every image request. The server-wide `enable_thinking` default is kept for image prompts and documented as a deliberate divergence, and both trace arms gain a switch (`MLXCEL_TRACE_ENABLE_THINKING`, `--enable-thinking`) so parity can be scored with thinking off; with it off, gemma-4-e2b and MiniCPM-V 4.6 build the reference ids exactly. A real MiniCPM-o checkpoint (`mlx-community/MiniCPM-o-4_5-4bit`) was added to the model store and smoked through the built-in ChatML template that #2750 had only covered with synthetic files.

## 1. Problem Statement

### 1.1 Two backends, two Gemma 3 expansions

transformers' `Gemma3Processor` expands `<start_of_image>` in the text before tokenizing, so the wrapper's `\n\n` fuses with the template's `user\n` into a single `\n\n\n` token. mlxcel expands at the token level, and #2750 fixed the MLX path by re-fusing adjacent newline runs after expansion (`merge_image_block_newline_runs`). The OpenXLA path (`Gemma3IreeHostPreprocessor::prepare_iree` in `host_preprocessor.rs`) called `apply_image_token_blocks` with the same `\n\n` wrapper and never merged, so it kept the 273-token prompt. The two backends disagreed on the prompt before either model ran, and the fix existed in one place only, so nothing stopped the next change from drifting them further.

The MLX path also rebuilt the newline-run table on every image request: `newline_run_token_ids` encodes `"\n"` repeated 1, 2, ... times until a run stops being one token, which on Gemma 3 is 32 single-string encodes (runs 1 to 31 plus the failing probe). The table depends only on the tokenizer, so it is a per-model constant recomputed per request.

### 1.2 The thinking default looked like a parity bug

mlxcel defaults `enable_thinking` on when the tokenizer has think markers (`apply_thinking_default` in `server::chat_front`, aligned with mlx-lm PR #1114); transformers and mlx-vlm leave it off. That accounts for gemma-4-e2b-it-4bit building 278 image-prompt tokens to mlx-vlm's 271 (a system `<|think|>` turn) and for part of the MiniCPM-V 4.6 gap with the upstream template (mlxcel opens `<think>` where transformers closes an empty block). #2750 left this open. The image trace (`examples/logit_trace/image.rs`) always rendered with the server default, so neither family could be scored against the reference with the same template setting, and a length mismatch there could not be told apart from a real processor bug.

### 1.3 The MiniCPM-o fallback had never seen a real checkpoint

#2750 added `MINICPM_CHATML_TEMPLATE` as the fallback for template-less `MiniCPMV46VLM` and `MiniCPMOVLM` checkpoints, because the server otherwise rendered the generic `User: ...\n\nAssistant: ` prompt and the image expansion wrapped it in a second ChatML turn. No MiniCPM-o checkpoint was in the store, so only `template_less_minicpmo_server_prompt_matches_the_cli_prompt` covered the MiniCPM-o arm, with synthetic files.

### 1.4 Risk of leaving them

| Risk | Impact | Likelihood |
|---|---|---|
| OpenXLA and MLX Gemma 3 build different prompts for the same request | Medium: different outputs per backend, and cross-backend traces measure the prompt, not the model | Certain before this PR |
| Per-request newline probing | Low: about 32 encodes of tiny strings per Gemma 3 image request | Certain, small |
| Thinking-default gap misread as a processor bug | Medium: wasted investigation or a wrong "fix" to the default | High on gemma-4 and MiniCPM-V 4.6 traces |
| MiniCPM-o fallback broken on real files | Medium: doubled framing on template-less MiniCPM-o servers | Unknown before this PR |

## 2. Technical Decisions

### 2.1 One expansion helper, called by both backends

`gemma3_vlm_prompt.rs` gains `expand_image_blocks_with_newline_fusion(tokens, info, image_count, newline_runs)`. It runs `apply_image_token_blocks` and, when blocks were expanded and the layout has a wrapper (`has_image_block_wrapper`: non-empty `block_prefix_tokens` or `block_suffix_tokens`), runs the merge. It returns what `apply_image_token_blocks` returns, so the call sites keep their stats handling. The MLX `Standard` arm in `vlm_runtime.rs` and a new free function `host_preprocessor::gemma3_iree_prompt_tokens` both call it; `prepare_iree` calls that function and then runs `validate_sequence_capacity` on the merged length as before.

The alternative was to copy the merge call into `prepare_iree`, the smallest diff. That is how the two paths diverged in the first place: #2750 fixed one call site and the other did not exist in its test surface. With one entry point the merge cannot be applied on one backend and forgotten on the other.

The OpenXLA token step and its block layout (`gemma3_iree_token_block_info`) are free functions under `cfg(any(feature = "xla-iree", test))`. `Gemma3IreeHostPreprocessor` exists only with `xla-iree`, so a method on it could not be tested in the default build. As free functions they are, and `xla_gemma3_prompt_matches_transformers_ids` also asserts that the OpenXLA layout equals the MLX loader's (`gemma3_info(256)`), which pins the `GEMMA3_VLM_NEWLINE_WRAPPER_TOKEN_ID` constant (108) against the MLX `vec![108]`.

Graph identity does not change: prompt ids are not part of `stable_identity` or `prepared_gemma3.rs`, and the sequence capacity check already ran after expansion.

### 2.2 Where the newline table lives on each backend

MLX: `VisionModule` gets `newline_runs: OnceLock<Vec<i32>>`, filled through `cached_newline_runs(cell, encode)` on the first image request that needs it and reused after. The MLX path has no tokenizer at load time inside the vision loaders, but every request carries an `encode` closure, so lazy fill is the natural seam. All 10 `VisionModule` literals (9 loaders plus `loaded_model_tests.rs`) gain `OnceLock::new()`. Layouts without a wrapper never touch the cell and pass an empty slice.

OpenXLA: `load_gemma3_iree_host_preprocessor` loads the tokenizer with `crate::tokenizer::load_tokenizer(model_path)`, probes once, and passes the table to `Gemma3IreeHostPreprocessor::from_parts`, which stores it. The preprocessor is built once at load and owns no request tokenizer, so building the table eagerly there is simpler than a lazy cell. The cost is one extra `tokenizer.json` parse at load; the path is the same one the server already requires.

A `OnceLock` rather than a `Mutex<Option<_>>` because the value is written once and read on every request after; a concurrent first request blocks only for the one probe, and a panic during the probe leaves the cell empty for the next request to retry.

### 2.3 An empty table warns instead of failing

Both encode closures map tokenizer errors to an empty vector, so a probe failure yields an empty table, the merge becomes a no-op, and the prompt silently falls back to 273 tokens. Before this PR that probe ran per request; now an empty result is cached for the model's lifetime (MLX) or stored at load (OpenXLA). The security review flagged this, and `warn_if_newline_runs_empty` now logs one `tracing::warn!` where the table is filled (in `cached_newline_runs` and in the OpenXLA loader), so it fires once per model, not per request. Failing the load or the request was not chosen: the unfused prompt is still a valid prompt that the model can answer, one token off the reference, and a working tokenizer encodes `\n` without error.

### 2.4 Keep the thinking default, make the harness match it

Three options were on the table:

| Option | Pros | Cons |
|---|---|---|
| Turn `enable_thinking` off for image requests only | Image prompts match transformers and mlx-vlm by default | The same checkpoint renders differently with and without an image; a text and an image request in one conversation disagree |
| Turn it off server-wide | Matches the references everywhere | Reverses the mlx-lm #1114 alignment and changes every text request on thinking models |
| **Chosen: keep the default, add trace switches** | No behavior change; parity is measurable with the setting matched | The default image prompt still differs from the references on thinking-capable checkpoints |

Clients can already turn thinking off per request with `chat_template_kwargs.enable_thinking`, so the divergence is a default, not a capability gap. `docs/benchmarks.md` now lists it as a deliberate divergence next to the existing ones.

`examples/logit_trace` reads `MLXCEL_TRACE_ENABLE_THINKING` through `parse_enable_thinking`: `0` and `1` map to off and on, unset or blank keeps the default, and anything else (`yes`, `true`) is an error rather than a silent default. The value is applied with `set_default_enable_thinking` after `load_model_chat_template` and printed as `# enable_thinking`. `ImageArm::from_env` now returns `Result<Option<Self>>` so the rejection propagates. `scripts/mlx_vlm_image_logit_trace.py` gains `--enable-thinking {true,false}`, forwarded to `apply_chat_template` as a template kwarg and also printed. Strict parsing matters here because a typo would otherwise produce a trace that looks valid but was rendered under the other setting.

### 2.5 A template-less MiniCPM-o copy built by the test

The issue asked for a manual working copy without the template. The test instead builds one per run (`template_less_copy` in `minicpm_chat_template_tests.rs`): it symlinks every checkpoint file except `chat_template.jinja`, `chat_template.json` and `tokenizer_config.json`, and writes `tokenizer_config.json` back without its `chat_template` key. The store keeps the checkpoint as published, the test needs no manual step, and symlinks avoid copying 6 GB of weights. On non-Unix targets it falls back to copying.

## 3. Change Summary

19 files, +480/-47: two commits (fd941589 fix, a36063ac follow-up tidy) plus a merge of origin/main (f2d0e9c5, which brought in #2773).

- `src/multimodal/gemma3_vlm_prompt.rs`: `expand_image_blocks_with_newline_fusion`, `cached_newline_runs`, `warn_if_newline_runs_empty`, `has_image_block_wrapper`.
- `src/multimodal/vlm_runtime.rs`: the `Standard` arm calls the shared helper with the cached table.
- `src/multimodal/host_preprocessor.rs`: `gemma3_iree_token_block_info` and `gemma3_iree_prompt_tokens`; `Gemma3IreeHostPreprocessor` stores `newline_runs` and `prepare_iree` uses the shared token step.
- `src/loading/vlm_gemma.rs`: the OpenXLA loader builds the table once from the checkpoint tokenizer.
- `src/vision/mod.rs`, six `src/loading/vlm_*.rs` files, `src/loaded_model_tests.rs`: the `newline_runs` field and its `OnceLock::new()` initializers.
- `examples/logit_trace.rs`, `examples/logit_trace/image.rs`: `MLXCEL_TRACE_ENABLE_THINKING`, `parse_enable_thinking`, the `# enable_thinking` line.
- `scripts/mlx_vlm_image_logit_trace.py`: `--enable-thinking`.
- `docs/benchmarks.md`: thinking-off rows for MiniCPM-V 4.6 (upstream template) and gemma-4-e2b, the MiniCPM-o row, the thinking-default divergence, the OpenXLA note. `docs/environment-variables.md`: the new variable.
- Model store (outside the repository): `minicpm-o-4.5-4bit` on spark-102, manifest row added, link views rebuilt on both hosts.

Tests added (7, two of them ignored local-checkpoint tests):

| Test | Pins |
|---|---|
| `shared_expansion_helper_fuses_the_wrapper_newlines` | the helper builds transformers' 272 ids |
| `xla_gemma3_prompt_matches_transformers_ids` | the OpenXLA token step builds 272 and its layout equals the MLX loader's |
| `newline_runs_are_probed_once_per_model` | a counting encoder runs one probe pass across two calls |
| `enable_thinking_accepts_zero_one_and_unset`, `enable_thinking_rejects_other_values` | the env parser |
| `local_gemma4_template_thinks_by_default_and_not_when_disabled` (ignored) | the `<|think|>` system turn under the loaded default, and its absence with the default off |
| `local_template_less_minicpmo_renders_one_chatml_turn` (ignored) | one ChatML user and assistant opener, no `User:`/`Assistant:`, server prompt equals the CLI prompt |

## 4. Results

Failing first on main 52e5fd52 plus the new tests: the three Gemma 3 tests fail to compile (the helper, `cached_newline_runs` and the free function do not exist). With main's behavior restored inside the new functions they fail with `left: 273, right: 272` and `left: 8, right: 4` (the second call probes again).

Mutation checks on the ignored tests:

- Forcing `set_default_enable_thinking` to store `false` fails the gemma-4 test's first assertion. Forcing only the getter `default_enable_thinking()` to `false` did not change the render, because rendering reads the field directly; the test now asserts the getter too.
- Removing `MiniCPMOVLM` from the `builtin_chat_template` arm fails the MiniCPM-o test with "generic framing leaked into the prompt".

GB10 smoke (`gpu-lock`, `MLXCEL_REQUIRE_MODELS=1`, `tests/fixtures/test_image_shapes.png`):

| Family | CLI prompt ids | Server `prompt_tokens` | Output |
|---|---|---|---|
| gemma-3-4b-it-4bit | 272 | 272 | describes the square, circle and triangle |
| gemma-4-e2b-it-4bit | 278 | 278 | coherent; all 64 tokens in the reasoning channel (thinking default on) |
| minicpm-v-4.6-bf16 (no template) | 82 | 82 | describes the three shapes |
| minicpm-o-4.5-4bit, template-less copy | 78 | 78 | reasoning-only at 64 tokens; with `-n 400` the CLI names the red square, blue circle and green triangle |

Trace `# prompt_tokens` with the thinking switch:

- gemma-4-e2b: default 278; `MLXCEL_TRACE_ENABLE_THINKING=0` gives 271, the same ids as mlx-vlm 0.7.8. mlx-vlm with `--enable-thinking true` gives 278, so the whole gap was the default.
- MiniCPM-V 4.6 with the upstream template: default 84, thinking off 86, the same ids as transformers (86). mlx-vlm stays at 84 because it keeps the parentheses of its `(<image>./</image>)` marker and drops the empty think block, a divergence #2750 already documented.

## 5. Trade-offs and Residuals

- **The OpenXLA Gemma 3 path is not smoked end to end.** The token step is unit-tested and `cargo check --features cuda,xla-iree --all-targets` passes, but no OpenXLA Gemma 3 image request was run on a real checkpoint in this PR.
- **The OpenXLA Gemma 3 mask still differs from MLX.** #2773 (merged into this branch from main) moved the MLX Gemma 3 image prefill to the transformers mask; `prepared_gemma3.rs` still uses `GEMMA3_VLM_MASK_MODE = "gemma3-vlm-bidirectional-padding-f32-min-v1"`. After this PR both backends build the same 272 ids, but they still prefill them under different masks, so Gemma 3 image outputs still differ between backends. No issue was open for this at the time of writing.
- **An empty newline table is sticky.** It now warns once, but the MLX cell does not recover without a restart. A working tokenizer does not produce one.
- **Table not validated against the image ids.** Already true before this PR (#2750): a malformed tokenizer that encoded a newline run as `<start_of_image>`, `<end_of_image>` or the image token id could let the merge shrink an image stretch. Only an operator-supplied checkpoint can trigger it.
- **The default image prompt still differs from the references on thinking-capable checkpoints.** Intended; documented in `docs/benchmarks.md`.
- **The probe moved slightly earlier on MLX.** It now runs before expansion for any wrapper layout, including a first request whose expansion finds no block; once per model, so negligible.

## 6. Learning Points

- When two backends implement the same transformation, give them one function, not two call sites of the same steps. The OpenXLA gap existed because #2750 fixed the MLX caller and the other caller was outside its test surface. A shared helper plus a test that compares the two layouts makes drift a compile or test failure.
- Put feature-gated logic in a free function compiled under `cfg(any(feature = "...", test))` when the owning type exists only under the feature. The default test build then covers the logic without the heavy dependency.
- Cache per-model constants derived from the tokenizer, and make the failure mode loud at the moment the cache is filled. Caching turns a per-request soft failure into a sticky one, so the warning belongs at fill time.
- Separate configuration differences from processor bugs before comparing prompt lengths. A harness switch that matches the reference's setting (`enable_thinking` off) turned an unexplained 7-token gap into identical ids and showed there was no bug to fix.
- Mutation checks show which entry point a test actually pins. Forcing the getter changed nothing because the render reads the field, so the test gained an assertion on the getter as well.
- Build test fixtures from real checkpoints with symlinks and a rewritten config instead of maintaining a hand-made copy in the store; the fixture stays in sync with the published checkpoint and costs no disk.

## 7. Related

- Issue #2761: this PR's issue (three residuals of #2750).
- PR #2750 / issue #2547: the MLX prompt parity work these residuals came from.
- PR #2773 / issue #2760: the MLX Gemma 3 image prefill mask, merged into this branch; its OpenXLA counterpart remains.
- mlx-lm PR #1114: the origin of the think-marker `enable_thinking` default.
