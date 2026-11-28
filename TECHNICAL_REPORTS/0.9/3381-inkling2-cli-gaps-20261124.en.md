# PR #3381: Inkling 2 CLI gaps (reasoning effort, banner count, 4-bit trace)

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3378). Narrow suites pass; validated on an M3 Ultra (512 GB) on both Inkling 2 Mini checkpoints

## Executive Summary

The PR closes three gaps #3367 left for Inkling 2. `mlxcel generate` and `mlxcel run` take `--reasoning-effort`, which reaches the chat template in the one-shot render, in `run -p` and in both chat REPLs. The KV banner counts the caches the loaded model keeps, so Mini prints `48 of 48` instead of the 24 physical layers. The remap that let mlx-vlm read the mlx-community 4-bit export is now committed, and it ran the 4-bit image trace that #3367 skipped: 600 identical prompt ids and 0 of 70 decided disagreements.

## 1. Problem Statement

- **Effort.** Inkling 2's template reads `reasoning_effort` (a level name or a number in [0.0, 0.99], default 0.9). The server takes it per request, but the CLI rendered with no kwargs, so the level could not be set.
- **Banner.** `kv_cache_mode_banner_suffix` read `num_hidden_layers` from `config.json` before the model loaded. Mini runs 24 physical layers as 48 visits with a cache each, so it printed `24 of 24`.
- **4-bit trace.** mlx-vlm 466069f5 reads only the native layout. #3367 used an uncommitted rename-only script for the text trace and ran no image trace.

## 2. Design

### 2.1 Reasoning effort

The server forwards the request field as a JSON string and never forwards a disabled level (`none`, `off`, `0`, ...); it sets `enable_thinking=false` instead. Inkling 2 needs the opposite for both: `none` must reach the template as a level name (it renders `Thinking effort level: 0`), and `0.5` must be a JSON number, because the string `"0.5"` raises `Unknown reasoning_effort`. `server::in_process::reasoning_effort` therefore does two things. `apply_to_chat_body` sets the portable `reasoning_effort` field, so the server's resolution still runs and still fills `reasoning_strength` or `enable_thinking` where a template wants them, and it also sets the explicit `chat_template_kwargs.reasoning_effort` with a typed value (number when the text parses as a finite number, else the lowercase string). The server's per-key precedence lets the explicit kwarg win. `cli_template_kwargs` builds a synthetic request the same way and calls the server's `resolve_effective_kwargs`, so the one-shot render and the chat path share one mapping and the CLI keeps no second table.

`CliPromptMedia` carries the level into `resolve_cli_prompt` and `apply_vlm_chat_template`; `CliServerSettings.reasoning_effort` carries it into every chat request body. The template's own `raise_exception` message is the validator and is shown unchanged (`Chat template rejected the request: Unknown reasoning_effort: bogus`). Only an empty value is refused at parse time.

A disabled level also sets `enable_thinking=false` on templates that read that key, which is what the server does for the same field. A template that reads neither key renders byte-identically for every level; one that reads `enable_thinking` does not, by design.

### 2.2 Banner

`print_kv_cache_mode_banner` now runs after `load_generation_model` on the normal path and counts `kv_cache_storage_modes()` when the model returns it: one entry per cache, and only entries that store the effective mode count as applied. Mini returns 48 FP16 entries, so Fp16 reads `48 of 48` and an int8 request reads `0 of 48`. A model that returns `None` uses `num_layers()` with the existing boundary-layer rule, which keeps a dense model's banner as it was. The pipeline and OpenXLA paths load no `LoadedModel`, so they still print before the load with the config count.

### 2.3 4-bit remap

`scripts/inkling2_mlx_community_remap.py` rewrites the export into mlx-vlm's converted layout (the names `Model.sanitize` passes through) rather than the native `model.llm.*` layout the issue text names; both load the same model, and the converted layout avoids re-running native-only transforms. It renames tensors with the table `src/models/inkling2/sanitize.rs` already uses, splits `gate_up_proj` with mlx-vlm's `_split_gate_up`, sets `weight_scale_2 = global_scale / 2688`, and writes per-expert modes into `config.json`, inferred from the ratio of packed width to scale width (2 is NVFP4, 8 is MXFP8). The export has no `processor_config.json`, so those files are linked from `models/inkling-2-mini`. Both trace scripts reach it through `--inkling2-mlx-community`.

## 3. Validation

| Check | Result |
|---|---|
| Effort at `none` / `0.5` / `XHigh` / default (rendered prompt ids decoded) | `Thinking effort level: 0` / `0.5` / `0.99` / `0.9` |
| `--reasoning-effort bogus` | `Chat template rejected the request: Unknown reasoning_effort: bogus` |
| `run -p` and the REPL at `none` against `high` | different replies, so the flag reaches the in-process chat request |
| Banner on Mini | `KV cache mode: fp16 (applied to 48 of 48 layers)` |
| 4-bit text trace through the remap (1024 positions) | 33 top-1 differences, 0 of 372 decided, perplexity -0.32% |
| 4-bit image trace, `tests/fixtures/sapiens2/person.jpg` | 600 prompt ids identical, 7 of 127 top-1 differences, 0 of 70 decided, perplexity -1.03% |

The text corpus differs from #3367's (which was not recorded), so the decided count differs from its 0/494; the result, no disagreement at a decided position, is the same.

Unit selectors: `server::in_process` 27, `cli::` 321, `coverage_tests` 18, `commands::generate` 119, `commands::run` 11, `commands::cli_server` 6, `commands::chat` 19; clippy with `-D warnings` on lib, bins and tests; `tests/test_inkling2_mlx_community_remap.py` 6.

## 4. Limitations and Follow-ups

- `--reasoning-effort` has no effect with `--no-chat-template`.
- The banner counts what a model reports before generation injects its layer-mode table. A family whose storage report needs that table returns `None` at that point and keeps the `num_layers()` count.
- Batched decode is #3377.
