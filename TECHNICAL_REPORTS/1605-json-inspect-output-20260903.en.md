# Technical Report: PR #1605 - feat: add JSON inspect output

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (CLI, library, clippy, and local real-checkpoint smoke are recorded in the PR; merge and broader integrated gates remain on the normal PR path)
**Languages**: Rust, Markdown
**Risk Level**: Medium (`mlxcel inspect` gains a new machine-readable contract, and resolver behavior now differs between text and JSON modes on cache misses)

---

## Executive Summary

Recipe builders and schedulers needed the existing memory estimator as stable byte fields, but `mlxcel inspect` only exposed a human-readable banner. Any consumer had to scrape prose, infer units, and stay in lock-step with a CLI format that was written for operators rather than tooling.

PR #1605 adds `mlxcel inspect --json` as a structured contract. The new path reuses the same estimator state as the banner, emits one JSON object with exact byte totals and effective inputs, derives `family` from the same registry-facing classifier mapping instead of raw `config.json` names, and resolves models quietly in offline mode so stdout stays machine-readable.

---

## 1. Problem Statement

### 1.1 Background

The repository already had a unified memory estimator behind `mlxcel inspect`, `generate --estimate-memory`, and `serve --estimate-memory`. That solved the sizing problem for humans, but not for automation. External recipe tooling needed byte-exact fields such as model weights, KV cache totals, activation reserve, headroom, and budget, while the CLI only printed a formatted text report.

### 1.2 Existing issues

- Tooling had to scrape human prose instead of reading a stable schema.
- The raw `model_type` string in `config.json` did not always match public registry identifiers, so a consumer could not reliably join inspect output with the architecture catalog.
- The normal model resolver prints expansion and downloader notices to stdout, which corrupts any command that promises "one JSON object" output.

### 1.3 Risk of leaving it unfixed

Without a structured surface, every downstream integration would have to duplicate fragile parsing logic and would break on presentation-only CLI changes. More importantly, a partially machine-readable command is worse than none at all: once stdout carries both data and resolver chatter, automation cannot safely distinguish success from noise.

---

## 2. Technical Review

### 2.1 New contract

`src/execution/memory_estimate.rs` now defines `InspectReport`, `InspectReportInputs`, and `InspectKvBytesPerToken`, and `InspectReport::from_estimate()` copies the same `MemoryEstimate` values the text banner already prints. The schema carries:

- version and resolved model path
- raw `model_type` and best-effort `family`
- effective inputs: `max_tokens`, `batch`, `kv_cache_mode`, `quant`
- byte fields: `weights_bytes`, `kv_bytes_total`, `activation_bytes`, `headroom_bytes`, `budget_bytes`, `total_bytes`
- optional fields serialized as `null` when geometry is unavailable: per-token KV rates, paged per-slot overhead, family/model type

This is the correct seam: the report is built from the estimator state rather than re-parsing rendered text, so the text and JSON forms cannot silently diverge.

### 2.2 Resolver behavior

`src/commands/inspect.rs` switches JSON mode onto `resolve_model_source_quietly_with_options()` and sets `offline: args.json`. That does two things deliberately:

- cache hits stay silent, so stdout is exactly one JSON object
- cache misses fail through stderr instead of auto-downloading with progress output

The human-readable banner path keeps the previous reuse-or-download behavior, so the PR widens the CLI surface without changing the interactive operator workflow.

### 2.3 Registry alignment

The initial JSON branch would have slugified the raw `model_type`, but that would have drifted from public registry ids such as `qwen3_5` versus `qwen3-5`. The fix commit moves `family` onto `inspect_family_slug()`, which derives the value through `get_model_type()` and the same registry-id mapping the architecture catalog uses. Raw `model_type` remains present as a separate field.

### 2.4 Compatibility and dependencies

- No new dependency is introduced.
- Existing `mlxcel inspect` banner output remains intact.
- The change is additive at the CLI level: `--json` is new, not a behavior change for existing invocations.
- JSON mode intentionally changes cache-miss semantics to offline failure, which should be treated as part of the contract, not an incidental implementation detail.

---

## 3. Technical Decisions

### 3.1 Build JSON from the estimator, not from the banner

The estimator already encoded the authoritative totals and fit decision. Reusing that state avoids dual logic and guarantees that the machine-readable and human-readable paths answer the same sizing question.

The rejected alternative was parsing or reformatting the existing banner. That would have left two sources of truth and made wording-only text changes risky for automation.

### 3.2 Keep stdout pure by making JSON mode quiet and offline

Machine-readable CLI output is only trustworthy if stdout contains the payload and nothing else. The PR therefore treats resolver chatter as a correctness bug for `--json`, not a cosmetic issue. Quiet resolution alone was not enough because a cache miss could still invoke the downloader; forcing offline behavior closes that gap.

The trade-off is that `mlxcel inspect --json -m <repo-id>` no longer auto-downloads on a miss. That is an acceptable constraint for automation, because scripts can now depend on a deterministic success shape and a deterministic failure channel.

### 3.3 Separate `family` from raw `model_type`

The contract exposes both fields because they solve different problems:

- `model_type` preserves the checkpoint's raw metadata
- `family` provides the registry-facing join key recipe tooling actually needs

Collapsing them into one field would either lose raw metadata or leak internal naming drift into public tooling.

### 3.4 Prefer nullable fields over invented defaults

The PR serializes unavailable geometry-dependent fields as `null` instead of `0` or placeholder strings. That makes absence explicit and prevents downstream code from mistaking "not computable" for a real zero-cost value.

---

## 4. Implementation Details

### 4.1 CLI surface

`src/main.rs` adds `json: bool` to `InspectArgs`, documenting that `mlxcel inspect --json` emits the same estimate as one JSON object.

### 4.2 Command flow

`src/commands/inspect.rs` now:

- resolves the model path through the quiet resolver when `--json` is set
- preserves effective KV-cache-mode resolution, so the report matches what `generate` and `serve` would actually build
- prints `serde_json::to_string_pretty(&report)` and returns early

The text banner path remains unchanged, including the over-budget note for non-fitting configurations.

### 4.3 Estimator extensions

`src/execution/memory_estimate.rs` adds the JSON structs plus helpers for:

- raw model-type extraction from `config.json`
- best-effort family classification aligned with registry ids
- paged per-slot overhead reporting for families that default to paged decode

The unit tests cover contract copying, stable key order, null serialization, raw-versus-classified family values, and known registry edge cases such as `qwen3_5` and `gemma3_text`.

### 4.4 Documentation

- `README.md` adds a `mlxcel inspect --json` example and enumerates the key byte fields.
- `docs/environment-variables.md` clarifies that `MLXCEL_MEMORY_LIMIT` also feeds the inspect and estimate-memory preflight figures through the same parser.

---

## 5. Validation

The PR body records these checks:

| Check | Result |
|---|---|
| `cargo fmt --check` | passed |
| `cargo check --lib --tests` | passed |
| `cargo test --lib -- memory_estimate` | passed |
| `cargo clippy --lib --tests -- -D warnings` | passed |
| Real checkpoint smoke on `/home/inureyes/models/mlx/qwen3-4b-4bit` | text banner preserved; JSON mode emitted one stdout object with empty stderr |
| Cache-miss smoke on `definitely-missing-model-for-pr1605` | stdout stayed empty; failure surfaced on stderr |

The validation is well matched to the change: it exercises the estimator unit boundary, the CLI integration path, and the structured-output cleanliness guarantee on both cache-hit and cache-miss branches.

---

## 6. Learning Points

**Structured CLI output is a separate product surface.** Once a command promises machine readability, resolver messages and downloader progress are no longer harmless UX details; they are contract violations.

**Raw metadata and public identifiers are not interchangeable.** `config.json` is useful provenance, but automation often needs a stable join key aligned with a curated catalog. Exposing both fields keeps those responsibilities separate.

**Nullable beats fake certainty.** Returning `null` for unavailable geometry communicates the real state of the estimator and keeps consumers from baking in incorrect assumptions about unsupported paths such as TurboQuant sizing.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 7 |
| Lines added | 561 |
| Lines removed | 20 |
| Commits | 3 |

### Related commits

- `f8d670e` feat: add JSON inspect output
- `c142dbd` fix: align inspect family with registry ids
- `85eea20` fix: keep inspect JSON stdout machine-readable

### Files of interest

| File | Change |
|---|---|
| `src/main.rs` | Adds `--json` to `inspect` |
| `src/commands/inspect.rs` | Routes JSON mode through quiet offline resolution and report serialization |
| `src/execution/memory_estimate.rs` | Adds the JSON contract and contract-focused tests |
| `src/downloader/resolver.rs` | Exposes quiet model resolution for structured-output callers |
| `README.md` | Documents the new JSON form |
| `docs/environment-variables.md` | Clarifies memory-limit interaction with inspect/preflight |

---

## 8. Follow-up Actions

- Replace the local `inspect_family_slug()` fallback with the canonical `mlxcel arch --json` registry id once issue `#1508` lands on `main`.
- If external tooling begins depending on the JSON shape, promote the schema to an explicit documentation page or compatibility note so future additions remain additive.
- Consider whether other structured CLI surfaces should adopt the same "quiet stdout, stderr on failure" rule, to keep automation semantics consistent across commands.
