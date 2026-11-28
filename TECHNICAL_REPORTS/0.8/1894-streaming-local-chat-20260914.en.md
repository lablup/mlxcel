# Technical report: PR #1894 — bounded local streaming chat

**Updated:** 2026-09-15 · **Status:** Implementation and scoped real acceptance complete; final epic/manual gates remain · **Risk:** Medium

## Integrated behavior

Chat uses the existing authenticated `/v1/chat/completions?autoload=false` transport and published npm `@lablup/ui-common@0.1.0-alpha.19` adapters. Each turn freezes opaque/inference identity, revision and validated numerical parameters. Canonical memory-only Settings defaults are combined with explicit single-next-turn overrides: blank inherits Settings, absence inherits the server, invalid drafts prevent sending, and accepted requests consume the override. Model selection changes observation and the next turn, never the running inference stream. Logout/session fences and the reader-abort/EOF race fix remain integrated with Settings, Activity and Models.

The IME-safe composer preserves partial cancelled/interrupted/error output, separates collapsed reasoning and non-executing tool inspection, and distinguishes reported usage from client timing estimates. Bounded escaped Markdown blocks raw HTML and remote images; highlighting is lazy and local. Image admission requires provider-confirmed vision capability and validates the complete UTF-8/base64 request against actual mode-specific server limits and the separate 16 MiB browser ceiling, not the administrative 2 MiB limit.

History remains memory-only unless explicitly enabled in bounded, versioned IndexedDB. Imports, quotas, Clear All and late callbacks are guarded; credentials are never persisted and image persistence requires separate consent. System prompts belong to conversations, not shared Settings.

## Verification

Measured source: `9385bcb7ef64386180ea62b3324540eef713a4e2`; server binary SHA256: `b32df976d255ca69d420368dd96130bb93748c5ade02d9f4596abbf580a26058`. Host: Apple M1 Ultra, macOS 27.0 build 26A428. Both test-fast executables built; features were Metal, Accelerate and default WebUI.

- Passed: 266 Vitest tests/34 files, 17 Node helper tests, type/lint, 49 strict contract fixtures, 18 constructed browser response shapes, deterministic bundle verification and 28 local browser tests. Initial JS gzip is 145,637 bytes; the raw-chunk advisory was not suppressed.
- Workspace all-target Clippy passed with WebUI enabled and disabled on `988493fc`; all Rust source and every served asset path/SHA256 remained identical at `9385bcb7`, except generated manifest metadata and test/harness changes. This is not a claim of identical executable binaries.
- Source CI [run 34922431893](https://github.com/lablup/mlxcel/actions/runs/34922431893) passed every executed job, including OpenXLA feature compile. MLX pin extraction, OpenXLA feature link and CUDA sm_70 compile were conditionally skipped, not passed runtime checks.

An isolated secured router with one loaded model, context 8192, parallelism 1 and autoload disabled ran the three checkpoints sequentially. Llama and Granite each answered “Hello.” correctly. Qwen3-VL correctly identified the red square, blue circle and green triangle in the inspected local image fixture. Each observed positive active requests before UI Stop, retained cancelled state, returned to zero without retry, explicitly unloaded with `worker_exit_observed`, and the server exited 0. Actual served CSP, completed/cancelled axe and layout checks passed with no violations or external requests. Selected Llama/Qwen screenshots were visually reviewed without visible clipping; this is not native accessibility acceptance.

| Checkpoint | Cached config revision | `model.safetensors` SHA256 |
|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit | `241a666dad6cb93c8ff213d39a7f34a36bf26db4` | `08eff50fa4eb1fe499ead69eaeb4c5177ae82f218da0b55e0e31d7a989debb92` |
| Granite-4.0-H-Tiny-4bit | `02c8783da0cb171942e30a7dde8acce11fe82e0d` | `fc076be2631a2a6b6ca15cbe742d034f9a2217a73ab1238c351b0358085738ea` |
| Qwen3-VL-2B-Instruct-4bit | Unknown | `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb` |

Cached config revisions are recorded provenance, not independently verified whole-checkpoint revisions. Qwen's cached config revision was unavailable and is not inferred. Maintainer-held `chat-9385-actual/result.json` preserves the original capture status; the separate `acceptance-review.json` records the subsequent semantic/visual PASS.

## Corrected instrumentation and remaining scope

The initial Llama acceptance at `988493fc` failed because screenshot caret cleanup left empty `style=""` attributes. The exact installed Playwright function reproduced this in jsdom; the initial axe hypothesis was not confirmed. The helper now exempts only trim-empty attributes while rejecting nonempty CSS, including invalid declaration text. Explicit private mode-0600 evidence files preserve replies before axe and Stop observations before later layout assertions. Product code and CSP strictness were unchanged; the failed attempt was not counted as complete acceptance.

Independent implementation/security/performance/accessibility reviews found no remaining scoped blocker. Broad final workspace Metal verification belongs to #1848; this report does not claim it passed. GPU timeout root-cause investigation is separately deferred. Safari/VoiceOver and actual native 200% page zoom remain explicitly deferred to the integrated manual session, not passed. Reports-only follow-up commits do not change the measured implementation.
