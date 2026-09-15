# Model library WebUI — PR #1891

Updated: 2026-09-15. Issue #1844, epic #1834. Measured implementation: `22f19d9eb7836c4b33333724294980a3ecfb9168`. Local browser and isolated real-model lifecycle acceptance passed; integrated/manual gates remain pending at this report revision.

## Implementation and operator control

The authenticated Models route uses the shared provider and npmjs `@lablup/ui-common@0.1.0-alpha.19` adapters, not copied component source or a second model/token store. Local search/filter/sort and 25-row pagination preserve opaque selection without remote marketplace queries or implicit loading. The inspector distinguishes architectural support, tested-checkpoint evidence, completeness, disk bytes, unknown memory estimates and revision-matched effective context.

Public downloads require explicit consent and disclose destination/network use. Operations expose progress, cancellation, failure and retry without treating HTTP 202 as completion. Unload observes worker release; disk deletion is separate, server-eligible managed-cache only, displays the checkpoint name plus opaque ID and requires that exact ID. User-provided roots never offer deletion.

Canonical Settings profiles are resolved for the actual explicit load target, including capacity recovery after selection changes. Model profiles replace reusable defaults; request values are copied, and saving does not load or mutate server settings. Explicit CLI/environment values remain authoritative. Capacity confirmation freezes both victim ID and revision; the server validates the pair during admission and queued execution and carries it through operation identity/idempotency. Stale victim errors name `eviction_target_expected_revision`; primary target errors remain `expected_revision`.

## Defects found through acceptance

An actual downloaded SmolLM reached Ready while Use in Chat stayed disabled. The UI gate was correct: catalog projection exposed only pre-load Chat/Completion facts. The backend now adds distinct provider-ready text facts only for supported declared output tasks with loaded-provider evidence and Ready lifecycle. Runtime projection removes stale text facts before rebuilding, preserving pre-load metadata and preventing duplicate or unloaded/error-state positives. Two regressions failed before the correction; all 30 CPU catalog tests passed afterward.

Combined browser testing also exposed a runtime fixture missing Activity's required `slots`. The exact response helper now derives from the canonical snapshot and has lifecycle/strict-schema regressions; no schema or pending=0 assertion was weakened. Root reviewed only the two signed-in Models baselines per Linux/Darwin platform; gallery/login baselines and comparison tolerances were unchanged. This is automated visual review, not native Safari acceptance.

## Validation evidence

| Gate | Observed result |
|---|---|
| Frontend | 182 Vitest tests, 17 Node parser/harness tests, typecheck and lint passed |
| Contract/bundle | 49 canonical fixtures; deterministic two-clean-build verification passed |
| Local workspace all-target Clippy | Metal/Accelerate with default WebUI and with `--no-default-features` both passed |
| Narrow backend | 30 catalog CPU tests and 5 focused eviction/error-field cases passed; independent reviews clear |
| Production build | Both binaries, test-fast, Metal/Accelerate/default WebUI, passed |
| Full local mocked browser suite | 26/26 passed |
| Source CI `34920687721` | All executable checks passed, including cargo-deny/fmt/clippy, WebUI bundle and OpenXLA compile; pin extraction, CUDA sm_70 compile and OpenXLA link were conditionally skipped |
| Isolated real checkpoint journey | Passed; generation deliberately not tested |

The primary checkout retains actual evidence under `target/epic-1834-recovery-20260913/models-22f19-actual/mlxcel-models-owned-nnin2wfs/`: `result.json`, `browser/result.json`, `browser/checkpoint.json` and `browser/responses.json`. The sibling `mlxcel-models-owned-w3x1gprc/` archive retains a second passing same-binary run with corrected scroll-top screenshot capture. Host: Apple M1 Ultra, macOS 27.0 build 26A428. Server binary SHA-256: `00bf95141dfd446f21e142cf6c04bec0faddc33d1bb5d92e38424e3ddd654048`. Checkpoint: `mlx-community/SmolLM-135M-Instruct-4bit`, pinned revision `642e06afe3fab57fd6cc518637c471af0a569e1e`, weights SHA-256 `e91560ee24b13eee6ddeb14879d728a90780053d69057b15ff199ccadfcfe33b`.

The production browser started with an empty isolated store, downloaded and verified that checkpoint, saved a profile without server mutation, explicitly loaded it, observed provider-ready Chat, and read actual context 1024/one slot. It navigated Use in Chat, unloaded through observed worker exit, deleted only the newly downloaded opaque ID, and returned to empty catalog/pending=0. Production CSP and axe checks passed with zero violations, no external browser requests or persisted session key; the owned server exited gracefully with code 0. Existing user checkpoints were untouched. Navigation is not an inference-quality or Chat-generation test.

Bundle digest: `96c48109c78a27b6eae26999603f0c596dc4abe474b24cb1b3af7d0fc065606c`. Compressed asset budgets passed unchanged; Vite's uncompressed main-chunk advisory remains visible.

## Remaining gates

Final local WebUI-on/off workspace all-target Clippy and source CI passed as recorded above; skipped jobs are not counted as executed validation. A new full-workspace Metal test run is not claimed and remains a final #1848 integration gate. Integrated Chat generation and cross-screen acceptance belong to #1845/#1848. Safari/VoiceOver and actual native 200% zoom remain deferred to the consolidated manual session. GPU-timeout root-cause investigation was deferred by the maintainer; this successful bounded lifecycle run does not establish GPU stability or resolve that investigation.
