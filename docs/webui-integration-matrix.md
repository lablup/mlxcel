# WebUI integration requirement matrix (#1848)

This matrix is the canonical handoff for the bundled WebUI release gate. Mock/unit success is never a substitute for the required hardware, installed-artifact, browser, CUDA, Safari/VoiceOver, or native-hidden evidence rows; rows marked `not-run` or `blocked` remain release blockers until root records final evidence.

| Requirement | Automated / committed gate | Evidence status for current branch | Remaining release evidence |
|---|---|---|---|
| One aggregate deterministic WebUI gate | `make verify-webui` runs contract drift, pure verifier helper unit tests, frontend type/lint/unit, Chromium/Firefox/WebKit Playwright scaffold, bundle reproducibility, and scoped Rust WebUI/router tests. | `not-run` in this agent; root must execute after source freeze. | Full workspace `test-fast`, clippy, and fmt are still separate root gates. |
| Installed artifact, not Vite-only | `make verify-webui-installed WEBUI_SERVER_BIN=... WEBUI_CLI_BIN=...` runs `scripts/webui/verify_installed_artifact.py`; CI builds `mlxcel-server`/`mlxcel` once with the GB10 CUDA feature set (`cuda,webui`, `MLX_CUDA_ARCHITECTURES=121`) and verifies the shipped Rust artifact from an isolated CWD/HOME. Separate installed helper targets cover startup/RSS (`verify-webui-startup`), loopback reverse proxy/TLS authority (`verify-webui-reverse-proxy`), and real single-model acceptance (`verify-webui-single-model`). | Root PASS evidence: installed HTTP+TLS/on/off/generated/feature-off in `integration-1848/installed-f09-52103-both-tls-pass`; relocated/source and outbound-network-denial negative controls in `integration-1848/offline-52103-pass`; reverse proxy for both commands in `integration-1848/proxy-fbf7-52103-pass`; startup/RSS in `integration-1848/startup-52103-pass`; single-model four-arm Hello capture in `integration-1848/single-model-52103-pass` with legacy own SIGINT return `-2` noted and no drain proof. | Root/CI must still attach final CI artifact URLs and binary SHA-256/features; pending all-three-engine CI remains separate. |
| Real secured router with fake model/downloader leaves | `make verify-webui-integration-fake` runs the ignored exact Rust test `server::router_server::router_webui_playwright_harness_tests::real_router_browser_harness`, which serves the embedded bundle over the real secured Axum router and invokes `webui/tests/router-real.harness.ts` through the dedicated Playwright config only. | PASS at `ff4463c6` under root-granted narrow run: `MLXCEL_WEBUI_ROUTER_ARTIFACTS=/private/tmp/mlxcel-router-restart-gap-2 cargo test --profile test-fast --features metal,accelerate server::router_server::router_webui_playwright_harness_tests::real_router_browser_harness -- --ignored --exact --nocapture`; evidence `/private/tmp/mlxcel-router-restart-gap-2/run-948b6f65-897b-4c31-a3d3-df88dd5b28ba/router-real-evidence.json`. | Root should archive the raw run directory; this remains fake model/downloader evidence and does not satisfy actual checkpoint rows. |
| Authentication, prefix, CSP, cache/offline, hostile routes | Installed-artifact smoke and real-router harness assert `/lab` prefix, private bearer auth, CSP without unsafe inline/eval, no external browser requests in the real-router harness, unauthenticated 401, hostile Host/Origin/Fetch Metadata 403 on UI adapters and legacy `/v1/chat/completions`, and missing asset 404. The installed-artifact smoke also disables Python proxies, sanitizes inherited `LLAMA_ARG_*`/`MLXCEL_*`/HF-style environment influence, writes logs to files instead of pipes, and removes private key files in cleanup. | Root PASS evidence for TLS/reverse-proxy authority, relocation/source separation, and outbound-denial negative controls is in the installed/proxy/offline archives named above. | Preserve exact artifact paths and build provenance in the final PR body; reverse proxy/TLS evidence is installed-artifact evidence, not Vite evidence. |
| UI disabled regression | Installed-artifact smoke starts the same binary with `--no-webui` and requires health to stay available while `/webui/` and UI API stay unavailable. | `not-run`; coverage exists in committed gate. | Feature-disabled build remains root/CI evidence, not implied by runtime `--no-webui`; add that row before release. |
| Model-free / no-network guarantee | Installed-artifact smoke uses empty models/cache roots and no checkpoint load; real-router harness fake downloader materializes only owned temp snapshots. | `not-run`; design is fail-closed rather than skip-green. | Root must verify no provider initialization in empty-router runs and retain logs. |
| Download → inspect → load → chat → Stop/drain → unload → remove flow | Real-router browser harness uses UI login/download/load/chat, local Stop with observed active request drain, authenticated unload worker-exit observation, and remove against fake model/downloader leaves. | `not-run`; unit/browser mocks remain separate and insufficient alone. | Root actual-model evidence must cover dense, hybrid/MoE, VLM image, and small public download. |
| State completion and races | Existing contract/unit suites plus `verify-webui-rust` cover bad transitions, operation list, restart/gap fixtures, download cancellation and schema errors; the real-router harness additionally exercises empty-library rescan, real same-port fresh server restart SSE, event-ring gap SSE, stale revision 409, unknown-field 400 and unsupported terminal-operation cancel on the production secured route. | PASS for the real-router fake harness at `ff4463c6`; unit/aggregate rerun remains root-owned. | Preserve failed attempts and exact commit/binary identity in the final evidence table. |
| Browser visual/accessibility engines | `pnpm --dir webui run browser` keeps the approved Chromium pixel-baseline/Chromium-AX suite, while `pnpm --dir webui run browser:all` uses `playwright.engines.config.ts` for Chromium/Firefox/WebKit DOM, keyboard, layout and axe coverage without invoking Chromium-only CDP or screenshot baselines. CI installs all three engines and runs both paths through `verify-webui-frontend`. | Root PASS on mock/Vite Chromium evidence: 28 default Chromium tests, 7 cross-surface/performance Chromium tests in `integration-1848/engines-f873-quiet-pass`, `chromium-prefix-default-pass`, and latest true DOM+paint timer evidence in `integration-1848/engines-browser-only-timing-pass`; 311 Vitest, 17 Node, 49 contract tests, typecheck and lint also passed. | **Native Safari on the installed artifact: USER-REPORTED PASS 2026-09-16**, against `mlxcel-server --webui` serving the embedded bundle over a loopback-forwarded session, not a Vite preview. Scope confirmed by the maintainer, and only this scope: screen and layout reviewed by eye across Models, Chat, Activity and Settings; keyboard Tab and Shift-Tab order with visible focus and Escape closing dialogs and sheets with focus restored to the opener; Safari native page zoom at 200% with no horizontal page scroll at 390 px, and readability retained with increased contrast. **Manual VoiceOver remains NOT RUN** and is a distinct row: it was not part of this pass and no other gate substitutes for it, since the axe and DOM checks in CI assert markup rather than what a screen reader announces. All-three-engine CI is covered by the `WebUI bundle` job. This supersedes the 2026-09-13 report only for the surfaces above; that one covered the design-system candidate under a Vite preview at `#gallery`. |
| Bundle budgets and deterministic assets | `make verify-webui-bundle` still enforces initial JS, total JS, embedded bundle size, deterministic checked-in assets, no source maps/local paths/active external origins, and license notice. | `not-run` here. | Root must record final manifest digest and bundle sizes for the integrated artifact. |
| Performance and large data | Existing frontend tests cover 1,000-entry catalog and 10,000-token transcript parsing; `make verify-webui-startup` records installed empty-router startup/RSS; `make verify-webui-activity-performance` runs the real Activity overhead/native-hidden harness with a checkpoint, including the genuine Linux GB10 headed Chrome minimize-to-`document.hidden` path when invoked with the approved host/display setup. | Root PASS evidence: installed startup/RSS `integration-1848/startup-52103-pass`; mock/Vite Chromium true DOM+paint timing `integration-1848/engines-browser-only-timing-pass` recorded cold 59.8 ms, navigation 25.8 ms, 1,000-entry search 25.9 ms, 10,000-token new conversation 32.0 ms, CLS 0. Prior visible Activity evidence is partial, tied to its source commit only. The GB10 job now runs the headed visible-mode observation overhead on every change (`--perf-mode visible-only-headed`) and summarizes it with `--expect-hidden-native not-run`, so the summary declares `status: incomplete` and `hidden_native: not-run` rather than reading as a complete acceptance. | **Native hidden acceptance: user-deferred 2026-09-16.** Scope of the deferral is the minimize-to-`document.hidden` measurement only; the headed visible-mode overhead is measured in CI and every numeric budget still applies to it. Reason, established on the GB10 runner and not inferred: Xvfb and openbox give headed Chromium a real 700x900 content area, the browser reports the window state as `minimized`, and `document.visibilityState` stays `visible` for 15 seconds anyway; foregrounding a sibling tab in the same context also leaves `document.hidden` false; both visibility-suppressing launch arguments (`--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`) were removed and their removal verified by reading a live browser process's arguments. The unverified step in that chain is whether Playwright's sibling page is a tab in the same window or a separate window, which would make the tab datum uninformative. This is a property of the automation browser, not of the runner or the product. The visibility backoff the row exists to protect stays covered by two unit tests in `webui/src/state/sync.test.ts` and the `visibilitychange` case in `webui/tests/browser.spec.ts`. Lifting the deferral means running `--perf-mode full` again, which needs a browser that propagates a real window-visibility change. |
| Actual inference/media | No committed mock gate can satisfy actual inference. Accepted Chat source9385 dense/hybrid/VLM evidence may be referenced only for the specific dense/hybrid/VLM generation paths whose source stayed identical through baseline6976 (`git diff --name-only 9385bcb7 6976e1df -- src Cargo.toml Cargo.lock build.rs webui/src webui/package.json webui/pnpm-lock.yaml` was empty). This is source-equivalence evidence for those paths, not an identical-binary or blanket current-branch runtime/UI claim: #1848 also includes production URL-prefix/public-health changes (`d6b7`, `c629`) plus test harness changes. | Root single-model installed four-arm Hello capture PASS in `integration-1848/single-model-52103-pass`, with semantic review scope limited to that small real checkpoint prompt and legacy own SIGINT `-2` noted; source-equivalent Chat dense/hybrid/VLM rows remain accepted only with the limitation above. | Root must still run or reference exact rows for small public download, VLM image, and genuine native-hidden Activity overhead with `MLXCEL_REQUIRE_MODELS=1`, recording checkpoint revisions, outputs, lifecycle order, and resource before/after. |
| CUDA host smoke | `make verify-webui-cuda WEBUI_CUDA_EVIDENCE=...` validates the root-run CUDA evidence JSON for UI-on, UI-off and installed-artifact rows. | `not-run`; not satisfied by macOS. | GB10/CUDA host must run UI-on/UI-off server smoke and record result; historic outage waivers are not blanket permission. |
| Settings / Chat / Activity cross-surface integration | Existing merged child gates provide source-specific evidence; this matrix points to final rerun rather than inheriting a fictional one-run pass. | Prior root notes: Models source22f19 lifecycle/profile/navigation/removal passed with generationTested=false; Chat source9385 dense/hybrid/VLM semantics, Stop active→0, CSP, unload passed; Activity visible paired overhead partial passed while native hidden did not run. | Final integrated artifact must rerun or explicitly reference accepted prior evidence by commit/source and limitation. Chat composer caps new turns at 100 while history import/storage permits 200 per conversation; Stop refreshes aggregate Activity, not a per-request cancellation receipt. |
| Compatibility endpoints / CLI manifests | `verify-webui-rust`, installed-artifact smoke, `verify-llama-compat`, and feature-off root build together cover compatibility surfaces. | `not-run` here. | Root must include `mlxcel-server` and `mlxcel serve`, UI-on/UI-off, single-model and empty-router, generated/explicit key, prefix, TLS/proxy where supported. |
| Security and privacy | Contract/frontend/server tests cover token redaction, no token persistence, history/image consent, body limits, XSS/Markdown sanitization and UI adapter errors; real-router/installed-artifact gates add production CSP/auth/legacy route attacks. | `not-run` here; no enabled placeholder is marked passed. | Security PR findings must be represented by exact test/evidence rows before epic closure. |
| Final publication | PR body and final report must link this matrix and list every `not-run`, `blocked`, or user-deferred item. | This branch owns the matrix; #1849 owns broader user guide/docs. | Parent epic must not close until required gates are verified or explicitly deferred by the user with concrete dates/scope. |

## Evidence row schema

For each final evidence row record: requirement id, source commit, binary SHA-256/features, UI manifest source and bundle digest, OS/hardware/browser, checkpoint revision when relevant, command or harness version, result (`passed`, `failed`, `not-run`, `blocked`, or explicit user deferral), raw evidence path, human visual/manual reviewer when required, owner, and residual limitation.

## Required root commands after source freeze

Pure validator self-test:

```bash
make verify-webui-helper-tests
```

Deterministic integration gates:

```bash
make verify-webui
make verify-webui-integration-fake
make verify-webui-installed WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_FEATURE_OFF_SERVER_BIN=/path/to/feature-off/mlxcel-server WEBUI_FEATURE_OFF_CLI_BIN=/path/to/feature-off/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_FEATURE_OFF_BUILD_SOURCE_HEAD=<40-hex> WEBUI_INSTALLED_EVIDENCE=/path/to/webui-installed-evidence.json
make verify-webui-startup WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_BUILD_FEATURES=metal,accelerate,webui WEBUI_STARTUP_EVIDENCE=/path/to/startup-evidence.json
make verify-webui-reverse-proxy WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_REVERSE_PROXY_EVIDENCE=/path/to/reverse-proxy-evidence.json
make verify-test
make verify-clippy
make verify-fmt
```

Root-owned actual-model and CUDA evidence commands, using existing harnesses rather than validator-only declarations:

```bash
# For each accepted dense, hybrid/MoE, and VLM checkpoint row, root starts an isolated bundled server with the model already loaded, then runs the existing real chat browser harness and stores its JSON/PNG artifacts referenced by hardware-evidence.json.
MLXCEL_CHAT_REAL_ISOLATED=1 \
MLXCEL_CHAT_REAL_URL=http://127.0.0.1:<port>/<prefix>/webui/ \
MLXCEL_CHAT_REAL_KEY_FILE=/path/to/private-0600-key \
MLXCEL_CHAT_REAL_MODEL_ID=mdl_... \
MLXCEL_CHAT_REAL_IMAGE_FILE=/path/to/image.png \
  pnpm --dir webui exec playwright test --config playwright.chat-real.config.ts

# For the small public download and native-hidden/startup rows, root runs the approved actual-router/download and Activity native-hidden/startup harnesses on the scheduled host and records raw JSON/log/PNG artifacts.
MLXCEL_REQUIRE_MODELS=1 \
WEBUI_SERVER_BIN=/path/to/mlxcel-server \
WEBUI_CLI_BIN=/path/to/mlxcel \
WEBUI_SINGLE_MODEL=/path/to/checkpoint \
WEBUI_BUILD_SOURCE_HEAD=<40-hex> \
WEBUI_BUILD_FEATURES=metal,accelerate,webui \
WEBUI_SINGLE_MODEL_OUTPUT_DIR=/path/to/single-model-output \
WEBUI_SINGLE_MODEL_STARTUP_REPORT=/path/to/startup-evidence.json \
  make verify-webui-single-model

MLXCEL_REQUIRE_MODELS=1 \
WEBUI_SERVER_BIN=/path/to/mlxcel-server \
WEBUI_ACTIVITY_MODEL=/path/to/checkpoint \
WEBUI_ACTIVITY_CHECKPOINT_REVISION=<checkpoint-revision> \
WEBUI_BUILD_SOURCE_HEAD=<40-hex> \
WEBUI_BUILD_FEATURES=metal,accelerate,webui \
WEBUI_ACTIVITY_EVIDENCE=/path/to/activity-evidence.json \
  make verify-webui-activity-performance

# Then validate that every required row is present, passed, typed, SHA-identified, and backed by existing artifact files.
MLXCEL_REQUIRE_MODELS=1 make verify-webui-hardware WEBUI_HARDWARE_EVIDENCE=/path/to/hardware-evidence.json

# On the GB10/CUDA host, root builds/runs UI-on/UI-off installed artifact smokes and records artifacts, then validates them.
make verify-webui-cuda WEBUI_CUDA_EVIDENCE=/path/to/cuda-evidence.json
```

`verify-webui-hardware` and `verify-webui-cuda` validate root-supplied evidence JSON and fail closed when evidence is missing, incomplete, non-passing, has duplicate rows, invalid `source_commit`/`binary_sha256`, wrong field types, missing runnable commands, or artifact paths that do not exist; they are not substitutes for the root-run hardware/CUDA commands themselves.
