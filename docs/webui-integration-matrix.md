# WebUI integration requirement matrix (#1848)

This matrix is the canonical handoff for the bundled WebUI release gate. Mock/unit success is never a substitute for the required hardware, installed-artifact, browser, CUDA, Safari/VoiceOver, or native-hidden evidence rows; rows marked `not-run` or `blocked` remain release blockers until root records final evidence.

| Requirement | Automated / committed gate | Evidence status for current branch | Remaining release evidence |
|---|---|---|---|
| One aggregate deterministic WebUI gate | `make verify-webui` runs contract drift, frontend type/lint/unit, Chromium/Firefox/WebKit Playwright scaffold, bundle reproducibility, and scoped Rust WebUI/router tests. | `not-run` in this agent; root must execute after source freeze. | Full workspace `test-fast`, clippy, and fmt are still separate root gates. |
| Installed artifact, not Vite-only | `make verify-webui-installed WEBUI_SERVER_BIN=... WEBUI_CLI_BIN=...` runs `scripts/webui/verify_installed_artifact.py`; CI builds `mlxcel-server`/`mlxcel` once with the GB10 CUDA feature set (`cuda,webui`, `MLX_CUDA_ARCHITECTURES=121`) and verifies the shipped Rust artifact from an isolated CWD/HOME. | `not-run` in this agent; CI job added. | Root/CI must attach `webui-installed-evidence.json` and binary SHA-256/features. |
| Real secured router with fake model/downloader leaves | `make verify-webui-integration-fake` runs the ignored exact Rust test `server::router_server::router_webui_playwright_harness_tests::real_router_browser_harness`, which serves the embedded bundle over the real secured Axum router and invokes `webui/tests/router-real.harness.ts` through the dedicated Playwright config only. | Source implemented; deliberately ignored by ordinary `cargo test`, not counted as pass. | Root must run the exact ignored target after Cargo/browser slot is released and preserve Playwright artifacts. |
| Authentication, prefix, CSP, cache/offline, hostile routes | Installed-artifact smoke and real-router harness assert `/lab` prefix, private bearer auth, CSP without unsafe inline/eval, no external browser requests in the real-router harness, unauthenticated 401, hostile Host/Origin/Fetch Metadata 403 on UI adapters and legacy `/v1/chat/completions`, and missing asset 404. The installed-artifact smoke also disables Python proxies, sanitizes inherited `LLAMA_ARG_*`/`MLXCEL_*`/HF-style environment influence, writes logs to files instead of pipes, and removes private key files in cleanup. | `not-run`; coverage exists in committed gates. | TLS/reverse-proxy authority edge, binary relocation outside `target/`, and OS-level network-denial evidence remain required installed-artifact/proxy rows; this branch does not mark them passed. |
| UI disabled regression | Installed-artifact smoke starts the same binary with `--no-webui` and requires health to stay available while `/webui/` and UI API stay unavailable. | `not-run`; coverage exists in committed gate. | Feature-disabled build remains root/CI evidence, not implied by runtime `--no-webui`; add that row before release. |
| Model-free / no-network guarantee | Installed-artifact smoke uses empty models/cache roots and no checkpoint load; real-router harness fake downloader materializes only owned temp snapshots. | `not-run`; design is fail-closed rather than skip-green. | Root must verify no provider initialization in empty-router runs and retain logs. |
| Download → inspect → load → chat → Stop/drain → unload → remove flow | Real-router browser harness uses UI login/download/load/chat, local Stop with observed active request drain, authenticated unload worker-exit observation, and remove against fake model/downloader leaves. | `not-run`; unit/browser mocks remain separate and insufficient alone. | Root actual-model evidence must cover dense, hybrid/MoE, VLM image, and small public download. |
| State completion and races | Existing contract/unit suites plus `verify-webui-rust` cover bad transitions, operation list, restart/gap fixtures, download cancellation and schema errors; the real-router harness additionally exercises stale revision 409, unknown-field 400 and unsupported terminal-operation cancel on the production secured route. | Existing child evidence is not inherited as a final pass; root rerun required. | Preserve failed attempts and exact commit/binary identity in the final evidence table. |
| Browser visual/accessibility engines | `pnpm --dir webui run browser` keeps the approved Chromium pixel-baseline/Chromium-AX suite, while `pnpm --dir webui run browser:all` uses `playwright.engines.config.ts` for Chromium/Firefox/WebKit DOM, keyboard, layout and axe coverage without invoking Chromium-only CDP or screenshot baselines. CI installs all three engines and runs both paths through `verify-webui-frontend`. | `not-run` here. | Native Safari on macOS 27, manual VoiceOver, real page zoom 200%, and Chromium pixel-baseline review remain distinct evidence rows; WebKit CI/axe does not replace them. |
| Bundle budgets and deterministic assets | `make verify-webui-bundle` still enforces initial JS, total JS, embedded bundle size, deterministic checked-in assets, no source maps/local paths/active external origins, and license notice. | `not-run` here. | Root must record final manifest digest and bundle sizes for the integrated artifact. |
| Performance and large data | Existing frontend tests cover 1,000-entry catalog and 10,000-token transcript parsing; Activity paired visible overhead evidence exists from prior root runs but native hidden did not run; the proposed host gate is a genuine Linux GB10 headed Chrome minimize-to-`document.hidden` run under a real window manager/Xvfb, with geometry/state recorded, not a synthetic visibility override. | Prior evidence is partial, tied to its source commit only. | Root must record cold shell ≤2 s, feedback ≤100 ms excluding backend work, startup RSS/time, no layout shift, and native-hidden overhead. |
| Actual inference/media | No committed mock gate can satisfy actual inference. Accepted Chat source9385 dense/hybrid/VLM evidence may be referenced for those three generation rows because root verified `git diff --name-only 9385bcb7 6976e1df -- src Cargo.toml Cargo.lock build.rs webui/src webui/package.json webui/pnpm-lock.yaml` is empty; this is source-equivalence evidence, not an identical-binary claim. Current #1848 runtime changes are test-only router factory hooks/new test module plus package scripts. | `not-run` for native-hidden/startup/public-download integration on this branch; source-equivalent Chat rows remain source-specific accepted evidence. | Root must still run or reference exact rows for small public download, startup/RSS/performance, and genuine native-hidden Activity overhead with `MLXCEL_REQUIRE_MODELS=1`, recording checkpoint revisions, outputs, lifecycle order, and resource before/after. |
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
python3 scripts/ci/check_webui_evidence.py --self-test
```

Deterministic integration gates:

```bash
make verify-webui
make verify-webui-integration-fake
make verify-webui-installed WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel
cargo test --workspace --profile test-fast --features metal,accelerate
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
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
# Then validate that every required row is present, passed, typed, SHA-identified, and backed by existing artifact files.
MLXCEL_REQUIRE_MODELS=1 make verify-webui-hardware WEBUI_HARDWARE_EVIDENCE=/path/to/hardware-evidence.json

# On the GB10/CUDA host, root builds/runs UI-on/UI-off installed artifact smokes and records artifacts, then validates them.
make verify-webui-cuda WEBUI_CUDA_EVIDENCE=/path/to/cuda-evidence.json
```

`verify-webui-hardware` and `verify-webui-cuda` validate root-supplied evidence JSON and fail closed when evidence is missing, incomplete, non-passing, has duplicate rows, invalid `source_commit`/`binary_sha256`, wrong field types, missing runnable commands, or artifact paths that do not exist; they are not substitutes for the root-run hardware/CUDA commands themselves.
