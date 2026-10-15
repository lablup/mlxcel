# mlxcel WebUI

The WebUI is a browser interface compiled into `mlxcel-server` and `mlxcel serve`. Start it with `--webui`, open `http://127.0.0.1:8080/webui/`, and sign in with the session key the server prints once (or the key from your `--api-key-file`). It manages the local model library (Models), chats with loaded models (Chat), shows runtime observations (Activity), and separates browser, request, model, and server settings (Settings). It is opt-in and experimental.

```bash
mlxcel-server --webui --models-dir ~/models/mlx
```

## For users

| Document | English | 한국어 |
|---|---|---|
| Quickstarts, sign-in and keys, remote access, models, settings and privacy, troubleshooting | [user-guide.md](user-guide.md) | [user-guide.ko.md](user-guide.ko.md) |
| What was verified for the release, and what is deferred | [../webui-integration-matrix.md](../webui-integration-matrix.md) | |
| How `--webui` relates to llama-server's `--ui` family | [../llama-server-compat.md](../llama-server-compat.md#web-ui-tools-mcp-cors-proxy-and-agent-mode-1435-1838) | |

## For contributors

| Document | English | 한국어 |
|---|---|---|
| Repository layout, contract-first workflow, frontend build, feature graph | [contributing.md](contributing.md) | [contributing.ko.md](contributing.ko.md) |
| Bundle pipeline, Cargo feature, static router | [bundling.md](bundling.md) | [bundling.ko.md](bundling.ko.md) |
| UI API contract, authentication model, limits | [architecture.md](architecture.md), [api.yaml](api.yaml) | |
| Lifecycle, download, and event replay state machines | [state-machines.md](state-machines.md) | |
| Screens, strings, keyboard, and layout floor | [ux-contract.md](ux-contract.md) | |
| Design tokens, themes, and shared components | [design-system.md](design-system.md), [ui-common.md](ui-common.md) | |
| Models catalog projection | [catalog.md](catalog.md) | [catalog.ko.md](catalog.ko.md) |
| Chat, Settings, Activity, and runtime observation | [chat.md](chat.md), [settings.md](settings.md), [activity.md](activity.md), [runtime.md](runtime.md) | |

The contract and screen documents describe the implementation for people changing it; the user guide is the operator-facing summary and links back to them rather than repeating them.
