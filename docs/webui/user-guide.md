# Using the bundled WebUI

[한국어](user-guide.ko.md) | [WebUI documentation index](README.md)

`mlxcel-server --webui` and `mlxcel serve --webui` serve a browser interface from the server binary itself: a model library (Models), a local chat client (Chat), runtime observations (Activity), and scoped settings (Settings). The interface is compiled into the binary, so it needs no Node.js, no CDN, and no network access beyond the server you started. It is opt-in and still marked experimental: without `--webui` the server exposes only its HTTP APIs, exactly as before.

The WebUI is an administrator surface. Anyone holding its key can load, unload, download, and delete models on that server, so treat the key like a root password for the model server.

## Quickstart

All examples below bind the default loopback address (`127.0.0.1:8080`) unless they say otherwise. `mlxcel serve` accepts the same flags as `mlxcel-server`; every `mlxcel-server --webui ...` line works unchanged as `mlxcel serve --webui ...`.

### Local session with a generated key

Run the server from an interactive terminal:

```bash
mlxcel-server --webui --models-dir ~/models/mlx
```

Because no API key is configured, the server generates a restart-local session key and prints it once to the controlling terminal:

```text
mlxcel WebUI session key (shown once): mlxcel_webui_<generated>
```

Open `http://127.0.0.1:8080/webui/`, paste the key into **Session key**, and press **Connect**. The key changes on every restart. It is never written to a file, a log line, or a URL.

### Key file (services, scripts, and non-interactive shells)

A server started without a terminal (a service manager, `nohup`, a CI job) cannot show a generated key, so it needs an explicit one:

```bash
mkdir -p ~/.config/mlxcel
(umask 077 && openssl rand -hex 32 > ~/.config/mlxcel/webui.key)
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx
```

The file holds one key per line; lines starting with `#` are comments. Paste the key from the file into the login form. The same key also authorizes ordinary API clients, so keep the file readable only by you. `--api-key <KEY>` works too, but it puts the key on the command line, where other local users can read it from the process list.

### Choosing what the library shows

The library lists checkpoints from three sources, in increasing precedence when two of them use the same name: the managed model store, the `--models-dir` directory, and sections of a `--models-preset` INI file.

| Command | Library contents |
|---|---|
| `mlxcel-server --webui --models-dir ~/models/mlx` | Every direct subdirectory of `~/models/mlx` that holds a `config.json`, plus the managed store. |
| `mlxcel-server --webui` | Only the managed store, `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models`. A store that does not exist yet is an empty library, not an error. |
| `mlxcel-server --webui --model-store-root /data/mlxcel-models` | The managed store at an explicit root. **Add model** downloads land here. An explicit root that cannot be read stops startup (see [Troubleshooting](#troubleshooting)). |
| `mlxcel-server --webui -m ~/models/mlx/qwen3-0.6b-4bit` | Single-model mode: that one checkpoint, loaded at startup. Load, Unload, Add model, and Delete are read-only; the model inspector lists the server's reason ("single-model mode does not own the router lifecycle pool" or "... the router managed cache") and the instruction to restart without `-m`/`--model`. |

With `--models-dir` or the managed store, the server starts with no model loaded and never loads one implicitly: browsing, filtering, selecting a model, opening Activity, and sending a chat request to an unloaded model all leave it unloaded. A model is loaded only when you press **Load**. The two exceptions are explicit: single-model mode loads its `-m` checkpoint at startup, and a preset section with `load-on-startup = true` starts loading before the server begins serving.

### Offline use

```bash
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx --offline
```

`--offline` (or `LLAMA_ARG_OFFLINE`) keeps the server off the network. Local checkpoints load and chat as usual; **Add model** is disabled with the reason "offline mode is enabled". The WebUI itself never needs the network: all of its scripts, styles, and license text are embedded in the binary.

### Live settings, metrics, and slots

The WebUI does not switch on any optional server endpoint. Start the server with the ones you want to see:

```bash
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key --models-dir ~/models/mlx --settings --props --metrics
```

`--settings` enables live sampling changes on the Settings **Model** tab, `--props` the observed context on the **Server** tab, and `--metrics` the request counters on Activity. Slot occupancy follows the existing `--slots` default (on). Without these flags the matching panels say that the endpoint is disabled or the value is unavailable; they never show zeros in its place.

### Remote access

A WebUI that is reachable from another machine must use an explicit key and TLS. There are two supported shapes.

**TLS on the server itself.** Bind the address the browser will use, since the WebUI only accepts requests whose `Host` matches the bound address:

```bash
mlxcel-server --webui --host 192.0.2.10 --port 8443 \
  --ssl-cert-file /etc/mlxcel/tls/cert.pem --ssl-key-file /etc/mlxcel/tls/key.pem \
  --api-key-file /etc/mlxcel/webui.key --models-dir /srv/models
```

Browse to `https://192.0.2.10:8443/webui/`. Binding `0.0.0.0` or another address the browser does not use makes every UI request fail its host check.

**Loopback backend behind a TLS reverse proxy.** Keep the server on loopback and let the proxy terminate TLS:

```bash
mlxcel-server --webui --host 127.0.0.1 --port 8080 --api-prefix /lab \
  --api-key-file /etc/mlxcel/webui.key --models-dir /srv/models
```

The proxy must forward `/lab/` to `http://127.0.0.1:8080/lab/`, send `Host: 127.0.0.1:8080` upstream, replace a browser `Origin` header with `http://127.0.0.1:8080` (and add none when the browser sent none), pass `Authorization` through unchanged, not buffer `text/event-stream` responses, and validate the public host and origin itself, because the server can only check the values the proxy sends. The WebUI is then at `https://<public host>/lab/webui/`. `scripts/webui/verify_reverse_proxy.py` (`make verify-webui-reverse-proxy`) is the tested reference for exactly this header contract.

`--api-prefix` moves the API and the WebUI under that path (`/lab/webui/`, `/lab/ui-api/v1/...`), while the health check at `/` stays at the root and needs no key.

## Signing in and keys

- The key you paste is held in the page's memory only. It is not stored in cookies, local storage, IndexedDB, or the URL, so a reload or a new tab asks for it again, and **Clear key** forgets it immediately.
- Every UI API request sends the key as an `Authorization: Bearer` header. A request without it gets `401 Unauthorized`; the shell page and its static files are public, the data behind them is not.
- A key the server rejects shows "The session key was rejected. Copy the latest key printed by the local server terminal." A generated key stops working when the server restarts, so copy the new one.
- Never put the key in a link, a screenshot, or a bug report. No example in this documentation contains a real key.

## Models: supported, downloaded, and loaded

Models keeps separate facts that are easy to conflate:

- **Downloaded / complete** describes files on disk. A complete checkpoint has the expected file layout; that is not proof that it loads.
- **Supported** means mlxcel's architecture registry recognizes the checkpoint, and **runnable** means that architecture is supported on this server's GPU backend (Metal, CUDA, or ROCm). Neither is a measured inference result.
- **Loaded / Ready** is the live state of the server's model worker. Only a Ready model can answer chat requests.

Selecting a model in Models or Chat only selects it. **Load** starts a server operation: the button returning means the request was accepted, not that the model is ready, and the badge moves through Loading to Ready (or Failed with a reason). When `--models-max` is reached, Load asks which loaded model to evict and evicts only the one you confirmed.

**Unload** stops new requests to the model, waits for the active ones to finish (draining), and only then releases the worker. Its memory counts as in use until the server observes the worker exit. Closing a browser tab never unloads a model.

**Delete** removes a checkpoint's files from the managed store. It is a different action from Unload: Unload keeps the files, Delete removes them. Only managed-store entries can be deleted, never `--models-dir` directories or preset paths, never a model that is loading, loaded, downloading, or draining, and only after you type the model's opaque ID (shown in the dialog, starting with `mdl_`), not its name, in the confirmation.

**Add model** downloads a public Hugging Face repository by its `owner/name` id into the managed store. URLs and local paths are refused before any request is made. Downloads are anonymous: the server sends no Hugging Face token, even when `HF_TOKEN` is set, so a private or gated repository fails at its metadata check, before any file is written. Private Hub authentication is not part of this release. A download can be cancelled; the cancellation finishes only once the downloader has stopped, and a partial download is never shown as complete. Loads and unloads cannot be cancelled once accepted.

Capabilities gate actions. The chat picker lists only chat-capable models, image attachment appears only after a loaded model confirms vision input, and other tasks (embeddings, reranking, audio, image generation) are listed as capabilities with API examples rather than as chat buttons.

## Chat, settings, and privacy

Settings separates values by how long they live and who they affect:

| Where | Scope | Lifetime |
|---|---|---|
| Settings, **Appearance** | This browser | Browser preferences (theme, glass, motion, contrast). Never sent to the server. |
| Settings, **Requests**, and Chat's **Parameters for next turn** | Your future requests | Memory only. Sent with each request; they never change the loaded model or other users. |
| Settings, **Model**, live values | The selected loaded model, for everyone | Applied on the server through `/settings` (needs `--settings`). Affects newly admitted requests. A partial apply reports which fields were rejected. |
| Settings, **Model**, next-load profile | The next Load of that model | Saved in this browser. Only `ctx_size`, `n_parallel`, and the KV cache mode. Explicit server command-line flags win over it. |
| Settings, **Server** | The running server | Read-only startup values; changing them needs a restart. |

Chat history is kept in memory by default and disappears on reload or sign-out. **Save conversations on this device (IndexedDB)**, under **Local history and privacy** in the Chat settings drawer, is an opt-in that saves conversations in this browser's IndexedDB for this origin; it has to be re-enabled after a reload before saved data is read. Turning saving off stops new writes but keeps old data; **Clear All** deletes saved and in-memory conversations. **Export JSON** writes a file you choose to save; import validates the file and asks before replacing current history. Images are saved only with separate consent. Keys and system prompts are never written by Settings.

The WebUI sends no telemetry and contacts no third-party server: the only requests it makes go to the server that served it. Model output is rendered as untrusted text (no raw HTML, no remote images). Tool calls a model emits are displayed and can be copied, but the WebUI and the server never execute them. **Stop** ends the stream and keeps the partial answer; it is not a guarantee that generation already stopped on the server.

## Not in this release

These are documented limits, not hidden buttons: private Hugging Face authentication, native desktop packaging, tool or MCP execution, and dedicated pages for non-chat tasks. A VoiceOver screen-reader pass on the installed artifact has not been run; keyboard, zoom, and visual review in Safari has. The [integration matrix](../webui-integration-matrix.md) records each open item with its scope.

## Troubleshooting

Each message below is printed verbatim by the server, prefixed with `Error: `, when startup is refused.

| Symptom | Message | Fix |
|---|---|---|
| No key and no interactive terminal | `--webui without --api-key is only allowed on an interactive loopback terminal so the generated session key can be shown once` | Pass `--api-key-file`, or start the server from a terminal. |
| Listening on a Unix socket | `--webui requires a TCP listener; use a loopback TCP listener behind a reverse proxy instead of a Unix socket` | Use `--host 127.0.0.1` and a reverse proxy. |
| Non-loopback address without key or TLS | `non-loopback --webui requires an explicit API key and TLS; alternatively bind the WebUI backend to loopback behind a TLS reverse proxy` | Add `--api-key-file` plus `--ssl-cert-file`/`--ssl-key-file`, or bind loopback behind a proxy. |
| `--models-dir` does not exist or cannot be read | `--models-dir /srv/missing: cannot open models directory: No such file or directory (os error 2)` | Point `--models-dir` at an existing directory. |
| Explicit store root does not exist or cannot be read | `--model-store-root /srv/missing: explicit WebUI/router cache root must be a readable directory: No such file or directory (os error 2)` | Create the directory or fix the path; `MLXCEL_MODELS_DIR` gets the same check. |
| Binary built without the WebUI | `` --webui requires a binary built with the Cargo `webui` feature; rebuild mlxcel with bundled WebUI support `` | Use an official build or a build without `--no-default-features`. |
| A llama-server UI flag | `--ui-config is not supported: it configures llama-server's web UI; mlxcel's bundled WebUI does not consume upstream UI JSON.` | Remove the flag. `--path`, `--tools`, `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`, `--ui-mcp-proxy`, and `--agent` are refused the same way. |

In the browser:

| What you see | Meaning |
|---|---|
| "Authentication required" on every page | The page has no key in memory. Paste the key again. |
| "The session key was rejected. ..." | Wrong key, or a generated key from before a restart. |
| "Could not reach the local WebUI API. Check that mlxcel-server is still running with --webui." | The server stopped or cannot be reached. |
| "The key authenticated, but this UI endpoint is forbidden for the current server session." | The server refused the request's `Host`, `Origin`, or fetch metadata: the page was opened through an address the server did not bind, or a reverse proxy did not rewrite `Host` and `Origin` as described above. |
| A metric reads "unknown" or "not yet measured" | The value is not available (missing `--metrics`, no loaded model, no sample yet). It is never shown as zero. |

A request to a UI endpoint without a key returns HTTP 401 with this body:

```json
{"error":{"code":"unauthorized","message":"authentication is required","retryable":false},"request_id":"req_<id>"}
```

A UI request that reaches the server under a host name or address it did not bind (for example a server bound to `0.0.0.0` and opened as `https://192.0.2.10:8443/webui/`) returns HTTP 403:

```json
{"error":{"code":"forbidden","message":"Host header is not allowed","retryable":false},"request_id":"req_<id>"}
```

## Release evidence

The [integration requirement matrix](../webui-integration-matrix.md) is the canonical record of what was verified for the bundled WebUI and how. Do not copy its numbers into other documents; link the row instead.

- **Bundle source hash and size:** `src/webui/assets/mlxcel-webui-manifest.json` records the source digest, per-file SHA-256, and gzip sizes against the budgets in [Developing the bundled WebUI](bundling.md); the matrix row "Bundle budgets and deterministic assets" tracks the release reading.
- **Offline moved-binary check:** the matrix row "Installed artifact, not Vite-only" holds the relocated-binary and outbound-network-denial evidence (`integration-1848/offline-52103-pass`).
- **Release feature inclusion:** the `webui` Cargo feature is on by default, so `--features metal,accelerate` and `--features cuda` release builds embed the bundle; the same row covers the installed artifact with the feature on and off.
- **Runtime dependencies:** embedding the WebUI does not make the binary self-contained. The platform GPU runtime is still required: the `mlx.metallib` artifact the macOS release packages, or the CUDA and cuDNN runtime libraries on Linux, as [installation.md](../installation.md) describes.
