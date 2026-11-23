# PR #3271: Server-side MCP tool loop on /v1/responses

**Date**: 2026-11-23
**Status**: PR open as a draft (closes #1457). Unit and route suites pass against a scripted stdio MCP server; a two-round tool task ran on `qwen3.5-4b-4bit` on an M3 Ultra in streaming, non-streaming, and approval modes
**Risk Level**: High (the server now spawns child processes that inherit its privileges)

## Executive Summary

Before this PR, `/v1/responses` answered an `mcp` tool with 400 and `--mcp-servers-config` / `--mcp-servers-json` failed startup pointing at #1457. With the new `--mcp-tools` opt-in, `mlxcel-server` starts the stdio MCP servers named in a Cursor-compatible `mcpServers` document and runs the tool loop itself (model turn, tool calls, results, next turn) until the model answers or a bound stops it. Clients get the standard `mcp_list_tools`, `mcp_call`, and `mcp_approval_request` output items and their streaming events. `/v1/chat/completions` is unchanged.

The feature is off by default and guarded at startup: it refuses a non-loopback listener without API keys, router, image-only, and distributed launches, and an inherited `LLAMA_ARG_MCP_SERVERS_*` variable cannot spawn anything on its own.

## 1. Problem Statement

### 1.1 Background

#1435 and #1494 classified llama-server b10621's MCP flags as not applicable: upstream runs its agent loop in the browser UI, and mlxcel's bundled WebUI has no MCP client. #1457 kept the product-level alternative: run the loop on the server and expose it through the OpenAI Responses built-in tool contract mlxcel already serves, so any OpenAI-compatible client gets tool execution without a bespoke UI.

### 1.2 Existing Issues

- `UNSUPPORTED_TOOL_TYPES` in `src/server/responses_translator.rs` listed `mcp`, so the request failed with 400.
- `UiCompatArgs::ensure_supported` refused both MCP flags at startup.
- Nothing in `src/` spoke JSON-RPC to an MCP server, managed a child process, or produced `mcp_*` items.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A network-reachable unauthenticated server runs tools for anyone | High | Prevented by the startup gate |
| A hung or chatty child blocks a worker thread or exhausts memory | High | Bounded by timeouts, byte caps, and admission control |
| A client forges approval for arguments the model did not produce | High | Prevented: only server-stored arguments run, once |

## 2. Technical Review

### 2.1 Security

- [x] Opt-in: `--mcp-tools` has no environment variable. The server-set flags without it fail startup, including when they come only from `LLAMA_ARG_MCP_SERVERS_*`.
- [x] Exposure: `validate_mcp_startup` refuses a non-loopback TCP listener without resolved API keys. It runs before any spawn and before the model load, and reuses `is_loopback_bind_host`. Unix sockets and loopback pass; an IPv6-mapped loopback address counts as network.
- [x] Children: argv only, no shell; `env_clear()` plus an allowlist plus the entry's own `env`, so the server's API keys never reach a child; `kill_on_drop` plus explicit kill and reap.
- [x] Logging: tool arguments and outputs are never logged. Spawn errors returned to callers carry only the OS reason. `McpServerSpec` has a manual `Debug` that prints env keys only, and `hide_env_values` is set on the JSON flag.
- [x] Config: strict parsing; `url`, `headers`, `transport`, a non-stdio `type`, or an unknown key fails startup. Per-request `server_url`, `headers`, `authorization`, and `connector_id` are a 400.

| Issue (found in review) | Severity | Status |
|-------|----------|--------|
| Refused calls did not advance `tool_rounds`, so a `store:false` request could loop model turns without limit | High | Fixed (81169ec4) |
| `max_tool_calls` was not applied to approved calls run on resume | High | Fixed (81169ec4) |
| Each tool-loop response holds a blocking thread outside queue accounting; concurrent slow tools could exhaust Tokio's blocking pool | High | Fixed (825bcfa9): admission capped at `--max-queue-depth` (503 past it), server lock waits bounded by `--mcp-tool-timeout` |
| Distributed launches silently ignored `--mcp-tools` | Low | Fixed (b17449bf): refused at startup |
| Derived `Debug` printed child env values | Low | Fixed (b17449bf) |
| No per-turn call cap without `max_tool_calls` | Medium | Fixed (b17449bf): 16 calls per turn |
| A hung grandchild behind a launcher (`npx`, `uvx`) survives `kill_on_drop` | Medium | Open, see Follow-up |
| Approvals are not scoped to an API key | Low | Accepted: every key already has the same tool access and the response store is not key-scoped either |
| A concurrent duplicate resume is rejected after the stream starts | Low | Accepted: the approval is consumed before the call runs, so nothing runs twice |

### 2.2 Performance

The loop adds no cost when the flag is off: the translator path is unchanged and `mcp` still returns 400. With the flag on, each response runs inside one `spawn_blocking` task, and at most `--max-queue-depth` such responses run at once. Tool waits are bounded per call (`--mcp-tool-timeout`, default 30 s, which also bounds server start and lock waits).

### 2.3 Compatibility & Dependencies

- **Breaking changes**: none. Flags that failed before now work only behind the new opt-in.
- **New dependencies**: none.
- **Compat manifest**: the two flags move from `not_applicable` to `by_design` in the `ui-tools-mcp-gcp` shard; `--tools`, `--tools-runtime`, `--ui-mcp-proxy`, and `--agent` stay refused.

## 3. Technical Decisions

### 3.1 A separate opt-in flag

| Option | Pros | Cons |
|--------|------|------|
| The server-set flag enables the feature | Matches b10621 | An inherited `LLAMA_ARG_MCP_SERVERS_*` variable would spawn processes silently |
| **Chosen: `--mcp-tools` plus the server set** | Spawning always needs an explicit mlxcel flag | One extra flag |

### 3.2 Approval defaults and resume

`require_approval` omitted means `always`, following OpenAI. Pending approvals live in an in-memory registry (4096 entries) holding the model's arguments. The resume (`previous_response_id` plus `mcp_approval_response`) must answer every pending request of that response; the server runs the stored arguments, consumes the approval atomically, and returns 400 for a replay, a wrong response id, or a tool the resuming request no longer allows. When approval cannot be collected (`store:false`, no store, or `conversation`), the call is refused and the model continues. Approvals do not survive a restart, matching the in-memory `ResponsesStore`.

### 3.3 Discovery failure is not fatal

A server that cannot start is logged at `warn` and reported in its `mcp_list_tools` item (`error`, plus `response.mcp_list_tools.failed`); the next request that names it retries. One broken MCP server does not take down inference.

## 4. Implementation Details

```
POST /v1/responses with tools[type=mcp]
  -> mcp::request::plan_request   (labels, allowed_tools, approval policy, collisions)
  -> routes::responses_mcp        (admission permit, prepare_generation)
  -> routes::responses_mcp_loop   (model turn -> calls -> results, bounds)
       -> mcp::runtime -> mcp::client (stdio JSON-RPC, caps, respawn)
       -> mcp::approvals          (store / consume pending requests)
  -> one event sequence -> JSON body or SSE stream
```

- `src/server/mcp/`: `config` (strict `mcpServers` parsing, startup gates), `client` (newline-delimited JSON-RPC over stdio, byte-capped stdout, 4 KiB stderr ring), `runtime` (per-server state, lazy respawn, admission), `approvals`, `request`.
- `src/server/routes/responses_mcp*.rs`: the loop, item builders, and per-round execution. `prepare_response_run` now delegates to a new `prepare_generation` with its body unchanged.
- `src/server/responses_mcp_lowering.rs`: lowers `mcp_call`, `mcp_approval_request`, and `mcp_approval_response` history items into chat turns for the template.
- Bounds: `--mcp-max-tool-rounds` (8), `--mcp-tool-timeout` (30 s), `--mcp-max-output-bytes` (65536), 16 calls per turn when `max_tool_calls` is unset. A bound hit ends the response `incomplete` with `max_tool_rounds`, `max_tool_calls`, or `max_output_tokens`; a new `usage.mcp` object reports counts and the bounds in force, and usage sums every model turn.

## 5. Learning Points

### 5.1 Count refused work toward loop bounds

A loop bound that counts only successful iterations can be bypassed by an input that fails every time. Here a call refused for lack of an approval channel produced no `tool_rounds` increment, so the model could call it forever. Count every round that produced tool results, executed or refused.

### 5.2 Blocking tasks need their own admission control

A `spawn_blocking` task that waits on external I/O is invisible to queue-depth accounting. Admission must be capped where the task is created, not inferred from the generation queue.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 52 |
| Lines | +7252 / -128 |
| Tests | about 60 unit and route tests for MCP plus startup-gate tests on both binaries |

| Hash | Type | Message |
|------|------|---------|
| `3562fe22` | feat | add server-side MCP tool loop on /v1/responses |
| `81169ec4` | fix | bound refused MCP rounds and resumed approved calls |
| `825bcfa9` | fix | cap concurrent MCP tool-loop runs and server lock waits |
| `b17449bf` | fix | tighten MCP gate, Debug redaction, per-turn call cap |

## 8. Follow-up Actions

- Kill process groups on timeout (`process_group(0)` plus `killpg`). This needs a SIGINT/SIGTERM handler in the single-model server first, or Ctrl-C would stop killing children.
- HTTP/SSE transports and per-request `server_url` behind an explicit allowlist (out of scope for #1457).
- The Responses `tool_choice: {"type":"mcp",...}` form is not parsed; arguments arrive in one delta after each model turn rather than incrementally.
- Windows: fixture tests are `cfg(unix)` and nothing was run on Windows.

## Appendix

### A. Test Results

- `cargo test --profile test-fast --features metal,accelerate --lib -- server::mcp responses_mcp ui_compat_args coverage_tests` plus the startup test: 104 passed. The fixture `tests/fixtures/mcp/fake_mcp_server.py` covers normal, hang, crash, oversized output, invalid JSON-RPC, tool error, stderr flood, and startup hang, crash, and garbage.
- `--test webui_cli_help`, `--test cli_help_consistency`, `--test llama_compat_manifest`, clippy on lib and tests, `cargo check --bins`, fmt, and the repository contract scripts pass.

### B. Real-model run (M3 Ultra, `qwen3.5-4b-4bit`, GPU lock held)

`mlxcel-server -m models/qwen3.5-4b-4bit --host 127.0.0.1 --mcp-tools --mcp-servers-json '{"mcpServers":{"vault":{"command":"python3","args":["tests/fixtures/mcp/fake_mcp_server.py"]}}}'`, prompt "Look up the secret numbers stored under the keys alpha and beta, then use the add tool to add them and tell me the sum."

- Non-streamed: `lookup(alpha)=17`, `lookup(beta)=25`, `add(17,25)=42`, final message with 42; `usage.mcp.tool_rounds: 2`, `tool_calls: 3`; 1515 input and 290 output tokens over three turns.
- Streamed: 235 events with gapless sequence numbers and the same eight output items.
- Approval: the default policy returned `mcp_approval_request add {"a":19,"b":23}`; the approved resume returned 42; replaying the answer returned 400.
- The non-streamed and streamed runs were repeated on the final head (`b17449bf`) with identical items, usage, and event counts; the approval run is from the first commit.
