# Continuous batching and disaggregated serving

`mlxcel-server` (and `mlxcel serve`) runs a continuous-batching scheduler: it
admits concurrent requests into a running batch, interleaves prefill and decode,
and streams each request's tokens as they are produced. This document covers the
batching scheduler, how it uses the paged KV cache, and the disaggregated
serving roles that split prefill and decode across processes.

## Continuous batching

The scheduler keeps up to `--parallel` sequences active at once. Each step it
either admits and prefills queued prompts (chunked at `--prefill-chunk-size` so a
long prompt does not stall decode) or advances the active batch by one decode
token, then streams the new tokens out. Relevant flags:

| Flag | Purpose |
|------|---------|
| `--parallel N` | Maximum active (in-flight) sequences. |
| `--max-batch-size N` | Maximum sequences decoded together in one batched step. |
| `--max-queue-depth N` | Maximum queued (not yet admitted) requests. |
| `--prefill-chunk-size N` | Token chunk size for prefill; bounds prefill's effect on decode latency. |
| `--enable-preemption` | Allow evicting a lower-priority sequence to admit a waiting one. |
| `--no-batch` | Disable batching and serve sequentially (the legacy single worker). |

### Paged decode and the prompt-prefix cache

Decode state and the cross-request prompt-prefix cache share one refcounted,
copy-on-write block pool (the default for batch-capable pool-backed families;
`--decode-storage-backend dense` opts out). Concurrent requests that share a
prompt prefix store that prefix's KV once and skip re-prefilling it: adoption
is non-consuming (clone-and-pin), so one stored prefix serves any number of
simultaneous borrowers, and Automatic Prefix Caching (on by default, disable
with `--apc-enabled=false`) lets requests that diverge after a shared prefix
reuse the common part. The mechanism, the measured memory and prefill-token
savings, the decode throughput, and `--kv-cache-budget` are documented in
[turbo-kv-cache.md](turbo-kv-cache.md#unified-paged-kv-cache). Paged decode is
byte-identical to the dense backend; it is the storage backend the disaggregated
roles below build on.

Recurrent and hybrid SSM / linear-attention families cannot safely reuse
arbitrary KV blocks, so they keep the hybrid-SSM/APC exclusion. Families that
opt into `supports_snapshot_reuse()` instead use a separate exact-prefix
snapshot bucket: on a healthy finish the scheduler copies the model-owned
state, and on the next turn it restores that state only when the stored token
vector is a whole prefix of the incoming request in the same session. The
unmatched suffix is still prefilled normally, with no recurrent state
truncation or cross-session sharing. The supported snapshot families are
Mamba, Mamba2, Jamba, Nemotron-H, Qwen 3.5 / 3.6 text, MoE, and VLM wrappers,
and Gemma 4 text, VLM, and Unified wrappers.

For multimodal servers, `--enable-vlm-prefix-cache` opts image requests into
prompt-prefix reuse across same-session follow-up turns with the same image.
The default stays off for VLM requests, and text-only prompt-cache behavior is
unchanged.

## Disaggregated serving

Prefill is compute-bound and decode is memory-bound, so a deployment can run them
on separate processes (or machines) with a router in front. `--node-role`
selects the role:

- **`prefill`**: runs prompt prefill, samples the first token, and hands the
  sequence's KV cache off to a decode node over TCP.
- **`decode`**: receives the KV handoff, continues autoregressive decode, and
  returns the continuation.
- **`router`**: a model-free HTTP front (it loads only a tokenizer and chat
  template, not model weights). It tokenizes the client request, routes it to a
  prefill node, and merges the prefill node's first token with the decode node's
  continuation into one SSE stream back to the client.
- **absent (hybrid)**: single-node serving, byte-identical to a server started
  with no distributed flags.

Request flow (topology A): client HTTP request -> router -> prefill node (TCP)
-> decode node (TCP) -> router merges the two halves -> client SSE.

Networking flags:

| Flag | Role | Purpose |
|------|------|---------|
| `--serving-bind <addr>` | prefill, decode, router | This node's own role-transport listener (`host:port`). |
| `--decode-peers <addr,...>` | prefill, router | On a router, the decode pool it balances; on a prefill node, the static fallback used only when the router does not pick a decode target for a request. |
| `--prefill-peers <addr,...>` | decode, router | On a router, the prefill pool it balances; on a decode node, the prefill node(s) it accepts handoffs from. |

### Multi-node routing, load balancing, and failover

With more than one prefill or decode node, the router balances both pools
(issue #201). For each request it picks a prefill node and, independently, a
decode node, both round-robin across the nodes the registry currently considers
online (the router has no live per-node load telemetry, so round-robin is the
strategy that actually spreads load). The chosen decode node travels to the
prefill node in the request frame's `decode_target` field, so the prefill node
hands the KV cache to the router-balanced decode node rather than to its own
`--decode-peers` config. The field is optional on the wire: a frame without it
(an older router) leaves the prefill node on its config fallback, and an older
prefill node ignores it and uses config, so mixed-version pools keep working.

As opt-in defense-in-depth (issue #389), a prefill node can validate the
router-chosen `decode_target` against an allowlist before connecting. The
allowlist source is the dedicated `MLXCEL_DECODE_ALLOWLIST` environment variable,
a comma-separated list of numeric `IP:port` values (parsed as `SocketAddr`) set to the full pool of router-selectable decode nodes (the shared cluster config); hostname:port entries are not resolved and are skipped with a warning. It is independent of `--decode-peers`,
which stays the static handoff fallback only, so enabling the allowlist does not
constrain router balancing. When `MLXCEL_DECODE_ALLOWLIST` is unset the prefill
stays permissive and logs a warning rather than rejecting, so balancing is never
silently broken; see [`docs/distributed.md`](distributed.md) for the security
rationale.

The router tracks node health and fails over without wedging:

- A transport error when sending a request to a prefill node marks that node
  unreachable, re-routes its in-flight requests, and retries the request on a
  healthy node. When no prefill node is left, the request fails cleanly with an
  error rather than hanging.
- A background health monitor TCP-probes every peer's serving address on an
  interval. A node that stops accepting connections is marked unreachable (so
  routing skips it, including a dead decode node the router never sends to
  directly), and a node that starts accepting again is restored to online.
- Admission control runs before every dispatch: when the prefill queue is full
  or no prefill node is available, the router returns HTTP 503 instead of
  attempting a dispatch that cannot succeed.

`GET /router/stats` reports the per-node dispatch counts for both pools, the
registered nodes with their current health status, and the routing-metrics
snapshot, so an operator can confirm the spread and see which nodes are down.

The handoff transfers the paged block contents (not just metadata) over the
transport, so the decode node reconstructs the exact KV the prefill node built;
see `src/distributed/kv_cache_serde/`. The disaggregated output is byte-identical
to a single-node run, verified end to end by `tests/disaggregated_router_e2e.rs`
(three real processes) and at the cache level by `tests/paged_handoff_parity.rs`.

Implementation entry points: `src/server/router_front.rs` (the router front),
`src/distributed/disaggregated/{coordinator,handoff_impl,serving_protocol}.rs`
(the role loops and wire protocol), and `src/server/batch/` (the scheduler
serving-role entries).

### Example: three processes on localhost

```bash
# Decode node: receives KV handoffs on :8304.
mlxcel-server -m models/<ckpt> --port 8302 \
    --parallel 2 --max-batch-size 2 --decode-storage-backend paged \
    --node-role decode --serving-bind 127.0.0.1:8304

# Prefill node: prefills, hands off to the decode node.
mlxcel-server -m models/<ckpt> --port 8301 \
    --parallel 2 --max-batch-size 2 --decode-storage-backend paged \
    --node-role prefill --serving-bind 127.0.0.1:8305 \
    --decode-peers 127.0.0.1:8304

# Router: model-free HTTP front on :8300.
mlxcel-server -m models/<ckpt> --port 8300 \
    --node-role router --serving-bind 127.0.0.1:8306 \
    --prefill-peers 127.0.0.1:8305 --decode-peers 127.0.0.1:8304

# Client talks only to the router.
curl http://127.0.0.1:8300/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"What is 2 + 2?"}],"stream":true,"temperature":0}'
```

### Scope and limitations

- Pool-backed Fp16 families only (the dense-natural backends such as qwen3 and
  llama3). Model-owned-state families and recurrent/hybrid SSM models are
  excluded from the paged handoff; the exact-prefix snapshot cache described
  above is a single-node prompt-cache optimization and is not serialized across
  disaggregated prefill/decode roles.
- Text-only. The router serves `POST /v1/chat/completions` and `POST /v1/completions`; multimodal requests are rejected.
- On `/v1/completions`, three option groups are rejected with HTTP 400 because the disaggregated wire protocol does not carry the data they require: `logprobs`, `response_format` (structured output), and explicit reasoning/thinking budgets. Requests without those fields work normally.
- `stop` sequences are not forwarded to the worker nodes by the router. This is a pre-existing limitation shared with the chat path.
- `completion_tokens` in the usage block is counted from emitted detokenized text pieces. For byte-level BPE tokenizers (Qwen, Llama) this equals the worker's token count exactly. For byte-fallback tokenizers (Gemma `<0xXX>` byte sequences) it can under-count, which may flip `finish_reason` between `"length"` and `"stop"`. A precise fix requires the disaggregated wire protocol to carry the worker's token count; a follow-up issue tracks the protocol extension.
- The router stream filter suppresses model-specific structural markers (`<think>`, tool-call delimiters) and routes thinking content to `reasoning_content`. Tool-call parsing is not yet supported on the router path: only `content` and `reasoning_content` are emitted.
- Co-locating the roles on one machine adds transport hops without the scaling
  benefit of separate prefill and decode hardware, so a single-machine setup is
  for validation. The throughput case for disaggregation is separate prefill and
  decode pools under load.
