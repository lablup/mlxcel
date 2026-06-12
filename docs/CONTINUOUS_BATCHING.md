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
opt into `supports_snapshot_reuse()` (Mamba, Mamba2, Jamba, Nemotron-H, and
Qwen 3.5 / Qwen3-Next variants) instead use a separate exact-prefix snapshot
bucket: on a healthy finish the scheduler copies the model-owned state, and on
the next turn it restores that state only when the stored token vector is a
whole prefix of the incoming request in the same session. The unmatched suffix
is still prefilled normally, with no recurrent state truncation or
cross-session sharing.

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
| `--decode-peers <addr,...>` | prefill | Decode node(s) a prefill node hands KV off to. |
| `--prefill-peers <addr,...>` | router | Prefill node(s) the router routes requests to. |

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
- Text-only. The router serves `/v1/chat/completions`; multimodal requests are
  rejected by the router.
- The router does not yet apply the chat stream filter that the single-node chat
  route uses, so reasoning/tool-call markers pass through verbatim. Use a
  non-reasoning prompt or `chat_template_kwargs.enable_thinking=false` for clean
  content.
- Co-locating the roles on one machine adds transport hops without the scaling
  benefit of separate prefill and decode hardware, so a single-machine setup is
  for validation. The throughput case for disaggregation is separate prefill and
  decode pools under load.
