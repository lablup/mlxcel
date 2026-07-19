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

| Flag | Default | Purpose |
|------|---------|---------|
| `--parallel N` | 4 | Maximum active (in-flight) sequences; caps the concurrent decode batch. |
| `--max-batch-size N` | (= `--parallel`) | Maximum sequences decoded together in one batched step. |
| `--max-batch-prefill N` | 4 | Requests batched into one prefill forward pass (families that support it). |
| `--max-batch-prefill-tokens N` | (derived) | Padded-token budget bounding one batched prefill's transient memory. Unset derives `2 * max_batch_prefill * prefill_chunk_size`; `0` disables the cap. |
| `--max-queue-depth N` | 32 | Maximum queued (not yet admitted) requests. |
| `--prefill-chunk-size N` | 512 | Token chunk size for prefill; bounds prefill's effect on decode latency. |
| `--enable-preemption` | off | Allow evicting a lower-priority sequence to admit a waiting one. |
| `--no-batch` | off | Disable batching and serve sequentially (the legacy single worker). |
| `--no-prompt-cache` | off | Disable the prompt-prefix KV cache (it is on by default). |

### Serving-throughput defaults

The shipped defaults are tuned for multi-client serving. Batched decode reads
each weight once per step regardless of how many sequences share the step, so
aggregate throughput scales with concurrency until the batch hits the compute
roofline, while per-request decode rate falls proportionally. The defaults:

- `--parallel 4`: admit a decode batch of up to 4. The worker clamps this to 1
  for families that cannot batch (SSM / hybrid / mixed-cache, i.e. any model
  where `supports_batching()` is false), so the default is safe for every
  architecture.
- `--max-batch-prefill 4`: batch up to 4 pending prompts into one prefill pass.
  Only families that opt into `supports_batched_prefill()` (Llama 3, Qwen 3,
  Qwen 3.5, and aliases such as Qwen 2.5) use it; others fall back to sequential
  prefill automatically.
- Prompt-prefix cache on, bounded to a 2 GiB KV budget (`--prompt-cache-*` to
  retune). A repeated shared prefix (for example a long system prompt) is
  prefilled once and reused.

Escape hatches restore the previous single-client behavior: `--parallel 1`
(single decode slot), `--no-batch` (legacy sequential worker, no scheduler),
`--max-batch-prefill 1` (sequential prefill), and `--no-prompt-cache`.

> Backend note (CUDA / Blackwell, e.g. GB10): batched decode used to be a
> throughput wash on CUDA because the `M*B < 8` quantized matmul fell back to
> per-row qmv, re-reading the weights once per sequence (aggregate flat at
> ~45-50 tok/s across B=1/2/4 on llama-3.1-8b-4bit). The multirow qmv path
> (issue #725, `MLXCEL_QMV_MULTIROW`, on by default) amortizes the weight reads
> across the batch, so aggregate decode now scales with concurrency on CUDA
> too: 40.4 / 68.5 / 74.2 tok/s at 1/2/4 clients on llama-3.1-8b-4bit on GB10
> (vs 40.5 / 49.8 / 49.6 with the kill switch off). Scaling stays sublinear
> versus Metal because attention KV reads still grow with B; the small-M
> `qmm_sm80` tile shape (`M*B >= 8`) remains upstream MLX territory. For the
> same reason, keep `--n-parallel` (mlxcel-server: `--parallel`) at 7 or below
> on CUDA: the multirow window covers 2-7 rows, and a full batch of 8+ decode
> rows crosses into the under-tiled `qmm_sm80` shape, which measures worse
> than per-row decode until the upstream small-M tile lands. The shipped
> default of 4 sits comfortably inside the window. See
> `docs/benchmark_results/qmv-multirow-gb10-2026-07-11.md`. `--max-batch-prefill`
> (large-M, tile-tuned, `docs/benchmark_results/qmm-sm121-tile-tuning-gb10-2026-07-10.md`)
> and the prompt cache remain independent wins on CUDA.

Memory sizing: the KV footprint grows with the active batch, so budget for up to
`--parallel` concurrent sequences' KV. When `--ctx-size` is set it is divided
across the active slots (`ctx_size / parallel` per slot, floor 512 tokens);
leave it at 0 to use the model default per slot. Admission control
(`--kv-cache-budget`, `--max-kv-size`) still sheds load under pressure, so a
large `--parallel` degrades to queueing rather than OOM.

Measured scaling (Apple M1 Ultra, Metal; `meta-llama-3.1-8b-instruct-4bit`,
512-token prompt, 128 new tokens; `scripts/bench_serving_concurrency.py`).
The baseline column is the previous default `--parallel 1`. The batched column
was measured with `--max-batch-prefill 4` and a batch ceiling high enough to
expose each concurrency level (`--parallel 8`), so the 8-client row reflects a
`--parallel 8` server; the shipped default is `--parallel 4`, whose batch caps
at 4 (the 4-client row is the default at saturation, and 8 clients would run 4
batched plus 4 queued). Aggregate is tokens/sec summed across concurrent
clients; TTFT is time-to-first-token:

| clients | aggregate tok/s (`-p1`) | aggregate tok/s (batched) | TTFT mean ms (`-p1`) | TTFT mean ms (batched) |
|--------:|------------------------:|--------------------------:|---------------------:|-----------------------:|
| 1 | 56.6 | 56.8 | 889 | 783 |
| 2 | 71.5 | 99.9 | 1150 | 114 |
| 4 | 65.6 | 107.9 | 3257 | 189 |
| 8 | 62.8 | 105.1 (`-p8`) | 7514 | 348 (`-p8`) |

At 4 concurrent clients the default delivers 1.90x the single-client aggregate
throughput and cuts mean TTFT under load ~17x (3257 ms to 189 ms), while
single-client throughput is unchanged. On a higher-bandwidth decode target (for
example GB10) the weights-read amortization headroom is larger; that number is
pending a CUDA measurement session.

The batched-decode default is paired with an `auto` paged KV budget
(`--kv-cache-budget auto`, the default): the #122 block-budget admission bounds
KV for the concurrent batch and returns backpressure instead of letting four
full-context sequences run into an OOM abort. On the dense decode backend the
budget is inert. Disable the guard with `--kv-cache-budget none`. Memory-
constrained hosts can also lower `--parallel` or cap `--ctx-size` (see the
context-sizing note in [environment-variables.md](environment-variables.md)).

### Bounding the batched-prefill transient (`--max-batch-prefill-tokens`)

The `--kv-cache-budget` guard above bounds steady-state KV, not the transient a
batched prefill allocates while it runs. When two or more cold prompts share one
padded batched prefill, the path pads every row to the window's longest prompt
`L` and materializes a stacked `[B, L, L]` FP32 attention mask plus a
`[B, L, hidden]` forward. Left uncapped, `B` long prompts arriving together
allocate an `O(B*L^2)` mask that ignores `--prefill-chunk-size`: four 8k prompts
build a `[4, 8192, 8192]` FP32 mask, about 1 GiB, a spike far above the working
set of the sequential chunked prefill those prompts take when batched prefill is
off. On the serving path an allocation failure is an uncatchable MLX C++ throw
that aborts the whole server, so this is an availability edge the KV budget does
not model.

`--max-batch-prefill-tokens N` caps the drained batched window by total padded
tokens (`rows * L`). Draining stops before a row that would push `rows * L` past
`N`; the remaining rows stay queued and prefill on later ticks (short ones
re-batch, long ones take the chunked single-sequence path). A head prompt too
long to join a two-row batch (`2 * head_len > N`) skips the batched path
entirely and prefills chunked. The bound follows from the token budget: a cohort
of `B >= 2` rows padded to `L` costs `B*L <= N` tokens, and since `L <= (B*L)/2`
the mask stays within `N^2 / 2` elements, i.e. `~N^2` bytes at FP16 and
`~2*N^2` at FP32.

The default budget is derived, not fixed: `2 * max_batch_prefill * prefill_chunk_size`
(the shipped `2 * 4 * 512 = 4096`; the 2x headroom absorbs the padding slop of
real chunk-sized prompts, whose chat template pushes them slightly over
`prefill_chunk_size`), so a full batch of chunk-sized prompts stays
eligible for batching while a window of longer prompts spills to the chunked
path. At the default the FP32 mask is bounded to `2 * 4096^2` bytes, about 34 MiB,
negligible beside model activation memory. `0` (the flag, or
`MLXCEL_MAX_BATCH_PREFILL_TOKENS=0`) disables the cap for the pre-#715 unbounded
behavior. The flag takes precedence over `MLXCEL_MAX_BATCH_PREFILL_TOKENS`, which
takes precedence over the derived default (see
[environment-variables.md](environment-variables.md)).

The analytic prediction for four concurrent 8k-token prompts on
`meta-llama-3.1-8b-instruct-4bit` (Apple M1 Ultra, Metal):

| config | prefill mask window | mask transient (analytic) | path |
|--------|--------------------:|--------------------------:|------|
| uncapped (`--max-batch-prefill-tokens 0`) | `[4, 8192, 8192]` FP32 | `4 * 8192^2 * 4 B` = 1024 MiB | single unchunked batched forward |
| default cap (4096) | four `[≤512, 8192]` chunk masks | `512 * 8192 * 4 B` = 16 MiB (one at a time) | 8k prompts spill to the chunked single-sequence path |

The empirical A/B (server phys-footprint peak: RSS does not capture MLX Metal
buffers on Apple Silicon, so use `/usr/bin/footprint -p <pid>` for
`phys_footprint_peak`) is pending a measurement session. Reproduce with a server
started at `--max-batch-prefill-tokens 0` versus the default, driven by four
concurrent 8k-token requests (`scripts/bench_serving_concurrency.py --concurrency 4
--prompt-tokens 8192`, `--max-tokens` small to isolate prefill).

Short-prompt concurrency (the #714 default motivation) is unaffected by
construction: at `--prompt-tokens 512 --concurrency 4` all four rows cost
`4 * 512 = 2048` padded tokens, half the default budget, so they batch with
headroom even when the chat template pushes each prompt slightly past 512; the
`2 * head_len <= budget` admission holds (`2 * 512 <= 4096`). TTFT and aggregate
throughput match the #714 table above (empirical re-run pending the same
session).

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

### Speculative decoding under continuous batching

Supplying a drafter checkpoint (`--draft-model` on `mlxcel serve`,
`--model-draft` on `mlxcel-server`; the spellings are cross-aliased) turns on
server-side speculative decoding. The kind is auto-detected from the drafter's
`config.json` (`--draft-kind mtp|dflash` overrides), and the draft block size
K defaults per kind (4 for MTP, 16 for DFlash; `--draft-block-size`).

Whether the singleton (B=1) speculative burst actually runs is decided by an
adaptive policy (issues #333/#736), not by the flag alone. The first 4
qualifying requests of each (target, drafter, hardware, K) pairing are
profiled with speculation forced on; each profiled burst also runs 2
classic-step probe rounds (drafterless single-token verifies that emit real
greedy tokens), so the policy can compare the measured speculative round cost
against the measured classic step time and settle to Enable, Decline, or the
static per-hardware default when the estimate is ambiguous. The verdict is
persisted under `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/` and
logged at settle time (`adaptive MTP policy: settled verdict ... est_speedup=...
[measured]`), so a restart skips profiling. Changing K re-profiles. Because
the decision is measured, the same binary enables a pairing where the
speculative round pays for itself (Gemma 4 12B + assistant: ~1.87x on M5 Max,
~1.5x on GB10 with the multirow qmv verify) and declines it where it does not
(the same pairing on M1 Ultra, or on CUDA with `MLXCEL_QMV_MULTIROW=0`).

Scheduling: for non-batchable targets (the 12B Unified family) an enabled B=1
burst equals their inherent single-slot serialization. For batch-capable
targets, the B=1 MTP arm is served tick-cooperatively by default (issue #734):
the request runs one speculative round per scheduler tick, alternating with
the classic decode/prefill actions, so concurrent classic-decode rows advance
between rounds instead of stalling for the whole request. This bounds the
head-of-line stall a speculative request imposes on concurrent rows to about
one round; the scheduler logs it as the max single-tick `burst_wall_ms` at
finalize (on the validation pairing this dropped a multi-second whole-burst
stall to about one round of steady-state occupancy, roughly 23x faster
time-to-first-token for the concurrent classic rows). The interleaving is not
free: on the validation pairing the strict alternation cost roughly 27% of
the combined two-request aggregate decode throughput, so
`MLXCEL_MTP_TICK_SLICE=0` restores the legacy run-to-completion burst for
deployments that serve speculative requests without concurrent classic traffic
and want raw speculative throughput. DFlash B=1 and every B>1 batched burst
still run to completion inside one tick regardless of the flag. Concurrent
speculative requests that share a prompt length, `max_tokens`, and sampling
config can instead run as one B>1 batched burst, but that path is experimental
and stays behind `MLXCEL_ENABLE_MTP_BATCH` (plus
`MLXCEL_ENABLE_MTP_BATCH_RAGGED` for mixed prompt lengths).

A tick-cooperative slice spans the request's whole generation, so the single
speculative slot would otherwise be held for that entire time. Since issue
#746 the slot rotates across concurrent speculative requests instead of
serving only the first one: while a slice is in flight, up to 2 further
tick-slice-eligible requests are parked in a grant backlog (any beyond that
cap fall back to classic decode, the pre-#746 behavior), and once the active
request has run `MLXCEL_MTP_SLICE_GRANT_ROUNDS` slices (default 8, slice 0
included) with the backlog non-empty, it is parked at the next round boundary
and the slot is granted to the next request, priority lane first and FIFO
within a lane, with an anti-starvation floor: an entry passed over by 2 grant
decisions becomes overdue and must be granted next regardless of lane, so a
sustained stream of higher-priority arrivals (priorities are client-assignable
via the `X-Priority` header) can delay a lower-priority request only by a
bounded number of grants, never stall its stream outright. The backlog cap
applies at admission; a park transiently holds cap + 1 backlog entries until
the next promotion, so total live speculative sessions (the active request
plus parked and waiting ones, each holding its per-sequence KV) peak at 3.
A parked request resumes from its saved session state on its
next grant, with the worker's one drafter handle handed over through the same
return/take plumbing the end of a request uses (safe because the MTP
assistant drafter's reset is a no-op and every round re-arms the drafter from
the session's own stored verify output), so the token streams are
byte-identical to uncontended runs. The budget binds only under contention:
a single speculative request never rotates and behaves exactly as under
#734. `MLXCEL_MTP_SLICE_GRANT_ROUNDS=0` disables rotation and restores the
whole-generation hold, including the classic fallback for every concurrent
speculative request.

Operator controls, all env-level (see
[environment-variables.md](environment-variables.md)): `MLXCEL_ENABLE_MTP_B1`
pins the B=1 decision in either direction and suppresses profiling,
`MLXCEL_MTP_ADAPTIVE=0` disables the adaptive policy in favor of the static
per-hardware gates, `MLXCEL_MTP_SLICE_GRANT_ROUNDS` sizes (or disables) the
speculative-slot grant rotation, and `MLXCEL_QMV_MULTIROW=0` restores the
stock per-row CUDA quantized-matmul kernel under the verify.

On CUDA, the same draft/verify shape diversity (multiplied by batch size and
sequence-length bucket) is also what drives MLX's own captured-graph-cache
LRU hardest; mlxcel raises `MLX_CUDA_GRAPH_CACHE_SIZE` to 2000 at startup by
default so a long-lived server does not hit MLX's fatal `Cache thrashing`
abort at its own 400 default (issue #818, `MLX_CUDA_GRAPH_CACHE_SIZE` in
[environment-variables.md](environment-variables.md)).

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
- `completion_tokens` in the usage block uses the worker's authoritative generated-token count carried over the disaggregated wire protocol (issue #387), so it is exact for both byte-level BPE tokenizers (Qwen, Llama) and byte-fallback tokenizers (Gemma `<0xXX>` byte sequences), and `finish_reason` matches single-node. A node that predates the wire field (a mixed-version cluster) falls back to counting emitted detokenized text pieces, which can under-count for byte-fallback tokenizers and flip `finish_reason` between `"length"` and `"stop"`.
- `/v1/chat/completions` reports the same `usage` shape as `/v1/completions` (`prompt_tokens`, `completion_tokens`, `total_tokens`): the non-streaming response always carries it, and the streaming response sends it as a trailing chunk with empty `choices` when the request sets `stream_options.include_usage`, matching single-node (issue #398).
- The router stream filter suppresses model-specific structural markers (`<think>`, tool-call delimiters) and routes thinking content to `reasoning_content`. Tool-call parsing is not yet supported on the router path: only `content` and `reasoning_content` are emitted.
- Co-locating the roles on one machine adds transport hops without the scaling
  benefit of separate prefill and decode hardware, so a single-machine setup is
  for validation. The throughput case for disaggregation is separate prefill and
  decode pools under load.
