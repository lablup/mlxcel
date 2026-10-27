# Distributed inference

`mlxcel` has three distributed or multi-device surfaces. They share code under
`src/distributed/`, but their maturity differs by mode and model family.

| Mode | Purpose | Maturity |
|------|---------|----------|
| Tensor parallelism (TP) | Shard tensor operations across in-process ranks. | Implemented for selected dense text families; validate per model. |
| Pipeline parallelism (PP) | Split layer ranges across stages. | Best validated on Llama-family text models and two-stage topologies. |
| Disaggregated inference (DI) | Split prefill/decode roles while each node holds the model. | Infrastructure exists; treat as experimental unless validated for your topology. |

None of the three modes runs on native Windows. `mlxcel generate` and
`mlxcel-server` exit at startup, before the model is resolved, with an error
naming the flag: `--tp-size` above 1 and `--pp-layers` for both, `--pp-size`
above 1 for `mlxcel generate`, and `--pp-auto`, `--pp-peer`,
`--distributed-config`, `--node-role`, `--peers`, `--prefill-peers`,
`--decode-peers` and `--serving-bind` for the server (lablup/mlxcel#58).
Use Linux, macOS or WSL2 for these modes.

## Choosing a mode

```text
Model fits on one device?
├── yes
│   ├── latency-sensitive single-user serving → single device
│   └── many concurrent users                 → consider DI after validation
└── no
    ├── high-bandwidth local devices          → TP or PP
    └── multi-host / uneven memory            → PP with explicit layer ranges
```

## Tensor parallelism

TP shards weights inside transformer layers and synchronizes row-parallel
outputs. The public knobs are:

```bash
mlxcel generate -m models/<checkpoint> \
    --tp-size 2 \
    -p "Hello" -n 100

mlxcel-server -m models/<checkpoint> \
    --tp-size 2 \
    --port 8080
```

Related options include `--tp-moe-mode`, `--tp-embedding-mode`, and
`--tp-lm-head-mode`. The current runtime requires replicated embedding and LM
head modes for many families.

The help text in `src/main.rs` and `src/bin/mlx_server.rs` is the source of
truth for the currently advertised TP family list. At the time of this docs
pass, it includes dense Llama, Qwen 2/2.5/3/3.5 text, Gemma 3/4 text,
ERNIE 4.5, and Hunyuan v1 Dense, with additional implementation pieces for
other families.

Under `mlxcel-server --parallel N`, the TP Llama, Qwen 3, Qwen 3.5, Gemma 3,
ERNIE 4.5, and Hunyuan v1 Dense runtimes execute one native multi-row decode
graph per rank and layer; startup reports `decode path: native batched forward`.
TP Gemma 4 remains on the per-row fallback until issue #2546 supplies its
batched rank-layer path.

Limitations:

- The model must shard cleanly across the selected rank count.
- Some server batching and VLM paths are intentionally conservative under TP.
- Benchmark and correctness validation should be repeated for every model family
  and rank count you intend to run.

## Pipeline parallelism

PP splits the model by layer range. It is useful when a model exceeds a single
device's memory or when hosts have uneven memory capacity.

Kimi K3 (`kimi_k3`) is the family that will need it outright: the full
93-layer, 2.8T-parameter checkpoint occupies about 1.4 TB at 4 bits (the mxfp4
experts plus bf16 attention, shared experts, embeddings and `lm_head`), more
than twice the memory of the largest single Apple Silicon host, so no single
host holds it and at least three 512 GB nodes are the smallest topology that
could. **PP cannot run this family yet.** What is in the tree is the partition
profile (`src/distributed/pipeline/partition_profile_heuristics.rs`), which
prices each layer by type so an automatic split accounts for the dense layer 0,
the cheaper q-LoRA MLA layers at every fourth position and the 15 to 16 GB MoE
layers everywhere else, rather than assuming a uniform per-layer cost. That is
a planning input; there is no `StageFamily` variant and no stage executor, so
`--pp-size` and `--pp-layers` refuse a `kimi_k3` model. Issue #1734 tracks the
executor and the full-model multi-node run, including how the Attention
Residual blocks cross a stage boundary. Until then the family is
single-process only, and a layer-truncated local copy (lower
`text_config.num_hidden_layers` in `config.json`; the sanitizer drops the
layers past it) is the single-host check for loading, shapes and finite logits.
Its output text is meaningless by construction.

### In-process CLI path

```bash
mlxcel generate -m models/<checkpoint> \
    --pp-size 2 \
    --pp-micro-batch-size 4 \
    -p "Hello" -n 100
```

You can provide explicit layer ranges instead of relying on auto partitioning:

```bash
mlxcel generate -m models/<checkpoint> \
    --pp-layers 0-15,16-31 \
    --pp-micro-batch-size 4 \
    -p "Hello" -n 100
```

### Server / multi-host path

The server uses `--distributed-config` with a TOML cluster configuration. The
repository includes helper scripts and examples under `examples/distributed/`
and `scripts/benchmark_pipeline_remote_rollout.sh`; inspect those before
operating a real cluster.

A minimal shape looks like this:

```bash
# Stage process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/pipeline_remote_2node_tcp.toml \
    --cluster-secret-file /etc/mlxcel/cluster.secret \
    --node-id stage-1 \
    --host 0.0.0.0 --port 18081 --no-warmup

# Coordinator / serving process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/pipeline_remote_2node_tcp.toml \
    --cluster-secret-file /etc/mlxcel/cluster.secret \
    --node-id coordinator \
    --host 0.0.0.0 --port 18080 \
    --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2
```

`--pp-auto N` can generate a zero-config pipeline plan and is mutually exclusive
with `--distributed-config`. For production, prefer checking in an explicit TOML
once the topology is known.

Both commands above also need the cluster secret described in the next
section; without it each process exits at startup with an error naming the
flag, the environment variables, and the TOML field.

### Cluster authentication

Every distributed role authenticates its peers with one pre-shared per-cluster
secret: the pipeline roles since issue #2637, and the disaggregated prefill,
decode and router roles since issue #2655. The coordinator, every remote stage,
every `--pp-peer` process, and every `--node-role prefill|decode|router`
process must hold the same value. Generate it once and copy it to each host
with owner-only permissions:

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
```

Supply it through one of these sources, listed from highest to lowest
precedence:

1. `--cluster-secret-file <PATH>` on `mlxcel-server` (and `mlxcel serve`).
2. `MLXCEL_CLUSTER_SECRET_FILE=<PATH>`.
3. `MLXCEL_CLUSTER_SECRET=<VALUE>` (the secret itself; prefer the file form on
   shared hosts). Setting both environment variables is an error.
4. `secret_file = "<PATH>"` under `[cluster]` in the cluster TOML. A relative
   path is resolved against the TOML's own directory. The secret value itself
   never goes into the TOML, because the coordinator writes the generated TOML
   to disk and it is copied between hosts. `--pp-auto` records the coordinator's
   secret file as an absolute path there when the secret came from a file.

The secret must be at least 16 bytes after trimming one trailing newline; the
server warns when the file is group- or world-readable, and never logs the
value.

What the secret protects:

- **Stage connections.** Every TCP connection between pipeline roles (plain
  TCP, Thunderbolt, and the RDMA wrapper all share the same core) runs a mutual
  HMAC-SHA256 challenge-response before a single frame is exchanged. A peer
  with the wrong secret is rejected whichever side opened the connection, the
  coordinator fails at startup instead of on the first request, and a frame
  from an unauthenticated connection never reaches the stage service or the
  coordinator. The accepted peer's listener address is bound to its TCP source
  IP and becomes the identity every frame is attributed to.
- **Command and reply authorization.** A stage accepts commands (`RunEntry`,
  `CancelRequest`, `BeginDrain`, `Shutdown`, ...) only from the coordinator
  node of its cluster config, forward activations only from its upstream
  stage, and reverse activations only from its downstream stage; it replies to
  the authenticated source, never to an address named inside the payload. The
  coordinator accepts a reply only from the stage it addressed, only with the
  fresh per-exchange nonce it sent, only for the matching request and
  micro-batch, and only if the returned logits have the model's shape
  (`[1, seq_len, vocab_size]`); everything else is dropped while the exchange
  keeps waiting against an absolute deadline.
- **Discovery beacons.** With `--cluster-discovery=mdns` every beacon carries a
  timestamp, a nonce, and an HMAC over its fields. The coordinator rejects
  beacons without a valid MAC, beacons more than 60 seconds off its clock or
  replayed within that window, beacons from a different mlxcel version, and
  beacons whose advertised address is not the datagram's source IP. Static
  discovery is unaffected.

- **Disaggregated role connections.** Since issue #2655 the router, prefill
  and decode role transports run the same connection handshake: the router's
  request frames, the prefill node's KV handoff to a decode node, and every
  result frame travel only over authenticated connections, and a raw TCP
  client that skips the handshake never gets a frame to any role's receive
  loop. Frames from authenticated members stay trusted (no per-frame source
  check inside the disaggregated protocol); `MLXCEL_DECODE_ALLOWLIST` is
  unchanged.

Fail-closed scope: `--pp-auto`, `--pp-peer`, any `--distributed-config` whose
`pipeline_parallel_size` is 2 or more, and `--node-role prefill`, `decode` and
`router` require the secret and refuse to start without one, naming every
source in the error. Default single-node serving (no `--node-role`, or
`--node-role hybrid`) and tensor parallelism inside one process do not read it
and behave exactly as before.

Handshake failures are reported distinctly so an operator can tell them apart
in the log:

- *Wrong secret on the connecting side*: the connecting process fails with
  "its authenticator does not verify under the configured cluster secret
  (secret mismatch)"; the accepting side logs "client closed the connection
  after receiving this server's authenticator; it most likely holds a
  different cluster secret".
- *Advertised identity not owned*: a node whose advertised listener IP is not
  the source IP the accepting side sees (a wrong bind or advertised address, or
  a NAT or port forward between the nodes) is refused with an explicit
  rejection record before any server nonce or authenticator is sent. The
  connecting side fails with "server refused this node's advertised identity
  `<host:port>`: the advertised IP must be the address the server sees this
  connection come from"; the accepting side logs "refusing to bind a listener
  address the peer does not own".
- *Closed before the handshake*: a connect-and-close liveness probe (the
  router health monitor, `probe_peers`) is logged at debug only.

The handshake wire protocol is versioned (`PROTOCOL_VERSION` 2 since #2655); a
pair of processes built from different versions fails with an explicit version
message rather than a confusing I/O error. Mixed versions in one cluster are
not supported.

Residual risks:

- The handshake authenticates connections; it does not encrypt or MAC individual frames. An on-path attacker who can hijack an already established TCP stream is outside what this provides. Keep the cluster ports on a trusted segment, and treat TLS or a per-frame MAC as the follow-up when that assumption does not hold.
- Connections that have not finished the handshake do not count against the 512 authenticated-connection cap. They draw from a separate pending pool (64 in flight, at most 8 per source IP, each for at most the 5 second handshake timeout); a socket beyond either bound is closed at accept. A flood of unauthenticated connections from enough source IPs (8 or more at the defaults) can fill the pending pool and so delay new connections from any source, including a real peer reconnecting, by up to the 5 second timeout per slot while the flood lasts; it cannot take slots from established authenticated connections or inject frames. Failed inbound handshakes and pending-pool refusals are logged at warn for the first few per minute and at debug afterwards.
- `MLXCEL_CLUSTER_SECRET` is inherited by child processes of the server. Prefer `--cluster-secret-file` or `MLXCEL_CLUSTER_SECRET_FILE` so the value never sits in the environment.
- The server identity is not bound to the socket IP on the connecting side, so a stage address that is reached through NAT or a port forward still authenticates by the secret alone.

## Transports

| Transport | Notes |
|-----------|-------|
| TCP | Default IP transport. |
| Thunderbolt | macOS Thunderbolt Bridge selection on top of the shared TCP core. |
| RDMA | Backend exists with capability probing and fallback behavior; validate on the target OS/hardware before relying on acceleration. |

mDNS/static discovery options are available for zero-config startup. Static
configuration is the safer choice across subnets or locked-down networks.

## Disaggregated inference

DI separates prefill and decode roles. Unlike PP, it does **not** reduce
per-node model memory: each role still needs the model loaded. The intended use
case is throughput tuning, not making an oversized model fit.

The code shares the same cluster config, registry, transport, and metrics
infrastructure as PP. Treat it as a topology-specific feature: run a live test
with your traffic shape before publishing performance claims.

Model support is narrower than PP: the paged prefill/decode handoff works only
for pool-backed Fp16 families (dense-natural backends such as qwen3 and
llama3). Model-owned paged families (gemma3, gemma4, llama4, qwen3_5,
qwen3_next) carry their KV state outside the shared block pool, so it cannot
be serialized across the handoff. A request against one of those families is
rejected at the prefill node with a clean per-request error before any
prefill work runs, and the node keeps serving subsequent requests (issue
#708). See [continuous batching and disaggregated serving](CONTINUOUS_BATCHING.md#scope-and-limitations)
for the full scope and limitations list.

A `--node-role router` front balances independent prefill and decode pools. It
picks both the prefill node and the decode node per request (round-robin over
the nodes the registry marks online), ships the chosen decode node to the
prefill node in the request frame so the prefill hands off to the
router-balanced decode node, marks unreachable peers down (via send-error
detection and a background liveness probe) and fails over to healthy nodes, and
applies admission control that returns HTTP 503 when no node can take the
request. `GET /router/stats` reports the per-node load spread and node health.
See [continuous batching and disaggregated serving](CONTINUOUS_BATCHING.md#multi-node-routing-load-balancing-and-failover)
for the flags and behavior.

### Security and trust model

The disaggregated role-transport connections (router to prefill, prefill to decode, and every result frame back to the router) authenticate with the same [cluster secret](#cluster-authentication) as the pipeline roles (issue #2655): every `--node-role prefill`, `decode` and `router` process must be given the secret and refuses to start without one, a peer holding a different secret is rejected at connect time, and a raw TCP client that skips the handshake never gets a frame to a role's receive loop. The handshake authenticates connections, not individual frames, and carries no TLS. The entire disaggregated cluster, including the router's client-facing HTTP port and all internal role-transport ports, should still run inside a trusted network segment or behind a network isolation boundary (VPC security group, firewall rule, or Kubernetes NetworkPolicy) that prevents untrusted callers from reaching the internal ports; the secret keeps outsiders off the role ports, it does not replace the isolation.

Launch shape with the secret (the same file on all three hosts; see [continuous batching and disaggregated serving](CONTINUOUS_BATCHING.md#example-three-processes-on-localhost) for the full flag set):

```bash
mlxcel-server -m models/<ckpt> --port 8302 --node-role decode \
    --serving-bind 10.0.0.2:8304 --cluster-secret-file /etc/mlxcel/cluster.secret ...
mlxcel-server -m models/<ckpt> --port 8301 --node-role prefill \
    --serving-bind 10.0.0.1:8305 --decode-peers 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret ...
mlxcel-server -m models/<ckpt> --port 8300 --node-role router \
    --serving-bind 10.0.0.3:8306 --prefill-peers 10.0.0.1:8305 --decode-peers 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

`--serving-bind` must name the IP the peers see this node connect from (not `0.0.0.0` on a multi-homed host), because the accepting side binds the advertised identity to the TCP source IP and refuses a mismatch with the identity rejection described under cluster authentication.

Two specific operational notes for the multi-node routing surface, both hardened with defense-in-depth beyond the trusted-segment baseline (issue #389):

- `GET /router/stats` is mounted on the router's client-facing HTTP port, the same surface as `/v1/chat/completions`. By default it redacts each registered peer's raw host:port and reports only the stable router-assigned node ids (`prefill-<i>` / `decode-<i>`), roles, health status, per-node dispatch counts, and the router metrics snapshot. That is enough to read the load spread and confirm a failed node is marked unreachable, without disclosing the internal cluster topology (the raw addresses) to a client that can reach the inference port. The response carries `"addresses_redacted": true` in this mode. Set `MLXCEL_ROUTER_STATS_VERBOSE=1` (also `true` / `yes` / `on`, case-insensitive) to opt a trusted-segment deployment back into the full address view (`"addresses_redacted": false`). The redaction is defense-in-depth, not a substitute for network isolation: keep the router's client port on the same trusted segment as the internal ports regardless.

- A prefill node connects to the address in the `decode_target` field of the incoming request frame to deliver the KV cache handoff, which encodes the prompt. The router populates that field only from its own configured node registry, so under the trusted-segment assumption the addresses are operator-controlled. As opt-in defense-in-depth the prefill node can validate `decode_target` against an allowlist before connecting: a target outside the allowlist is rejected (the request fails with a clear error and the KV cache never leaves for an unknown address) instead of connected to, so a forged frame from a breached segment cannot redirect the handoff off-cluster. The allowlist source is the dedicated `MLXCEL_DECODE_ALLOWLIST` environment variable, a comma-separated list of `host:port` addresses set to the FULL set of router-selectable decode nodes (the shared cluster config).

  Why a dedicated variable rather than `--decode-peers`: the router may hand any prefill node a `decode_target` for any decode node in its pool, so the allowlist must cover every router-selectable decode node, while a prefill's `--decode-peers` is only its static handoff fallback (its first entry, used when the router omits `decode_target`) and may legitimately list just one peer. Sourcing the allowlist from `--decode-peers` would reject every router-balanced target outside that static set and break the multi-node decode balancing added in issue #201. Keep the two independent: set `MLXCEL_DECODE_ALLOWLIST` to the full pool on each prefill node, and leave `--decode-peers` as the local fallback. When `MLXCEL_DECODE_ALLOWLIST` is unset (or parses to no valid entry), the prefill stays permissive and logs a warning rather than rejecting, so a missing allowlist source never silently disables balancing. Entries must be numeric `IP:port` values (parsed as `SocketAddr`); hostname:port entries do not parse and are skipped with a warning, so they will never match a router-chosen target. An entry that does not parse is skipped with a warning rather than failing startup. A request frame without `decode_target` (an older router) still falls back to the prefill's first static `--decode-peers` entry, which needs no validation because it is operator config rather than a wire-supplied address.

## Common limitations

- Distributed support is not uniform across model families.
- VLM partitioning is partial; text-only paths are better covered.
- Multi-host CI coverage is limited compared with single-host unit tests.
- Transport performance depends heavily on the physical interconnect and OS
  network configuration.

See [supported models](supported-models.md) for the maintained support summary.
