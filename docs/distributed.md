# Distributed inference

`mlxcel` has three distributed or multi-device surfaces. They share code under
`src/distributed/`, but their maturity differs by mode and model family.

| Mode | Purpose | Maturity |
|------|---------|----------|
| Tensor parallelism (TP) | Shard tensor operations across in-process ranks. | Implemented for selected dense text families; validate per model. |
| Pipeline parallelism (PP) | Split layer ranges across stages. | Best validated on Llama-family text models and two-stage topologies. |
| Disaggregated inference (DI) | Split prefill/decode roles while each node holds the model. | Infrastructure exists; treat as experimental unless validated for your topology. |

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

Limitations:

- The model must shard cleanly across the selected rank count.
- Some server batching and VLM paths are intentionally conservative under TP.
- Benchmark and correctness validation should be repeated for every model family
  and rank count you intend to run.

## Pipeline parallelism

PP splits the model by layer range. It is useful when a model exceeds a single
device's memory or when hosts have uneven memory capacity.

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
    --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \
    --node-id stage-1 \
    --host 0.0.0.0 --port 18081 --no-warmup

# Coordinator / serving process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \
    --node-id coordinator \
    --host 0.0.0.0 --port 18080 \
    --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2
```

`--pp-auto N` can generate a zero-config pipeline plan and is mutually exclusive
with `--distributed-config`. For production, prefer checking in an explicit TOML
once the topology is known.

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

The disaggregated role-transport connections (router to prefill, prefill to decode) carry no authentication or TLS, the same assumption the pipeline-parallelism transport already makes. The entire cluster, including the router's client-facing HTTP port and all internal role-transport ports, should run inside a trusted network segment or behind a network isolation boundary (VPC security group, firewall rule, or Kubernetes NetworkPolicy) that prevents untrusted callers from reaching the internal ports.

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
