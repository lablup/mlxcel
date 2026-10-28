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

Both commands above also need the cluster secret described in the
[Cluster authentication](#cluster-authentication) section; without it each
process exits at startup with an error naming the flag, the environment
variables, and the TOML field.

### Zero-config bring-up

`--pp-auto N` and `--pp-peer` replace the hand-written TOML for a pipeline of
N stages. The coordinator (`--pp-auto N`) resolves N peer processes, plans
the cluster, and pushes one stage assignment to each peer over the
authenticated cluster handshake; a peer (`--pp-peer`) listens for that
assignment and then serves its stage. The coordinator runs no stage itself,
so `--pp-auto 2` needs two `--pp-peer` processes. `--pp-auto` is mutually
exclusive with `--distributed-config`; a peer started with
`--distributed-config` and `--node-id` skips the assignment wait and uses the
TOML as is.

What a peer checks before it accepts an assignment, in this order: the
assignment protocol version, the mlxcel version, the cluster name
(`--cluster-name`, default `mlxcel-cluster`), the model identity (SHA-256 of
`config.json`, the names and sizes of every `*.safetensors` file, the layer
count), the layer range the coordinator computed for the stage against the
range the peer derives from its own checkpoint, that the offered node id is
`stage-<i>` for the offered stage, that the offered cluster places this peer
at its listener port, and that both the coordinator's control address and the
coordinator node in the offered cluster match the identity the handshake
bound to the connection. A
protocol, mlxcel version, model, or layer range mismatch ends the peer
process with `stage assignment refused`, because a retry against the same
coordinator cannot succeed; a cluster name, address, or cluster TOML problem
sends it back to waiting. The coordinator commits only after every
peer accepted, and activates only after every peer committed: a peer writes
its cluster TOML and starts its stage on `Activate`, and a peer that
committed while another peer failed the commit phase receives `Abort` with
the failure summary and goes back to waiting, so a failed bring-up starts no
stage anywhere and its error names each peer that did not commit. A
coordinator that dies between two `Activate` writes leaves the peers it
already activated running, the same as a crash right after a successful
bring-up. Bounds: 5 s per handshake, 10 s per frame, 60 s for the
coordinator's whole push (its offer phase ends 10 s early, so the commit
phase always keeps a full 10 s frame window), 60 s for a peer's wait between
`Accepted` and `Commit` and again between `Committed` and `Activate`, 600 s
for a peer's wait for any assignment. The assignment protocol version is 2;
a coordinator or peer on version 1 is refused with `assignment protocol
mismatch`.

Three facts shape the commands below. The coordinator's control address
defaults to the `--host` IP on port 19000, and stages reply to it, so a
coordinator serving on `127.0.0.1` cannot drive peers on another host (it
refuses to start when discovery resolves a non-loopback peer). The model
directory must be the same checkpoint on every host, because the model
identity is compared byte for byte on `config.json`. And every host needs the
same secret file.

Secret distribution, once:

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
scp /etc/mlxcel/cluster.secret host-b:/etc/mlxcel/cluster.secret
scp /etc/mlxcel/cluster.secret host-c:/etc/mlxcel/cluster.secret
```

**Static discovery** (the coordinator knows the peer addresses). Each peer
names its own listen address with `--cluster-control-addr`, and the
coordinator lists exactly those addresses in `--cluster-peers`. Peers may
start before or after the coordinator; the coordinator retries the connect
and the handshake every 500 ms within its 60 s push window (a secret or
identity mismatch is not retried).

```bash
# host-b (peer, LAN IP 192.168.1.11)
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-control-addr 192.168.1.11:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-c (peer, LAN IP 192.168.1.12)
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-control-addr 192.168.1.12:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-a (coordinator, LAN IP 192.168.1.10)
mlxcel-server -m models/<checkpoint> --pp-auto 2 \
    --cluster-peers 192.168.1.11:19001,192.168.1.12:19001 \
    --cluster-control-addr 192.168.1.10:19000 \
    --host 0.0.0.0 --port 8080 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

**mdns discovery** (the coordinator learns the peer addresses from beacons).
Each peer sends an HMAC-signed UDP beacon every second to
`--cluster-discovery-target` on `--cluster-discovery-port` (default 49555)
until its assignment is committed; the coordinator listens on that port on
all interfaces for 30 s. A peer may omit `--cluster-control-addr`; it then
listens on `0.0.0.0` at an OS-assigned port and announces that port, so pass
a fixed address (as below) when a firewall has to allow the peer's port. The
default target is the limited broadcast `255.255.255.255`, which leaves only
through the default-route interface; a host whose cluster traffic uses a
second NIC or a Thunderbolt bridge passes that subnet's directed broadcast
(`192.168.1.255`), and a network that drops broadcast takes the
coordinator's unicast IP. Peers must be beaconing when the coordinator
starts, or within its 30 s discovery window.

```bash
# host-b and host-c (peers); the control port is OS-assigned only when
# --cluster-control-addr is omitted
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-discovery=mdns --cluster-discovery-target 192.168.1.255 \
    --cluster-control-addr 0.0.0.0:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-a (coordinator)
mlxcel-server -m models/<checkpoint> --pp-auto 2 \
    --cluster-discovery=mdns \
    --cluster-control-addr 192.168.1.10:19000 \
    --host 0.0.0.0 --port 8080 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

Firewall: open UDP 49555 (or `--cluster-discovery-port`) inbound on the
coordinator for mdns, TCP 19000 (or the `--cluster-control-addr` port)
inbound on the coordinator, and each peer's `--cluster-control-addr` TCP
port inbound on that peer (the coordinator connects to it for the assignment
and again for stage traffic, and the neighbouring stages connect to it for
activations). The HTTP port is only needed by clients.

`--dry-run` is for the `--pp-auto` coordinator only (it plans the cluster and
exits without assigning stages); `--pp-peer` combined with `--dry-run` is
refused at startup, before any socket is bound.

The coordinator writes its plan to `--cluster-config-out` (default
`.mlxcel/cluster.toml`) and each peer writes the cluster it received to its
own `--cluster-config-out` (default `.mlxcel/cluster-<node-id>.toml`). The
two files share the same body; their headers differ, because only the
coordinator's file is reproduced byte for byte by rerunning its command,
while the peer's file is overwritten by the next assignment it commits. Those
files are the input for a later manual run with `--distributed-config` and
`--node-id`; the stage ids are `stage-0`, `stage-1`, ... in the lexicographic
order of the peer addresses. A single-host loopback rehearsal of either mode
is `tests/zero_config_cluster_init.rs` (`zero_config_cli_*`), which drives
the real binaries with `--cluster-discovery-target 127.0.0.1`.

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
- **Every frame after the handshake.** Each connection derives its own
  directional keys from the handshake (HKDF-SHA256 over the secret and both
  nonces), and every frame on it is encrypted and authenticated with
  AES-256-GCM under an implicit per-direction sequence number. Activations,
  logits, KV handoffs and control commands are not readable on the wire, and a
  frame that was altered, replayed, reordered, or recorded on another
  connection fails authentication and closes the connection. See
  [Threat model and per-frame protection](#threat-model-and-per-frame-protection).
- **Command and reply authorization.** A stage accepts commands (`RunEntry`,
  `CancelRequest`, `BeginDrain`, `Shutdown`, ...) only from the coordinator
  node of its cluster config and forward activations only from its upstream
  stage; it replies to the authenticated source, never to an address named
  inside the payload. The forward activation carries the coordinator's
  exchange nonce, and the final stage sends the logits once, as the
  `RunResult` of that nonce, straight to the coordinator (3 frames per decoded
  token: `RunEntry`, forward activation, `RunResult`; a reverse activation
  reaching a stage is dropped). The coordinator accepts an `Ack`, `State` or
  `Error` only from the stage it addressed and a `RunResult` only from the
  final stage, only with the fresh per-exchange nonce it sent, only for the
  matching request and micro-batch, and only if the returned logits have the
  model's shape (`[1, seq_len, vocab_size]`); everything else is dropped while
  the exchange keeps waiting against an absolute deadline.
- **Discovery beacons and stage assignment.** With `--cluster-discovery=mdns`
  every beacon carries a timestamp, a nonce, and an HMAC over its fields. The
  coordinator rejects beacons without a valid MAC, beacons more than 60
  seconds off its clock or replayed within that window, beacons from a
  different mlxcel version, and beacons whose advertised address is not the
  datagram's source IP. Static discovery is unaffected. The stage assignment
  a `--pp-auto` coordinator pushes to each `--pp-peer` travels over the same
  connection handshake: the coordinator proves the secret before the peer
  reads an offer, and the peer proves it before the coordinator sends one,
  so a coordinator with another secret cannot assign a stage or a model, and
  the peer keeps waiting for the real one. The peer's assignment listener
  draws unauthenticated sockets from a pending pool of 8 (4 per source: an
  IPv4 address or an IPv6 /64) and closes the excess at accept.

- **Disaggregated role connections.** Since issue #2655 the router, prefill
  and decode role transports run the same connection handshake: the router's
  request frames, the prefill node's KV handoff to a decode node, and every
  result frame travel only over authenticated connections, and a raw TCP
  client that skips the handshake never gets a frame to any role's receive
  loop. The router refuses to start on a transport bound without the secret,
  and its result demux delivers a frame only when the sender's
  handshake-bound identity is one of its `--prefill-peers` or
  `--decode-peers` addresses (canonical IP and port). Only the source IP is
  authenticated (the handshake verifies it against the TCP connection); the
  port is the sender's advertised listener port and is not verified, so the
  check tells cluster members apart by source IP plus claimed port, not by
  port-level proof. Request ids are
  sequential, so any other holder of the secret, such as a pipeline stage
  that shares it, could otherwise inject tokens into a client's response. A
  frame from another member is dropped with a warning (`router: dropping
  result frame from ...`) and the request fails with the handoff timeout. The
  bound identity is the sender's `--serving-bind` address with an unspecified
  bind IP replaced by the connection's source IP, so a peer must be listed
  under the address it connects from (a multi-homed node, or one bound to
  `0.0.0.0` that reaches the router over another interface, needs that
  interface's address in the router's peer list). The decode role's handoff
  intake and the prefill role's request intake keep trusting any
  authenticated member; `MLXCEL_DECODE_ALLOWLIST` is unchanged.

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
  different cluster secret". A reset or abort after the server's authenticator
  is classified the same way as an orderly close.
- *Advertised identity not owned*: a node whose advertised listener IP is not
  the source IP the accepting side sees (a wrong bind or advertised address, or
  a NAT or port forward between the nodes) is refused with an explicit
  rejection record before any server nonce or authenticator is sent. The
  connecting side fails with "server refused this node's advertised identity
  `<host:port>`: the advertised IP must be the address the server sees this
  connection come from"; the accepting side logs "refusing to bind a listener
  address the peer does not own". An advertised identity that is not a socket
  address at all comes from no mlxcel client, so the connection is closed
  without a rejection record.
- *Closed before the handshake*: a connect-and-close liveness probe (the
  router health monitor, `probe_peers`) is logged at debug only.
- *At capacity*: a peer that authenticates while the accepting node already
  holds its `max_connections` authenticated connections is answered with the
  at-capacity verdict inside the handshake. The connecting side fails with
  "server authenticated this node but refused the connection: it already
  holds its maximum number of authenticated connections; retry later", which
  is not treated as fatal like a handshake rejection: the connection pool
  returns it without backoff and the next `acquire` reconnects, and
  `Transport::connect` logs a warning and continues; the accepting side logs "max connections (N)
  reached". The verdict is sent only after the client's authenticator
  verified, so a party without the secret never learns the capacity state.

These warnings are rate limited in two classes so that junk connections (bad
magic, a malformed hello, a stall to the 5 second timeout, a pending-pool
refusal) cannot push the typed refusals (probable secret mismatch, identity
not owned, suite mismatch, a frame that failed authentication) down to debug:
each class may log 5 warnings per minute, the rest go to debug, and the first
warning of a new minute reports how many lines of its class were suppressed
in the previous one (`N cluster handshake warnings (diagnostic) were logged
at debug in the last 60s`).

The handshake wire protocol is versioned (`PROTOCOL_VERSION` 4: 3 added the
frame protection suite and the derived per-frame keys in #2656, 4 added the
at-capacity verdict in #2692); a pair of processes built from different
versions fails with an explicit version message rather than a confusing I/O
error. Mixed versions in one cluster are not supported. Every version bump so
far landed after v0.8.0-beta.1, so no released version is affected.

Residual risks:

- Per-frame protection stops an on-path attacker from reading, altering, injecting, replaying or reordering frames on an authenticated connection; it does not stop one from dropping or delaying them. A dropped frame surfaces as the existing stage timeout or absolute deadline, exactly like a lost connection.
- Frame lengths and kinds (`TensorData`, `Control`, RPC request or response) are visible on the wire, as is the timing of every decode step. An observer can tell how many tokens a cluster is producing even though it cannot read any of them.
- Frames from authenticated members stay trusted inside the pipeline protocol and on the prefill and decode intakes of the disaggregated protocol: a host that holds the secret is a full member there. The router's result demux is the exception (it accepts frames only from its configured peers). Rotate the secret when a member is decommissioned.
- Connections that have not finished the handshake do not count against the 512 authenticated-connection cap. They draw from a separate pending pool (64 in flight, at most 8 per source, each for at most the 5 second handshake timeout); a socket beyond either bound is closed at accept. A source is an IPv4 address or an IPv6 /64 (a host that owns a /64 would otherwise get one pending slot per address), which also means cluster nodes inside one IPv6 /64 share the per-source budget and a large IPv6 cluster in a single /64 may need a higher `max_pending_handshakes_per_ip`. A flood of unauthenticated connections from enough sources (8 or more at the defaults) can fill the pending pool and so delay new connections from any source, including a real peer reconnecting, by up to the 5 second timeout per slot while the flood lasts; it cannot take slots from established authenticated connections or inject frames. Failed inbound handshakes and pending-pool refusals are logged at warn for the first few per minute (per class, see above) and at debug afterwards.
- A party without the secret can still produce the "probable secret mismatch" warning (send a well-formed hello naming its own IP, then close after the server's authenticator). The "identity not owned" and "suite mismatch" refusals are decided from the hello alone, before the client authenticates, so it can trigger those as well. Any of them can saturate the diagnostic warning budget. The per-minute summary of suppressed warnings is what keeps the operator informed then.
- The receiver refuses an oversize `len` from the header alone, but a frame whose `len` is within the cap (256 MiB by default) makes it allocate and read the full body before the tag is checked. Only a peer that completed the handshake can reach this path, so it is a resource bound for authenticated members, not an exposure to the open network.
- `MLXCEL_CLUSTER_SECRET` is inherited by child processes of the server. Prefer `--cluster-secret-file` or `MLXCEL_CLUSTER_SECRET_FILE` so the value never sits in the environment.
- The server identity is not bound to the socket IP on the connecting side, so a stage address that is reached through NAT or a port forward still authenticates by the secret alone.

### Threat model and per-frame protection

The cluster transports assume one of two networks:

- **Trusted segment** (a private VLAN, a Thunderbolt bridge, a lab switch): the
  only parties on the wire are the cluster's own hosts. The handshake exists
  here to keep a misconfigured or stray process out, and per-frame protection
  costs a little latency for no attacker in particular.
- **Untrusted network** (a shared office LAN, a cloud VPC with other tenants,
  anything reachable from hosts the operator does not control): an attacker
  can be on the path (ARP or DHCP spoofing, a compromised switch, a host on
  the same segment) and can read, inject, alter, drop, replay or reorder
  packets, and can relay one peer's connection to the real server.

What each layer covers:

| Layer | Covers | Does not cover |
|-------|--------|----------------|
| Connect-time handshake (`PROTOCOL_VERSION` 4) | Only a holder of the secret can open or accept a connection; the accepted identity is bound to the TCP source IP; the frame suite is agreed inside the authenticated transcript; a full connection cap is reported to an authenticated peer as a typed verdict that is not fatal. | Anything that happens on the TCP stream after the handshake. |
| Per-frame protection (#2656) | Every frame after the handshake: confidentiality of the payload, integrity of header and payload, replay and reorder rejection through an implicit sequence number, binding to the connection whose handshake derived the keys. Defeats the relay, which cannot derive keys without the secret. Covers pooled outbound connections, `send_stream`, RPC requests and responses, the Thunderbolt and RDMA wrappers (same TCP core), the disaggregated router, prefill and decode transports, and the `--pp-auto` stage assignment channel. | Dropping or delaying frames; frame lengths, kinds and timing. |
| Discovery beacons (`--cluster-discovery=mdns`) | Each UDP beacon carries an HMAC, a timestamp and a nonce; the coordinator rejects an invalid MAC, a 60 s clock skew, a replay inside that window, another mlxcel version, or an advertised address that is not the datagram's source. | Hiding the fact that a cluster is forming; beacon payloads are readable. |

Design (chosen from the measurements below). After the handshake each side
derives two 32-byte keys with HKDF-SHA256: IKM is the cluster secret, the salt
is the client nonce followed by the server nonce, and the info strings are
`mlxcel-cluster-frame-v1/<channel>/c2s` and `mlxcel-cluster-frame-v1/<channel>/s2c`,
where `<channel>` is `transport` for the TCP transport core and `assignment`
for the `--pp-auto` stage assignment channel, so the two never share keys even
when their handshake inputs match. A frame is
`[kind][len][ciphertext][tag]`: the payload is encrypted in place with
AES-256-GCM under the direction's key, the 96-bit nonce is the frame's
sequence number (starting at 0, never sent, never wrapped: a sender that
reaches the end closes the connection), and the clear-text `kind || len`
header is the associated data, so the receiver refuses an oversize `len`
before any decryption work. A frame that fails to decrypt closes the
connection; the event is logged through the diagnostic class of the same
rate limiter as a failed handshake, never with payload bytes. Keys are per
connection, so a frame recorded on one connection fails on every other. The suite is announced in
the handshake hello and covered by both authenticators; a node configured for
a different suite is refused with an explicit record. Nothing new is
configured: the secret sources above are the only input, and there is no
opt-out.

Rejected candidates:

- *Per-frame HMAC-SHA256 with the same keys and sequence (candidate 1)*: gives
  integrity, replay and reorder rejection and connection binding, but leaves
  activations, prompts in KV handoffs and commands readable on the wire. The
  measurements put AES-256-GCM inside the decision rule's margin, so the
  variant without confidentiality was not needed.
- *ChaCha20-Poly1305 (the other AEAD of candidate 2)*: the same guarantees;
  measured as the alternative AEAD and kept in the code for the benchmark.
  Both GB10 (ARM Neoverse with AES and PMULL extensions) and Apple silicon
  have AES instructions, so AES-256-GCM is the default.
- *TLS 1.3 through `tokio-rustls` with an ephemeral certificate per node and
  the HMAC handshake bound to the TLS exporter (candidate 3)*: not prototyped.
  Its record layer runs the same AEAD primitives as candidate 2 plus 16 KiB
  record framing, a second (TLS) handshake per connection and certificate
  generation at startup, so it cannot be faster than candidate 2, and the
  decision rule prefers 2 unless 3 is no slower. It would also add `rcgen` as
  a runtime dependency.

Measurements (loopback on one NVIDIA GB10 host, 2026-10-27, nothing else
running; one-way latency is half the round trip of a `TensorData` message the
peer echoes back over a second authenticated connection, so it includes both
ends' frame work):

| Mode | Payload | Iterations | p50 | p95 | p99 | One-way bandwidth |
|------|--------:|-----------:|----:|----:|----:|------------------:|
| unprotected (baseline, main before #2656) | 8 KiB | 4000 | 5.5 us | 5.6 us | 6.2 us | 1.38 GiB/s |
| unprotected | 64 KiB | 2000 | 11.1 us | 11.2 us | 12.0 us | 5.51 GiB/s |
| unprotected | 1 MiB | 400 | 411 us | 418 us | 419 us | 2.37 GiB/s |
| unprotected | 2 MiB | 200 | 1.040 ms | 1.059 ms | 1.096 ms | 1.87 GiB/s |
| unprotected | 16 MiB | 60 | 16.39 ms | 16.52 ms | 16.62 ms | 0.95 GiB/s |
| candidate 1: HMAC-SHA256 | 8 KiB | 4000 | 12.2 us | 12.4 us | 12.9 us | 0.63 GiB/s |
| candidate 1: HMAC-SHA256 | 64 KiB | 2000 | 64.1 us | 64.8 us | 65.3 us | 0.95 GiB/s |
| candidate 1: HMAC-SHA256 | 1 MiB | 400 | 1.008 ms | 1.016 ms | 1.025 ms | 0.97 GiB/s |
| candidate 1: HMAC-SHA256 | 2 MiB | 200 | 2.115 ms | 2.150 ms | 2.216 ms | 0.92 GiB/s |
| candidate 1: HMAC-SHA256 | 16 MiB | 60 | 27.22 ms | 27.56 ms | 28.29 ms | 0.57 GiB/s |
| candidate 2: AES-256-GCM (chosen) | 8 KiB | 4000 | 7.4 us | 7.6 us | 8.2 us | 1.02 GiB/s |
| candidate 2: AES-256-GCM (chosen) | 64 KiB | 2000 | 27.4 us | 27.9 us | 28.5 us | 2.23 GiB/s |
| candidate 2: AES-256-GCM (chosen) | 1 MiB | 400 | 422 us | 426 us | 434 us | 2.31 GiB/s |
| candidate 2: AES-256-GCM (chosen) | 2 MiB | 200 | 0.926 ms | 0.944 ms | 0.954 ms | 2.10 GiB/s |
| candidate 2: AES-256-GCM (chosen) | 16 MiB | 60 | 17.93 ms | 18.09 ms | 18.38 ms | 0.87 GiB/s |
| candidate 2: ChaCha20-Poly1305 | 8 KiB | 4000 | 14.0 us | 14.4 us | 14.9 us | 0.56 GiB/s |
| candidate 2: ChaCha20-Poly1305 | 64 KiB | 2000 | 75.4 us | 76.2 us | 76.8 us | 0.81 GiB/s |
| candidate 2: ChaCha20-Poly1305 | 1 MiB | 400 | 1.192 ms | 1.204 ms | 1.220 ms | 0.82 GiB/s |
| candidate 2: ChaCha20-Poly1305 | 2 MiB | 200 | 2.479 ms | 2.489 ms | 2.504 ms | 0.79 GiB/s |
| candidate 2: ChaCha20-Poly1305 | 16 MiB | 60 | 30.16 ms | 30.55 ms | 30.65 ms | 0.52 GiB/s |
| any mode | any | two-node link (Thunderbolt, 10/25 GbE) | not measured on this run | | | |

The 1 MiB and 2 MiB rows bracket the per-token logits reply under main's JSON
number-array encoding; with the base64 encoding below that reply is about
0.4 MB, between the 64 KiB and 1 MiB rows. At those sizes the AES-256-GCM rows sit inside loopback noise of the
unprotected ones (hardware AES and PMULL run faster than the loopback copy),
while HMAC-SHA256 adds 0.6 and 1.2 ms per one-way transfer and
ChaCha20-Poly1305 0.8 and 1.4 ms.

Decision rule (from #2656): choose a confidentiality-providing option when its
added p50 one-way latency at 8 KiB is within 20 us or 5 percent of candidate 1,
whichever is larger, and end-to-end decode drops by at most 2 percent. At
8 KiB AES-256-GCM is 4.8 us *faster* than the HMAC candidate (ring's GCM uses
the AES and PMULL instructions, its SHA-256 path is slower on this core) and
1.9 us above the unprotected baseline, so the latency clause holds with room to
spare; ChaCha20-Poly1305 is 1.8 us slower than HMAC and was not preferred.

End-to-end greedy decode of the two-stage remote pipeline (qwen3-0.6b-4bit,
128 tokens, both stages and the coordinator in one process over loopback,
`tests/pipeline_remote_decode_bench.rs`), runs interleaved baseline and
candidate:

| Binary | Runs | Median of per-run medians | Per-run medians (min to max) |
|--------|-----:|--------------------------:|------------------------------|
| main 4144ecfb, unprotected, array-encoded tensors | 5 | 68.3 tok/s | 65.4 to 72.7 tok/s |
| main cfa14e7b (HEAD), unprotected, array-encoded tensors | 5 | 68.7 tok/s | 60.7 to 69.7 tok/s |
| this branch with the codec disabled (attribution only): compact tensors, no protection | 5 | 138.8 tok/s | 136.5 to 146.7 tok/s |
| this branch: compact tensors, AES-256-GCM on every frame | 5 | 133.2 tok/s | 122.2 to 145.1 tok/s |

Against the main baseline the protected pipeline decodes 94 percent faster,
but that gain comes from the compact encoding shipped with it. Read on the
protection alone, the 2 percent clause is not met: the cost of the protection
itself, read from the two compact rows, is 4 percent of a 7.3 ms step (about
0.3 ms per token for two 405 KB frames plus the small ones), with overlapping
spreads; it is a fixed per-token cost that shrinks in proportion on the models
that need pipeline parallelism. The rule's fallback, candidate 1, costs more
than AES-256-GCM at every payload size above (1.5x to 2.4x) and leaves payloads
in clear text, so falling back would lower throughput further. Decision:
AES-256-GCM, no opt-out.

Attribution (throwaway instrumentation, not committed): a frame trace of the
bench showed four frames per decoded token, a 275-byte `RunEntry`, a 7.5 KB
forward activation, and two frames of 1.1 MB each carrying the same 304 KB of
f16 logits as a JSON array of numbers (the final stage returns them to the
entry stage, which forwards them to the coordinator). In-process AES-256-GCM
over those 2.2 MB costs 0.54 ms per token (seal 275 us, open 268 us;
HMAC-SHA256 would cost 1.77 ms), and a framing-only variant of the codec
decoded no faster than the protected one, so the first measured gap (71.3
versus 63.5 tok/s on the array encoding) was the AEAD cost plus run-to-run
noise, not the framing. Encoding those fields compactly (below) removed most
of the step time of main itself, which was spent encoding and parsing JSON
number arrays, and left the protected pipeline well ahead of the unprotected
baseline.

Wire change in the same release: the bulk byte fields of the stage protocol
(`RunEntry.input_ids` and `attention_mask`, `RunResult.logits`, and the
activation message's `tensor_data`, `attention_mask` and `position_ids`) are
base64 strings inside the JSON documents instead of JSON arrays of numbers
(1.33 instead of about 3.6 characters per byte). A decode token of
qwen3-0.6b-4bit used to cross the wire twice as a 1.1 MB frame (the final
stage returned the logits to the entry stage, which forwarded them to the
coordinator); it became about 0.4 MB each way, which cuts both the baseline
step time and the bytes the AEAD covers. A build with this encoding does not
interoperate with one without it; mixed versions were already unsupported.

Follow-up (#2694): the final stage now sends the `RunResult` straight to the
coordinator, so a decoded token is 3 frames instead of 4 and the logits cross
the wire once; the sender also seals a message in place instead of copying
the payload into a second buffer before sealing. The forward activation
carries the exchange nonce in a new `reply_nonce` field (absent means `None`
on an older receiver, which then drops the logits with a warning rather than
guessing an exchange).

The two-node link rows (Thunderbolt or 10/25 GbE) were not measured on this
run. The benchmark (`cargo test --test cluster_frame_bench -- --ignored
--nocapture`) binds both transports on 127.0.0.1, so a two-node measurement
needs `measure_activation_transfer` (round-trip mode) run against a transport
on the second host.

## Transports

| Transport | Notes |
|-----------|-------|
| TCP | Default IP transport. |
| Thunderbolt | macOS Thunderbolt Bridge selection on top of the shared TCP core. |
| RDMA | Backend exists with capability probing and fallback behavior; validate on the target OS/hardware before relying on acceleration. |

Static and mdns discovery are both available for the
[zero-config bring-up](#zero-config-bring-up). Static discovery is the choice
across subnets or on networks that drop UDP broadcast; mdns needs the peers'
beacons to reach the coordinator's `--cluster-discovery-port`.

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

The disaggregated role-transport connections (router to prefill, prefill to decode, and every result frame back to the router) authenticate with the same [cluster secret](#cluster-authentication) as the pipeline roles (issue #2655): every `--node-role prefill`, `decode` and `router` process must be given the secret and refuses to start without one, a peer holding a different secret is rejected at connect time, and a raw TCP client that skips the handshake never gets a frame to a role's receive loop. Every frame on those connections is then encrypted and authenticated with keys derived from the handshake (issue #2656, see [Threat model and per-frame protection](#threat-model-and-per-frame-protection)); there is no TLS. The entire disaggregated cluster, including the router's client-facing HTTP port and all internal role-transport ports, should still run inside a trusted network segment or behind a network isolation boundary (VPC security group, firewall rule, or Kubernetes NetworkPolicy) that prevents untrusted callers from reaching the internal ports; the secret keeps outsiders off the role ports, it does not replace the isolation.

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
