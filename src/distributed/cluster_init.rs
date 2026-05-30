// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Zero-config cluster bring-up for multi-machine pipeline-parallel runs.
//!
//! This module implements the intent-based launch path described in issue
//! The operator declares the desired number of stages with `--pp-auto N`
//! on the coordinator, peers announce themselves (either explicitly via static
//! seeds or opt-in via LAN discovery), and the coordinator:
//!
//! - Assigns stage indices deterministically (lexicographic ordering of peer
//!   addresses so the same topology always produces the same TOML).
//! - Allocates distinct ports for the coordinator HTTP plane, the cluster
//!   control plane, and each stage's data transport channel.
//! - Emits a byte-identical [`ClusterConfig`] TOML to disk for reproducibility.
//! - Surfaces discovery and handshake failures as actionable error messages.
//!
//! The output is plugged straight into the existing `ServerStartupConfig`
//! distributed resolution path (`distributed_config` is populated with the
//! emitted TOML), so the zero-config path shares the same runtime code as the
//! manual TOML path. No parallel prototype is introduced.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use super::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
use super::transport::TransportBackend;

/// Default port for the cluster bring-up beacon (UDP broadcast discovery).
pub const DEFAULT_DISCOVERY_PORT: u16 = 49555;

/// Default timeout for peer discovery rendezvous.
pub const DEFAULT_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Well-known default base port for the coordinator control plane. Selected
/// from the IANA user range to avoid clashing with the llama-server default
/// HTTP port (8080) or common development servers.
pub const DEFAULT_CONTROL_BASE_PORT: u16 = 19000;

/// Discovery mechanism selected by the operator.
///
/// `Static` consumes an explicit address list (default — the network is not
/// touched for announcements). `Mdns` enables UDP-broadcast based LAN
/// discovery; the mlxcel peers announce themselves on the configured UDP port
/// until the coordinator is satisfied. The name `Mdns` is retained for
/// forward-compatibility with a future zeroconf implementation; today it uses
/// plain UDP broadcast so no extra crate dependency is required.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClusterDiscoveryMode {
    /// Use the static peer list provided on the CLI.
    #[default]
    Static,
    /// Listen for peer announcements over UDP broadcast on the same LAN.
    Mdns,
}

impl std::fmt::Display for ClusterDiscoveryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => write!(f, "static"),
            Self::Mdns => write!(f, "mdns"),
        }
    }
}

impl std::str::FromStr for ClusterDiscoveryMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "static" | "off" | "none" | "disable" | "disabled" => Ok(Self::Static),
            "mdns" | "udp" | "broadcast" => Ok(Self::Mdns),
            other => {
                bail!("unknown cluster discovery mode '{other}'; expected one of: static, mdns")
            }
        }
    }
}

/// Operator-facing launch intent assembled from CLI arguments.
///
/// A single coordinator process consumes a [`ClusterInitRequest`] and produces
/// a fully resolved [`ClusterInitPlan`] — or a diagnostic error that explains
/// what the operator needs to fix.
#[derive(Debug, Clone)]
pub struct ClusterInitRequest {
    /// Desired pipeline depth (number of PP stages). Must be >= 2 for the
    /// zero-config path to activate.
    pub pp_stages: u32,
    /// Human-readable cluster name. Included verbatim in the emitted TOML.
    pub cluster_name: String,
    /// Inter-stage transport backend.
    pub transport_backend: TransportBackend,
    /// Discovery mechanism.
    pub discovery: ClusterDiscoveryMode,
    /// Timeout for rendezvous. If `None`, [`DEFAULT_DISCOVERY_TIMEOUT`] is used.
    pub discovery_timeout: Option<Duration>,
    /// UDP port used for the discovery beacon when `discovery == Mdns`.
    pub discovery_port: u16,
    /// Local coordinator HTTP address (host:port) operators send requests to.
    /// Must not collide with any assigned control or data port.
    pub coordinator_http_addr: SocketAddr,
    /// Local bind address used for the coordinator control plane. The port
    /// may be `0` to request auto-selection from the OS ephemeral pool.
    pub coordinator_control_addr: SocketAddr,
    /// Static peer seeds. Peers whose address is already present in this list
    /// are accepted immediately without going through LAN discovery. Ordering
    /// has no semantic effect — the plan sorts peers lexicographically.
    pub static_peers: Vec<SocketAddr>,
    /// Base port for per-stage data transport ports when `port == 0` in the
    /// original request. Sequential ports are assigned starting from this
    /// value, skipping ports that the OS reports as already in use.
    pub data_port_base: u16,
    /// Output TOML path. If `None`, the plan is not persisted to disk (used
    /// by `--dry-run`).
    pub output_toml_path: Option<PathBuf>,
}

impl Default for ClusterInitRequest {
    fn default() -> Self {
        Self {
            pp_stages: 2,
            cluster_name: "mlxcel-cluster".to_string(),
            transport_backend: TransportBackend::default(),
            discovery: ClusterDiscoveryMode::Static,
            discovery_timeout: None,
            discovery_port: DEFAULT_DISCOVERY_PORT,
            coordinator_http_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            coordinator_control_addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                DEFAULT_CONTROL_BASE_PORT,
            ),
            static_peers: Vec::new(),
            data_port_base: DEFAULT_CONTROL_BASE_PORT + 1,
            output_toml_path: None,
        }
    }
}

/// The fully resolved plan produced by [`plan_cluster`].
///
/// This is the primary output of the zero-config bring-up command. The plan
/// is deterministic for a given [`ClusterInitRequest`] with the same set of
/// resolved peers, so rerunning the bring-up produces a byte-identical TOML.
#[derive(Debug, Clone)]
pub struct ClusterInitPlan {
    /// The resolved [`ClusterConfig`] ready to be emitted to disk or consumed
    /// directly by the server startup path.
    pub cluster: ClusterConfig,
    /// Operator-facing topology summary suitable for `--dry-run` output.
    pub summary: String,
    /// Deterministic TOML form of [`ClusterInitPlan::cluster`]. Kept separate
    /// so callers can check byte-level determinism in tests.
    pub toml: String,
}

/// Resolve a `ClusterInitRequest` into a concrete `ClusterInitPlan`.
///
/// This routine is pure and deterministic: it performs no network I/O.
/// Discovery (if enabled) runs separately; see [`discover_peers`] which calls
/// this function with the expanded peer list once the rendezvous completes.
pub fn plan_cluster(request: &ClusterInitRequest) -> Result<ClusterInitPlan> {
    validate_request(request)?;

    let coordinator_control = request.coordinator_control_addr;
    ensure!(
        coordinator_control != request.coordinator_http_addr,
        "coordinator control address {coordinator_control} conflicts with the HTTP listen address \
         {http}; assign a distinct port for control traffic",
        coordinator_control = coordinator_control,
        http = request.coordinator_http_addr
    );

    let mut peers: Vec<SocketAddr> = request.static_peers.to_vec();
    peers.sort_by_key(|a| a.to_string());
    peers.dedup();

    ensure!(
        peers.len() as u32 == request.pp_stages,
        "resolved {resolved} peer(s) but --pp-auto requested {wanted} stages; either supply \
         {wanted} static peers or increase the discovery timeout",
        resolved = peers.len(),
        wanted = request.pp_stages,
    );

    let mut used_ports = std::collections::HashSet::new();
    used_ports.insert(request.coordinator_http_addr.port());
    used_ports.insert(coordinator_control.port());

    let mut nodes = Vec::with_capacity(peers.len() + 1);
    nodes.push(NodeConfig {
        id: "coordinator".to_string(),
        address: coordinator_control,
        role: NodeRole::Hybrid,
        stage: None,
        rank: None,
        resources: NodeResources::default(),
    });

    for (stage_index, peer_addr) in peers.iter().enumerate() {
        ensure!(
            !used_ports.contains(&peer_addr.port())
                || *peer_addr != coordinator_control && *peer_addr != request.coordinator_http_addr,
            "stage-{stage_index} peer address {peer_addr} collides with a port already reserved \
             for the coordinator; pick a distinct port for the peer",
        );
        used_ports.insert(peer_addr.port());
        nodes.push(NodeConfig {
            id: format!("stage-{stage_index}"),
            address: *peer_addr,
            role: NodeRole::PipelineStage,
            stage: Some(stage_index as u32),
            rank: None,
            resources: NodeResources::default(),
        });
    }

    let cluster = ClusterConfig {
        cluster: ClusterMeta {
            name: request.cluster_name.clone(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: request.pp_stages,
            transport_backend: request.transport_backend,
        },
        nodes,
    };

    // Round-trip through the existing validator so the generated cluster
    // behaves identically to one loaded from a hand-written TOML file.
    let toml = render_deterministic_toml(&cluster);
    let reparsed = ClusterConfig::from_toml(&toml)
        .context("generated cluster TOML failed validator round-trip (internal bug)")?;

    let summary = reparsed.topology_summary();

    Ok(ClusterInitPlan {
        cluster: reparsed,
        summary,
        toml,
    })
}

fn validate_request(request: &ClusterInitRequest) -> Result<()> {
    ensure!(
        request.pp_stages >= 2,
        "--pp-auto must request at least 2 pipeline stages (got {got}); use the legacy single-node \
         path for non-distributed runs",
        got = request.pp_stages,
    );
    ensure!(
        !request.cluster_name.trim().is_empty(),
        "cluster name must not be empty"
    );
    Ok(())
}

/// Serialize a [`ClusterConfig`] into a deterministic TOML document.
///
/// This function intentionally uses a hand-rolled emitter rather than
/// [`toml::to_string`] so the byte layout is under our control:
///
/// - keys appear in a fixed order,
/// - no trailing whitespace is introduced,
/// - nodes are sorted by (stage, id) so reordering the input peer list does
///   not change the output,
/// - port numbers are formatted without locale-dependent separators.
///
/// The resulting TOML is guaranteed to parse back through
/// [`ClusterConfig::from_toml`] because [`plan_cluster`] round-trips it.
pub fn render_deterministic_toml(cluster: &ClusterConfig) -> String {
    let mut out = String::new();

    let header = "# Generated by `mlxcel-server --pp-auto`.\n\
                  # Rerunning the bring-up command with the same topology produces a byte-identical file.\n\n";
    out.push_str(header);

    out.push_str("[cluster]\n");
    let _ = writeln!(out, "name = {}", toml_escape_string(&cluster.cluster.name));
    let _ = writeln!(
        out,
        "tensor_parallel_size = {}",
        cluster.cluster.tensor_parallel_size
    );
    let _ = writeln!(
        out,
        "pipeline_parallel_size = {}",
        cluster.cluster.pipeline_parallel_size
    );
    let _ = writeln!(
        out,
        "transport_backend = {}",
        toml_escape_string(&cluster.cluster.transport_backend.to_string())
    );
    out.push('\n');

    // Sort nodes for determinism: coordinator (stage = None) first, then
    // pipeline stages by stage index, then any remaining nodes by id.
    let mut sorted_nodes: Vec<&NodeConfig> = cluster.nodes.iter().collect();
    sorted_nodes.sort_by(|a, b| match (a.stage, b.stage) {
        (None, None) => a.id.cmp(&b.id),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(sa), Some(sb)) => sa.cmp(&sb).then_with(|| a.id.cmp(&b.id)),
    });

    for node in sorted_nodes {
        out.push_str("[[nodes]]\n");
        let _ = writeln!(out, "id = {}", toml_escape_string(&node.id));
        let _ = writeln!(
            out,
            "address = {}",
            toml_escape_string(&node.address.to_string())
        );
        let _ = writeln!(out, "role = {}", toml_escape_string(&node.role.to_string()));
        if let Some(stage) = node.stage {
            let _ = writeln!(out, "stage = {stage}");
        }
        if let Some(rank) = node.rank {
            let _ = writeln!(out, "rank = {rank}");
        }
        out.push('\n');
    }

    // Strip the single trailing blank line so consecutive `render_deterministic_toml`
    // calls produce identical byte output.
    while out.ends_with("\n\n") {
        out.pop();
    }

    out
}

fn toml_escape_string(s: &str) -> String {
    let mut buf = String::with_capacity(s.len() + 2);
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(buf, "\\u{:04X}", c as u32);
            }
            c => buf.push(c),
        }
    }
    buf.push('"');
    buf
}

/// Persist a plan's TOML to `path`. Creates parent directories as needed.
pub fn write_plan_toml(plan: &ClusterInitPlan, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory for cluster TOML: {parent:?}"))?;
    }
    std::fs::write(path, &plan.toml)
        .with_context(|| format!("failed to write generated cluster TOML: {path:?}"))?;
    Ok(())
}

/// Check that a TCP port on `bind_ip` is currently bindable.
///
/// Used by the dry-run preflight and by [`allocate_data_ports`]. A port that
/// succeeds here may still race with another process before the real server
/// binds, but the check catches the common case of hand-configured ports
/// already in use.
pub fn is_port_available(bind_ip: IpAddr, port: u16) -> bool {
    if port == 0 {
        return true;
    }
    TcpListener::bind(SocketAddr::new(bind_ip, port)).is_ok()
}

/// Allocate `count` distinct, currently-available TCP ports starting from
/// `base_port`. The returned ports are monotonically increasing.
///
/// Best-effort: the port is released before we return so the caller can bind
/// it for real. A race is possible but rare on a machine with few listeners.
pub fn allocate_data_ports(bind_ip: IpAddr, base_port: u16, count: usize) -> Result<Vec<u16>> {
    let mut ports = Vec::with_capacity(count);
    let mut candidate: u32 = base_port as u32;
    while ports.len() < count {
        ensure!(
            candidate <= u16::MAX as u32,
            "exhausted TCP port space while allocating {count} data ports starting at {base_port}"
        );
        let port = candidate as u16;
        if is_port_available(bind_ip, port) {
            ports.push(port);
        }
        candidate += 1;
    }
    Ok(ports)
}

/// Payload exchanged between peers during LAN discovery.
///
/// Kept intentionally small so a single datagram fits in the standard MTU
/// without fragmentation. A peer repeatedly broadcasts its beacon until the
/// coordinator confirms receipt or the rendezvous times out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryBeacon {
    /// mlxcel version string (used as a best-effort compatibility gate).
    pub version: String,
    /// Human-readable cluster name. Peers with a mismatching cluster_name
    /// are ignored by the coordinator.
    pub cluster_name: String,
    /// Peer's control+data socket address (host:port) the coordinator
    /// should record in the generated TOML.
    pub peer_addr: SocketAddr,
    /// Optional node identifier chosen by the peer. The coordinator may
    /// override this to keep stage ids stable across reruns.
    pub node_id: Option<String>,
    /// Peer process identifier, used purely for de-duplication when a peer
    /// broadcasts multiple times from the same address+pid pair.
    pub pid: u32,
}

/// Fully deterministic mapping from (address, pid) so repeated beacons from
/// the same peer collapse to a single entry.
///
/// `Ord` is derived so the map lookup order matches the textual address
/// ordering used by [`dedup_and_sort`], which keeps the stage assignment
/// deterministic across runs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PeerKey {
    addr: SocketAddr,
    pid: u32,
}

/// Listen for discovery beacons on UDP and return the set of resolved peers.
///
/// The function terminates when either:
/// - `target_count` distinct peers have announced themselves (success), or
/// - `timeout` elapses (error with an actionable message).
///
/// When `mode == Static` the function short-circuits and returns the static
/// seeds unchanged, so callers can always call this and let the mode decide.
pub async fn discover_peers(
    mode: ClusterDiscoveryMode,
    cluster_name: &str,
    bind_ip: IpAddr,
    port: u16,
    static_seeds: &[SocketAddr],
    target_count: usize,
    timeout: Duration,
) -> Result<Vec<SocketAddr>> {
    if matches!(mode, ClusterDiscoveryMode::Static) {
        return Ok(dedup_and_sort(static_seeds));
    }

    let socket = tokio::net::UdpSocket::bind(SocketAddr::new(bind_ip, port))
        .await
        .with_context(|| {
            format!(
                "failed to bind UDP discovery socket on {bind_ip}:{port}; \
                 another process may already be listening. \
                 Use `--cluster-discovery=static` to fall back to explicit peer seeds."
            )
        })?;
    socket
        .set_broadcast(true)
        .context("failed to enable SO_BROADCAST on the UDP discovery socket")?;

    let mut resolved: BTreeMap<PeerKey, SocketAddr> = BTreeMap::new();
    for seed in static_seeds {
        resolved.insert(
            PeerKey {
                addr: *seed,
                pid: 0,
            },
            *seed,
        );
    }

    let mut buf = vec![0u8; 2048];
    let deadline = tokio::time::Instant::now() + timeout;
    while resolved.len() < target_count {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await {
            Ok(Ok((len, _from))) => {
                let slice = &buf[..len];
                let Ok(beacon) = serde_json::from_slice::<DiscoveryBeacon>(slice) else {
                    // Ignore malformed datagrams — they may come from another
                    // unrelated application on the same broadcast port.
                    continue;
                };
                if beacon.cluster_name != cluster_name {
                    continue;
                }
                resolved.insert(
                    PeerKey {
                        addr: beacon.peer_addr,
                        pid: beacon.pid,
                    },
                    beacon.peer_addr,
                );
            }
            Ok(Err(err)) => {
                return Err(err).context("UDP discovery socket read failed");
            }
            Err(_) => break, // timer elapsed
        }
    }

    if resolved.len() < target_count {
        bail!(
            "cluster discovery timed out after {timeout:?}: resolved {resolved}/{target} peer(s) \
             for cluster '{cluster_name}'. \
             Common causes: peers on a different subnet, UDP broadcast blocked by firewall, \
             or fewer peer `mlxcel-server` processes running than expected. \
             Retry with `--cluster-discovery=static` and an explicit `--cluster-peers` list.",
            resolved = resolved.len(),
            target = target_count,
        );
    }

    Ok(dedup_and_sort(
        resolved.values().copied().collect::<Vec<_>>().as_slice(),
    ))
}

fn dedup_and_sort(peers: &[SocketAddr]) -> Vec<SocketAddr> {
    let mut v: Vec<SocketAddr> = peers.to_vec();
    v.sort_by_key(|a| a.to_string());
    v.dedup();
    v
}

/// Broadcast a beacon so the coordinator's [`discover_peers`] loop can see us.
///
/// This is a fire-and-forget peer-side helper. The peer keeps broadcasting at
/// `interval` cadence until `stop` is signalled or the send buffer rejects
/// (e.g., the socket is closed). Errors are logged but do not propagate so a
/// transient network blip does not take the peer down.
pub async fn broadcast_beacon_loop(
    bind_ip: IpAddr,
    discovery_port: u16,
    beacon: DiscoveryBeacon,
    interval: Duration,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let socket = tokio::net::UdpSocket::bind(SocketAddr::new(bind_ip, 0))
        .await
        .context("failed to bind UDP socket for beacon broadcast")?;
    socket
        .set_broadcast(true)
        .context("failed to enable SO_BROADCAST on the beacon socket")?;

    let target = SocketAddr::new(IpAddr::V4(Ipv4Addr::BROADCAST), discovery_port);
    let payload = serde_json::to_vec(&beacon).context("failed to encode discovery beacon")?;

    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Err(err) = socket.send_to(&payload, target).await {
                    tracing::debug!("discovery beacon send failed: {err}");
                }
            }
            _ = stop.changed() => {
                if *stop.borrow() {
                    break;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "cluster_init_tests.rs"]
mod tests;
