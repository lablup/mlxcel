// Copyright 2025-2026 Lablup Inc.
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

//! Stage push-assignment for the zero-config pipeline bring-up (#2658).
//!
//! `mlxcel-server --pp-auto N` resolves N peers (static list or discovery
//! beacons), plans the cluster, and then pushes one stage assignment to each
//! peer over an authenticated TCP connection; `mlxcel-server --pp-peer`
//! listens for that assignment instead of needing a hand-copied TOML and a
//! guessed `--node-id`. The channel reuses the cluster handshake
//! ([`client_handshake`] / [`server_handshake`]): the coordinator dials the
//! peer's listener and advertises its control address, the peer advertises
//! its listener address, and the peer binds the coordinator's identity to the
//! TCP source IP exactly as the stage transport does.
//!
//! Frames after the handshake are `u32` big-endian length + JSON, at most
//! [`MAX_ASSIGNMENT_FRAME_BYTES`], every read bounded by
//! [`ASSIGNMENT_IO_TIMEOUT`]. The exchange has two phases so a bring-up that
//! fails on one peer commits no peer:
//!
//! ```text
//! coordinator -> peer : Offer(StageAssignmentOffer)
//! peer -> coordinator : Accepted { node_id } | Rejected { reason, detail }
//! coordinator -> peer : Commit            (only after every peer accepted)
//! peer -> coordinator : Committed { node_id }
//! ```
//!
//! The offer carries the model identity (config hash, weight file names and
//! sizes, layer count) and the coordinator's layer range for the stage. The
//! peer re-derives both from its own checkpoint and refuses a mismatch with
//! a typed error, because a retry against the same coordinator cannot
//! succeed. The peer never acts on the offer's `cluster_toml` beyond parsing
//! it and checking that it names this peer as the offered stage at the
//! listener's port.
//!
//! Used by: `server::startup::run_zero_config_bring_up` (both branches).

use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use super::cluster_auth::ClusterAuth;
use super::cluster_handshake::{PeerClosedBeforeHandshake, client_handshake, server_handshake};
use super::cluster_init::{
    ClusterDiscoveryMode, ClusterInitPlan, DiscoveryBeacon, broadcast_beacon_loop,
    render_deterministic_toml,
};
use super::config::{ClusterConfig, NodeRole};
use super::pending_handshakes::PendingHandshakeLimiter;
use super::pipeline::{
    resolve_in_process_pipeline_num_layers, resolve_in_process_stage_assignments_for_model,
};
use super::tcp_transport::handshake_warn_allowed;

/// Wire version of the assignment frames (independent of the handshake's).
pub const ASSIGNMENT_PROTOCOL_VERSION: u16 = 1;
/// Largest frame either side accepts (length prefix excluded).
pub const MAX_ASSIGNMENT_FRAME_BYTES: usize = 64 * 1024;
/// Bound on every single frame read or write.
pub const ASSIGNMENT_IO_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a peer that answered `Accepted` waits for `Commit` before it
/// discards the offer and listens again.
pub const ASSIGNMENT_COMMIT_TIMEOUT: Duration = Duration::from_secs(60);
/// Bound on the coordinator's whole push (connect retries, handshakes,
/// offers, commits).
pub const ASSIGNMENT_ACK_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a `--pp-peer` process waits for a committed assignment.
pub const PEER_ASSIGNMENT_TIMEOUT: Duration = Duration::from_secs(600);
/// Interval between two discovery beacons from a waiting peer.
pub const PEER_BEACON_INTERVAL: Duration = Duration::from_secs(1);
/// Handshakes the peer's assignment listener runs at once across all
/// sources, and from one source IP. The listener accepts unauthenticated
/// sockets, so it uses the same pending pool as the stage transport.
pub const MAX_PENDING_ASSIGNMENT_HANDSHAKES: usize = 8;
pub const MAX_PENDING_ASSIGNMENT_HANDSHAKES_PER_IP: usize = 4;
/// Retry cadence for the coordinator's connect to a peer that has not
/// started listening yet (a static peer may start after the coordinator).
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(500);

/// What a peer and the coordinator must agree on before a stage can serve a
/// slice of the model. Built from the checkpoint directory without reading
/// weight bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Hex SHA-256 of the `config.json` bytes.
    pub config_sha256: String,
    /// Every `*.safetensors` file name with its byte size, sorted by name.
    pub weight_files: Vec<(String, u64)>,
    /// `resolve_in_process_pipeline_num_layers` of the checkpoint.
    pub num_layers: usize,
}

impl ModelIdentity {
    /// Hash the config and list the weight files of `model_dir`.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
        let mut weight_files = Vec::new();
        let entries = std::fs::read_dir(model_dir)
            .with_context(|| format!("failed to list {}", model_dir.display()))?;
        for entry in entries {
            let entry = entry
                .with_context(|| format!("failed to list an entry of {}", model_dir.display()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".safetensors") {
                continue;
            }
            // `metadata` follows symlinks, so a linked shard counts by its
            // real size.
            let metadata = std::fs::metadata(entry.path())
                .with_context(|| format!("failed to stat {}", entry.path().display()))?;
            if metadata.is_file() {
                weight_files.push((name, metadata.len()));
            }
        }
        weight_files.sort();
        let num_layers = resolve_in_process_pipeline_num_layers(model_dir)?;
        Ok(Self {
            config_sha256,
            weight_files,
            num_layers,
        })
    }

    /// One line naming what differs between two identities, for a rejection
    /// detail. Never includes file contents.
    fn describe_difference(&self, other: &Self) -> String {
        let mut parts = Vec::new();
        if self.config_sha256 != other.config_sha256 {
            parts.push(format!(
                "config.json sha256 {}.. vs {}..",
                &self.config_sha256[..self.config_sha256.len().min(12)],
                &other.config_sha256[..other.config_sha256.len().min(12)]
            ));
        }
        if self.num_layers != other.num_layers {
            parts.push(format!(
                "{} vs {} layers",
                self.num_layers, other.num_layers
            ));
        }
        if self.weight_files != other.weight_files {
            let mine: BTreeSet<_> = self.weight_files.iter().collect();
            let theirs: BTreeSet<_> = other.weight_files.iter().collect();
            let only_mine: Vec<_> = mine
                .difference(&theirs)
                .map(|(n, s)| format!("{n} ({s} B)"))
                .collect();
            let only_theirs: Vec<_> = theirs
                .difference(&mine)
                .map(|(n, s)| format!("{n} ({s} B)"))
                .collect();
            parts.push(format!(
                "weight files differ (only here: [{}]; only there: [{}])",
                only_mine.join(", "),
                only_theirs.join(", ")
            ));
        }
        if parts.is_empty() {
            "identical".to_string()
        } else {
            parts.join("; ")
        }
    }
}

/// The coordinator's proposal for one stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageAssignmentOffer {
    /// [`ASSIGNMENT_PROTOCOL_VERSION`].
    pub protocol: u16,
    /// `env!("CARGO_PKG_VERSION")` of the coordinator.
    pub mlxcel_version: String,
    pub cluster_name: String,
    /// `stage-{i}` as planned by `plan_cluster`.
    pub node_id: String,
    pub stage_index: u32,
    pub num_stages: u32,
    /// The coordinator's partition for this stage. Not an instruction (each
    /// stage derives its own range from its checkpoint) but a second
    /// identity check.
    pub layer_range: std::ops::Range<usize>,
    /// Where the stage will reach the coordinator. An unspecified IP is
    /// replaced by the handshake-bound identity on the peer.
    pub coordinator_control_addr: SocketAddr,
    pub model: ModelIdentity,
    /// `render_deterministic_toml` of the plan with `secret_file` cleared.
    pub cluster_toml: String,
}

/// Why a peer refused an offer. `ProtocolMismatch`, `VersionMismatch`,
/// `ModelMismatch`, and `LayerRangeMismatch` end the peer process (a retry
/// against the same coordinator cannot succeed, see [`Self::is_fatal`]); the
/// rest return the peer to its listening loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AssignmentRejectReason {
    ProtocolMismatch,
    VersionMismatch,
    ClusterNameMismatch,
    ModelMismatch,
    LayerRangeMismatch,
    AddressMismatch,
    InvalidClusterConfig,
}

impl AssignmentRejectReason {
    /// Whether this rejection ends the peer process instead of returning it
    /// to the listening loop.
    pub fn is_fatal(self) -> bool {
        matches!(
            self,
            Self::ProtocolMismatch
                | Self::VersionMismatch
                | Self::ModelMismatch
                | Self::LayerRangeMismatch
        )
    }
}

impl std::fmt::Display for AssignmentRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ProtocolMismatch => "assignment protocol mismatch",
            Self::VersionMismatch => "mlxcel version mismatch",
            Self::ClusterNameMismatch => "cluster name mismatch",
            Self::ModelMismatch => "model mismatch",
            Self::LayerRangeMismatch => "layer range mismatch",
            Self::AddressMismatch => "address mismatch",
            Self::InvalidClusterConfig => "invalid cluster config",
        };
        f.write_str(name)
    }
}

/// One frame of the assignment channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case")]
pub enum AssignmentFrame {
    Offer(StageAssignmentOffer),
    Accepted {
        node_id: String,
    },
    Rejected {
        reason: AssignmentRejectReason,
        detail: String,
    },
    Commit,
    Committed {
        node_id: String,
    },
}

impl AssignmentFrame {
    /// Frame name for logs and errors (never the payload, which may hold the
    /// cluster TOML).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Offer(_) => "Offer",
            Self::Accepted { .. } => "Accepted",
            Self::Rejected { .. } => "Rejected",
            Self::Commit => "Commit",
            Self::Committed { .. } => "Committed",
        }
    }
}

/// Encode one frame as length prefix + JSON.
pub fn encode_frame(frame: &AssignmentFrame) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(frame).context("failed to encode assignment frame")?;
    ensure!(
        body.len() <= MAX_ASSIGNMENT_FRAME_BYTES,
        "assignment frame of {} bytes exceeds the {MAX_ASSIGNMENT_FRAME_BYTES}-byte limit",
        body.len()
    );
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode one complete frame from `bytes` (length prefix included). A
/// truncated buffer, trailing bytes, or an oversize length is an error.
pub fn decode_frame(bytes: &[u8]) -> Result<AssignmentFrame> {
    ensure!(
        bytes.len() >= 4,
        "assignment frame is shorter than its length prefix"
    );
    let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    ensure!(
        len <= MAX_ASSIGNMENT_FRAME_BYTES,
        "assignment frame announces {len} bytes, above the {MAX_ASSIGNMENT_FRAME_BYTES}-byte limit"
    );
    ensure!(
        bytes.len() == 4 + len,
        "assignment frame announces {len} bytes but {} follow the prefix",
        bytes.len() - 4
    );
    serde_json::from_slice(&bytes[4..]).context("failed to decode assignment frame")
}

/// Write one frame, bounded by [`ASSIGNMENT_IO_TIMEOUT`].
pub async fn write_frame(stream: &mut TcpStream, frame: &AssignmentFrame) -> Result<()> {
    let bytes = encode_frame(frame)?;
    tokio::time::timeout(ASSIGNMENT_IO_TIMEOUT, async {
        stream.write_all(&bytes).await?;
        stream.flush().await
    })
    .await
    .map_err(|_| anyhow::anyhow!("timed out writing {} frame", frame.kind()))?
    .with_context(|| format!("failed to write {} frame", frame.kind()))
}

/// Read one frame, bounded by [`ASSIGNMENT_IO_TIMEOUT`].
pub async fn read_frame(stream: &mut TcpStream) -> Result<AssignmentFrame> {
    tokio::time::timeout(ASSIGNMENT_IO_TIMEOUT, read_frame_unbounded(stream))
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "timed out after {ASSIGNMENT_IO_TIMEOUT:?} waiting for an assignment frame"
            )
        })?
}

/// Read one frame with no time bound of its own. Every caller wraps it in a
/// timeout: [`read_frame`] uses [`ASSIGNMENT_IO_TIMEOUT`], and the peer's
/// wait for `Commit` uses the longer commit timeout, because an accepted
/// peer legitimately idles while the coordinator is still connecting to the
/// other peers.
async fn read_frame_unbounded(stream: &mut TcpStream) -> Result<AssignmentFrame> {
    let mut prefix = [0u8; 4];
    stream
        .read_exact(&mut prefix)
        .await
        .context("connection closed before an assignment frame arrived")?;
    let len = u32::from_be_bytes(prefix) as usize;
    ensure!(
        len <= MAX_ASSIGNMENT_FRAME_BYTES,
        "peer announced an assignment frame of {len} bytes, above the \
         {MAX_ASSIGNMENT_FRAME_BYTES}-byte limit"
    );
    let mut body = vec![0u8; len];
    stream
        .read_exact(&mut body)
        .await
        .context("connection closed inside an assignment frame")?;
    serde_json::from_slice::<AssignmentFrame>(&body).context("malformed assignment frame")
}

/// The peer refused an offer for a reason a retry cannot fix.
#[derive(Debug, Clone, thiserror::Error)]
#[error("stage assignment refused ({reason}): {detail}")]
pub struct AssignmentRefused {
    pub reason: AssignmentRejectReason,
    pub detail: String,
}

/// One peer the coordinator could not commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerAssignmentFailure {
    pub node_id: String,
    pub addr: SocketAddr,
    pub reason: String,
}

/// The coordinator's push did not commit every planned peer. No `Commit`
/// was sent when any peer failed the offer phase, so every peer is still
/// waiting for an offer.
#[derive(Debug, Clone, thiserror::Error)]
pub struct AssignmentPushFailed {
    pub failures: Vec<PeerAssignmentFailure>,
}

impl std::fmt::Display for AssignmentPushFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stage assignment failed for {} peer(s): ",
            self.failures.len()
        )?;
        for (index, failure) in self.failures.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(
                f,
                "{} at {}: {}",
                failure.node_id, failure.addr, failure.reason
            )?;
        }
        f.write_str(
            ". No stage was committed; check that every peer runs `mlxcel-server --pp-peer` \
             with the same cluster secret, cluster name, model, and mlxcel version",
        )
    }
}

/// Everything a `--pp-peer` process needs to wait for its assignment.
#[derive(Debug, Clone)]
pub struct PeerJoinConfig {
    /// Where the assignment listener binds; its local address is what the
    /// stage service binds afterwards. Static mode needs a concrete port.
    pub listen_addr: SocketAddr,
    pub discovery: ClusterDiscoveryMode,
    /// UDP port the coordinator's receiver listens on (mdns only).
    pub discovery_port: u16,
    /// Where beacons are sent: the limited broadcast, a directed broadcast,
    /// or the coordinator's unicast IP (mdns only).
    pub discovery_target: IpAddr,
    pub cluster_name: String,
    /// `--node-id` as typed; advertised in the beacon, overridden by the
    /// assignment.
    pub node_id_hint: Option<String>,
    pub model_dir: PathBuf,
    pub auth: Arc<ClusterAuth>,
    /// Bound on the whole wait ([`PEER_ASSIGNMENT_TIMEOUT`] in production).
    pub assignment_timeout: Duration,
    /// Bound on the wait for `Commit` after `Accepted`.
    pub commit_timeout: Duration,
}

/// The one assignment a peer committed to.
#[derive(Debug, Clone)]
pub struct CommittedAssignment {
    pub node_id: String,
    pub stage_index: u32,
    /// The offered cluster with the coordinator's address replaced by the
    /// handshake-bound identity when it was unspecified, this node's address
    /// replaced by the listener's local address, and no `secret_file`.
    pub cluster: ClusterConfig,
    /// The coordinator's identity as bound by the handshake (`host:port`).
    pub coordinator_identity: String,
}

/// Facts about the local peer the offer is checked against.
struct PeerLocalFacts {
    cluster_name: String,
    model: ModelIdentity,
    model_dir: PathBuf,
    listener_addr: SocketAddr,
}

/// Check an offer in the documented order and return the parsed cluster, or
/// the first failing reason with a detail suitable for the coordinator's
/// error message.
fn verify_offer(
    offer: &StageAssignmentOffer,
    local: &PeerLocalFacts,
    bound_coordinator: SocketAddr,
) -> std::result::Result<ClusterConfig, (AssignmentRejectReason, String)> {
    use AssignmentRejectReason as R;
    if offer.protocol != ASSIGNMENT_PROTOCOL_VERSION {
        return Err((
            R::ProtocolMismatch,
            format!(
                "coordinator speaks assignment protocol {}, this peer expects {ASSIGNMENT_PROTOCOL_VERSION}",
                offer.protocol
            ),
        ));
    }
    let local_version = env!("CARGO_PKG_VERSION");
    if offer.mlxcel_version != local_version {
        return Err((
            R::VersionMismatch,
            format!(
                "coordinator runs mlxcel {}, this peer runs {local_version}",
                offer.mlxcel_version
            ),
        ));
    }
    if offer.cluster_name != local.cluster_name {
        return Err((
            R::ClusterNameMismatch,
            format!(
                "offer is for cluster '{}', this peer joined '{}' (--cluster-name)",
                offer.cluster_name, local.cluster_name
            ),
        ));
    }
    if offer.model != local.model {
        return Err((
            R::ModelMismatch,
            format!(
                "the peer's checkpoint differs from the coordinator's: {}",
                local.model.describe_difference(&offer.model)
            ),
        ));
    }
    if offer.num_stages < 2 || offer.stage_index >= offer.num_stages {
        return Err((
            R::LayerRangeMismatch,
            format!(
                "stage index {} of {} stages is not a valid pipeline position",
                offer.stage_index, offer.num_stages
            ),
        ));
    }
    let expected_range = resolve_in_process_stage_assignments_for_model(
        &local.model_dir,
        local.model.num_layers,
        Some(offer.num_stages as usize),
        None,
    )
    .ok()
    .and_then(|(assignments, _)| {
        assignments
            .into_iter()
            .find(|assignment| assignment.stage_index == offer.stage_index as usize)
            .map(|assignment| assignment.layer_range)
    });
    match expected_range {
        Some(range) if range == offer.layer_range => {}
        Some(range) => {
            return Err((
                R::LayerRangeMismatch,
                format!(
                    "coordinator partitions stage {} as layers {:?}, this peer derives {:?} for \
                     {} stages",
                    offer.stage_index, offer.layer_range, range, offer.num_stages
                ),
            ));
        }
        None => {
            return Err((
                R::LayerRangeMismatch,
                format!(
                    "this peer cannot derive a partition for stage {} of {}",
                    offer.stage_index, offer.num_stages
                ),
            ));
        }
    }
    let cluster = match ClusterConfig::from_toml(&offer.cluster_toml) {
        Ok(cluster) => cluster,
        Err(err) => {
            return Err((
                R::InvalidClusterConfig,
                format!("offered cluster TOML does not parse: {err:#}"),
            ));
        }
    };
    if cluster.cluster.pipeline_parallel_size != offer.num_stages {
        return Err((
            R::InvalidClusterConfig,
            format!(
                "offered cluster TOML declares pipeline_parallel_size {} but the offer names {} stages",
                cluster.cluster.pipeline_parallel_size, offer.num_stages
            ),
        ));
    }
    let Some(own_node) = cluster.find_node(&offer.node_id) else {
        return Err((
            R::InvalidClusterConfig,
            format!("offered cluster TOML has no node '{}'", offer.node_id),
        ));
    };
    if own_node.role != NodeRole::PipelineStage || own_node.stage != Some(offer.stage_index) {
        return Err((
            R::InvalidClusterConfig,
            format!(
                "node '{}' in the offered cluster TOML is not pipeline stage {}",
                offer.node_id, offer.stage_index
            ),
        ));
    }
    if own_node.address.port() != local.listener_addr.port() {
        return Err((
            R::AddressMismatch,
            format!(
                "offered cluster TOML places '{}' at {} but this peer listens on {}",
                offer.node_id, own_node.address, local.listener_addr
            ),
        ));
    }
    if cluster
        .nodes
        .iter()
        .all(|node| node.role == NodeRole::PipelineStage)
    {
        return Err((
            R::InvalidClusterConfig,
            "offered cluster TOML has no coordinator node".to_string(),
        ));
    }
    let offered = offer.coordinator_control_addr;
    let bound_ip = bound_coordinator.ip().to_canonical();
    let offered_ip = offered.ip().to_canonical();
    if offered.port() != bound_coordinator.port()
        || (!offered_ip.is_unspecified() && offered_ip != bound_ip)
    {
        return Err((
            R::AddressMismatch,
            format!(
                "offer names coordinator control address {offered} but the coordinator \
                 authenticated as {bound_coordinator}"
            ),
        ));
    }
    Ok(cluster)
}

/// Rewrite the offered cluster for the local stage: the coordinator's
/// address becomes the bound identity when the offer left it unspecified,
/// and this node's address becomes the listener's local address.
fn localize_cluster(
    mut cluster: ClusterConfig,
    node_id: &str,
    listener_addr: SocketAddr,
    bound_coordinator: SocketAddr,
) -> ClusterConfig {
    for node in &mut cluster.nodes {
        if node.id == node_id {
            node.address = listener_addr;
        } else if node.role != NodeRole::PipelineStage && node.address.ip().is_unspecified() {
            node.address = bound_coordinator;
        }
    }
    cluster.cluster.secret_file = None;
    cluster
}

/// State shared by every connection handler of one peer wait.
struct PeerShared {
    facts: PeerLocalFacts,
    auth: Arc<ClusterAuth>,
    local_identity: String,
    commit_timeout: Duration,
    committed: AtomicBool,
    results: mpsc::Sender<std::result::Result<CommittedAssignment, AssignmentRefused>>,
}

/// Serve one inbound connection: handshake, read the offer, answer, and wait
/// for the commit. Returns nothing; outcomes that end the wait go through
/// `shared.results`, everything else is logged and the peer keeps listening.
async fn handle_coordinator_connection(
    mut stream: TcpStream,
    slot: super::pending_handshakes::PendingHandshakeSlot,
    shared: Arc<PeerShared>,
) {
    let source_ip = stream.peer_addr().map(|a| a.ip()).ok();
    let bound = match server_handshake(&mut stream, &shared.auth, &shared.local_identity).await {
        Ok(bound) => bound,
        Err(err) => {
            if err.downcast_ref::<PeerClosedBeforeHandshake>().is_some() {
                tracing::debug!("assignment listener: {err}");
            } else if handshake_warn_allowed() {
                tracing::warn!(
                    "assignment listener: rejecting connection from {}: {err:#}",
                    source_ip.map(|ip| ip.to_string()).unwrap_or_default()
                );
            } else {
                tracing::debug!(
                    "assignment listener: rejecting connection from {}: {err:#}",
                    source_ip.map(|ip| ip.to_string()).unwrap_or_default()
                );
            }
            return;
        }
    };
    drop(slot);
    let bound_coordinator: SocketAddr = match bound.parse() {
        Ok(addr) => addr,
        Err(_) => {
            tracing::warn!(
                "assignment listener: bound coordinator identity {bound:?} is not a socket address"
            );
            return;
        }
    };
    let offer = match read_frame(&mut stream).await {
        Ok(AssignmentFrame::Offer(offer)) => offer,
        Ok(other) => {
            tracing::warn!(
                "assignment listener: coordinator {bound} sent {} instead of an offer",
                other.kind()
            );
            return;
        }
        Err(err) => {
            tracing::debug!("assignment listener: coordinator {bound} sent no offer: {err:#}");
            return;
        }
    };
    tracing::debug!(
        "assignment listener: offer from {bound} for {} (stage {} of {})",
        offer.node_id,
        offer.stage_index,
        offer.num_stages
    );
    let cluster = match verify_offer(&offer, &shared.facts, bound_coordinator) {
        Ok(cluster) => cluster,
        Err((reason, detail)) => {
            let _ = write_frame(
                &mut stream,
                &AssignmentFrame::Rejected {
                    reason,
                    detail: detail.clone(),
                },
            )
            .await;
            tracing::warn!(
                "assignment listener: rejected offer from {bound} for {}: {reason}: {detail}",
                offer.node_id
            );
            if reason.is_fatal() {
                let _ = shared
                    .results
                    .send(Err(AssignmentRefused { reason, detail }))
                    .await;
            }
            return;
        }
    };
    if let Err(err) = write_frame(
        &mut stream,
        &AssignmentFrame::Accepted {
            node_id: offer.node_id.clone(),
        },
    )
    .await
    {
        tracing::info!(
            "assignment listener: coordinator {bound} went away before Accepted was sent: {err:#}"
        );
        return;
    }
    // Not `read_frame`: its 10 s frame bound would cut this wait short of the
    // commit timeout while the coordinator is still reaching a later peer.
    match tokio::time::timeout(shared.commit_timeout, read_frame_unbounded(&mut stream)).await {
        Ok(Ok(AssignmentFrame::Commit)) => {}
        Ok(Ok(other)) => {
            tracing::warn!(
                "assignment listener: coordinator {bound} sent {} instead of Commit; discarding the offer for {}",
                other.kind(),
                offer.node_id
            );
            return;
        }
        Ok(Err(err)) => {
            tracing::info!(
                "assignment listener: coordinator {bound} dropped the offer for {} before commit ({err:#}); waiting for another assignment",
                offer.node_id
            );
            return;
        }
        Err(_) => {
            tracing::info!(
                "assignment listener: no Commit from {bound} within {:?}; discarding the offer for {} and waiting for another assignment",
                shared.commit_timeout,
                offer.node_id
            );
            return;
        }
    }
    if shared
        .committed
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        tracing::info!(
            "assignment listener: already committed to another assignment; closing the connection from {bound}"
        );
        return;
    }
    if let Err(err) = write_frame(
        &mut stream,
        &AssignmentFrame::Committed {
            node_id: offer.node_id.clone(),
        },
    )
    .await
    {
        // The coordinator did not learn of the commit and will fail its
        // push, so this peer must not start a stage for it.
        shared.committed.store(false, Ordering::Release);
        tracing::warn!(
            "assignment listener: coordinator {bound} went away before Committed was delivered ({err:#}); waiting for another assignment"
        );
        return;
    }
    let cluster = localize_cluster(
        cluster,
        &offer.node_id,
        shared.facts.listener_addr,
        bound_coordinator,
    );
    let _ = shared
        .results
        .send(Ok(CommittedAssignment {
            node_id: offer.node_id,
            stage_index: offer.stage_index,
            cluster,
            coordinator_identity: bound,
        }))
        .await;
}

/// The unspecified address of `ip`'s family.
pub fn unspecified_for(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    }
}

/// Peer side of the bring-up: listen on `config.listen_addr`, beacon in
/// mdns mode, and return the first committed assignment. Fails with
/// [`AssignmentRefused`] on a fatal rejection and with a timeout error
/// naming the listen address, discovery mode, and cluster name when
/// `config.assignment_timeout` elapses. Exactly one assignment is committed
/// per call: the listener is closed before this returns.
pub async fn await_stage_assignment(config: PeerJoinConfig) -> Result<CommittedAssignment> {
    let model = ModelIdentity::from_model_dir(&config.model_dir)?;
    let listener = TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| {
            format!(
                "failed to bind the stage assignment listener on {} (--cluster-control-addr)",
                config.listen_addr
            )
        })?;
    let listener_addr = listener
        .local_addr()
        .context("stage assignment listener has no local address")?;
    let limiter = PendingHandshakeLimiter::new(
        MAX_PENDING_ASSIGNMENT_HANDSHAKES,
        MAX_PENDING_ASSIGNMENT_HANDSHAKES_PER_IP,
    )?;

    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let beacon_task = match config.discovery {
        ClusterDiscoveryMode::Mdns => {
            let beacon = DiscoveryBeacon {
                version: env!("CARGO_PKG_VERSION").to_string(),
                cluster_name: config.cluster_name.clone(),
                peer_addr: listener_addr,
                node_id: config.node_id_hint.clone(),
                pid: std::process::id(),
                timestamp_ms: 0,
                nonce: 0,
                mac: String::new(),
            };
            tracing::info!(
                "Announcing this peer at {listener_addr} to {}:{} every {:?} (cluster '{}')",
                config.discovery_target,
                config.discovery_port,
                PEER_BEACON_INTERVAL,
                config.cluster_name
            );
            Some(tokio::spawn(broadcast_beacon_loop(
                unspecified_for(config.discovery_target),
                config.discovery_port,
                beacon,
                PEER_BEACON_INTERVAL,
                stop_rx,
                Arc::clone(&config.auth),
                config.discovery_target,
            )))
        }
        _ => None,
    };

    let (results_tx, mut results_rx) = mpsc::channel(4);
    let shared = Arc::new(PeerShared {
        facts: PeerLocalFacts {
            cluster_name: config.cluster_name.clone(),
            model,
            model_dir: config.model_dir.clone(),
            listener_addr,
        },
        auth: Arc::clone(&config.auth),
        local_identity: listener_addr.to_string(),
        commit_timeout: config.commit_timeout,
        committed: AtomicBool::new(false),
        results: results_tx,
    });
    tracing::info!(
        "Waiting up to {:?} for a stage assignment on {listener_addr} (discovery {}, cluster '{}')",
        config.assignment_timeout,
        config.discovery,
        config.cluster_name
    );

    let deadline = tokio::time::Instant::now() + config.assignment_timeout;
    let mut handlers = tokio::task::JoinSet::new();
    let outcome = loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, peer_addr)) => match limiter.try_acquire(peer_addr.ip()) {
                    Ok(slot) => {
                        handlers.spawn(handle_coordinator_connection(
                            stream,
                            slot,
                            Arc::clone(&shared),
                        ));
                    }
                    Err(refusal) => {
                        if handshake_warn_allowed() {
                            tracing::warn!(
                                "assignment listener: rejecting unauthenticated connection from {}: {refusal}",
                                peer_addr.ip()
                            );
                        } else {
                            tracing::debug!(
                                "assignment listener: rejecting unauthenticated connection from {}: {refusal}",
                                peer_addr.ip()
                            );
                        }
                        drop(stream);
                    }
                },
                Err(err) => {
                    tracing::warn!("assignment listener: accept failed: {err}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            },
            Some(_) = handlers.join_next(), if !handlers.is_empty() => {}
            Some(result) = results_rx.recv() => break result.map_err(anyhow::Error::from),
            _ = tokio::time::sleep_until(deadline) => break Err(anyhow::anyhow!(
                "no stage assignment received within {:?} on {listener_addr} (discovery {}, \
                 cluster '{}'); check that `mlxcel-server --pp-auto N` runs with the same \
                 cluster secret and name, that N peers are up, and that this address is \
                 reachable from the coordinator",
                config.assignment_timeout,
                config.discovery,
                config.cluster_name
            )),
        }
    };
    handlers.abort_all();
    drop(listener);
    if let Some(task) = beacon_task {
        let _ = stop_tx.send(true);
        if tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .is_err()
        {
            tracing::debug!("discovery beacon loop did not stop within 1s");
        }
    }
    match &outcome {
        Ok(committed) => tracing::info!(
            "Committed stage assignment {} (stage {}) from coordinator {}",
            committed.node_id,
            committed.stage_index,
            committed.coordinator_identity
        ),
        Err(err) => tracing::error!("stage assignment wait ended: {err:#}"),
    }
    outcome
}

/// Build the per-stage offers for `plan`. Shared between the push and its
/// tests.
fn build_offers(
    plan: &ClusterInitPlan,
    coordinator_control_addr: SocketAddr,
    model_dir: &Path,
) -> Result<Vec<(SocketAddr, StageAssignmentOffer)>> {
    let model = ModelIdentity::from_model_dir(model_dir)?;
    let num_stages = plan.cluster.cluster.pipeline_parallel_size;
    ensure!(
        num_stages >= 2,
        "stage assignment needs a pipeline of at least 2 stages (plan has {num_stages})"
    );
    let (assignments, _report) = resolve_in_process_stage_assignments_for_model(
        model_dir,
        model.num_layers,
        Some(num_stages as usize),
        None,
    )?;
    let mut cluster = plan.cluster.clone();
    cluster.cluster.secret_file = None;
    let cluster_toml = render_deterministic_toml(&cluster);
    let mut offers = Vec::new();
    for node in cluster.pipeline_stage_nodes() {
        let stage_index = node.stage.ok_or_else(|| {
            anyhow::anyhow!(
                "planned pipeline stage node '{}' has no stage index",
                node.id
            )
        })?;
        let layer_range = assignments
            .iter()
            .find(|assignment| assignment.stage_index == stage_index as usize)
            .map(|assignment| assignment.layer_range.clone())
            .ok_or_else(|| {
                anyhow::anyhow!("no partition entry for stage {stage_index} of {num_stages}")
            })?;
        offers.push((
            node.address,
            StageAssignmentOffer {
                protocol: ASSIGNMENT_PROTOCOL_VERSION,
                mlxcel_version: env!("CARGO_PKG_VERSION").to_string(),
                cluster_name: cluster.cluster.name.clone(),
                node_id: node.id.clone(),
                stage_index,
                num_stages,
                layer_range,
                coordinator_control_addr,
                model: model.clone(),
                cluster_toml: cluster_toml.clone(),
            },
        ));
    }
    ensure!(
        !offers.is_empty(),
        "plan has no pipeline_stage nodes to assign"
    );
    Ok(offers)
}

/// Connect to one peer (retrying until `deadline`), handshake, send the
/// offer, and wait for `Accepted`. The returned stream carries the commit.
async fn offer_to_peer(
    addr: SocketAddr,
    offer: &StageAssignmentOffer,
    auth: &ClusterAuth,
    local_identity: &str,
    deadline: tokio::time::Instant,
) -> std::result::Result<TcpStream, String> {
    let mut last_connect_error: Option<String> = None;
    let timed_out = |last: &Option<String>| {
        format!(
            "timed out before the peer accepted the offer (last connect error: {})",
            last.as_deref().unwrap_or("none; connected but no answer")
        )
    };
    let mut stream = loop {
        match tokio::time::timeout_at(deadline, TcpStream::connect(addr)).await {
            Ok(Ok(stream)) => break stream,
            Ok(Err(err)) => {
                last_connect_error = Some(err.to_string());
                if tokio::time::timeout_at(deadline, tokio::time::sleep(CONNECT_RETRY_INTERVAL))
                    .await
                    .is_err()
                {
                    return Err(timed_out(&last_connect_error));
                }
            }
            Err(_) => return Err(timed_out(&last_connect_error)),
        }
    };
    let exchange = async {
        client_handshake(&mut stream, auth, local_identity)
            .await
            .map_err(|err| format!("cluster handshake failed: {err:#}"))?;
        write_frame(&mut stream, &AssignmentFrame::Offer(offer.clone()))
            .await
            .map_err(|err| format!("{err:#}"))?;
        match read_frame(&mut stream)
            .await
            .map_err(|err| format!("no answer to the offer: {err:#}"))?
        {
            AssignmentFrame::Accepted { node_id } if node_id == offer.node_id => Ok(()),
            AssignmentFrame::Accepted { node_id } => Err(format!(
                "peer accepted as '{node_id}' instead of '{}'",
                offer.node_id
            )),
            AssignmentFrame::Rejected { reason, detail } => {
                Err(format!("peer rejected the offer ({reason}): {detail}"))
            }
            other => Err(format!(
                "peer answered the offer with {} instead of Accepted",
                other.kind()
            )),
        }
    };
    match tokio::time::timeout_at(deadline, exchange).await {
        Ok(Ok(())) => Ok(stream),
        Ok(Err(reason)) => Err(reason),
        Err(_) => Err(
            "timed out during the handshake or offer (the peer connected but did not accept in \
             time)"
                .to_string(),
        ),
    }
}

/// Send `Commit` on an accepted stream and wait for `Committed`.
async fn commit_peer(
    stream: &mut TcpStream,
    node_id: &str,
    deadline: tokio::time::Instant,
) -> std::result::Result<(), String> {
    let exchange = async {
        write_frame(stream, &AssignmentFrame::Commit)
            .await
            .map_err(|err| format!("{err:#}"))?;
        match read_frame(stream)
            .await
            .map_err(|err| format!("no Committed after Commit: {err:#}"))?
        {
            AssignmentFrame::Committed { node_id: got } if got == node_id => Ok(()),
            AssignmentFrame::Committed { node_id: got } => {
                Err(format!("peer committed as '{got}' instead of '{node_id}'"))
            }
            other => Err(format!(
                "peer answered Commit with {} instead of Committed",
                other.kind()
            )),
        }
    };
    match tokio::time::timeout_at(deadline, exchange).await {
        Ok(result) => result,
        Err(_) => Err("timed out waiting for Committed".to_string()),
    }
}

/// Coordinator side of the bring-up: offer every planned `pipeline_stage`
/// node its assignment concurrently, and commit all of them only after every
/// one accepted. Bounded by `timeout` as a whole. Any failure is an
/// [`AssignmentPushFailed`] naming each peer that did not commit; no
/// `Commit` is sent when any peer failed the offer phase.
pub async fn push_stage_assignments(
    plan: &ClusterInitPlan,
    coordinator_control_addr: SocketAddr,
    model_dir: &Path,
    auth: &ClusterAuth,
    timeout: Duration,
) -> Result<()> {
    let offers = build_offers(plan, coordinator_control_addr, model_dir)?;
    let deadline = tokio::time::Instant::now() + timeout;
    let local_identity = coordinator_control_addr.to_string();
    tracing::info!(
        "Pushing stage assignments to {} peer(s) as {local_identity} (timeout {timeout:?})",
        offers.len()
    );

    let local_identity = &local_identity;
    let offer_results = futures::future::join_all(offers.iter().map(|(addr, offer)| async move {
        let result = offer_to_peer(*addr, offer, auth, local_identity, deadline).await;
        (offer, *addr, result)
    }))
    .await;
    let mut accepted = Vec::new();
    let mut failures = Vec::new();
    for (offer, addr, result) in offer_results {
        match result {
            Ok(stream) => {
                tracing::info!("Peer {addr} accepted stage assignment {}", offer.node_id);
                accepted.push((offer, addr, stream));
            }
            Err(reason) => failures.push(PeerAssignmentFailure {
                node_id: offer.node_id.clone(),
                addr,
                reason,
            }),
        }
    }
    if !failures.is_empty() {
        // Dropping `accepted` closes those connections without a Commit, so
        // every peer returns to its listening loop.
        drop(accepted);
        bail!(AssignmentPushFailed { failures });
    }

    let commit_results = futures::future::join_all(accepted.into_iter().map(
        |(offer, addr, mut stream)| async move {
            let result = commit_peer(&mut stream, &offer.node_id, deadline).await;
            (offer, addr, result)
        },
    ))
    .await;
    for (offer, addr, result) in commit_results {
        match result {
            Ok(()) => tracing::info!("Peer {addr} committed stage assignment {}", offer.node_id),
            Err(reason) => failures.push(PeerAssignmentFailure {
                node_id: offer.node_id.clone(),
                addr,
                reason,
            }),
        }
    }
    if !failures.is_empty() {
        bail!(AssignmentPushFailed { failures });
    }
    Ok(())
}

#[cfg(test)]
#[path = "stage_assignment_tests.rs"]
mod tests;
