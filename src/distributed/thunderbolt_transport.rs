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

//! Thunderbolt transport backend.
//!
//! This backend reuses the battle-tested TCP framing and connection-pool
//! implementation, but validates that the chosen bind address belongs to a
//! macOS Thunderbolt Bridge interface before bringing the transport up.
//!
//! Used by: remote pipeline runtime, remote stage service

use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::pin::Pin;
use std::process::Command;

use anyhow::{Context, Result, anyhow, ensure};
use bytes::Bytes;

use super::tcp_transport::{TcpTransport, TcpTransportConfig};
use super::transport::{BoxedAsyncRead, RpcHandler, Transport, TransportBackend, TransportMessage};

const NETWORKSETUP_BIN: &str = "/usr/sbin/networksetup";
const IFCONFIG_BIN: &str = "/sbin/ifconfig";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThunderboltInterfaceInfo {
    pub interface: String,
    pub addresses: Vec<IpAddr>,
}

/// Error returned when a Thunderbolt backend is requested on an unsupported
/// host or bind address.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ThunderboltUnavailable {
    message: String,
}

impl ThunderboltUnavailable {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Configuration for the Thunderbolt transport backend.
#[derive(Debug, Clone)]
pub struct ThunderboltTransportConfig {
    /// Thunderbolt bridge interface name (for example `"bridge0"`).
    pub interface: String,
    /// Port to bind on the Thunderbolt bridge.
    pub port: u16,
    /// Reserved for future zero-copy / RDMA-style transfers.
    pub use_shared_memory: bool,
    /// Maximum in-flight transfer size in bytes (default 1 GiB).
    pub max_transfer_size: usize,
}

impl ThunderboltTransportConfig {
    pub fn from_bind_address(bind_address: &str) -> Result<Self> {
        let bind_addr: SocketAddr = bind_address
            .parse()
            .with_context(|| format!("invalid thunderbolt bind address '{bind_address}'"))?;
        let interfaces = discover_thunderbolt_interfaces()?;
        let resolved = resolve_bind_address(bind_addr, &interfaces)?;
        Ok(Self {
            interface: resolved.interface,
            port: resolved.bind_address.port(),
            ..Default::default()
        })
    }
}

impl Default for ThunderboltTransportConfig {
    fn default() -> Self {
        Self {
            interface: "bridge0".to_string(),
            port: 9200,
            use_shared_memory: true,
            max_transfer_size: 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedThunderboltBind {
    interface: String,
    bind_address: SocketAddr,
}

/// Working Thunderbolt transport built on the shared TCP core.
pub struct ThunderboltTransport {
    config: ThunderboltTransportConfig,
    bind_address: SocketAddr,
    inner: TcpTransport,
}

impl ThunderboltTransport {
    /// Bind the Thunderbolt transport after validating the local bridge
    /// interface and address.
    pub async fn bind(config: ThunderboltTransportConfig) -> Result<Self> {
        let interfaces = discover_thunderbolt_interfaces()?;
        let resolved = resolve_config(&config, &interfaces)?;
        let inner = TcpTransport::bind(TcpTransportConfig {
            bind_address: resolved.bind_address.to_string(),
            ..Default::default()
        })
        .await?;
        Ok(Self {
            config,
            bind_address: resolved.bind_address,
            inner,
        })
    }

    /// Return the configured interface name.
    pub fn interface(&self) -> &str {
        &self.config.interface
    }

    /// Return the configured port.
    pub fn port(&self) -> u16 {
        self.config.port
    }

    /// Check whether shared-memory mode is configured.
    pub fn use_shared_memory(&self) -> bool {
        self.config.use_shared_memory
    }

    /// Return the validated bind address used by the backend.
    pub fn bind_address(&self) -> SocketAddr {
        self.bind_address
    }

    /// Detect whether at least one Thunderbolt Bridge interface with an IP
    /// address is available on this machine.
    pub fn is_available() -> bool {
        discover_thunderbolt_interfaces()
            .map(|interfaces| !interfaces.is_empty())
            .unwrap_or(false)
    }

    /// Enumerate local Thunderbolt Bridge interfaces with routable addresses.
    pub fn available_interfaces() -> Result<Vec<ThunderboltInterfaceInfo>> {
        discover_thunderbolt_interfaces()
    }
}

impl Transport for ThunderboltTransport {
    fn connect(
        &self,
        peers: &[String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.connect(peers)
    }

    fn send(
        &self,
        peer: &str,
        message: TransportMessage,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.send(peer, message)
    }

    fn recv(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, TransportMessage)>> + Send + '_>>
    {
        self.inner.recv()
    }

    fn send_stream(
        &self,
        peer: &str,
        data: Bytes,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.send_stream(peer, data)
    }

    fn recv_stream(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(String, BoxedAsyncRead)>> + Send + '_>>
    {
        self.inner.recv_stream()
    }

    fn rpc_call(
        &self,
        peer: &str,
        request: &[u8],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        self.inner.rpc_call(peer, request)
    }

    fn serve_rpc(
        &self,
        handler: RpcHandler,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.serve_rpc(handler)
    }

    fn shutdown(&self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        self.inner.shutdown()
    }

    fn backend(&self) -> TransportBackend {
        TransportBackend::Thunderbolt
    }

    fn local_addr(&self) -> Result<String> {
        self.inner.local_addr()
    }
}

fn resolve_config(
    config: &ThunderboltTransportConfig,
    interfaces: &[ThunderboltInterfaceInfo],
) -> Result<ResolvedThunderboltBind> {
    ensure!(
        config.max_transfer_size > 0,
        "thunderbolt max_transfer_size must be > 0"
    );
    let requested = interfaces
        .iter()
        .find(|candidate| candidate.interface == config.interface)
        .ok_or_else(|| {
            ThunderboltUnavailable::new(format!(
                "Thunderbolt interface '{}' is unavailable; detected interfaces: {}",
                config.interface,
                format_interface_list(interfaces)
            ))
        })?;
    let bind_ip = requested
        .addresses
        .iter()
        .find(|addr| addr.is_ipv4())
        .copied()
        .or_else(|| requested.addresses.first().copied())
        .ok_or_else(|| {
            ThunderboltUnavailable::new(format!(
                "Thunderbolt interface '{}' has no routable address",
                config.interface
            ))
        })?;
    Ok(ResolvedThunderboltBind {
        interface: config.interface.clone(),
        bind_address: SocketAddr::new(bind_ip, config.port),
    })
}

fn resolve_bind_address(
    bind_address: SocketAddr,
    interfaces: &[ThunderboltInterfaceInfo],
) -> Result<ResolvedThunderboltBind> {
    ensure!(
        !interfaces.is_empty(),
        "{}",
        ThunderboltUnavailable::new(
            "Thunderbolt transport is unavailable: no Thunderbolt Bridge interface with an IP address was detected",
        )
    );
    if bind_address.ip().is_unspecified() {
        let interface = interfaces
            .first()
            .ok_or_else(|| anyhow!("missing thunderbolt interface after availability check"))?;
        let bind_ip = interface
            .addresses
            .iter()
            .find(|addr| addr.is_ipv4())
            .copied()
            .or_else(|| interface.addresses.first().copied())
            .ok_or_else(|| {
                ThunderboltUnavailable::new(format!(
                    "Thunderbolt interface '{}' has no routable address",
                    interface.interface
                ))
            })?;
        return Ok(ResolvedThunderboltBind {
            interface: interface.interface.clone(),
            bind_address: SocketAddr::new(bind_ip, bind_address.port()),
        });
    }

    let requested = interfaces
        .iter()
        .find(|candidate| candidate.addresses.contains(&bind_address.ip()))
        .ok_or_else(|| {
            ThunderboltUnavailable::new(format!(
                "bind address '{}' does not belong to a Thunderbolt Bridge interface; detected interfaces: {}",
                bind_address.ip(),
                format_interface_list(interfaces)
            ))
        })?;
    Ok(ResolvedThunderboltBind {
        interface: requested.interface.clone(),
        bind_address,
    })
}

fn discover_thunderbolt_interfaces() -> Result<Vec<ThunderboltInterfaceInfo>> {
    let mut interfaces = Vec::new();
    for interface in discover_thunderbolt_interface_names()? {
        let addresses = query_interface_addresses(&interface)?;
        if !addresses.is_empty() {
            interfaces.push(ThunderboltInterfaceInfo {
                interface,
                addresses,
            });
        }
    }
    interfaces.sort_by(|left, right| left.interface.cmp(&right.interface));
    interfaces.dedup_by(|left, right| left.interface == right.interface);
    Ok(interfaces)
}

fn discover_thunderbolt_interface_names() -> Result<Vec<String>> {
    let mut interfaces = if Path::new(NETWORKSETUP_BIN).exists() {
        let output = Command::new(NETWORKSETUP_BIN)
            .arg("-listallhardwareports")
            .output()
            .context("failed to query networksetup for Thunderbolt interfaces")?;
        if output.status.success() {
            parse_networksetup_hardware_ports(&String::from_utf8_lossy(&output.stdout))
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if interfaces.is_empty() && Path::new(IFCONFIG_BIN).exists() {
        let output = Command::new(IFCONFIG_BIN)
            .arg("-l")
            .output()
            .context("failed to list interfaces via ifconfig")?;
        if output.status.success() {
            interfaces.extend(
                parse_ifconfig_interface_list(&String::from_utf8_lossy(&output.stdout))
                    .into_iter()
                    .filter(|name| name.starts_with("bridge")),
            );
        }
    }

    interfaces.sort();
    interfaces.dedup();
    Ok(interfaces)
}

fn query_interface_addresses(interface: &str) -> Result<Vec<IpAddr>> {
    let output = Command::new(IFCONFIG_BIN)
        .arg(interface)
        .output()
        .with_context(|| format!("failed to inspect interface '{interface}' via ifconfig"))?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(parse_ifconfig_addresses(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn format_interface_list(interfaces: &[ThunderboltInterfaceInfo]) -> String {
    if interfaces.is_empty() {
        return "<none>".to_string();
    }
    interfaces
        .iter()
        .map(|interface| {
            let addrs = interface
                .addresses
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{} [{}]", interface.interface, addrs)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_networksetup_hardware_ports(output: &str) -> Vec<String> {
    let mut current_port: Option<String> = None;
    let mut interfaces = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(port) = trimmed.strip_prefix("Hardware Port:") {
            current_port = Some(port.trim().to_string());
            continue;
        }
        if let Some(device) = trimmed.strip_prefix("Device:")
            && current_port
                .as_deref()
                .is_some_and(|port| port.contains("Thunderbolt"))
        {
            interfaces.push(device.trim().to_string());
        }
    }
    interfaces
}

fn parse_ifconfig_interface_list(output: &str) -> Vec<String> {
    output
        .split_whitespace()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect()
}

fn parse_ifconfig_addresses(output: &str) -> Vec<IpAddr> {
    let mut addresses = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("inet ") {
            if let Some(token) = rest.split_whitespace().next()
                && let Ok(addr) = token.parse::<IpAddr>()
                && !addr.is_loopback()
            {
                addresses.push(addr);
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("inet6 ")
            && let Some(token) = rest.split_whitespace().next()
        {
            let normalized = token.split('%').next().unwrap_or(token);
            if let Ok(addr) = normalized.parse::<IpAddr>()
                && !addr.is_loopback()
            {
                addresses.push(addr);
            }
        }
    }
    addresses
}

#[cfg(test)]
#[path = "thunderbolt_transport_tests.rs"]
mod tests;
