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

//! OS-specific capability probes for the RDMA-aware transport backend.
//!
//! The probes here never fail hard: every function returns either a positive
//! detection or a reason string that the caller surfaces in a single-line
//! fallback log entry. See [`probe_capabilities`] for the composition.

use std::fmt;
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::Path;

/// Local peer / node RDMA protocol version exchanged during negotiation.
///
/// The value is bumped whenever the on-wire fast-path framing or handshake
/// semantics change so mismatched peers can fall back cleanly.
pub const RDMA_PROTOCOL_VERSION: u16 = 1;

/// OS family name for logging, expanded per target for clarity.
pub(crate) fn os_family() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    }
}

/// Sender-side acceleration mode negotiated for this backend instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmaAcceleration {
    /// `io_uring` registered buffers on Linux.
    LinuxIoUring,
    /// macOS `kqueue` + `writev` batched send with registered memory regions.
    MacosKqueueRegistered,
    /// No acceleration available — the abstraction hands the request back to
    /// the TCP core.
    TcpFallback,
}

impl fmt::Display for RdmaAcceleration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinuxIoUring => write!(f, "io_uring_registered"),
            Self::MacosKqueueRegistered => write!(f, "kqueue_registered"),
            Self::TcpFallback => write!(f, "tcp_fallback"),
        }
    }
}

/// Structured result of a capability probe.
///
/// When `acceleration` is `TcpFallback`, `reason` explains why, and the RDMA
/// transport emits a single-line log entry before handing the request off to
/// TCP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaCapabilities {
    pub acceleration: RdmaAcceleration,
    pub protocol_version: u16,
    pub reason: Option<String>,
}

impl RdmaCapabilities {
    pub fn accelerated(acceleration: RdmaAcceleration) -> Self {
        Self {
            acceleration,
            protocol_version: RDMA_PROTOCOL_VERSION,
            reason: None,
        }
    }

    pub fn tcp_fallback(reason: impl Into<String>) -> Self {
        Self {
            acceleration: RdmaAcceleration::TcpFallback,
            protocol_version: RDMA_PROTOCOL_VERSION,
            reason: Some(reason.into()),
        }
    }

    /// Whether the negotiation selected an accelerated path.
    pub fn is_accelerated(&self) -> bool {
        !matches!(self.acceleration, RdmaAcceleration::TcpFallback)
    }
}

/// Linux kernel knob that disables `io_uring` system-wide.
///
/// Exposed as `/proc/sys/kernel/io_uring_disabled` on kernels >= 6.6.
/// `0` = allowed, `1` = permitted processes only, `2` = fully disabled.
#[cfg(target_os = "linux")]
const IO_URING_DISABLED_PATH: &str = "/proc/sys/kernel/io_uring_disabled";

/// Top-level capability probe.
///
/// The probe chooses an acceleration mode per target platform, defaulting to
/// [`RdmaAcceleration::TcpFallback`] with a reason whenever detection fails.
pub fn probe_capabilities() -> RdmaCapabilities {
    match probe_os_capabilities() {
        Ok(cap) => cap,
        Err(reason) => RdmaCapabilities::tcp_fallback(reason),
    }
}

#[cfg(target_os = "linux")]
fn probe_os_capabilities() -> Result<RdmaCapabilities, String> {
    check_linux_io_uring_enabled()?;
    Ok(RdmaCapabilities::accelerated(
        RdmaAcceleration::LinuxIoUring,
    ))
}

#[cfg(target_os = "macos")]
fn probe_os_capabilities() -> Result<RdmaCapabilities, String> {
    // kqueue has shipped on every supported macOS version since 10.3, so the
    // only feasible fallback reason here is a hostile sandbox. Report success
    // and let the transport use `writev`-batched sends on top of the TCP core.
    Ok(RdmaCapabilities::accelerated(
        RdmaAcceleration::MacosKqueueRegistered,
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_os_capabilities() -> Result<RdmaCapabilities, String> {
    Err(format!(
        "os={}, driver=unsupported: no zero-copy primitive on this platform",
        os_family()
    ))
}

#[cfg(target_os = "linux")]
fn check_linux_io_uring_enabled() -> Result<(), String> {
    // Kernel knob is optional; if the file is not present we treat io_uring as
    // allowed and rely on the syscall probe.
    let path = Path::new(IO_URING_DISABLED_PATH);
    if path.exists() {
        match fs::read_to_string(path) {
            Ok(contents) => {
                let trimmed = contents.trim();
                if trimmed == "2" {
                    return Err("os=linux, driver=io_uring_disabled_sysctl: \
                         /proc/sys/kernel/io_uring_disabled = 2 (forbidden)"
                        .to_string());
                }
            }
            Err(err) => {
                return Err(format!(
                    "os=linux, driver=io_uring_probe_failed: \
                     cannot read /proc/sys/kernel/io_uring_disabled: {err}"
                ));
            }
        }
    }

    // Best-effort syscall probe: try to set up a tiny ring. This is the
    // definitive test — if `io_uring_setup` is unavailable or seccomp denies
    // it, the fallback kicks in with a precise reason.
    probe_io_uring_setup_syscall()
}

/// Try to issue an `io_uring_setup(1, NULL)` to confirm kernel + seccomp
/// availability. Returns `Ok` on success. On failure returns a string reason
/// suitable for the single-line fallback log.
#[cfg(target_os = "linux")]
fn probe_io_uring_setup_syscall() -> Result<(), String> {
    // SYS_io_uring_setup is 425 on all Linux architectures that support it.
    // Use raw `syscall()` instead of a crate to avoid adding a dependency.
    const SYS_IO_URING_SETUP: libc::c_long = 425;

    // SAFETY: Passing NULL for the params pointer is documented behaviour for
    //   probing the syscall existence — the kernel still validates `entries`
    //   but rejects NULL params with EINVAL, confirming the syscall exists.
    //   Any errno other than ENOSYS / EPERM still counts as "available" since
    //   the syscall is reachable.
    let fd: libc::c_long = unsafe {
        libc::syscall(
            SYS_IO_URING_SETUP,
            1u32,
            std::ptr::null_mut::<libc::c_void>(),
        )
    };

    if fd >= 0 {
        // SAFETY: fd is a valid file descriptor returned by the kernel.
        unsafe {
            libc::close(fd as libc::c_int);
        }
        return Ok(());
    }

    // Inspect errno to classify the failure. Use thread-safe errno accessor.
    // SAFETY: __errno_location returns a pointer to the thread-local errno.
    let errno = unsafe { *libc::__errno_location() };
    match errno {
        libc::ENOSYS => Err(
            "os=linux, driver=io_uring_missing: kernel does not export io_uring_setup (ENOSYS)"
                .to_string(),
        ),
        libc::EPERM => Err(
            "os=linux, driver=io_uring_permission_denied: seccomp/LSM blocks io_uring_setup (EPERM)"
                .to_string(),
        ),
        other if other == libc::EINVAL => {
            // EINVAL with NULL params actually means the syscall exists and
            // accepts our invocation shape — treat as success.
            Ok(())
        }
        other => Err(format!(
            "os=linux, driver=io_uring_probe_failed: io_uring_setup returned errno={other}"
        )),
    }
}

/// Negotiate the on-wire protocol version with a peer's advertised version.
///
/// Returns the agreed version when both sides support at least
/// [`RDMA_PROTOCOL_VERSION`]; otherwise returns a `reason` string describing
/// the mismatch so the caller can log it once and fall back to TCP.
pub fn negotiate_protocol_version(peer_version: u16) -> Result<u16, String> {
    if peer_version == 0 {
        return Err(format!(
            "os={}, peer_version=0: peer did not advertise RDMA protocol version",
            os_family()
        ));
    }
    if peer_version != RDMA_PROTOCOL_VERSION {
        return Err(format!(
            "os={}, peer_version={}: incompatible with local RDMA protocol version {}",
            os_family(),
            peer_version,
            RDMA_PROTOCOL_VERSION
        ));
    }
    Ok(RDMA_PROTOCOL_VERSION)
}

#[cfg(test)]
#[path = "rdma_capabilities_tests.rs"]
mod tests;
