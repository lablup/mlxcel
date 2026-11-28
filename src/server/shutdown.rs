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

//! Graceful shutdown on SIGTERM and SIGINT (#3483).
//!
//! In a container the server runs as PID 1, where the kernel applies no
//! default action to SIGTERM or SIGINT: without a handler both are ignored and
//! `docker stop` or a pod deletion waits out the grace period before a
//! SIGKILL. This module owns the process-wide handler and the connection drain
//! the accept loop in [`crate::server::listen`] runs when it fires.
//!
//! The handler is installed once, early in startup ([`install`]), and latches
//! the first signal in a [`CancellationToken`]. A signal that arrives while the
//! model is still loading is therefore not lost: the listener sees the token
//! already cancelled and returns before serving, and the process exits through
//! the usual worker join. The same task keeps listening, and a second signal
//! during the drain ends the process at once with exit code 1.

use std::sync::OnceLock;
use std::time::Duration;

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Exit code of the process when a second signal arrives during the drain.
const SECOND_SIGNAL_EXIT_CODE: i32 = 1;

/// Upper bound on waiting for aborted connection tasks to unwind after the
/// drain deadline. An aborted task stops at its next poll, so this only
/// guards against a task stuck in a synchronous call.
const ABORT_SETTLE: Duration = Duration::from_secs(1);

/// Which signal asked the process to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShutdownSignal {
    /// SIGTERM: `docker stop`, Kubernetes, systemd, a supervisor.
    Terminate,
    /// SIGINT: Ctrl-C (the only signal off Unix).
    Interrupt,
}

impl ShutdownSignal {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Terminate => "SIGTERM",
            Self::Interrupt => "SIGINT",
        }
    }
}

/// The process-wide latch: cancelled on the first signal, with the signal
/// that did it.
struct Latch {
    token: CancellationToken,
    received: OnceLock<ShutdownSignal>,
}

static LATCH: OnceLock<Latch> = OnceLock::new();

fn latch() -> &'static Latch {
    LATCH.get_or_init(|| Latch {
        token: CancellationToken::new(),
        received: OnceLock::new(),
    })
}

/// Install the SIGTERM / SIGINT handler and return the shutdown token.
///
/// Idempotent: the first call spawns the listener task on the current tokio
/// runtime and later calls return the same token. Call it as early in startup
/// as possible, so a signal during model load is latched instead of killing
/// the process (or, as PID 1, being ignored).
///
/// Must be called from inside a tokio runtime.
pub(crate) fn install() -> CancellationToken {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    let latch = latch();
    INSTALLED.get_or_init(|| {
        // The listener is created here, synchronously, so the handlers are
        // registered before `install` returns rather than whenever the spawned
        // task is first polled.
        let mut listener = SignalListener::new();
        tokio::spawn(async move {
            let first = listener.recv().await;
            let _ = latch.received.set(first);
            tracing::info!("shutdown signal received: {}", first.name());
            latch.token.cancel();
            let second = listener.recv().await;
            tracing::error!(
                "second shutdown signal ({}); exiting without drain",
                second.name()
            );
            exit_immediately(SECOND_SIGNAL_EXIT_CODE);
        });
    });
    latch.token.clone()
}

/// Resolve on the first SIGTERM or SIGINT the process receives.
///
/// Installs the handler if [`install`] has not run yet, and resolves at once
/// when a signal was already latched.
pub(crate) async fn shutdown_signal() -> ShutdownSignal {
    let token = install();
    token.cancelled().await;
    latch()
        .received
        .get()
        .copied()
        .unwrap_or(ShutdownSignal::Interrupt)
}

/// [`shutdown_signal`] shaped for callers that take an `io::Result<()>`
/// signal future (the remote pipeline stage).
pub(crate) async fn shutdown_signal_io() -> std::io::Result<()> {
    shutdown_signal().await;
    Ok(())
}

/// End the process without running exit handlers.
///
/// `std::process::exit` runs the C runtime's static destructors while a model
/// worker may still be freeing MLX arrays on its own thread, which aborts with
/// `pure virtual method called` (#2738) and would turn exit code 1 into
/// SIGABRT. `_exit` skips them; nothing Rust-side runs afterwards.
fn exit_immediately(code: i32) -> ! {
    #[cfg(unix)]
    {
        // SAFETY: `_exit` terminates the process without running exit
        // handlers or destructors, which is the point; no Rust state is
        // touched afterwards.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// The installed signal streams. Each is created once, so a signal that
/// arrives between two `recv` calls is still delivered.
struct SignalListener {
    #[cfg(unix)]
    terminate: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    interrupt: Option<tokio::signal::unix::Signal>,
}

impl SignalListener {
    /// Register the handlers. A handler that cannot be installed is logged
    /// and left out; the server keeps serving on the other one rather than
    /// refusing to start.
    fn new() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let terminate = signal(SignalKind::terminate())
                .map_err(|err| {
                    tracing::error!(
                        "cannot install the SIGTERM handler: {err}; serving with SIGINT only"
                    )
                })
                .ok();
            let interrupt = signal(SignalKind::interrupt())
                .map_err(|err| tracing::error!("cannot install the SIGINT handler: {err}"))
                .ok();
            Self {
                terminate,
                interrupt,
            }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }

    /// Wait for the next signal. Pends forever when no handler is installed.
    async fn recv(&mut self) -> ShutdownSignal {
        #[cfg(unix)]
        {
            async fn next(signal: &mut Option<tokio::signal::unix::Signal>) {
                match signal {
                    Some(signal) => {
                        if signal.recv().await.is_none() {
                            std::future::pending::<()>().await;
                        }
                    }
                    None => std::future::pending::<()>().await,
                }
            }
            tokio::select! {
                () = next(&mut self.terminate) => ShutdownSignal::Terminate,
                () = next(&mut self.interrupt) => ShutdownSignal::Interrupt,
            }
        }
        #[cfg(not(unix))]
        {
            match tokio::signal::ctrl_c().await {
                Ok(()) => ShutdownSignal::Interrupt,
                Err(err) => {
                    tracing::error!("cannot listen for Ctrl-C: {err}");
                    std::future::pending().await
                }
            }
        }
    }
}

/// Live connections of one listener, drained when it stops accepting.
///
/// Every accepted stream runs as a task in this set from the moment it is
/// accepted, so a TLS handshake still in progress counts as in flight.
/// `graceful` is the signal each connection watches to call hyper's
/// `graceful_shutdown`: an idle keep-alive connection closes at once, and a
/// request in progress (an SSE stream included) runs to completion before its
/// connection closes.
pub(crate) struct ConnectionDrain {
    tasks: JoinSet<()>,
    graceful: CancellationToken,
}

impl ConnectionDrain {
    pub(crate) fn new(graceful: CancellationToken) -> Self {
        Self {
            tasks: JoinSet::new(),
            graceful,
        }
    }

    /// The token a connection watches for the graceful close.
    ///
    /// A child token, so each connection's pending wait registers on its own
    /// lock instead of one mutex every connection shares on every IO wake;
    /// cancelling the parent still reaches every child.
    pub(crate) fn graceful(&self) -> CancellationToken {
        self.graceful.child_token()
    }

    /// Run one connection as a tracked task, reaping finished ones first so a
    /// long-lived listener does not accumulate completed entries.
    pub(crate) fn spawn<F>(&mut self, connection: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        while self.tasks.try_join_next().is_some() {}
        self.tasks.spawn(connection);
    }

    /// Signal every connection to close gracefully and wait up to `deadline`
    /// for them; abort the rest.
    ///
    /// Returns the number of connections aborted at the deadline. A zero
    /// deadline aborts everything still open at once.
    pub(crate) async fn drain(mut self, deadline: Duration) -> usize {
        self.graceful.cancel();
        while self.tasks.try_join_next().is_some() {}
        let open = self.tasks.len();
        if open == 0 {
            return 0;
        }
        tracing::info!(
            "shutdown: draining {open} connection(s) for up to {} s",
            deadline.as_secs()
        );
        let tasks = &mut self.tasks;
        let drained = tokio::time::timeout(deadline, async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        if drained.is_ok() {
            tracing::info!("shutdown: all connections drained");
            return 0;
        }
        while self.tasks.try_join_next().is_some() {}
        let aborted = self.tasks.len();
        tracing::warn!(
            "shutdown: drain timeout after {} s, aborting {aborted} connection(s)",
            deadline.as_secs()
        );
        self.tasks.abort_all();
        let tasks = &mut self.tasks;
        let _ = tokio::time::timeout(ABORT_SETTLE, async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        aborted
    }
}

/// Identity of a bound socket file; see [`socket_identity`].
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SocketIdentity {
    dev: u64,
    ino: u64,
    ctime: i64,
    ctime_nsec: i64,
}

#[cfg(unix)]
impl SocketIdentity {
    fn of(entry: &std::fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;

        Self {
            dev: entry.dev(),
            ino: entry.ino(),
            ctime: entry.ctime(),
            ctime_nsec: entry.ctime_nsec(),
        }
    }
}

/// The filesystem identity of the socket file at `path`: device, inode and
/// change time. The change time guards against inode reuse, since a
/// filesystem such as ext4 may hand a path re-created right after an unlink
/// the inode number it just freed.
///
/// Captured right after the bind, so [`remove_own_socket`] can tell this
/// server's socket from one a replacement server bound on the same path
/// while this one was draining. `None` when the entry cannot be read.
#[cfg(unix)]
pub(crate) fn socket_identity(path: &std::path::Path) -> Option<SocketIdentity> {
    std::fs::symlink_metadata(path)
        .ok()
        .map(|entry| SocketIdentity::of(&entry))
}

/// Remove the socket file this server bound, once it stops accepting.
///
/// Called while the listener is still open, so a replacement server's
/// `bind_unix` sees the socket as live and cannot take the path between the
/// check and the unlink. Only this server's own socket is removed: the entry
/// must still be a socket with the identity captured at bind time (`bound`,
/// from [`socket_identity`]). Another server's socket at this path, a path
/// replaced by something else (a regular file, a symlink), and a socket whose
/// identity was never captured are all left alone. A failure is logged; the next start clears a
/// stale socket in `bind_unix` anyway.
#[cfg(unix)]
pub(crate) fn remove_own_socket(path: &std::path::Path, bound: Option<SocketIdentity>) {
    use std::os::unix::fs::FileTypeExt;

    let Some(bound) = bound else {
        tracing::warn!(
            "{}: the socket identity was not captured at bind time; it is left as it is",
            path.display()
        );
        return;
    };
    match std::fs::symlink_metadata(path) {
        Ok(entry) if entry.file_type().is_socket() && SocketIdentity::of(&entry) == bound => {
            match std::fs::remove_file(path) {
                Ok(()) => tracing::info!("removed Unix socket {}", path.display()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!("failed to remove Unix socket {}: {error}", path.display())
                }
            }
        }
        Ok(entry) if entry.file_type().is_socket() => tracing::info!(
            "{} now belongs to another server; it is left as it is",
            path.display()
        ),
        Ok(_) => tracing::warn!(
            "{} is no longer a socket; it is left as it is",
            path.display()
        ),
        Err(_) => {}
    }
}

/// Run `work` unless a shutdown signal arrives first.
///
/// Returns `None` when the signal won; `work` is then dropped. For startup
/// waits that hold no model yet (pipeline discovery, the `--pp-peer`
/// assignment wait), where the latched signal would otherwise be ignored
/// until the wait timed out on its own.
pub(crate) async fn unless_shutdown<F: std::future::Future>(work: F) -> Option<F::Output> {
    let token = install();
    tokio::select! {
        biased;
        () = token.cancelled() => None,
        output = work => Some(output),
    }
}

#[cfg(test)]
#[path = "shutdown_tests.rs"]
mod shutdown_tests;
