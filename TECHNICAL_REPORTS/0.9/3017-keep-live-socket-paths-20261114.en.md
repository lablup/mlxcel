# Technical Report: PR #3017 - fix(python): keep live sockets and non-socket entries at the managed path

**Date**: 2026-11-14
**Status**: Completed
**Languages**: Python
**Risk Level**: Medium (behavior change at startup)

---

## Executive Summary

Before spawning `mlxcel serve`, `ManagedServer.start()` unlinked whatever existed at the Unix socket path. A regular file, a directory entry, a symlink, or another server's live socket was silently removed; a running server kept its descriptor but became unreachable by path. Startup now inspects the entry first and removes only a socket nobody listens on. Anything else raises `MlxcelServerError` before a child is spawned and leaves the entry untouched. The rule matches the Rust server's `bind_unix` fix in #3021, so both layers agree on what counts as a stale socket.

---

## 1. Problem Statement

### 1.1 Background

Managed mode on POSIX talks to the spawned server over a Unix domain socket. A socket file outlives the process that bound it, so a crashed server leaves a file that makes the next `bind()` fail with `EADDRINUSE`. The client therefore cleared the path before every start.

### 1.2 Existing Issues

- The clearing was unconditional: `os.path.exists(path)` then `os.unlink(path)`.
- A user who passed `socket=` pointing at a path in use by another server took that path away from it.
- A typo in `socket=` that hit a regular file deleted the file.
- #3005 (merged just before) stopped `close()` from unlinking the path when no child was spawned. Without it, the error this PR raises would still be followed by an unlink in the constructor's cleanup.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A start that used to succeed by deleting an entry now fails | Callers see `MlxcelServerError` | Low, and the old behavior was destructive |
| A live listener misread as stale | Path removed as before | macOS with a completely full accept queue only |
| Race between the probe and the unlink | Two simultaneous starts on one stale path | Low, documented as not a lock |

---

## 2. Technical Review

### 2.1 Security

The change narrows a destructive operation. `lstat` is used, so a symlink at the path is reported as "not a socket" and its target is never touched. An undeterminable probe result (`EACCES`, `EPROTOTYPE`) is treated as "in use", not as stale, so the client never deletes on a guess.

### 2.2 Performance

One non-blocking `connect()` on an existing socket at startup. Negligible.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: a start on a path holding a file, directory, symlink, live socket, or socket of another type now raises instead of deleting it. A leftover socket from a crashed server is still replaced.
- **New Dependencies**: none.
- **Compatibility**: POSIX only; Windows uses TCP and the new tests skip there.

### 2.4 Code Quality

`python/tests/test_startup_path.py` runs each case through `ManagedServer.start()`, `LLM(...)`, and `AsyncLLM(...)`. The stand-in child only writes a marker file, so the test proves no process started and that the entry's device, inode, and mode are unchanged. A live listener must still accept a new connection afterwards. A stale-socket case uses a child that binds the path itself and would fail with `EADDRINUSE` if the leftover were still there.

---

## 3. Technical Decisions

### 3.1 Classify with one non-blocking connect

**Context:** a socket file carries no record of whether its owner is alive.

| Probe result | Reading |
|---|---|
| `ECONNREFUSED`, path vanished | stale, remove |
| connected, or `EAGAIN` (Linux full queue) | live listener, refuse |
| anything else (`EACCES`, `EPROTOTYPE`) | unknown, refuse |

Non-blocking matters: a blocking connect to a full accept queue waits for the listener to accept. The same three-way rule is used by the Rust `probe_unix_socket` in #3021.

### 3.2 Refuse instead of falling back to a fresh path

Silently picking another path would hide the conflict and break callers that connect to the path they passed. Raising with the path in the message lets the user stop the other server, pick another path, or connect to the running server by passing `socket=` without a model.

### 3.3 Review follow-up: dropped probes during startup

Running the suite on macOS exposed a separate gap in `_probe_once`. When the child exits while a readiness request is queued, httpx raises `RemoteProtocolError` on macOS (Linux reports `ReadError`). It was not caught, so `start()` leaked a raw httpx exception instead of `MlxcelServerError` with the child's stderr. It is now treated as "still starting" and the next liveness check reports the exit. The live-listener fixture also moved from a backlog of 1 to 8, because the probe's queued connection filled the backlog and macOS refused the test's own reachability check.

---

## 4. Change Summary

| File | Change |
|------|--------|
| `python/src/mlxcel/_server.py` | `_clear_stale_socket()` classifies the existing entry before spawn; `_probe_once()` also tolerates `RemoteProtocolError` |
| `python/README.md` | One paragraph on what a managed start does with an existing path |
| `python/tests/test_startup_path.py` | New: 26 to 29 cases (full-backlog case is Linux-only) |

---

## 5. Learning Points

- Unix socket semantics differ between Linux and macOS in exactly the corner this code depends on: a full accept queue is `EAGAIN` on Linux and `ECONNREFUSED` on macOS, so a saturated listener on macOS cannot be told from a stale socket.
- A probe connection that is never accepted still occupies a backlog slot. Tests that probe and then connect again need headroom in the backlog.
- Health-check loops should treat every "connection ended without a response" variant as transient and let the process-liveness check produce the error; httpx maps the same event to different exception classes per platform.
- Unverified: the check and the unlink are not atomic, and a socket bound but not yet listening reads as stale. Both are documented in the README.
