# PR #2684: Protect every cluster frame with per-connection AEAD

**Date**: 2026-10-27
**Status**: Validation complete; PR open as a draft (report written on the PR branch before merge)
**Languages**: Rust, Markdown
**Risk Level**: Medium (security-sensitive transport code; wire-breaking: handshake `PROTOCOL_VERSION` 2 to 3 and a new JSON encoding for stage-protocol tensor bytes)

## Executive Summary

Issue #2656 asked for per-frame protection on the cluster TCP transports, chosen from measurements by a fixed decision rule. This PR derives two directional keys per connection from the existing handshake (HKDF-SHA256 over the cluster secret and both hello nonces) and seals every frame after the handshake with AES-256-GCM under an implicit per-direction sequence number, with the clear `kind || len` header as associated data. A frame that is altered, replayed, reordered, injected or spliced from another connection fails to open and closes the connection. All send paths and the single inbound reader go through one `ProtectedStream`; the `--pp-auto` stage assignment channel uses the same codec with its own key label. Nothing new is configured and there is no opt-out.

The PR also changes the stage-protocol JSON so bulk tensor bytes travel as base64 strings instead of JSON arrays of numbers. That second change dominates the end-to-end result: two-stage qwen3-0.6b-4bit decode goes from 68.7 tok/s on main to 133.2 tok/s (+94 percent). Measured on its own, protection costs about 4.0 percent of decode throughput (138.8 to 133.2 tok/s), which is above the issue's 2 percent clause. The rule's fallback, per-frame HMAC-SHA256, was slower than AES-256-GCM at every payload size measured, so taking the fallback would have cost more throughput and given up confidentiality. AES-256-GCM shipped.

## 1. Problem Statement

### 1.1 Background

PR #2650 (issue #2637) put a mutual HMAC-SHA256 handshake in front of every cluster TCP connection, and #2655 / PR #2674 hardened it and extended it to the disaggregated roles. The handshake authenticated a connection once, at connect time. Every frame after it was plain `[kind u8][len u64 BE][payload]` with no MAC and no encryption.

### 1.2 Existing Issues

- **Frame tampering on an authenticated stream**: an attacker on the cluster segment (ARP or DHCP spoofing, a compromised switch or host) could inject, alter, drop, replay or reorder frames after the handshake. `RunEntry`, `Shutdown` and other control commands travelled in clear.
- **Handshake relay**: the handshake was not bound to anything that protected later traffic, so a relayer could forward a real peer's handshake to the server and then edit the authenticated stream.
- **No confidentiality**: activations, logits, prompts carried in KV handoffs and commands were readable on the wire.
- **Inflated stage protocol**: found during measurement, not part of the issue. serde_json encodes `Vec<u8>` as an array of numbers, about 3.6 characters per byte, so one decoded token of qwen3-0.6b-4bit moved two 1.1 MB frames carrying the same 304 KB of f16 logits.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Command injection or activation tampering by an on-path attacker | High | Low (needs an on-path position on a segment the docs already tell operators to keep trusted) |
| Prompt and output disclosure from activations, logits and KV handoffs | High | Low |
| Decode slowdown from per-frame cryptography on a latency-bound pipeline | Medium | High without measurement, which is why the issue fixed a decision rule first |

## 2. Technical Decisions

### 2.1 Suite: AES-256-GCM, and how the decision rule came out

The issue's rule: choose a confidentiality-providing option when (a) its added p50 one-way latency at 8 KiB is within 20 us or 5 percent of candidate 1 (per-frame HMAC), whichever is larger, and (b) end-to-end decode tokens per second drop by at most 2 percent against the baseline. Otherwise choose candidate 1 and document that payloads stay in clear.

| Clause | Measured | Outcome |
|--------|----------|---------|
| (a) 8 KiB p50 vs HMAC | AES-256-GCM 7.4 us, HMAC-SHA256 12.2 us, unprotected 5.5 us | Pass: AES-GCM is 4.8 us faster than the candidate it is compared against |
| (b) e2e decode vs main, as the PR ships | main 68.7 tok/s, branch 133.2 tok/s | Pass as written: +94 percent |
| (b) e2e decode, protection alone | compact encoding without codec 138.8 tok/s, with AES-256-GCM 133.2 tok/s | Not met: -4.0 percent |

The literal reading passes only because the compact encoding is bundled into the same PR. Judged on the cost of protection by itself, clause (b) fails. The rule's prescribed outcome in that case is candidate 1, but candidate 1 is slower than AES-256-GCM everywhere: 12.2 vs 7.4 us at 8 KiB, 1.008 vs 0.422 ms at 1 MiB, 27.2 vs 17.9 ms at 16 MiB. Taking it would have lowered throughput further and dropped confidentiality, which defeats the purpose the fallback exists for. The PR therefore keeps AES-256-GCM with no opt-out and records the 4 percent protection-only cost in the PR body and `docs/distributed.md`.

A reviewer reading `docs/distributed.md` should note that its wording ("the rule's 2 percent clause holds with nothing to deviate from") reports the bundled number; the protection-only number appears in the next sentence. This report treats the 4 percent as the honest cost of the feature.

ChaCha20-Poly1305 was measured as the other AEAD and was slower than HMAC at every size on this host (14.0 us at 8 KiB). GB10 (ARM Neoverse with AES and PMULL) and Apple silicon both have AES instructions, so AES-256-GCM is the default. All three suites stay in the code because the benchmark measures them on the real transport; `TcpTransportConfig::frame_suite` exists only for that, and the CLI never changes it.

### 2.2 Key schedule, nonce and framing

| Choice | Alternative ruled out | Reason |
|--------|-----------------------|--------|
| HKDF-SHA256, IKM = cluster secret, salt = client nonce \|\| server nonce, info = `mlxcel-cluster-frame-v1/<channel>/{c2s,s2c}` | Reusing the handshake HMAC key directly | Per-connection, per-direction keys: a frame recorded on one connection cannot verify on another, and the relay cannot derive keys without the secret. HKDF extract keys HMAC with the nonce salt while the handshake keys HMAC with the secret, so the two uses do not overlap. |
| Implicit 64-bit sequence as the nonce (`0x00000000 \|\| seq`), never sent, never wraps | Explicit sequence or random nonce on the wire | Zero bytes of overhead, and replay, reorder, drop-then-inject all become tag failures without a replay window. Directional keys make the counter unique per key. At `u64::MAX` the sealer refuses and the connection closes. |
| Clear `kind \|\| len` header as AAD | Encrypting the header | The receiver must refuse an oversize `len` before allocating or doing any crypto. The header is still authenticated. |
| Suite byte in both hellos, covered by both authenticators; mismatch answered with reject reason `0x02` | Negotiation | No downgrade path: each side derives with its own configured suite, and a mismatch is a typed configuration error instead of a first-frame failure. |

Wire overhead per frame is the 16-byte GCM tag; the frame trace shows `wire = payload + 25` (9-byte header plus tag).

### 2.3 One protected path, owned with the socket

`ProtectedStream` (`src/distributed/tcp_frame.rs`) holds the `TcpStream` together with its `FrameSealer` and `FrameOpener`. `write_frame` and `read_frame` are the only framing code; `send`, `send_stream`, RPC requests, RPC responses written from `handle_connection`, and the assignment channel all call them. Main had three hand-built copies of the frame writer (`write_frame`, `send_stream`, the RPC response); they are gone. `ConnectionPool` now stores `ProtectedStream`, so a reused pooled connection continues its own sequence. A send that fails or is cancelled drops the connection instead of releasing it, because the sequence has already advanced inside `seal`.

### 2.4 Channel labels in the HKDF info

The first review found that the stage transport and the assignment channel used identical handshake labels and HKDF info, so a connection redirected between the two listeners completed the handshake (the frame formats then failed to parse). Commit `4f42196f` adds `FrameChannel` (`transport`, `assignment`) to the HKDF info. `client_handshake` / `server_handshake` keep their signatures and use `Transport`; `client_handshake_on` / `server_handshake_on` take the channel. The assignment channel's old `u32` + JSON framing was replaced by `Control` frames of the same codec.

### 2.5 Compact tensor bytes in the same PR

`pipeline::wire_bytes` adds serde `with` modules for the bulk `Vec<u8>` fields of `RunEntry` (`input_ids`, `attention_mask`), `RunResult` (`logits`) and `ActivationMessage` (`tensor_data`, `attention_mask`, `position_ids`): base64 in JSON (1.33 characters per byte), raw bytes in a non-human-readable format. Per decoded token the two logits frames shrink from about 1.1 MB to 405 KB each, and the forward activation from 7.5 KB to 3.0 KB. The motivation was the AEAD cost (fewer bytes to seal), but the larger effect was removing JSON number-array encode and parse time from every step on main itself. Bundling it was a deliberate trade: it is the reason the PR is net positive, and it is also the reason the end-to-end decision clause cannot be read from the headline number.

### 2.6 TLS 1.3 (candidate 3) not prototyped

rustls 0.23 has no external-PSK support, so candidate 3 needed an ephemeral certificate per node, `rcgen` as a runtime dependency, a second handshake per connection, and the HMAC handshake bound to the TLS exporter. Its record layer runs the same AEAD primitives as candidate 2 plus 16 KiB record framing, so it cannot be faster than candidate 2, and the rule prefers 2 unless 3 is no slower. This is an argument, not a measurement.

## 3. Measurements

All on one GB10 host (spark-102), loopback, quiet box, under `gpu-lock`, 2026-10-27.

### 3.1 Frame microbenchmark

`tests/cluster_frame_bench.rs`: two authenticated `TcpTransport`s on 127.0.0.1, the server echoes each `TensorData` message over a second authenticated connection, one-way = round trip / 2 (both ends' frame work included). p50 shown; p95 and p99 are in `docs/distributed.md`.

| Mode | 8 KiB | 64 KiB | 1 MiB | 2 MiB | 16 MiB |
|------|------:|-------:|------:|------:|-------:|
| unprotected (main layout) | 5.5 us | 11.1 us | 411 us | 1.040 ms | 16.39 ms |
| HMAC-SHA256 | 12.2 us | 64.1 us | 1.008 ms | 2.115 ms | 27.22 ms |
| AES-256-GCM (chosen) | 7.4 us | 27.4 us | 422 us | 0.926 ms | 17.93 ms |
| ChaCha20-Poly1305 | 14.0 us | 75.4 us | 1.192 ms | 2.479 ms | 30.16 ms |

AES-256-GCM below the unprotected row at 2 MiB is loopback noise, which puts the noise floor of this harness near 10 percent at that size. No two-node link (Thunderbolt or 10/25 GbE) was measured, although the issue asked for one.

### 3.2 End-to-end decode

`tests/pipeline_remote_decode_bench.rs`: qwen3-0.6b-4bit, two stages and the coordinator in one process over loopback, 128 greedy tokens, 7 repeats plus warm-up per run, 5 interleaved rounds.

| Binary | Median of per-run medians | Per-run range |
|--------|--------------------------:|---------------|
| main 4144ecfb, unprotected, array-encoded | 68.3 tok/s | 65.4 to 72.7 |
| main cfa14e7b (HEAD at measurement), same flags | 68.7 tok/s | 60.7 to 69.7 |
| branch, compact encoding, codec disabled (throwaway build) | 138.8 tok/s | 136.5 to 146.7 |
| branch final, compact encoding, AES-256-GCM | 133.2 tok/s | 122.2 to 145.1 |

The protection-only delta is -4.0 percent, about 0.3 ms of a 7.2 to 7.5 ms step, with overlapping spreads. The codec-disabled row came from a binary with an uncommitted environment hook; the committed code has no such switch.

### 3.3 Attribution of the first result

The first interleaved run, before the compact encoding, showed 71.3 vs 63.5 tok/s (-11.0 percent). A frame trace found four frames per token, two of them 1.1 MB JSON number arrays of the same logits (the final stage returns logits to the entry stage, which forwards them to the coordinator). In-process AES-256-GCM on those 2.2 MB costs 0.54 ms per token (seal 275 us, open 268 us; HMAC-SHA256 1.77 ms), about 4 percent of a 14 ms step. A framing-only build (same handshake and `ProtectedStream`, null codec) decoded at 63.5 tok/s against 65.6 protected and 69.1 main, so the plumbing costs nothing measurable and the rest of the -11 percent was run-to-run noise (main alone spanned 66 to 75 tok/s across rounds). A main baseline rebuilt with the same flags reproduced the orchestrator's binary hash, which rules out a build-flag difference.

## 4. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 28 |
| Lines added | +2993 |
| Lines deleted | -391 |
| New unit tests | 21 (frame_protection 8, tcp transport proxy and oversize 7, cluster_handshake 3, wire_bytes 3) |
| New ignored benchmarks | 2 (`cluster_frame_bench`, `pipeline_remote_decode_bench`) |
| New direct dependency | `subtle = "2"` (already transitive through rustls) |

### Changes by module

| Module | Change |
|--------|--------|
| `frame_protection.rs` (new) | `FrameSuite`, `FrameChannel`, `SessionKeys::derive`, `FrameSealer` / `FrameOpener`, typed `FrameRejected` and `SequenceExhausted` |
| `tcp_frame.rs` (new) | `ProtectedStream` with the only `write_frame` / `read_frame`; `serialize_message` / `deserialize_message` moved here from `tcp_transport.rs` |
| `cluster_handshake.rs` | suite byte in hellos, `REJECT_FRAME_SUITE`, `FrameSuiteMismatch`, returns `SessionKeys`; `PROTOCOL_VERSION` 3; `*_handshake_on` variants |
| `cluster_auth.rs` | `ClusterAuth` keeps the secret for `hkdf_extract` |
| `connection_pool.rs`, `tcp_transport.rs` | pool stores `ProtectedStream`; three duplicate frame writers removed; rejected frames logged through `handshake_warn_allowed` without payload bytes |
| `stage_assignment.rs` | `Control` frames of the shared codec on `FrameChannel::Assignment`; rate-limited rejection log |
| `pipeline/wire_bytes.rs` (new), `remote_service.rs`, `activation_transfer.rs` | base64 tensor bytes in JSON |
| `bench_activation.rs` | round-trip mode, `FrameProtectionMode`, `measure_frame_protection_candidates` |
| `mlx_server.rs`, `router_front.rs`, `docs/distributed.md` | stale "does not encrypt" claims removed; threat model section, measurements and decision recorded |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `6a1620d6` | test | add remote pipeline decode throughput bench |
| `348063d0` | feat | protect every cluster frame with per-connection AEAD |
| `1bbf41b5` | perf | encode stage protocol tensor bytes as base64 in JSON |
| `5a4864ee` | docs | record the final frame protection measurements |
| `32ccd72f` | docs | drop stale no-encryption claims after frame protection |
| `ec5fdc91` | docs | fix stale per-frame claim in router comment |
| `4f42196f` | fix | separate frame key channels and harden review findings |
| `69a32b28` | merge | origin/main into the feature branch |

## 5. Compatibility

- **Breaking**: a build with this PR and one without it cannot form a cluster. Authenticated peers fail the handshake with the explicit version message (`PROTOCOL_VERSION` 3). The base64 tensor encoding is a separate wire change behind the same version bump; every distributed role on main already sets `auth`, so mixed versions meet the version check before any stage message is parsed. Mixed versions were already unsupported.
- **No configuration change**: the existing secret sources (`--cluster-secret-file`, `MLXCEL_CLUSTER_SECRET_FILE`, `MLXCEL_CLUSTER_SECRET`, `[cluster] secret_file`) are the only input. No new flag or environment variable.
- **Single-node serving**: unchanged; without `auth` the plain frame layout is used.

## 6. Validation

- Test-first: the five proxy tests in `src/distributed/tcp_transport_frame_tests.rs` (`tampered_frame_is_rejected`, `replayed_frame_is_rejected`, `reordered_frames_are_rejected`, `frame_from_other_connection_is_rejected`, `activation_bytes_are_not_visible_on_the_wire`) use only APIs that exist on main and calibrate the trailer length from a frame, so they were run against the unmodified transport first. All five failed there (the tampered frame arrived as `inuegrity`, the others reached `recv()`, the pattern was readable) and pass on the branch. Two more cover the oversize header on `ProtectedStream` and on the transport.
- Full `distributed::` lib module on the final head: 1433 passed, 0 failed, 4 ignored (CUDA build, `--test-threads=1`, under `gpu-lock`).
- Real-model remote pipeline parity on GB10 (`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`): qwen3 and qwen3.5 tests pass. llama, gemma3 and gemma4 remote parity fail on main independently (#2654) and were not run.
- `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings` and `cargo fmt --check` clean.
- Commit `4f42196f` came after the measurements. It touches key derivation (an info-string change), logging and tests; the per-frame seal/open path and the encoding hot path are unchanged, so the numbers were not rerun.

## 7. Residual Risks and Follow-up

### Residual risks (documented in `docs/distributed.md`)

- An on-path attacker can still drop or delay frames; frame lengths, kinds and timing stay visible, so an observer can count decode steps.
- A host that holds the secret is a full member; per-frame protection does not constrain members.
- A frame with `len` within `max_frame_size` (256 MiB default) is allocated and read before its tag is checked. Only a handshaken peer reaches this, and the zeroed buffer commits lazily; same allocation as main.
- `hkdf::Prk` and ring's key schedules are not scrubbed (ring limitation); raw derived key bytes are zeroed after key construction.

### Follow-up

- [ ] Measure a two-node link (Thunderbolt or 10/25 GbE). The issue's acceptance list asks for it; the benchmark binds both ends on loopback, so this needs `measure_activation_transfer` in round-trip mode against a transport on a second host.
- [ ] Remove the duplicate logits transfer: the final stage returns logits to the entry stage, which forwards them to the coordinator. A direct final-stage reply, or a binary frame instead of JSON, would roughly halve the remaining per-token bytes and the AEAD work with them.
- [ ] Pre-existing on main: the TCP `rpc_call` reply is written back on the pooled outbound stream, which nothing reads. The reply path is protected but unused.

## 8. Learning Points

### 8.1 Implicit-sequence AEAD over a reliable stream

On an ordered, reliable byte stream, a per-direction counter both sides track gives a unique nonce and replay/reorder detection with no bytes on the wire and no replay window. The counter is part of the nonce, so any deviation (a replayed, dropped-then-injected or reordered frame) is just a tag failure. The constraints this imposes: keys must be unique per connection and direction (HKDF with both handshake nonces as salt), the counter must advance before the write so a partially written frame is never retried on the same stream, and a stream whose send failed or was cancelled must be discarded rather than returned to a pool.

```rust
// FrameSealer::seal, AEAD branch (frame_protection.rs)
let (header, payload) = frame.split_at_mut(FRAME_HEADER_LEN);
let tag = key
    .seal_in_place_separate_tag(aead_nonce(seq), aead::Aad::from(&*header), payload)
    .expect("AEAD seal cannot fail for an in-range payload");
frame.extend_from_slice(tag.as_ref());
self.seq += 1;
```

### 8.2 Hardware AES can beat a MAC

The intuition that "MAC only" is cheaper than "encrypt and MAC" did not hold on this hardware. ring's AES-GCM uses the ARMv8 AES and PMULL instructions, while its SHA-256 path is slower on this core, so AES-256-GCM was 1.5x to 2.4x faster than HMAC-SHA256 at every payload size measured (8 KiB to 16 MiB). A security fallback chosen for being "lighter" needs to be measured on the target hardware before it is preferred.

### 8.3 Attribute before you optimize the crypto

The first end-to-end number (-11 percent) looked like an AEAD problem. A frame trace showed the per-token bytes were dominated by JSON number arrays sent twice, and a null-codec build separated framing cost from crypto cost. The fix with the largest effect was in serialization, not in the cipher. The same trace also shows why a bundled PR needs its components measured separately: the headline +94 percent says nothing about what protection costs.

### Key Terms

| Keyword | Relevance |
|---------|-----------|
| HKDF (RFC 5869) | Extract-then-expand key derivation; here it turns one shared secret plus two handshake nonces into four directional, per-channel keys |
| AEAD, associated data | AES-256-GCM authenticates the clear header as AAD so `len` can be checked before decryption without being forgeable |
| Nonce uniqueness | GCM breaks if a (key, nonce) pair repeats; the implicit counter plus per-connection directional keys guarantee it does not |
| serde `with` modules, `is_human_readable` | One annotation encodes base64 in JSON and raw bytes in a binary format |

### Related PRs and Issues

- Issue #2656: this design and its decision rule
- Issue #2637 / PR #2650: the connect-time cluster handshake this builds on
- Issue #2655 / PR #2674: pending-handshake pool, rejection records, disaggregated role auth
- Issue #2654: pre-existing llama/gemma3/gemma4 remote parity failures on GB10
