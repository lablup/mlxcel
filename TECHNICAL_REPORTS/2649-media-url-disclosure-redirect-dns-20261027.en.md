# PR #2649: Stop disclosing internal DNS in media URL errors

**Date**: 2026-10-27
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

The SSRF guard for remote media URLs told the caller which internal names exist and what they resolve to: a refused host rendered as `media URL host {host} resolves to {address} ... set MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1 ...`, a missing one as `media URL host {host} does not resolve`, and both reached clients verbatim as 400 text on four routes. A public origin could also redirect the fetch onto an internal hostname, and reqwest sent a real GET there before the post-connect check discarded the response. Both refusals now render one sentence, `media URL host is not allowed or could not be resolved` (`MEDIA_HOST_REFUSED`), with the detail moved to a `debug` log. The shared media client now resolves every name through a policy resolver, so a hostname redirect hop is refused at DNS time and no request reaches the internal host. The same pass also refuses NAT64 and 6to4 spellings of private IPv4 addresses and bounds every lookup at 5 s (issue #2644).

## 1. Problem Statement

`MediaUrlError::BlockedAddress { host, address }` and `MediaUrlError::Unresolvable { host }` in `src/server/media_net.rs` put the attacker-supplied host, the internally resolved address and the opt-in variable into their `Display` text. Four surfaces return that text to the client unchanged:

- `fetch_labeled_image`, used by `POST /v1/embeddings`, rerank and image generation, formats the whole error chain into the 400 message.
- Chat with audio input reaches the same shared fetch through `AudioAcquisitionError::Source`, which the chat route renders with `to_string()`.

Chat images and video swallow fetch errors, so they did not leak. One POST per candidate name (`http://intranet-admin.internal/x.png`) answered "exists or not", and when it existed, its internal address. That is recon-grade DNS mapping against any deployment with an image- or audio-capable model loaded, reachable without a key in the default no-auth setup.

The second defect was in the redirect policy. reqwest's redirect callback is synchronous, so `media_redirect_policy` only re-ran `ensure_media_url_shape` (scheme, credentials, IP-literal) on each hop. A hop to `http://localhost:PORT/` passed the shape check, reqwest connected and sent the GET, and only then did the post-connect `remote_addr` check refuse the response, after which the 400 disclosed the internal peer address. The module doc's claim that "a public origin cannot bounce the fetch to an internal one" held for IP-literal targets only.

The codebase already had a precedent for the disclosure half: issue #1612 removed the configured `--media-path` candidate from client-facing text and logged it at `debug` instead (`src/server/media_root.rs`). The remote path had diverged from that rule.

## 2. Technical Decisions

### One generic sentence, detail at debug

`BlockedAddress` and `Unresolvable` now both render `#[error("{MEDIA_HOST_REFUSED}")]`. The variants keep their `host` and `address` fields, but only `MediaUrlError::log_detail(url)` reads them: it emits a `debug` event with the URL, host, address and the opt-in hint. The fetch boundaries in `media.rs` call it (`fetch_remote_bytes_with_limit_cancel` through a local `rejected` closure, and `fetch_remote_video`). The fix sits in the error type and the one shared fetch, so all four leaking routes inherit it without handler changes.

Collapsing the two cases is deliberate. If "does not resolve" and "is refused" read differently, the response still answers whether an internal name exists even with the host and address removed.

Other variants are unchanged: `Malformed { url }` echoes the caller's own input, and `TooManyRedirects { url }` names a hop that the policy already accepted as a public shape.

### Finding refusals inside reqwest errors

A refusal raised by the redirect policy or by the resolver reaches `send()` as a `reqwest::Error` whose source chain contains the `MediaUrlError`. Before this PR that path took the generic transport branch and printed `Failed to fetch {kind} URL {url}: {err}` in reqwest's wording, a different message from a pre-flight refusal and one that never reached `log_detail`. `MediaUrlError::find_in` walks `source()` and downcasts each link; when it finds a policy refusal, the caller logs its detail and returns the same `{kind} URL rejected: {MEDIA_HOST_REFUSED}` text as a pre-flight refusal. Without it, a resolver refusal would surface as a connect failure: the client would see a message that differs from the pre-flight one (and so distinguishes "refused on redirect" from other outcomes), and the operator would get no debug line naming the refused host.

### Redirect enforcement: a DNS resolver on the client

The issue offered two approaches: re-check the peer after each hop's connect, or refuse redirects that change authority. The PR takes a third, stronger one. `media_net::PolicyResolver` implements `reqwest::dns::Resolve` and is installed on the shared client with `ClientBuilder::dns_resolver`. For every name the client connects to, initial request or redirect hop, it resolves through `resolve_allowed_addresses`, keeps only addresses that pass `is_public_address` (`filter_allowed_addresses`), and fails the resolution with `BlockedAddress` when none is left or `Unresolvable` when the answer is empty. The connection never opens, so the internal host receives nothing.

The alternatives were rejected for these reasons:

- A per-hop post-connect check still connects and sends the request line and headers to the internal host before refusing it. It removes the response, not the request.
- Refusing cross-authority redirects breaks common legitimate flows (a public URL redirecting to a CDN host) and still does nothing about a same-authority name whose DNS changes between hops.

The resolver also closes the time-of-check gap between the pre-flight lookup and the connect for the first hop, because the address the client dials is the one the policy just accepted.

`PolicyResolver` reads `private_addresses_allowed()` on every resolution rather than capturing it at construction. The client is built once per process in a `OnceCell`, and tests toggle the opt-in per test; capturing it once would freeze whichever value the first fetch saw.

A mixed answer set (one public and one private address) is handled differently by the two DNS checks. The pre-flight check on the original host still refuses the whole name if any address is private. The resolver, which also sees redirect hops, filters the set down to its public addresses and lets the connection proceed to those only. Neither path can dial the private address.

### What still guards IP literals and the post-connect peer

A literal such as `http://169.254.169.254/` never reaches a resolver, so the per-hop shape check in `media_redirect_policy` remains the only guard for IP-literal redirect targets and stays in place. The post-connect `remote_addr` check also stays, now as defence in depth and as the check that sees a proxy's address. It now pairs the peer with `response.url().host_str()` (the final hop's host) instead of the original request's host, so the debug line names the host that was actually contacted after a redirect. The module doc and `docs/llama-server-compat.md` now list four checks (pre-flight, resolver, per-hop shape, post-connect) instead of the old three.

### NAT64 and 6to4

`is_public_address` already unwrapped IPv4-mapped and IPv4-compatible IPv6 forms. It treated `64:ff9b::/96` (RFC 6052 NAT64 well-known prefix) and `2002::/16` (RFC 3056 6to4) as ordinary global IPv6. Behind a NAT64 gateway, which cloud IPv6-only subnets use to reach IPv4 resources in the same VPC, an AAAA record of `64:ff9b::a00:5` reaches `10.0.0.5`. Both prefixes now pass their embedded IPv4 address (bits 96..128 for NAT64, bits 16..48 for 6to4, via `embedded_ipv4`) through the IPv4 rules. Public neighbours such as `64:ff9b::808:808` and `2002:808:808::1` stay allowed. This predates #2644 but is the same policy surface; the security review on the PR found it.

### Bounded lookups

The pre-flight lookup and the resolver share `lookup_bounded`, a `tokio::time::timeout` of 5 s (`DNS_LOOKUP_TIMEOUT`, equal to the client's connect timeout) around `tokio::net::lookup_host`. An error or timeout maps to `Unresolvable`. Before this, a slow authoritative server for an attacker-chosen name could hold a request for as long as the system resolver retried.

### Proxies stay enabled

The media client still honours `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`, which the corporate-proxy deployments that `MLXCEL_EXTRA_CA_CERTS` exists for depend on. In reqwest 0.12.28 (no `socks` feature) the client resolves and dials only the proxy's own host, so the resolver and the post-connect check apply to the proxy endpoint. A proxy on a private address already needed the opt-in, and that is unchanged.

## 3. Change Summary

- `src/server/media_net.rs`: `MEDIA_HOST_REFUSED`; generic `Display` for `BlockedAddress` and `Unresolvable`; `MediaUrlError::log_detail` and `MediaUrlError::find_in`; `PolicyResolver`, `resolve_allowed_addresses`, `filter_allowed_addresses`, `lookup_bounded` with `DNS_LOOKUP_TIMEOUT`; NAT64 and 6to4 handling in `is_public_address` with `embedded_ipv4`; module doc rewritten around the four checks, what the client is told, and the proxy caveat; redirect-policy doc corrected.
- `src/server/media_net_test_guard.rs` (new): the test-only opt-in guards moved out of `media_net.rs`, which is 506 lines after the change.
- `src/server/media.rs`: `log_detail` at both fetch boundaries, `find_in` on the `send()` error, the post-connect check keyed on the final response's host, and `.dns_resolver(Arc::new(PolicyResolver))` on the shared `http_image_client`. The video path keeps its `warn` line, which now prints only the generic text.
- `docs/llama-server-compat.md`: the remote-media divergence now lists four checks, the generic client text, the proxy behaviour, and the NAT64/6to4 forms.
- Tests (new files): `media_url_disclosure_tests.rs` (labeled-image text, chat-audio text, and a redirect from a loopback origin onto `localhost` through the real shared client with a listener that records any connection), `media_net_resolver_tests.rs` (resolver deny, allow, opt-in read per resolution, trailing-dot names, and `filter_allowed_addresses` on mixed, all-private, empty and opt-in sets), `routes/embeddings_media_url_tests.rs` (the real embeddings handler with an image-capable stub). Additions to `media_net_tests.rs`: Display equality of the two refusals, the post-connect error text, NAT64/6to4 refusal, and numeric IPv4 spellings (`0x7f.1`, `2130706433`, `0177.0.0.1`, `127.1`) checked as literals.

## 4. Trade-offs and Residuals

- **Operators lose the hint in the 400.** The response no longer names `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS`. An operator who expects internal media to load and gets the generic error has to enable `RUST_LOG=mlxcel::server::media_net=debug` to see the host, the address and the opt-in hint.
- **Proxy deployments rely on the proxy for hostname redirects.** Through a proxy, a hostname redirect target is resolved by the proxy, not by `PolicyResolver`. The pre-flight check still resolves the requested host locally, and the per-hop shape check still refuses literal targets, but refusing a hop onto an internal name depends on the proxy's own egress policy.
- **DNS timing.** A name that resolves to a private address and a name that returns NXDOMAIN produce the same text, but not necessarily the same latency. A caller measuring response time can still separate them. Neither reveals an address.
- **Transport errors for public hosts.** A name that resolves publicly but fails to connect or to complete TLS still returns `Failed to fetch {kind} URL {url}: {reqwest error}`. That text describes a public destination the caller chose; it does not carry an internal address.
- **NAT64 prefixes beyond the well-known one.** The local-use prefix `64:ff9b:1::/48` (RFC 8215) and operator-chosen NAT64 prefixes cannot be identified from the address alone and remain allowed.
- **Two lookups for the first host.** The pre-flight check and the resolver each resolve the original host, so a fetch issues one more DNS query than before. Both are bounded at 5 s.

## 5. Validation

**Revert proof** (recorded in `revert-evidence.txt` in the PR author's session scratchpad). With the fix as an uncommitted patch, the six disclosure and redirect tests (`server::routes::embeddings::tests::media_url::`, `server::media::tests::url_disclosure::`, `server::media_net::tests::a_blocked*`) passed under `cargo test --profile test-fast --features cuda -p mlxcel --lib -- --test-threads=1`. Reverting only `media.rs` and `media_net.rs` (358-line patch, test files kept) made all six fail. Representative failures: the route test received `input[0]: failed to read image: image URL rejected: media URL host localhost resolves to 127.0.0.1, which is a loopback, ... set MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1 ...`; the redirect test panicked with `redirect was followed to http://localhost:36649/x.png (status 200 OK)`, which is the original defect observed directly. Re-applying the patch made all six pass again. Per the PR body and commit 2a9a062e, the NAT64 test likewise fails with `media_net.rs` reverted (`64:ff9b::7f00:1 must be refused`).

**Real server** (recorded in `real-server.txt`, plus a repeat at the pre-rebase final head 73fbd1c9). `mlxcel-server` with `colsmol-256m-merged` on GB10, `RUST_LOG=info,mlxcel::server::media_net=debug`, opt-in unset, and a recording TCP listener standing in for an internal service:

| Request `image_url` to `/v1/embeddings` | Result |
|---|---|
| `http://localhost:18391/x.png` | 400, generic text |
| `http://intranet-admin.internal/x.png` | 400, identical text |
| `http://10.0.0.7/x.png` | 400, identical text |
| `https://httpbin.org/redirect-to?url=http://localhost:18391/x.png` | 400, identical text |
| `http://[64:ff9b::7f00:1]:18393/x.png` (final-head run) | 400, identical text |
| `data:` URL of `tests/fixtures/test_image.png` | 200 |

The recording listener logged only its `LISTENING` line, so no case reached it, the public redirect included. The server's debug log named the refused host for each case; for the httpbin redirect it logged `host="localhost" address=127.0.0.1`, showing that the refusal happened at the redirect hop's resolution. With `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`, `http://localhost:18392/x.png` (through the resolver) and `http://127.0.0.1:18392/y.png` (literal) both returned 200 and the listener logged both GETs, so the opt-in still works on both paths.

**Suites.** At the pre-rebase head: `server::media_net`, `server::media::`, `server::routes::embeddings`, `server::routes::rerank` (119 passed), `cargo clippy --lib --tests -D warnings` and `cargo fmt --check` clean.

**Rebase.** The branch was rebased onto main 11463e50 before this report. `git range-diff` shows the three commits are patch-identical, and the upstream commits pulled in (#2645, #2646, #2648) touch neither `src/server/` nor `Cargo.lock`. The real-server and revert runs above predate the rebase and were not repeated on it.

**Not verified.** No test or run went through an HTTP proxy, so the proxy behaviour rests on reading reqwest 0.12.28's `connect_via_proxy`. The 5 s lookup bound has no test with a slow DNS server. A real NAT64 gateway was not available; NAT64 coverage is the unit test and the literal-host real-server case.

## 6. Learning Points

- Removing identifiers from an error is not enough if the error still has two distinguishable outcomes. "Refused" and "does not resolve" have to read the same, or the response remains an existence oracle.
- reqwest's redirect policy is synchronous and runs before resolution, so it cannot make address decisions for hostnames. `ClientBuilder::dns_resolver` is the hook that sees every name the client connects to, redirect hops included, and refusing there stops the request before a socket opens.
- A policy error raised inside reqwest (resolver or redirect policy) comes back wrapped. Walk `source()` and downcast before deciding which text the client sees, or the wrapper's transport wording will carry what the policy error was built to hide.
- An IPv6 address can carry an IPv4 destination in more forms than IPv4-mapped: NAT64 (`64:ff9b::/96`) and 6to4 (`2002::/16`) both route to an embedded IPv4 address and need the IPv4 rules applied to that address.
