# PR #2741: Harden media fetch errors, DNS bound and NAT64 prefixes

**Date**: 2026-10-30 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at b0c76bed (commits 420c3a7c, c7c65f85 and d1596463, plus a merge of main at 060b5051). The targeted `server::media` tests, clippy, fmt and the two doc verifiers passed on GB10 (spark-102) per the PR body. At the time of writing, the hosted fast checks were green, `cargo-clippy`, `CPU-only link`, `WebUI installed artifact` and `OpenXLA feature compile` were still pending, and a pre-merge local gate was running in the worktree
**Languages**: Rust, Markdown
**Risk Level**: Low (one startup behavior change: `MLXCEL_MEDIA_NAT64_PREFIXES` now refuses a prefix outside `2000::/3` and `64:ff9b:1::/48`; media DNS lookups now share a 32-permit pool)

## Executive Summary

PR #2741 (closes #2728) fixes three LOW residuals that the security review of #2718 left in remote media URL fetching (`src/server/media.rs`, `src/server/media_net*.rs`). First, client-facing media errors no longer name a redirect hop: reqwest's error text carries the URL it failed on, which after a redirect is whatever the origin's `Location` header chose, so the PR strips it with `reqwest::Error::without_url()` where the error is created and drops the hop from the `TooManyRedirects` message. Second, media DNS lookups are bounded: a new `media_net_dns` module runs `getaddrinfo` on `spawn_blocking` while holding one of 32 process-wide semaphore permits, and the permit stays held until the resolver returns, so lookups abandoned by a timeout still count. Third, `Nat64Prefix::parse` refuses prefixes outside global unicast `2000::/3` and the RFC 8215 local-use block `64:ff9b:1::/48`, because a listed NAT64 prefix outranks the global-unicast gate.

None of the three lets a client reach a private address on main with the default configuration. Item 1 reflects text chosen by the origin server into the server's own error responses; item 2 is a denial of service against the Tokio blocking pool; item 3 needs an operator to configure a non-global prefix.

## 1. Problem Statement

Issue #2728 collected the three residuals and verified them against main 59c07652. The failing-first runs in section 4.1 were taken on main 8385f40b.

### 1.1 Redirect hop URLs reached client-facing errors

reqwest's `Display` for an error appends `for url (...)`. When the media client follows a redirect, that URL is the hop named by the origin's `Location` header. Three paths rendered it into the 400 returned by the embeddings, rerank and image-generation routes: a transport failure on the hop (`Failed to fetch image URL <url>: error sending request for url (<hop>)`), a non-2xx final response (`image URL returned error status <url>: HTTP status client error (404 Not Found) for url (<hop>)`), and a body-read error. A fourth path surfaced during implementation: `MediaUrlError::TooManyRedirects` rendered `media URL redirect chain exceeded 5 hops at <hop>`. The client's own URL is its own to see; the hop is not, because it can encode anything the origin wants reflected (internal hostnames learned elsewhere, tokens, or text meant to mislead a reader of the error).

### 1.2 Media DNS lookups could pin the whole blocking pool

`lookup_bounded` in `media_net.rs` wrapped `tokio::net::lookup_host` in a 5 s `DNS_LOOKUP_TIMEOUT`. `lookup_host` calls `getaddrinfo` on Tokio's blocking pool, and `getaddrinfo` cannot be cancelled: the timeout drops the future, not the thread. Requests naming hosts whose nameservers never answer could therefore occupy up to the whole blocking pool (512 threads by default) until the system resolver gave up, starving every other `spawn_blocking` user in the server. Both the pre-flight and per-hop check (`ensure_media_url_allowed`) and the connect-time `PolicyResolver` (`resolve_allowed_addresses`) went through this function.

### 1.3 NAT64 prefix configuration accepted non-global prefixes

`Nat64Prefix::parse` checked syntax, RFC 6052 lengths, host bits, the well-known overlap and the reserved `u` octet, but not where the prefix lies. By design since #2718, an address inside a listed prefix is judged by the IPv4 address it embeds, ahead of the `2000::/3` global-unicast gate. Listing a link-local (`fe80::/10`) or unique-local (`fc00::/7`) prefix therefore let any address inside it pass whenever it embedded a public IPv4 address, which re-opens those private ranges if the prefix is not really a translator.

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `media_net_redirect.rs` | `transport_or_refusal`, the only constructor of `MediaSendError::Transport`, logs the hop at debug (`hop = ?err.url()`) and stores `err.without_url()`. A wrapped `MediaUrlError` refusal is still returned as `Refused`. |
| 1 | `media.rs` | `fetch_remote_bytes_with_limit_cancel` formats the non-2xx error as `{kind} URL returned error status {url}: {err.without_url()}` and maps body-stream errors through `reqwest::Error::without_url` before `with_context`. The client's `{url}` stays. The video path only logs and is unchanged. |
| 1 (review) | `media_net.rs` | `MediaUrlError::TooManyRedirects` renders `media URL redirect chain exceeded 5 hops`; the hop stays in the variant and `log_detail` gains an arm that logs it at debug. |
| 2 | `media_net_dns.rs` (new) | `lookup_bounded` keeps the 5 s `DNS_LOOKUP_TIMEOUT` and adds `MAX_CONCURRENT_MEDIA_DNS_LOOKUPS = 32`, a process-wide `Arc<Semaphore>` in a `OnceLock`. `lookup_bounded_with(resolve, host, port, timeout, permits)` acquires an `OwnedSemaphorePermit` under a single deadline, moves it into a `spawn_blocking` closure that calls `resolve`, and awaits the handle under the same deadline. Saturation (logged at debug), a closed semaphore, a resolver error, a join error and a timeout all return `MediaUrlError::Unresolvable`. The resolver is `(host, port).to_socket_addrs()`, the system resolver. |
| 2 | `media_net.rs` | The old `lookup_bounded` and `DNS_LOOKUP_TIMEOUT` move out; the module imports `dns::lookup_bounded` (and `lookup_bounded_with` under `cfg(test)`). Call sites are unchanged. |
| 3 | `media_net_nat64.rs` | After the existing checks, `parse` calls `is_global_translator_space(network, len)`: top three bits `001` (`2000::/3`), or `len >= 48` with the first 48 bits equal to `64:ff9b:1::` (`64:ff9b:1::/48`), or exactly `64:ff9b::/96`. Otherwise the error is `"<entry>" lies outside 2000::/3 and 64:ff9b:1::/48; a link-local, unique-local or other non-global prefix would let synthesized addresses reach private hosts`, which `configure_media_nat64_prefixes_from_env` already turns into a startup failure naming the variable. |
| Docs | `docs/environment-variables.md`, `docs/llama-server-compat.md` | The `MLXCEL_MEDIA_NAT64_PREFIXES` row states the range rule and the behavior change (a ULA translator prefix is no longer accepted; renumber into global or `64:ff9b:1::/48` space). The compat page's address-policy paragraph states the same rule. |
| Tests | 3 test files | 6 new tests (section 4.1). The disclosure tests gained a `serve_canned` loopback origin and a `redirected_fetch_outcome` helper that goes through `fetch_labeled_image`, the path the routes use to build their 400. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 10 (5 source including the new `media_net_dns.rs`, 3 test, 2 docs) |
| Lines added / deleted | +414 / -26 against main |
| Tests added | 6 (`media_url_disclosure_tests.rs` 4, `media_net_resolver_tests.rs` 1, `media_net_tests.rs` 1) |
| New dependencies | None |
| Commits | 420c3a7c (items 1 to 3), c7c65f85 (`TooManyRedirects` hop), d1596463 (body-read guard test, DNS test made robust), b0c76bed (merge of main) |

## 3. Technical Decisions

### 3.1 Strip the URL where the error is created, not in the rendered text

**Context.** The hop reaches the client through any code that formats a `reqwest::Error`. Scrubbing the final message (for example, removing `for url (...)` with a pattern) would depend on reqwest's wording and would miss a new formatter.

**Choice.** `transport_or_refusal` is the single place a `MediaSendError::Transport` is built, so stripping there with `without_url()` makes every later renderer safe. The two errors that do not pass through it (non-2xx status and body read, both produced after the redirect loop returns) strip at their own creation sites. The hop is not lost: it goes to a debug log line next to the strip, so an operator can still see where a chain went.

**Trade-off.** A client debugging its own redirect chain sees less. That is intended; the server log has the hop.

### 3.2 `TooManyRedirects` keeps the hop as data but does not display it

The issue listed three paths. The acceptance criterion was broader ("no client-facing media error contains a URL other than the one the client sent"), and a test written against the criterion (`too_many_redirects_omits_hop_url`) found the fourth. The variant keeps its `url` field so `log_detail` can log it, and the `#[error]` string drops it. This follows the pattern `Unresolvable { host }` already used: the field is for logs, the `Display` is for clients.

### 3.3 A semaphore around `spawn_blocking`, not an async resolver

| Option | Pros | Cons |
|---|---|---|
| Keep `tokio::net::lookup_host` with a timeout | No change | Timeout drops the future, not the thread; the bug |
| Async resolver (hickory-resolver) | Cancellable, no blocking thread | New dependency; bypasses `/etc/hosts` and nsswitch, which operators rely on and the address policy honours |
| **Chosen: system resolver on `spawn_blocking` under a 32-permit semaphore** | Keeps resolver semantics; bounds threads including abandoned ones; no dependency | One noisy client can fill the pool (section 5) |

The permit is moved into the blocking closure (`let _permit = permit;`), so it is released when `getaddrinfo` returns, not when the caller's future is dropped. That is the property the bound needs: counting callers would let abandoned threads pile up exactly as before. Permit acquisition and the lookup share one deadline (`timeout_at`), so a caller never waits longer than `DNS_LOOKUP_TIMEOUT` in total. A saturated pool returns `Unresolvable`, the same client-facing text as a refused or non-resolving host, so saturation is not a new oracle.

32 is far below Tokio's default 512 blocking threads, so media resolution can no longer starve file I/O or other `spawn_blocking` users, and well above the lookup concurrency a healthy server needs (each lookup normally returns in milliseconds).

### 3.4 Inject the resolver and the semaphore for tests

`lookup_bounded_with` takes the blocking resolver closure and the `Arc<Semaphore>` as parameters; `lookup_bounded` passes `system_resolve` and the process-wide pool. The test can then use a resolver that sleeps 2 s and counts concurrent calls, with its own 4-permit semaphore, without touching the global pool other tests share and without network access.

### 3.5 Where a NAT64 prefix may lie

The rule accepts a prefix wholly inside `2000::/3` or `64:ff9b:1::/48`. The local-use check requires `len >= 48`, so `64:ff9b:100::/40` (which covers more than the local-use block) is refused. The well-known `64:ff9b::/96` is accepted as a special case even though it lies outside both: an existing test parses it, it is globally reachable per the IANA registry, and the address policy always decodes it whether listed or not, so listing it changes nothing.

The check is a range test on the top bits, not a lookup in the IANA special-purpose registry. `2001:db8:64::/96` still parses (the issue lists it as a prefix that must keep working, and the tests use it). Section 5 records what that leaves open.

## 4. Verification

### 4.1 Failing-first evidence

The tests were written before the fix and run on main 8385f40b:

```
redirect_hop_transport_error_omits_hop_url: the redirect hop URL leaked: ... Failed to fetch image URL http://127.0.0.1:39655/start.png: error sending request for url (http://127.0.0.1:40097/hop-secret-path)
redirect_hop_error_status_omits_hop_url: the redirect hop URL leaked: ... image URL returned error status http://127.0.0.1:45529/start.png: HTTP status client error (404 Not Found) for url (http://127.0.0.1:40087/hop-secret-path)
too_many_redirects_omits_hop_url: ... image URL rejected: media URL redirect chain exceeded 5 hops at http://127.0.0.1:46355/hop-secret-path
non_global_nat64_prefixes_are_refused: non-global NAT64 prefixes were accepted: ["fe80::/64 ...", "fd00:64::/96 ...", "fc00::/96 ...", "fec0::/96 ...", "64:ff9b:100::/40 ..."]
dns_lookups_are_bounded_by_the_permit_pool: error[E0425]: cannot find function `lookup_bounded_with` in this scope
```

`redirect_hop_body_read_error_omits_hop_url` (a hop that promises 4096 bytes, sends `partial` and closes) passes with and without the body-read strip, because reqwest builds body-stream errors without a URL today. It is a regression guard for the defensive strip, not failing-first evidence.

`dns_lookups_are_bounded_by_the_permit_pool` runs 100 lookups against a resolver that sleeps 2 s, with 4 permits and a 200 ms timeout. It asserts that peak resolver concurrency never exceeds 4, that every lookup returns `Unresolvable` naming its host within 200 ms plus 1 s of slack (still under the 2 s sleep, so a lookup that waited for the resolver would fail), that all permits come back once the abandoned resolver calls finish, and that a later lookup succeeds. The issue specified a 500 ms resolver; d1596463 raised it to 2 s so the slack can absorb a loaded runner while still separating "timed out" from "waited".

`non_global_nat64_prefixes_are_refused` also checks that `64:ff9b:1::/96`, `64:ff9b:1:abcd::/64` and `2001:db8:64::/96` still parse, and that `parse_media_nat64_prefixes("64:ff9b:1::/96, fd00:64::/96")` fails naming `MLXCEL_MEDIA_NAT64_PREFIXES` and the bad entry.

### 4.2 Local tests

On GB10 with `--profile test-fast --features cuda` at the final head: `server::media_net::` 36 passed, `server::media::tests::url_disclosure::` 9 passed, `server::media::tests::net_proxy::` 10 passed, and all of `server::media` 127 passed. `cargo clippy --lib --tests -D warnings`, `cargo fmt --check`, `make verify-env-var-docs` and `make verify-llama-compat` were clean.

### 4.3 Manual check

`mlxcel-server -m llama-nemotron-embed-vl-1b-v2` with `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS` unset: a `/v1/embeddings` image URL that redirects to the unreachable `http://1.1.1.1:81/hop-secret-path` returns 400 `Failed to fetch image URL <submitted URL>: error sending request`, without the hop. Starting with `MLXCEL_MEDIA_NAT64_PREFIXES=fe80::/64` fails with `"fe80::/64" lies outside 2000::/3 and 64:ff9b:1::/48; ...`.

### 4.4 Hosted CI

At the time of writing, the fast checks (fmt, cargo-deny, env var docs, llama-compat manifest, license headers and others) had passed; `cargo-clippy`, `CPU-only link`, `WebUI installed artifact` and `OpenXLA feature compile` were pending. The CUDA, ROCm and Windows jobs were skipped by the change filter.

## 5. Remaining Gaps and Risks

- **One client can fill the DNS pool.** A client that submits many URLs whose nameservers never answer holds all 32 permits for up to 5 s at a time, so other clients' hostname media fetches wait and then fail as `Unresolvable`. Before this PR the same traffic exhausted the whole blocking pool and hurt unrelated work, so the blast radius is smaller, but media fetches by hostname are still affected. Per-request fairness or a negative cache would be a separate change. IP-literal URLs skip the pre-flight lookup.
- **The deadline includes the permit wait.** Under contention a lookup gets less than 5 s of resolver time. That is the intended bound, but a slow-but-healthy resolver under load can now produce `Unresolvable` where it previously succeeded.
- **Each hostname fetch takes permits more than once.** The pre-flight check and the connect-time `PolicyResolver` each resolve, and every redirect hop repeats the pre-flight. Effective concurrent fetches by hostname are therefore lower than 32 when the pool is busy.
- **The body-read strip is not provable by test.** reqwest omits the URL from body-stream errors today; the strip protects against a future reqwest change, and the guard test would catch that change.
- **The NAT64 rule is a range check, not the IANA registry.** Prefixes inside `2000::/3` that the address policy otherwise refuses (documentation `2001:db8::/32`, the protocol-assignment block `2001::/23`, 6to4 `2002::/16`) still parse. Listing one makes addresses inside it judged by their embedded IPv4 instead of refused. None of those blocks routes to the operator's private hosts the way link-local or unique-local space does, so this was left as an operator decision.
- **Behavior change.** A deployment whose translator uses a unique-local prefix (for example `fd00:64::/96`) now fails startup. The fix is to renumber the translator into global or `64:ff9b:1::/48` space; `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` remains the blanket opt-out for deployments that need private targets.

## 6. Learning Points

### 6.1 Error values carry request context, and request context can be attacker-chosen

reqwest records the URL on its errors for good reason, but after a redirect that URL comes from the remote server. Any error that crosses a trust boundary should be checked for fields the other side controls. `reqwest::Error::without_url()` exists for this; applying it where the error is constructed protects every formatter downstream, which a scrub of the final string does not.

### 6.2 A timeout does not cancel blocking work

`tokio::time::timeout` around a future that delegates to `spawn_blocking` only stops waiting. The thread keeps running. To bound the resource, tie the permit to the thread's lifetime by moving it into the blocking closure, not to the caller's lifetime.

```rust
let permit = tokio::time::timeout_at(deadline, permits.acquire_owned()).await??;
let lookup = tokio::task::spawn_blocking(move || {
    let _permit = permit; // released when resolve returns, not when the caller gives up
    resolve(&owned_host, port)
});
tokio::time::timeout_at(deadline, lookup).await
```

### 6.3 An entry that outranks a gate needs the gate's check at configuration time

A listed NAT64 prefix is evaluated before the global-unicast test, so the configuration parser has to enforce what that test would have enforced. The same applies to any allow-list or override that bypasses a later policy check.

### 6.4 Write tests against the acceptance criterion, not only the listed sites

The issue named three leak paths. A test written for the broader criterion found the fourth (`TooManyRedirects`). Conversely, a test that passes without the fix should be labelled a regression guard, as d1596463 does for the body-read case, so it is not read as proof of a bug.

## 7. References

- Issue #2728: Media URL fetch error text, DNS thread bound and NAT64 prefixes
- PR #2718 and issue #2713: the security review that recorded these residuals
- Issue #2693: NAT64 prefix decoding (`MLXCEL_MEDIA_NAT64_PREFIXES`)
- Issue #2644: `PolicyResolver`, the connect-time address check
- RFC 6052 (IPv6 addressing of IPv4/IPv6 translators), RFC 8215 (local-use IPv4/IPv6 translation prefix)
- `docs/environment-variables.md` (`MLXCEL_MEDIA_NAT64_PREFIXES`), `docs/llama-server-compat.md` (remote media URL address policy)
