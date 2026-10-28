# PR #2702: Check every media redirect hop and refuse local-use NAT64

**Date**: 2026-10-28 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2649 left two SSRF gaps in the remote media fetch, and its own report listed both as unverified. Behind an HTTP proxy, a redirect from a public origin onto an internal hostname was never checked by mlxcel, because reqwest resolves only the proxy's host and the proxy resolved the hop. NAT64 addresses outside the well-known `64:ff9b::/96` passed as public, so `64:ff9b:1::a9fe:a9fe` reached `169.254.169.254` through a local translator. This PR (issue #2693) moves redirect following out of reqwest: the media client is built with `Policy::none()`, and `send_following_redirects` in the new `src/server/media_net_redirect.rs` runs the full `ensure_media_url_allowed` check on every hop before sending it, proxied or not, inside one shared 10 s deadline. The RFC 8215 local-use block `64:ff9b:1::/48` is refused by default. A new variable, `MLXCEL_MEDIA_NAT64_PREFIXES`, lists RFC 6052 prefixes whose addresses are judged by their embedded IPv4 address, and a bad entry fails startup. The security pass added two fail-closed rules: all of `0.0.0.0/8` is refused, and a configured prefix shorter than /96 that covers `64:ff9b::/96` is rejected at parse time.

## 1. Problem Statement

After #2649 the media fetch had four checks: a pre-flight check of the requested URL (`ensure_media_url_allowed`: shape, local resolution, every address), `PolicyResolver` installed as the client's DNS resolver, a redirect policy callback, and a post-connect check of the peer address. Two of them stop working when a proxy carries the request.

- `media_redirect_policy()` was a reqwest redirect callback. Those callbacks are synchronous, so it could only re-run `ensure_media_url_shape` on the next URL. It caught IP-literal targets (`http://169.254.169.254/`) and left hostname targets to `PolicyResolver`.
- `http_image_client()` never calls `.proxy()` or `.no_proxy()`, so reqwest 0.12.28 installs its system proxy matcher (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, plus macOS and Windows system settings). Through a proxy, reqwest resolves only the proxy's host. A hop onto `http://intranet.internal/` was resolved by the proxy and never reached `PolicyResolver`. The post-connect check saw only the proxy's address.

As a result, on any deployment whose environment set a proxy, a public URL that answered `302 Location: http://localhost:PORT/` made the proxy fetch the internal target. The failing-first test on main 54d63b0a shows this directly: the recording proxy's log had a second entry, `http://localhost:42731/x.png`, and the fetch returned the proxy's 502.

The NAT64 gap was in `is_public_address`. It decoded the embedded IPv4 of `64:ff9b::/96` and `2002::/16` (6to4) and treated every other IPv6 address under the generic rules. RFC 8215 reserves `64:ff9b:1::/48` for local-use NAT64, which is not globally reachable, and operators may also translate through a network-specific prefix (NSP) carved from their own global unicast space. Addresses in either passed as public.

## 2. Technical Decisions

### Follow redirects manually for every fetch, not only proxied ones

The client now uses `reqwest::redirect::Policy::none()`, and `send_following_redirects(client, first)` follows the chain. On each followed 3xx it joins `Location` against `response.url()`, runs `ensure_media_url_allowed(next)` (shape, local resolution, every resolved address), and only then sends the next request. A refused hop never produces a CONNECT or an absolute-form request line toward its host.

The issue considered switching to manual following only when a proxy is in effect, and rejected it. reqwest does not expose its proxy matcher, and the decision is per URL: `NO_PROXY`, scheme-specific variables, CGI `REQUEST_METHOD` suppression and the macOS/Windows system settings all affect it, so one chain can mix proxied and direct hops. Reproducing the decision would mean owning a `hyper_util` matcher and keeping two redirect paths, one of which would run only in proxy deployments. A single manual path behaves the same for environment proxies, system proxies and any proxy configured later, and the existing non-proxy redirect test keeps exercising it.

`PolicyResolver` and the post-connect check stay. For a direct hop the resolver closes the window between `check_hop`'s own lookup and the connect, and it is now the defence-in-depth layer for hostnames. The mutation run in the PR shows that layering: with `check_hop` removed, both proxied refusal tests failed, but the direct-path disclosure test still passed, because `PolicyResolver` refused the hop on its own.

### One deadline for the whole chain

reqwest's client-level `.timeout(10 s)` used to cover the redirect chain and the body read together. A manual loop that sent each hop with that timeout would stretch the worst case to 50 s over five requests. The helper takes `deadline = Instant::now() + budget` once and sends every hop with `.timeout(deadline - now)`. The per-request timeout in reqwest also covers the body read of the returned response, so the final body is bounded by the same deadline. The hop check's local lookup is bounded too (`tokio::time::timeout_at(deadline, ...)`), so a slow resolver for an attacker-chosen name cannot extend the chain. `send_following_redirects_within(client, url, budget)` exists as the test seam for this: a 300 ms budget against an origin whose second hop never answers returns `Transport` with `is_timeout()` true.

A lookup that runs out of time is reported as `Refused(Unresolvable)`, not as a timeout. The client sees the generic `MEDIA_HOST_REFUSED` text either way, so this does not leak anything, but the operator's warning line says "refused" for what was a timeout. The review recorded this as LOW.

### Same chain semantics as reqwest's policy

The loop keeps the observable behavior of the removed policy where it mattered:

- Only 301, 302, 303, 307 and 308 are followed. Any other status, 3xx included (for example 304), is returned to the caller unchanged.
- At most `MAX_MEDIA_REQUESTS = 5` requests. When the fifth answer is still a followed redirect, the result is `Refused(TooManyRedirects { url: next })`, as `attempt.previous().len() >= 5` produced before. The count check runs before `check_hop`, so the sixth target is neither resolved nor contacted.
- Every hop is sent as a body-less GET. The media fetch only issues GETs, so 307 and 308 lose nothing.
- No `Referer` header. reqwest's automatic policy added one; nothing depended on it, and not sending the previous URL to the next host is the safer default.
- A followed 3xx whose `Location` is missing, not UTF-8, or does not join is a new `BadRedirect { status }` error, rendered as `{kind} URL returned redirect {status} without a usable Location header`. reqwest used to return such a response to the caller as-is; `error_for_status()` does not reject a 3xx, so the redirect's body was then read and failed later as undecodable media.

The intermediate response is dropped unread. When it still has an unread body the connection is closed; an empty redirect body can let the connection return to the pool. The comment on `drop(response)` was corrected during review to say exactly this. Either outcome is safe because every hop is checked anyway.

### Error mapping at the two call sites

`MediaSendError` has three variants: `Refused(MediaUrlError)`, `Transport(reqwest::Error)` and `BadRedirect`. `transport_or_refusal` still runs `MediaUrlError::find_in` on reqwest errors, so a refusal raised by `PolicyResolver` comes back as `Refused`. In `fetch_remote_bytes_with_limit_cancel` the new `media_send_failure` turns a refusal into `{kind} URL rejected: {MEDIA_HOST_REFUSED}` with the detail at `debug`, the same text as a pre-flight refusal. A hop refusal therefore cannot be told apart from a first-URL refusal, which preserves #2649's single-message rule. `fetch_remote_video` keeps its `warn` and `None` shape for each variant. `media_send_failure` repeats the rendering of the existing `rejected` closure instead of calling it; the text is identical and asserted by tests (LOW, left as is).

`media_client_builder()` was split out of `http_image_client()` so tests build the production client configuration (timeouts, `Policy::none()`, `PolicyResolver`) and add their own `.proxy(..)`. `http_image_client()` adds the extra CA certificates and builds the shared client from it.

### NAT64: fail closed by default, decode only what the operator names

An address in `64:ff9b:1::/48` does not say where its IPv4 address sits. RFC 8215 allows any RFC 6052 length inside the block (/48, /56, /64, /96), and each length puts the IPv4 bytes at a different position. Decoding under every candidate length does not work: `64:ff9b:1::5db8:d822`, a /96 synthesis of the public 93.184.216.34, decodes to `0.0.0.0` under /48 and would be refused. Operator NSPs are worse: they cannot be distinguished from ordinary global IPv6 at all. So the PR adds configuration and makes the default the safe one:

- With nothing configured, every address in `64:ff9b:1::/48` is refused. The block is not globally reachable, so no public origin legitimately lives there.
- `MLXCEL_MEDIA_NAT64_PREFIXES` lists prefixes as `addr/len`, `len` in {32, 40, 48, 56, 64, 96}. An address inside a listed prefix is judged by the IPv4 address `Nat64Prefix::embedded_ipv4` extracts per RFC 6052 section 2.2, skipping octet 8 (the `u` octet) for lengths below 96. A non-zero `u` octet embeds nothing and is refused, since no conforming translator produces it. Listing `64:ff9b:1::/96` restores DNS64-synthesized public hosts for IPv6-only deployments; listing an NSP makes its addresses decode instead of passing as public.
- Parsing trims whitespace and skips empty entries. A value that does not parse, uses another length, sets bits past its length, or sets the `u` octet in a /96 prefix fails startup with an error naming the variable and the entry. A typo must not silently leave the translator open. Both server entry points (`src/bin/mlx_server.rs` and `src/commands/serve.rs`) call `configure_media_nat64_prefixes_from_env()` next to `configure_private_media_urls` and map the error into `anyhow`.

The IPv6 rule order in `is_public_address_with(addr, nat64)` is: loopback, unspecified and multicast refused; IPv4-mapped judged as IPv4; the rest of `::/96` refused; the first configured prefix that contains the address decides; `64:ff9b::/96` decoded as before; `64:ff9b:1::/48` refused; 6to4 decoded; ULA, link-local and documentation refused. Configured prefixes come before the well-known rule so that an operator listing is authoritative.

`is_public_address_with` is pure and is what the tests call. `is_public_address(addr)` reads the process-wide list from a `std::sync::RwLock<Vec<Nat64Prefix>>` through `with_configured_prefixes`, so `PolicyResolver`, the pre-flight check and the post-connect check all see the same configuration. The lock is written once at startup and read once per checked address; poisoning is recovered with `PoisonError::into_inner`.

One addition beyond the issue: a /96 prefix whose `u` octet is set is rejected at parse. RFC 6052 requires that octet to be zero, and every address under such a prefix would be refused anyway, so accepting it would only hide an operator mistake.

### Review fix: misconfigured lengths must fail closed

The security pass found a MEDIUM fail-open path. Configured prefixes are consulted before the well-known rule, and `parse` accepted a prefix covering `64:ff9b::/96` at a shorter length, such as `64:ff9b::/64`. Under that entry `64:ff9b::a9fe:a9fe` decodes to `0.0.0.169`. The IPv4 rules refused only `0.0.0.0`, so the address passed, and a /96 translator forwarded it to `169.254.169.254`. Commit 0bd4fad0 adds two rules:

- The IPv4 rules refuse all of `0.0.0.0/8` (RFC 791 "this network"), not just `0.0.0.0`. Any listing that is shorter than the translator's real length and whose extra bits are zero now decodes into `0.0.0.x` and is refused. This also covers `64:ff9b:1::/64` listed for a /96 translator.
- `Nat64Prefix::parse` rejects a prefix shorter than /96 that contains `64:ff9b::/96`, which RFC 6052 section 2.1 defines at /96 only.

The same commit added the DNS rebinding sentence to the proxy documentation.

## 3. Change Summary

- `src/server/media_net_redirect.rs` (new, 151 lines): `MEDIA_FETCH_TIMEOUT`, `MAX_MEDIA_REQUESTS`, `MediaSendError`, `send_following_redirects` and `send_following_redirects_within`, with `check_hop` bounding the hop's lookup by the chain deadline.
- `src/server/media_net_nat64.rs` (new, 197 lines): `Nat64Prefix` (`parse`, `contains`, `embedded_ipv4`), `parse_media_nat64_prefixes`, the process-wide prefix list, `configure_media_nat64_prefixes_from_env`, and `with_configured_prefixes`.
- `src/server/media_net.rs`: both modules mounted with `#[path]` (the pattern `media_net_test_guard.rs` uses; the file is 533 lines); `is_public_address` split into a reader and the pure `is_public_address_with`; configured-prefix and local-use rules; `0.0.0.0/8` refused; `media_redirect_policy()` deleted; module doc for checks 2 to 4 and the proxy paragraph rewritten.
- `src/server/media.rs`: `media_client_builder()` extracted; both fetch paths call `send_following_redirects`; `media_send_failure` maps the error at the labeled-image path; the video path maps it inline.
- `src/server/mod.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`: re-export and startup wiring of `configure_media_nat64_prefixes_from_env`.
- `docs/environment-variables.md`: `MLXCEL_MEDIA_NAT64_PREFIXES` row (`make verify-env-var-docs` enforces it). `docs/llama-server-compat.md`: checks 2 to 4, the proxy paragraph (split-horizon DNS and DNS rebinding as the residual for the proxy's egress policy), and the refused list with `0.0.0.0/8` and the local-use block.
- Tests: `src/server/media_net_proxy_tests.rs` (new, 436 lines) runs a recording forward proxy, an origin and an internal listener on loopback, and covers hostname and literal redirect refusals behind a proxy, a proxied chain reaching 200, the five-request limit, a missing `Location`, an unfollowed 3xx, and two deadline cases. `media_net_tests.rs` adds eleven NAT64 tests, including all six RFC 6052 section 2.4 vectors for 192.0.2.33, operator NSP decoding, parse rejections, env-list trimming, and the two review-fix tests. `media_url_disclosure_tests.rs` rewrites the direct-path redirect test to use the helper with `media_client_builder()`.

## 4. Trade-offs and Residuals

- **The proxy still resolves the name again.** mlxcel checks each hop against its own resolver, then the proxy resolves the same name for the actual connection. A proxy whose DNS view differs from the host's (split horizon), or a low-TTL name that answers mlxcel with a public address and the proxy with a private one (rebinding), still depends on the proxy's own egress policy. This applies to the first URL as well as to later hops, and the docs now say so.
- **A loopback proxy refuses its own successful fetches.** The post-connect check sees the proxy's address. With a proxy on `127.0.0.1`, a fully successful proxied fetch is refused unless `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` is set, and that variable disables every check, the new per-hop one included. Exempting an operator-configured proxy from the post-connect check was out of scope in the issue and is unchanged.
- **One extra lookup per direct hop.** A direct hop is resolved once by `check_hop` and once more by `PolicyResolver` at connect time. Both lookups are bounded and share the chain deadline. A host that cannot resolve public names without the proxy already failed the pre-flight check on main; that is unchanged.
- **Mis-sized NAT64 listings fail closed only when the extra bits are zero.** The `0.0.0.0/8` rule covers listings shorter than the translator's length whose surplus bits are zero, which is the common case (`64:ff9b::/64`, `64:ff9b:1::/64` for a /96 translator). It does not cover every mismatch. A listing longer than the real length (an NSP listed at /96 for a /64 translator) does not contain the synthesized addresses, so they fall through to the generic rules and pass as public, which is the pre-PR behavior for NSPs. A covering listing shorter than the real length whose surplus network bits are non-zero decodes a network byte into the IPv4 result: with `2001:db9:64::/48` listed for a translator on `2001:db9:64:1200::/56`, the synthesis of `169.254.169.254` decodes to `18.169.254.169` and passes. Both cases need an operator to list the wrong length, and neither can be detected from the address. The listed length has to match the translator's.
- **Pre-existing IPv6 gaps.** `fec0::/10` (deprecated site-local), Teredo `2001::/32`, RFC 2765 `::ffff:0:0:0/96` and `100::/64` still count as public, as on main. None is reachable on a typical stack.
- **DNS timing and public transport errors.** As after #2649, NXDOMAIN and a refused private answer render the same text but not the same latency, and a public host that fails to connect still returns reqwest's transport text.

## 5. Validation

**Failing first on main 54d63b0a.** The new tests were written against main's API (`media_redirect_policy()` plus `.send()` behind `Proxy::all`, and `is_public_address`). The proxied redirect test returned `Ok(("http://localhost:42731/x.png", 502))` with the proxy log `["http://media.test/start.png", "http://localhost:42731/x.png"]`; `local_use_nat64_forms_are_refused` and `an_operator_nat64_prefix_form_of_a_refused_address_is_refused` failed on their first address. The `Nat64Prefix`, deadline and `BadRedirect` tests use APIs that do not exist on main.

**Mutation check.** With `check_hop` skipped and each hop given the full budget, three proxy tests failed (`behind_a_proxy_a_redirect_onto_an_internal_hostname_never_leaves`, `behind_a_proxy_a_redirect_onto_an_internal_literal_never_leaves`, `hops_that_each_fit_the_budget_still_share_it`). With both review-fix rules disabled, `this_network_ipv4_block_is_refused_whole` (`0.1.2.3 must be refused`) and `a_prefix_shorter_than_96_covering_the_well_known_prefix_is_rejected` failed.

**Suites** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, one module at a time under `gpu-lock`, at 0bd4fad0): `server::media` 115 passed, `server::routes::embeddings` 23, `server::routes::rerank` 20. `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, and `make verify-env-var-docs` were clean.

**Real server** (GB10, run on 945d43d7; the review commit does not touch the redirect path). `mlxcel-server` with `colsmol-256m-merged`, `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS` unset, `HTTP_PROXY` and `HTTPS_PROXY` pointing at a recording forward proxy on `127.0.0.1:19652`, and a recording listener on `127.0.0.1:19651`:

| Case | Result |
|---|---|
| `MLXCEL_MEDIA_NAT64_PREFIXES=64:ff9b:1::/50` | startup exit 1, error names the variable and the entry |
| `https://postman-echo.com/redirect-to?url=http://localhost:19651/x.png` | 400, `image URL rejected: media URL host is not allowed or could not be resolved` |
| `http://[64:ff9b:1::7f00:1]:19651/x.png` | 400, same text |
| `data:` PNG | 200, embedding list |

The proxy logged only `CONNECT postman-echo.com:443`, the listener logged no connection, and the debug log recorded the refusal with `host="localhost" address=127.0.0.1` for the redirect case. Two substitutions from the issue's plan: `postman-echo.com/redirect-to` replaced `httpbin.org/redirect-to`, because httpbin now answers 403 when the redirect target is internal and so never issues the 302, and the listener used port 19651 instead of 18391.

**Not verified.** The successful proxied chain (200 at the end) is covered only by `a_redirect_chain_through_the_proxy_reaches_its_final_response` under the test opt-in; the real-server run cannot show it without the opt-in because of the loopback-proxy residual above. No real NAT64 translator was available, so NAT64 coverage is the unit tests (including the RFC 6052 vectors) and the literal-host case. The suites and real-server run predate the merge of origin/main (fdfe7f6f), which brought in #2687, #2688, #2700 and #2701; none of them touches `src/server/media*`, the startup sites, or the docs changed here.

## 6. Learning Points

- reqwest's redirect policy is synchronous and runs before resolution, and through a proxy the client never resolves the target at all. A security check that has to resolve names cannot live in that callback. Turning redirects off (`Policy::none()`) and following them in an async loop is the only place where every hop can be resolved and checked before anything is sent, whatever the transport.
- When replacing a client-level timeout with a manual loop, carry the deadline, not the timeout. A per-request `.timeout(remaining)` from one `Instant` keeps the worst case at the original bound, and bounding the per-hop lookup with `timeout_at` closes the remaining gap.
- An RFC 6052 address does not encode its own prefix length. Where the length cannot be known, refuse; where an operator supplies it, decode only under what they named, and make a bad value fail startup.
- When a decode can be fed the wrong length, check what the wrong answers look like. A zero-filled mismatch lands in `0.0.0.0/8`, so refusing that whole block turns a class of misconfiguration from fail-open into fail-closed. It does not cover mismatches with non-zero surplus bits.
- A mutation test that removes the new check is more informative when another layer still passes. Here the direct-path test kept passing because `PolicyResolver` caught the hop, which shows which layer actually carries each case.
