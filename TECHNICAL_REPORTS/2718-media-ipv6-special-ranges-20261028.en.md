# PR #2718: Refuse special-purpose IPv6 ranges in the media URL policy

**Date**: 2026-10-28 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low to Medium (tightens an SSRF allow-list; an address the policy used to admit may now be refused)

## Executive Summary

The remote media URL policy (#1451, #2644, #2649, #2693) decides which resolved or literal addresses an `image_url`, `audio_url` or video URL, and every redirect hop, may reach. Its IPv6 arm refused only unique-local, link-local, `2001:db8::/32` and the translated forms of refused IPv4 addresses, so site-local `fec0::/10`, the IPv4-translated `::ffff:0:0:0/96`, discard-only `100::/64`, the dummy prefix `100:0:0:1::/64`, SRv6 SIDs `5f00::/16`, the reserved space outside `2000::/3`, Teredo, benchmarking, deprecated ORCHID and the new documentation prefix `3fff::/20` all passed as public. #2702's report recorded these as pre-existing gaps. This PR (issue #2713) closes them with one structural rule rather than a list: after the IPv4-mapped decode, the listed NAT64 prefixes and the well-known `64:ff9b::/96` have been decided, every IPv6 address outside `2000::/3` is refused. Inside `2000::/3`, the `2001::/23` IETF protocol-assignment block is refused except the entries the IANA special-purpose registry marks globally reachable, and both documentation prefixes are refused. The predicate moved into its own file, `src/server/media_net_address.rs`.

The PR also closes two LOW findings from the #2702 review. `{kind} URL rejected:` is now rendered by one helper, `media_url_rejected`, for the pre-flight, redirect-hop and post-connect refusals. A redirect hop whose policy lookup outlives the chain deadline is now `MediaSendError::DeadlineExceeded` and reaches the client as a timeout that names no host, instead of the generic host refusal.

## 1. Problem Statement

### IPv6 blocks that passed as public

On main (752db4f9) `is_public_address_with` handled IPv6 in this order: loopback, unspecified and multicast refused; IPv4-mapped judged as IPv4; the rest of `::/96` refused; operator-listed NAT64 prefixes judged by their embedded IPv4; `64:ff9b::/96` decoded; `64:ff9b:1::/48` refused; 6to4 decoded; and finally only `fc00::/7`, `fe80::/10` and `2001:db8::/32` refused. Every other IPv6 address was public.

None of the missing blocks is globally reachable, but each is reachable on some hosts: a corporate network that still routes `fec0::/10`, a host with a local Teredo interface, an SRv6 deployment where a SID addresses a local function. On such a host, a client-supplied media URL or a redirect `Location` naming one of these addresses was fetched by the server. The failing-first run on 752db4f9 shows the extent: all 17 refused rows of the new table test passed as public.

### Two rendering issues in the fetch path

- `fetch_remote_bytes_with_limit_cancel` built the refusal text in a local closure, and `media_send_failure` built the same `"{kind} URL rejected: {refusal}"` plus the debug log call a second time. #2644's disclosure rule depends on the pre-flight refusal and a hop refusal rendering identically, and two copies of the format string can drift.
- `check_hop` mapped the chain deadline expiring during a hop's lookup to `Refused(Unresolvable)`. The client got `MEDIA_HOST_REFUSED` and the operator's log said "refused" for what was a timeout.

## 2. Technical Decisions

### A global-unicast gate instead of an enumeration

The new rule is one line, `if segments[0] & 0xe000 != 0x2000 { return false; }`. IANA has allocated only `2000::/3` for global unicast (RFC 4291, RFC 3587); everything outside it that reaches this point is local, special-purpose or reserved. The issue rejected enumerating the blocks outside `2000::/3` because the registry keeps growing out of reserved space: `5f00::/16` (RFC 9602, 2024) and `100:0:0:1::/64` (RFC 9780, 2025) were both carved from it, and a list misses the next one. A gate refuses future carve-outs by default. The previous `unique_local` and `link_local` terms became redundant and were deleted.

### Where the gate sits

The gate runs after every rule that decodes an embedded IPv4 address outside `2000::/3`, and before 6to4:

1. The IPv4-mapped decode (`::ffff:0:0/96`) and the `::/96` refusal run first, so `::ffff:127.0.0.1` is still judged as `127.0.0.1`. The IPv4-translated block `::ffff:0:0:0/96` differs from the mapped form in segment 5, so it is not decoded and falls to the gate.
2. Operator-listed NAT64 prefixes still win. `64:ff9b:1::/96`, listed under `MLXCEL_MEDIA_NAT64_PREFIXES`, lies outside `2000::/3`, and `a_listed_local_use_prefix_is_judged_by_its_embedded_address` passes unchanged.
3. The well-known `64:ff9b::/96` is decoded before the gate. The rest of `64:ff9b::/32`, which used to pass as public, is now refused by the gate. The security pass confirmed this ordering.
4. 6to4 (`2002::/16`) is inside `2000::/3`, so its decode moved below the gate without changing behavior.

The explicit `64:ff9b:1::/48` rule above the gate is now subsumed by it (`0x0064` is outside `2000::/3`). It was kept; its comment explains why the block cannot be judged without a listed prefix.

### Mirror the registry inside `2000::/3`

`ietf_protocol_assignment_not_global(segments)` is true for `2001::/23` (`segments[0] == 0x2001 && segments[1] < 0x0200`) unless the address is one of the registry's globally reachable entries: the three anycast addresses `2001:1::1`, `2001:1::2` and `2001:1::3` (RFC 7723, RFC 8155, RFC 9665), AMT `2001:3::/32` (RFC 7450), AS112-v6 `2001:4:112::/48` (RFC 7535), ORCHIDv2 `2001:20::/28` (RFC 7343) and DETs `2001:30::/28` (RFC 9374). This is the approach nightly `Ipv6Addr::is_global` takes; it is not available on stable, which the IPv4 arm already works around the same way. The final expression refuses this helper's result, `2001:db8::/32` and `3fff::/20` (`segments[0] == 0x3fff && segments[1] & 0xf000 == 0`).

The tests pin both sides of every exception: `2001:4::1`, `2001:4:113::1` and `2001:40::1` must be refused, and `2001:200::1` (first address past the `/23`) and `3fff:1000::1` (first past the `/20`) must stay public. Those edge rows came from the implementation review and were added in 70d3829e, so a future edit that widens an exception mask fails a test.

### Refuse Teredo whole

Teredo `2001::/32` carries the server's IPv4 in bits 32..64 and the client's IPv4, XOR-obfuscated, in the last 32 bits. Decoding them, as the policy does for 6to4, would mean judging two embedded addresses to admit traffic no media origin sends; the registry does not mark the block globally reachable. Teredo falls out of the `/23` rule with no special case.

### Move the predicate to `media_net_address.rs`

`media_net.rs` was 533 lines before this change grew the predicate further, and the predicate is pure apart from reading the configured NAT64 list, so it splits cleanly from the resolver, URL-shape and environment code around it. `is_public_address`, `is_public_address_with`, `embedded_ipv4` and the new helper moved to `src/server/media_net_address.rs` (185 lines), mounted with `#[path]` as `media_net_nat64.rs` and `media_net_redirect.rs` already are. `media_net.rs` is now 425 lines. Only `is_public_address` is re-exported for production; `is_public_address_with` and `Nat64Prefix` are re-exported under `#[cfg(test)]`, because no production caller outside the submodules needs them.

### One place renders a refusal

`media_url_rejected(kind, url, err)` calls `err.log_detail(url)` (host and address at `debug`) and returns `anyhow!("{kind} URL rejected: {err}")`. The pre-flight closure and the `Refused` arm of `media_send_failure` both call it, and the peer check goes through the same closure. This is a refactor with no behavior change; `a_hop_refusal_renders_like_the_preflight_refusal` asserts that the two paths produce the same text for `BlockedAddress` and `Unresolvable`.

### A deadline during a hop lookup is a timeout

`check_hop` now delegates to `check_hop_with(host, deadline, check)`, which wraps any policy future in `timeout_at(deadline, ...)`. A result from `check`, including `Unresolvable`, stays `Refused`; expiry of the deadline is `DeadlineExceeded { host }`. Taking the future as a parameter gives tests a seam: `std::future::pending()` exercises the deadline in 50 ms without a slow resolver.

The client text is `Failed to fetch {kind} URL {url}: the fetch deadline ran out while resolving a redirect target`. It names only the URL the client sent; the hop host goes to a `debug` log line. #2644's rule is that a refused host and an unresolvable host render identically, so that the response does not reveal which internal names exist. A deadline does not break it: the deadline belongs to the chain, not to the hop's name, and a transport timeout on the same chain was already reported distinctly. `fetch_remote_video` logs the same debug line plus a `warn` and returns `None`, as its other error arms do.

## 3. Change Summary

- `src/server/media_net_address.rs` (new, 185 lines): the address predicate with the global-unicast gate, `ietf_protocol_assignment_not_global`, the documentation prefixes, and RFC citations for every block.
- `src/server/media_net.rs` (+7/-115): mounts the new submodule and adjusts the re-exports (`check_hop_with` added under `#[cfg(test)]`).
- `src/server/media_net_redirect.rs`: `MediaSendError::DeadlineExceeded { host }`; `check_hop` split into `check_hop` and `check_hop_with`.
- `src/server/media.rs`: `media_url_rejected`, `REDIRECT_DEADLINE_EXCEEDED`, `log_redirect_deadline`; `media_send_failure` and `fetch_remote_video` handle the new variant.
- `docs/llama-server-compat.md`: the refused list now says only `2000::/3` is treated as IPv6 global unicast, names each newly refused block and the globally reachable exceptions. There is no Korean twin of this document.
- Tests: `ipv6_special_purpose_ranges_are_refused` (20 rows, every failing row reported with its block and RFC) and `ipv6_globally_reachable_neighbours_stay_public` (14 addresses) in `media_net_tests.rs`; two `check_hop_with` tests in `media_net_proxy_tests.rs`; two rendering tests in `media_url_disclosure_tests.rs`.

## 4. Trade-offs and Residuals

- **Behavior change for unusual deployments.** A deployment that fetched media from a site-local, Teredo, or other non-`2000::/3` address now gets a 400. `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` restores the old reachability (and disables every other address check). An IPv6-only deployment whose DNS64 synthesizes outside `2000::/3` lists that prefix in `MLXCEL_MEDIA_NAT64_PREFIXES`, as before.
- **`lookup_bounded`'s own timeout still reads as a refusal.** The per-lookup 5 s bound inside `ensure_media_url_allowed` reports `Unresolvable`, for the first URL and for hops. Only expiry of the chain deadline becomes `DeadlineExceeded`. The issue kept this out of scope.

The security pass on 5a4d137b found no CRITICAL or HIGH issues and left three LOW items open:

- **Hop URL in transport and status error text.** After a redirect, a `Transport` or `error_for_status` error renders reqwest's `Display`, which includes the hop URL. The hop already passed the policy and its host came from the origin's `Location`, so it reveals nothing about internal names, but `err.without_url()` would keep hop URLs out of client text entirely.
- **Blocking resolver threads outlive the timeout.** `lookup_host` runs `getaddrinfo` on Tokio's blocking pool. The 5 s bound and the chain deadline drop the future, not the thread, so many requests naming hanging DNS names can hold blocking threads until the system resolver gives up. The issue ruled out changing `lookup_bounded` here.
- **A listed NAT64 prefix can sit in local space.** A listed prefix wins over the gate by design, so an operator who lists a link-local or unique-local prefix makes addresses in it reachable whenever they embed a public IPv4. Refusing `fe80::/10` and `fc00::/7` prefixes at parse time would close that misconfiguration.

## 5. Validation

**Failing first on 752db4f9.** With only the new `media_net_tests.rs` tests added, `server::media_net::tests::ipv6_` gave 1 passed, 1 failed: all 17 rows of the issue table passed as public. The three exception-edge rows were added later (70d3829e) and were not part of that run. The rendering and deadline tests do not compile on 752db4f9, because `media_url_rejected`, `check_hop_with` and `DeadlineExceeded` do not exist there; on that commit the deadline case returned `Refused(Unresolvable)`.

**Suites** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, under `gpu-lock`): `server::media` 121 passed, `server::routes::embeddings` 23, `server::routes::rerank` 20. The implementation review and the security pass each re-ran `server::media` (121 passed). `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, `make verify-env-var-docs` and `make verify-llama-compat` were clean. The 70d3829e commit (test rows and RFC citations only) reports the same 121.

**Real server** (`target/test-fast/mlxcel-server`, `colsmol-256m-merged`, `127.0.0.1:19651`, `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS` unset):

| Request to `/v1/embeddings` | Result |
|---|---|
| `image_url` `http://[fec0::1]/x.png` | 400, `image URL rejected: media URL host is not allowed or could not be resolved`; debug log `address=fec0::1` |
| `data:` URL of `tests/fixtures/test_image.png` | 200, one embedding of shape (79, 128) |

**Not verified.** No host in the test setup had a route to site-local, Teredo or SRv6 space, so the refusal is shown by the unit table and the `fec0::1` literal, not by a blocked live fetch. The deadline path is covered by a stalled future in a unit test, not by a real slow resolver behind a real redirect. The logged suites and the real-server run were on 5a4d137b, before the merge of origin/main (844241d0), which brought in #2711 (ROCm imagegen parity tests) and touches none of the files here; a pre-merge test run on the merge commit was in progress when this report was written.

## 6. Learning Points

- For an allow-list over an address space that keeps getting carved up, gate on the allocation that is known to be global (`2000::/3`) and enumerate only the exceptions inside it. An enumeration of what is not global goes stale the next time IANA carves a block out of reserved space.
- The order of rules in an address predicate is part of its contract. Every rule that decodes an embedded address, and every operator override, has to run before a gate that would otherwise refuse its whole range; tests that pin those orderings (listed NAT64 prefix outside `2000::/3`, IPv4-mapped loopback) are what keep a later refactor from reordering them.
- When a predicate mirrors a registry's exception rows, test both sides of each exception's mask. A one-bit mask error widens an exception silently unless an address just outside it is asserted refused.
- Keep "refused" and "timed out" as distinct error variants even when the client text has to stay generic. The distinction is what lets the operator's log say what happened, and passing the policy future into `check_hop_with` makes the timeout testable without a slow resolver.
