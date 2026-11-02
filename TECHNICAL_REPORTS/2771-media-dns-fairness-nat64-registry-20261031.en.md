# PR #2771: Cap media DNS per domain, redact hop schemes, check the NAT64 registry

**Date**: 2026-10-31 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at a955fe97 (commits fdc89ca2 and 72d5fc54, plus a merge of main). The targeted `server::media` tests, clippy, fmt and the two doc verifiers passed on GB10 (spark-102) per the PR body, and a real `mlxcel-server` check covered the NAT64 startup refusal and the hop-scheme redaction. At the time of writing every hosted check except `license/cla` was pending and a pre-merge local gate was running in the worktree
**Languages**: Rust, Markdown
**Risk Level**: Low (one startup behavior change: `MLXCEL_MEDIA_NAT64_PREFIXES` now refuses prefixes inside four IANA special-purpose blocks; the worst case for one media DNS lookup grows from 5 s to 7 s)

## Executive Summary

PR #2771 (closes #2764) fixes the five residuals that #2741 recorded in the remote-media SSRF guard (`src/server/media_net*.rs`). The main one is fairness: #2741 bounded media DNS lookups with a 32-permit pool, but one client naming many subdomains of a domain with silent nameservers could hold all 32 permits and make every other client's hostname fetch fail as `Unresolvable`. Lookups now take one of 4 permits for their domain key (the last two labels of the host) before a global permit, so filling the pool needs at least 8 such domains. The other four changes give the permit wait its own 2 s bound so the resolver keeps its full 5 s, pin the pre-flight DNS answer for 30 s so each hop resolves once instead of twice, stop echoing a redirect target's scheme (chosen by the origin's `Location`) to the client, and refuse NAT64 prefixes inside `2001::/23`, `2001:db8::/32`, `2002::/16` and `3fff::/20`.

None of the five lets a client reach a private address on main. Item 1 is a denial of service against other clients' media fetches (MEDIUM priority in the issue); items 2 to 5 are LOW.

## 1. Problem Statement

Issue #2764 collected the residuals from the security review of #2741 and section 5 of `TECHNICAL_REPORTS/2741-media-fetch-errors-dns-nat64-20261030.en.md`. The failing-first runs in section 4.1 were taken against main 5c8ef80a.

### 1.1 One client could fill the DNS permit pool

`lookup_bounded_with` took one of `MAX_CONCURRENT_MEDIA_DNS_LOOKUPS = 32` process-wide permits and moved it into the `spawn_blocking` closure, so an abandoned lookup kept its permit until `getaddrinfo` returned. That is what bounds blocking threads, but it also means a client naming hosts whose nameserver never answers (random subdomains of one attacker domain cost nothing) holds every permit for the system resolver's timeout. Every other client's hostname fetch then waited 5 s and failed as `Unresolvable`. IP-literal URLs were unaffected.

### 1.2 The permit wait ate the resolver's time

One deadline, `DNS_LOOKUP_TIMEOUT` (5 s), covered both waiting for a permit and the resolver call. Under contention a healthy but slow resolver got whatever was left, so a lookup that would have succeeded failed.

### 1.3 Each hop resolved twice

The pre-flight check `ensure_media_url_allowed` resolved the host, and the connect-time `PolicyResolver` (`resolve_allowed_addresses`) resolved it again; `check_hop` repeats the pre-flight on every redirect hop. Each hostname hop therefore took two permits, so the effective concurrency under load was below 32, and the connect could use a different answer from the one the pre-flight approved (it was still filtered, so this was a consistency issue, not a bypass).

### 1.4 A hop's scheme reached the client

`check_hop` ran `ensure_media_url_allowed(next.as_str())` on the hop, and `MediaUrlError::Scheme { scheme }` renders `media URL scheme {scheme:?} is not supported`. After a redirect the scheme comes from the origin's `Location`, so an origin could put a chosen token into the 400 the server returns. `Malformed { url }` would echo a whole hop URL the same way, though a hop has already been parsed, so that path is theoretical. This is the same class of leak #2741 closed for transport errors and `TooManyRedirects`.

### 1.5 NAT64 prefixes were checked against ranges, not the registry

#2741 made `Nat64Prefix::parse` accept only prefixes inside `2000::/3` or `64:ff9b:1::/48`. Inside `2000::/3` that still admitted the documentation prefix `2001:db8::/32`, the IETF protocol-assignment block `2001::/23`, 6to4 `2002::/16` and the documentation prefix `3fff::/20`. The address policy refuses addresses in all four, but a listed prefix outranks the policy, so listing one made those addresses judged by their embedded IPv4 instead of refused.

## 2. Change Summary

| Item | Area | Change |
|---|---|---|
| 1 | `media_net_dns.rs` | New `DnsPool { global, domains: Mutex<DomainTable> }`. `lookup_bounded_with` computes `domain_key(host)` (last two labels, lowercased, trailing dot ignored; a host of two labels or fewer is its own key), takes a permit from that key's semaphore (`MAX_CONCURRENT_LOOKUPS_PER_DOMAIN = 4`), then a global permit, and moves both into the blocking closure as `let _permits = (domain_permit, global_permit);`. |
| 1 | `media_net_dns.rs` | `DnsPool::domain` prunes idle keys (strong count 1 and all permits available) when an insert would reach `prune_at`. `prune_at` starts at 1024 and becomes `max(1024, 2 * len)` after each pass. |
| 2 | `media_net_dns.rs` | `DnsLimits { permit_wait, lookup_timeout, per_domain }` with `DnsLimits::DEFAULT` (2 s, 5 s, 4). Both permits are acquired under one `permit_wait` deadline (`acquire_by`, which logs the saturated pool at debug); the resolver then runs under `tokio::time::timeout(lookup_timeout, ..)`. |
| 3 | `media_net_pin.rs` (new) | `PinStore`: lowercased host to `(Vec<SocketAddr>, expiry)`, TTL 30 s, at most 1024 hosts (expired evicted first, then the oldest), ports cleared to 0 so the connector substitutes the URL's port. |
| 3 | `media_net.rs` | `ensure_media_url_allowed` pins the answer after every address passes. `resolve_allowed_addresses` reads a live pin when private addresses are refused and still runs `filter_allowed_addresses` on it; on a miss it resolves as before. With `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` nothing is resolved in the pre-flight, pinned or read. |
| 4 | `media_net.rs`, `media_net_redirect.rs` | New variants `RedirectScheme { scheme }` ("media URL redirect target uses an unsupported scheme; only http and https are followed") and `RedirectMalformed { url }` ("media URL redirect target is not a valid absolute URL"), both with debug arms in `log_detail`. `MediaUrlError::for_redirect_target` maps `Scheme` and `Malformed`; `check_hop` applies it to the hop's pre-flight result. The client's own URL keeps the old messages. |
| 5 | `media_net_nat64.rs` | `SPECIAL_PURPOSE_BLOCKS` lists the four blocks with their RFC; `parse` refuses a prefix whose network lies in one, naming the block (`"2001:db8:64::/96" lies inside 2001:db8::/32 (documentation, RFC 3849), ...`). |
| Docs | `docs/environment-variables.md`, `docs/llama-server-compat.md` | The `MLXCEL_MEDIA_NAT64_PREFIXES` row and the address-policy section state the new refusal and the behavior change; the compat page gains a paragraph on the DNS bounds (32 global, 4 per domain, 2 s plus 5 s) and the multi-label public suffix limit, and notes the pin and the hop-scheme redaction. |
| Tests | 4 test files | 11 new tests (section 4.1). A `cfg(test)` `test_answers` hook in `system_resolve` serves canned answers per exact host and counts lookups. The 11 existing NAT64 tests moved from `media_net_tests.rs` (573 lines) to the new `media_net_nat64_tests.rs`; the RFC 6052 decode examples now build their `2001:db8::` prefixes with `parse_unchecked_for_test`. |

### Statistics

| Item | Value |
|---|---|
| Files changed | 11 (5 source including the new `media_net_pin.rs`, 4 test including the new `media_net_nat64_tests.rs`, 2 docs) |
| Lines added / deleted | +1087 / -263 against main (192 of the deletions and most of `media_net_nat64_tests.rs` are the moved NAT64 tests) |
| Tests added | 11 (`media_net_resolver_tests.rs` 9, `media_net_nat64_tests.rs` 1, `media_url_disclosure_tests.rs` 1) |
| New dependencies | None |
| Commits | fdc89ca2 (items 1 to 5), 72d5fc54 (amortized pruning and the redirect-variant unit test, from review), a955fe97 (merge of main) |

## 3. Technical Decisions

### 3.1 Key fairness by domain, not by client or public suffix

| Option | Pros | Cons |
|---|---|---|
| Per-client fairness | Targets the actual abuser | The fetch layer (`fetch_labeled_image`, used by chat, embeddings, rerank and image routes) has no client identity; threading one through is a larger change |
| Public-suffix-list key | Correct registrable domain (`example.co.uk`) | New dependency and list maintenance for a bound that is approximate anyway |
| **Chosen: last two labels of the host** | No dependency; random subdomains of one domain collapse into one key | Hosts under a multi-label public suffix share the suffix's key (`co.uk`) |

The bound changes the cost of the attack rather than removing it: holding all 32 global permits now needs 8 registered domains with silent nameservers instead of one. The known limit is written in the module doc of `media_net_dns.rs` and in `docs/llama-server-compat.md`.

The domain permit is acquired before the global one. A lookup that waits for a global permit therefore already counts against its domain, so one domain cannot queue more than 4 lookups in front of the global pool. Both permits move into the blocking closure, keeping the #2741 property that an abandoned lookup still counts until `getaddrinfo` returns.

### 3.2 Pruning the domain table without a quadratic path

The semaphores live in a `HashMap<String, Arc<Semaphore>>` under a std `Mutex`. An entry is idle when `Arc::strong_count == 1` and all permits are available; a pending `acquire_owned` holds a clone of the `Arc`, so a key with waiters is never dropped. The first version pruned whenever the table held 1024 keys. The security check in review found that a caller keeping more than 1024 keys busy (each held up to the 2 s permit wait while the global pool is saturated) made every new key run a full `retain` under the mutex on a Tokio worker: 6476 prune passes for 6000 inserts. 72d5fc54 makes the threshold `max(1024, 2 * surviving len)` after each pass, so scans happen once per doubling (at most 8 passes for the same 7500 busy keys) and idle keys are still pruned once a surge drains.

### 3.3 Separate deadlines for the permit wait and the resolver

A single deadline is the simplest bound but penalises healthy lookups exactly when the server is busy. Splitting it costs 2 s of worst-case latency (7 s instead of 5 s) and gives the resolver its full 5 s once it actually runs. Passing the three numbers through `DnsLimits`, and the pool as `&DnsPool`, lets the tests run the same code with 300 ms and 500 ms bounds and a private pool, so they never share the process-wide domain map.

### 3.4 Pin the pre-flight answer instead of caching DNS

The pin is not a general DNS cache. Only `ensure_media_url_allowed` writes it, only after every address passed, and the pre-flight never reads it, so every request still resolves its URL and each hop fresh. The connect-time `PolicyResolver` reads it and still filters it, so a pin cannot widen what is reachable. The result is one lookup per hop (one permit instead of two) and a connection that goes to the address set that was checked; on a rebinding name the connect now uses the approved answer instead of failing on a refused second answer. With `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` the pre-flight does not resolve, so there is nothing to pin and the path is unchanged.

TTL 30 s and 1024 hosts bound the store; pins are keyed by lowercased host with port 0, matching what the connector does with a fresh lookup.

### 3.5 Redact at the hop, keep the client's own messages

`for_redirect_target` is applied in `check_hop`, the one place hop refusals are produced, rather than changing `Scheme` and `Malformed` themselves. A client that submits `ftp://...` still learns which scheme it sent and why it was refused; only text the origin chose moves to the debug log. Both variants render through `media_url_rejected` as the same 400, and no code matches `MediaUrlError` exhaustively, so adding variants did not touch callers.

### 3.6 Registry blocks as a short table, not the full IANA registry

The four blocks are the ones the address policy already refuses inside `2000::/3`. Every allowed RFC 6052 length is /32 or longer and every listed block is /32 or shorter, so a prefix overlaps a block exactly when its network address lies inside it; one mask comparison per block is the whole test. Globally reachable carve-outs inside `2001::/23` (AMT, AS112-v6, ORCHIDv2) are not translator space either, so refusing the whole /23 for configuration costs nothing real.

## 4. Verification

### 4.1 Failing-first evidence

The new acceptance tests were run against unchanged main sources (5c8ef80a). The DNS tests ran in a main-compatible form (`lookup_bounded_with(.., Duration, Arc<Semaphore>)`, same scenario and numbers); the connect test needed the test-only `system_resolve` counter added.

| Test | Result on main |
|---|---|
| `one_domain_cannot_fill_the_dns_pool` (40 lookups of `a{i}.slow.test`, resolver sleeping 2 s, global 8, per domain 4) | `ok.other.test` failed `Unresolvable` after the 500 ms timeout; the pool was held by `slow.test` |
| `permit_wait_does_not_eat_resolver_time` (global 1, wait 300 ms, timeout 500 ms; first lookup holds 200 ms, second resolves in 400 ms) | second lookup failed `Unresolvable` |
| `connect_reuses_the_preflight_answer` (hostname origin redirecting once to a hostname hop) | lookups `(origin, hop)` were `(2, 2)`, expected `(1, 1)` |
| `redirect_to_a_non_http_scheme_omits_the_scheme` (`Location: secret-token-123://x/y`) | client text was `image URL rejected: media URL scheme "secret-token-123" is not supported; only http and https are fetched` |
| `special_purpose_nat64_prefixes_are_refused` | `2001:db8:64::/96` parsed |
| `pruning_is_amortized_while_more_than_the_idle_cap_of_keys_stay_busy` (1500 + 6000 busy keys) | 6476 prune passes against the first version of this PR; at most 8 required |

`dns_lookups_are_bounded_by_the_permit_pool` from #2741 passed on main and passes adapted to `DnsLimits`. The remaining new tests (`the_domain_key_is_the_last_two_labels`, `idle_domain_keys_are_pruned_and_busy_ones_kept`, `idle_keys_are_still_pruned_after_a_busy_surge_drains`, `a_redirect_hop_refusal_never_carries_the_origin_chosen_text`, `pins_expire_and_the_oldest_is_evicted_past_capacity`) cover the new helpers directly. The redirect-scheme test runs with the private-URL opt-in on because its origin is on loopback; the scheme check runs in `ensure_media_url_shape_with` before the opt-in is consulted, so the refusal does not depend on it.

### 4.2 Local tests

On GB10 with `--profile test-fast --features cuda`: `server::media` 138 passed with `--test-threads=1`; the resolver and disclosure tests passed 5 more times serially and 3 times with `--test-threads=8`. `cargo clippy --lib --tests -D warnings`, `cargo fmt --check`, `make verify-env-var-docs` and `make verify-llama-compat` were clean.

### 4.3 Real server

`mlxcel-server -m llama-nemotron-embed-vl-1b-v2` on 127.0.0.1:20600 under `gpu-lock`:

- `MLXCEL_MEDIA_NAT64_PREFIXES=2001:db8:64::/96` fails startup with `invalid MLXCEL_MEDIA_NAT64_PREFIXES entry: "2001:db8:64::/96" lies inside 2001:db8::/32 (documentation, RFC 3849), ...`; `64:ff9b:1::/96` starts.
- A `/v1/embeddings` image URL whose origin answers `302 Location: secret-token-123://x/y` returns HTTP 400 `input[0]: failed to read image: image URL rejected: media URL redirect target uses an unsupported scheme; only http and https are followed`.

### 4.4 Hosted CI

At the time of writing all hosted checks except `license/cla` were pending.

## 5. Remaining Gaps and Risks

- **Multi-label public suffixes share a key.** All hosts under `co.uk`, `com.au` and similar suffixes share 4 slots, so an attacker on such a suffix can make other hosts there wait. Documented; a public-suffix-list key is the fix if it ever matters.
- **The pool can still be filled.** Eight cheap domains with silent nameservers still hold all 32 permits for up to 7 s at a time. Per-client fairness needs a client identity in the fetch layer, and a negative cache for timed-out names was out of scope.
- **Latency.** The worst case for one lookup is now 7 s. Redirect hops remain inside the chain's shared 10 s deadline, so a slow hop lookup is cut there first.
- **Pin store eviction scans.** When the store holds 1024 live pins, each new host runs a `retain` plus a `min_by_key` over at most 1024 entries under the mutex. That is a constant bound, not the growth case 3.2 fixed, but it is per insert.
- **Pins can outlast the DNS TTL by up to 30 s.** Because the pre-flight always resolves afresh and overwrites the pin, this matters only between a request's pre-flight and its connect, or when concurrent requests for the same host interleave. Any pinned answer is re-filtered, so the effect is on which approved address is used, not on what is allowed.
- **`RedirectMalformed` is covered by a unit test only.** A hop reaching `check_hop` has already been parsed, so no end-to-end path produces it today.
- **Behavior change.** A deployment that listed a translator prefix inside `2001::/23`, `2001:db8::/32`, `2002::/16` or `3fff::/20` now fails startup and must renumber the translator into other global or `64:ff9b:1::/48` space. Documentation prefixes should never carry production traffic, so this is expected to affect test setups only.

## 6. Learning Points

### 6.1 A global bound needs a fairness key below it

A single semaphore bounds the resource but lets one tenant take all of it. Adding a smaller per-key semaphore, acquired first and held with the global permit, turns "one caller starves everyone" into "a caller needs N independent keys to starve everyone". The choice of key decides how cheap N is; here a registered domain, not a subdomain.

### 6.2 Amortize cleanup of a map that an attacker can keep busy

"Prune when the map exceeds K" is quadratic when the attacker can keep more than K entries live: every insert triggers a full scan that removes nothing. Moving the threshold to a multiple of the surviving size after each pass bounds the scans to the log of the key count, the same trick a growable vector uses.

```rust
if table.map.len() >= table.prune_at {
    table.map.retain(|_, s| Arc::strong_count(s) > 1 || s.available_permits() < per_domain);
    table.prune_at = MAX_IDLE_DOMAIN_KEYS.max(2 * table.map.len());
}
```

### 6.3 Reuse the checked answer instead of checking twice

When a check and the action it guards each resolve a name, they can disagree, and each resolution costs a bounded resource. Handing the checked answer to the action (still re-filtered, as defense in depth) removes both the cost and the disagreement without adding a cache that serves unchecked data.

### 6.4 Every field an upstream controls is a reflection channel

#2741 removed hop URLs from errors; this PR found the scheme was a separate field with the same origin. When an error is built from data a remote party chose, audit each field of the variant, not only the obvious URL.

## 7. References

- Issue #2764: Fair DNS budget, hop scheme redaction, NAT64 registry
- PR #2741 and issue #2728: the DNS permit pool, hop URL redaction and NAT64 range rule this PR extends; `TECHNICAL_REPORTS/2741-media-fetch-errors-dns-nat64-20261030.en.md` section 5 lists the residuals
- Issue #2644: `PolicyResolver`, the connect-time address check
- Issue #2693: redirect hop checks and NAT64 prefix decoding
- RFC 2928 (IETF protocol assignments), RFC 3056 (6to4), RFC 3849 and RFC 9637 (IPv6 documentation prefixes), RFC 6052 (IPv4-embedded IPv6 addresses), RFC 8215 (local-use translation prefix)
- `docs/environment-variables.md` (`MLXCEL_MEDIA_NAT64_PREFIXES`), `docs/llama-server-compat.md` (remote media URL address policy)
