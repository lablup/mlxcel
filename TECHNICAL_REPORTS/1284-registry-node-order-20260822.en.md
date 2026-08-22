# Technical Report: PR #1284 - A defined order for the distributed registry's node accessors

## Executive Summary

`RegistryInner.nodes` is a `HashMap`, and four of the registry's list accessors returned its `.values()` unordered while their consumers depended on position. The largest consequence was on the primary request path: `all_nodes` feeds `select_prefill_node` and `select_decode_node`, where `RoundRobin` indexes positionally, `LeastLoaded` uses `min_by_key` (first minimum wins) and `MemoryAware` uses `max_by_key` (last maximum wins). A cold or idle cluster is exactly the tie case, and the shipped defaults are `LeastLoaded` for prefill and `MemoryAware` for decode, so on an idle cluster the "least loaded" node was whichever one `HashMap` iteration placed first.

The fix sorts inside the accessors by `config.id`. This is the fourth and last instance of one root-cause class found in a single review sweep, after #1265 (test fixtures), #1267 (`lang_bias.rs`) and #1276 (RT-DETRv2 layout sniffing).

## 1. Problem Statement

Two sibling accessors in the same file, `nodes_at_stage` and `nodes_at_rank`, already ended with a `sort_by_key` because their callers depend on position. `all_nodes` and `nodes_with_role` had callers that depend on position just as much and had never been given the same treatment.

Request counts stayed balanced, because round-robin distributes evenly over whatever order it is handed. That is why the existing tests passed: `nodes_with_role_filter` asserts an index into a single-element list, which cannot distinguish a fixed order from an arbitrary one. What was not stable was the assignment, so a failover incident could not be reproduced from an identical cluster state, and on a heterogeneous cluster the quality of a selection varied for reasons no operator could see.

## 2. Technical Decisions

### 2.1 Sort in the accessors, with `sort_by` rather than `sort_by_key`

Sorting at the source means every consumer inherits determinism rather than each one remembering to sort. `sort_by(|a, b| a.config.id.cmp(&b.config.id))` was chosen over `sort_by_key(|n| n.config.id.clone())` because the key is a `String` that cannot be borrowed out of the element, so `sort_by_key` clones per key evaluation. Measured on a 12-element scrambled vector: 78 key evaluations, each a clone, against 39 comparisons and zero allocations.

Switching `nodes` to a `BTreeMap` was considered and rejected in the issue, because it would penalize the O(1) `get_node` path for a property only the list accessors need. Nothing found during implementation argued against that.

### 2.2 Sorting the candidates was not sufficient on its own

This corrects the issue's own analysis. `handle_node_failure` collects the affected request ids from `self.requests`, which is also a `HashMap`, and the re-routing loop hands candidates out round-robin **in that sequence**. With sorted candidates but an unsorted request list, the request-to-node pairing still moves between runs, which is precisely the property an operator is trying to reproduce. `affected.sort()` was added alongside the accessor fix.

### 2.3 Two more accessors were in scope than the issue named

`peer_addresses` and `topology_summary` also return or render `.values()` unordered and were not named in the issue. `topology_summary` is operator-facing, printed during discovery and cluster init, so re-running it against an unchanged cluster reshuffled the node list. Both are now sorted, giving four ordered accessors in total.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/distributed/registry.rs` | `all_nodes`, `nodes_with_role`, `peer_addresses`, `topology_summary` sort by `config.id` |
| `src/distributed/disaggregated/request_router.rs` | `handle_node_failure` sorts the affected request ids before the round-robin walk |
| `src/distributed/registry_tests.rs`, `.../request_router_tests.rs` | 9 new tests |

## 4. Review Findings

Two claims in the issue body were wrong and are corrected here rather than carried forward.

`routing.rs` was listed among the order-sensitive consumers. It is not: `RoundRobinRouter` already performs `online.sort_by(|a, b| a.node_id.cmp(&b.node_id))` before indexing, with a comment saying so. Someone had already made that path deterministic.

`find_pp_tp_node` is safe for a reason worth recording rather than assuming: `ClusterConfig::validate` rejects duplicate `(stage, rank)` pairs, so its `.values().find()` has a unique match regardless of order.

The cleared list in the issue was otherwise correct, and one consumer it omitted, `router_front.rs:1218`, takes only `.len()` and is insensitive.

The regression tests were demonstrated failing against the unfixed accessors before being finalized, since this bug class readily produces tests that pass both before and after. All nine failed, most on iteration 0 or 1, for example:

```
---- all_nodes_ordered_by_id_across_registries ----
  left: ["yankee-prefill", "alpha-decode", "mike-prefill", "bravo-decode", "zulu-hybrid"]
 right: ["alpha-decode", "bravo-decode", "mike-prefill", "yankee-prefill", "zulu-hybrid"]
---- round_robin_selection_sequence_is_stable_across_routers ----
  left: ["prefill-0", "prefill-1", "prefill-0", "prefill-1"]
 right: ["prefill-1", "prefill-0", "prefill-1", "prefill-0"]
```

The registry tests build 64 fresh registries each and the router tests 32 fresh routers each, rather than reusing one, because `RandomState` randomizes per `HashMap` instance and not merely per process. That property was measured while filing #1267: ten maps from the same five keys gave nine distinct orders inside one process.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64).

- `make verify-test-cuda`: recorded in the PR thread.
- Narrow filters, all exit 0: `distributed::registry` 18 passed, `distributed::disaggregated::request_router` 29 passed, `distributed::scheduler` 16, `distributed::routing` 19, `distributed::heartbeat` 9, `distributed::discovery` 3, `distributed::cluster_init` 14, `server::router_front` 20.
- `cargo fmt --all -- --check`: exit 0.
- `cargo clippy --lib --tests --features cuda -- -D warnings -A clippy::err_expect`: exit 0. Without the allow it exits 101 on a pre-existing `clippy::err_expect` at `src/multimodal/host_preprocessor_tests.rs:416` that predates this branch and is tracked as #1283; none of this PR's files appear in that output.

Not verified by observation: the issue's manual multi-node check, which needs three or more prefill nodes and repeated router restarts. This is a single-node machine. The accessor ordering and the determinism reaching `select_prefill_node` and `select_decode_node` under the tie cases are covered by test; the claim that a defined accessor order suffices for reproducible selection in production rests on each strategy arm being a pure function of the candidate sequence and the atomic counters, which is true by inspection and confirmed for ties by test, but live metrics break most ties before order matters.

## 6. Related Work

- #1277: the issue this closes.
- #1265 and PR #1266, #1267 and PR #1269, #1276 and PR #1281: the sibling instances of the same class.
- #1283: the pre-existing clippy failure encountered while validating this branch.

A sweep of the remaining `.values()` and `.iter()` collection sites under `src/distributed/` found three further instances, deliberately left for their own issue because they change eviction behavior rather than routing: `tensor_parallel/cache_manager.rs:711`, where `check_pressure` breaks early once the memory target is met so hash order decides which tied sequences are evicted; `pipeline/cache_manager.rs:697`, which publishes `PreemptionSignal.sequence_ids` documented as eviction priority order; and `request_tracker.rs:333`, where terminal requests sorted by `created_at` take a prefix and same-`Instant` ties fall back to hash order.
