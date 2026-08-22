# Technical Report: PR #1302 - Capacity buckets for the OpenXLA engine

## Executive Summary

The OpenXLA context capacity is a compiled graph shape and also the sequence length every decode step attends over, so it trades image capability against text throughput and no single value wins. The engine now holds one shape per capacity and routes each request to the smallest that admits it. On the pinned Molmo2 4B checkpoint, text-only decode returns to the small-shape speed while images continue to be served from the same process.

Two results are worth more than the feature itself. Weights, not shapes, dominate the memory cost, which is what makes buckets affordable at all. And the hardest of the issue's four design questions, migrating a sequence that outgrows its bucket, dissolved rather than being answered, because routing on the whole generation budget makes outgrowing impossible.

## 1. Problem Statement

Measured before designing anything, on the pinned checkpoint at `/home/inureyes/models/mlx/molmo2-4b`, 64-token greedy generations:

| capacity | decode | relative |
| --- | --- | --- |
| 256 | 3.18 tok/s | baseline |
| 1024 | 2.17 tok/s | 1.47x slower |
| 2048 | 1.41 tok/s | 2.26x slower |

One image on that checkpoint expands to between 424 tokens (square) and 1834 (tall). A capacity that admits the worst case therefore made every text-only request on the same process more than twice as slow, and a capacity tuned for text could serve no images at all.

## 2. Technical Decisions

### 2.1 Share weights, duplicate only KV

The first question was whether a second engine could exist without a second copy of the weights, because a 4B model widened to f32 is roughly 16 GB and duplicating it would cost more than every bucket's KV combined.

Inspection settled it: the C shim passes weights as call arguments (`iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i])`), not as module constants, so two sessions on the same device can take the same buffer views. Buckets therefore share the upload and duplicate only their cache:

| capacity | KV at B_max 4, f32 |
| --- | --- |
| 256 | 0.30 GB |
| 2048 | 2.42 GB |

Adding a small text bucket beside an image bucket costs 0.30 GB, against 16 GB for a second weight upload. Without that property this feature would not be worth building.

### 2.2 Refcount, not borrowing

Sharing could have been expressed with a "borrowed" flag and a conditional free. It is expressed by retaining instead: a bucket retains the instance, the device and every weight buffer view, so the existing `xla_llama_free` releases exactly what the bucket retained and needs no special case.

The reason is not tidiness. A borrowed flag makes correctness depend on the base outliving its buckets, which nothing in the type system enforces and which a future refactor would silently violate. Refcounting removes the ordering requirement entirely.

### 2.3 The migration question dissolved

The issue asked what happens when a sequence admitted to a small bucket generates past it, and whether KV can be copied between graph shapes without a re-prefill.

It cannot arise. A request is routed on its prompt **plus its whole generation budget**, which is the invariant `validate_request_capacity` already enforced for a single shape, so a request that fits at admission cannot exceed its bucket later. The alternative, admitting on prompt length alone and migrating on growth, would have required copying KV between differently shaped caches; it was rejected because conservative admission costs nothing the single-shape path did not already cost.

This is recorded because the cheap answer is easy to lose: a later change that admits on prompt length "to fill slots better" would reintroduce a problem this design does not have, and the unit tests pin the invariant precisely so that change fails loudly.

### 2.4 Batching does not span buckets

Each bucket has its own graph and its own KV cache, so rows in different buckets cannot be advanced by one call. `pump` drives every non-empty bucket in turn, and slots are per bucket rather than pooled. That is a real cost and it is stated rather than hidden: the alternative, promoting every request to the largest shape so all rows share a call, is exactly the behavior this change exists to remove.

## 3. Change Summary

| File | Change |
| --- | --- |
| `csrc/xla_iree.c` | `xla_llama_create_bucket` sharing weights and device by refcount; bundle append and entry probing extracted into `xla_append_bundle` so a bucket session cannot be validated differently from its base |
| `src/iree.rs` | `IreeRaggedLlama::load_bucket`, and an optional base threaded through `load_inner` so config, emit and compile are reused rather than reimplemented |
| `src/batch.rs` | `XlaBucketSet` with routing, cancel, pump and prepared submission; `route_capacity` as a pure rule; scheduler id striding |
| `src/context.rs` | `context_capacity_buckets_from_env`, `derive_context_buckets`, `MLXCEL_XLA_CONTEXT_BUCKETS` |
| `src/server/batch/xla_worker.rs` | `XlaServingEngine` for the bucket set; `XlaServeWorker::new` made generic, which it already was in its struct but not its constructor |
| `src/server/model_worker.rs` | Resolve the capacity set, guard on the largest, report dropped buckets |

## 4. Review Findings

Two defects were found while building, both of the kind a compiler does not catch and both failing quietly rather than loudly.

**Request ids collided across buckets.** Every scheduler issued ids from zero, so two buckets would hand the same id to different requests. The server matches cancellations and completions by id, so the collision would have acted on the wrong request rather than erroring. Ids now stride by the bucket count, which makes them unique and makes an id name its own bucket, so cancellation needs no side table that could drift out of step.

**The bucket context copied two flags it should have derived.** `has_deepstack` and `has_prefill_diagnostics` decide which modules a session appends and which entry points it probes. Copying them from the base meant a base carrying a diagnostics bundle would make a bucket append a module the caller never passed. They are derived from the arguments now, exactly as the owning path does, and a deepstack mismatch against the base is rejected explicitly, since the shared weights are read under the base's contract.

## 5. Validation

Unit tests, 29 across two modules: smallest-admitting-bucket selection, routing on the whole budget rather than the prompt, rejection above the largest bucket, single-bucket equivalence with the pre-bucket path, id non-collision, the derived set, and an explicit list.

The routing rule is a pure function over capacities rather than a method needing a device, for the same reason `Scheduler` was split out: this crate's tests cannot link the IREE runtime.

Live on GB10, buckets derived from the checkpoint as `[256, 1834]` with both shapes compiled and the server up:

| configuration | text decode, 64 tokens, 3 repeats | serves images |
| --- | --- | --- |
| single 256 | 3.17, 3.20 tok/s | no |
| single 2048 | 1.40, 1.41 tok/s | yes |
| buckets [256, 1834] | 3.190, 3.155, 3.198 tok/s | yes |

Three repeats rather than one because single-run decode on this host is bimodal by roughly plus or minus 25 percent even at pinned clocks, so one measurement cannot separate the effect from the noise floor. The image request in the same process returned a correct, content-dependent answer.

## 6. Related Work

Issue #1271 is closed by this PR. The bucket set consumes `xla_image_context_floor` from #1272 rather than recomputing the image expansion, and the startup guard from #916 now asks whether any shape admits an image rather than whether the single shape does.

Two things this does not do. Buckets are not chosen by observed traffic, only by the checkpoint or an explicit list, so a deployment whose real prompts cluster somewhere else pays for shapes it does not use. And slots are per bucket, so a set with many buckets spreads its concurrency across them rather than pooling it; that is acceptable for the two-bucket case this derives, and would want revisiting before anyone configures a long list.
