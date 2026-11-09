# Per-layer host synchronisation cost of SSD expert streaming (issue #2882)

Host: Mac Studio, Apple M3 Ultra, 512 GB. **Another mlxcel implementation job (a sibling worktree build) was running during these measurements**, so treat the absolute numbers as a lower bound on what an idle host gives. Load average during the runs was about 9 to 13.

Model: `Qwen3.8-Flash-Next-OptiQ-2bit`, 48 layers, 41.60 GB of routed experts. Prompt: "Write a short story about a lighthouse." (61 prompt tokens), `-n 128 --temp 0`. Decode tok/s as the CLI reports it.

Method: one release binary with a temporary switch that called `eval` on the router's top-k indices after every layer's router, on resident weights, so the measurement is the synchronisation alone: no arena, no reads, a 100% hit rate by construction. Runs were interleaved (off, on, on, off, off, on, on, off), four per arm.

| Arm | tok/s (4 runs) | Median | ms per token |
|---|---|---|---|
| Resident | 42.06, 42.02, 41.98, 41.56 | 42.0 | 23.8 |
| Sync after every router | 25.13, 25.78, 25.75, 25.37 | 25.6 | 39.1 |

The per-layer synchronisation costs about 15.3 ms per decode token here, about 0.32 ms per layer, a 1.64x slowdown of an otherwise idle decode. Token ids were identical in all eight runs.

Decision: the MLX graph approach stays. The mode engages only when a checkpoint cannot be kept resident, where the alternative is not running at all. The cost is a fixed amount per token, so its share falls on slower GPUs whose token time is larger. A custom Metal path with a shared-event handshake (as in Strata) could reduce this; it is a possible follow-up, not a precondition. The figure above is the 100% hit-rate cost that the acceptance criteria ask to report separately; it does not include I/O wait.

Paired five-run medians for decode, 4K prefill, hit rate and I/O wait across arena sizes and with the contiguous file have not been recorded; they were not run on this host while a sibling job was active.
