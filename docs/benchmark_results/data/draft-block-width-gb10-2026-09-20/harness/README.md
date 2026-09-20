# #1797 draft block width harness

Reproduces `docs/benchmark_results/draft-block-width-default-gb10-2026-09-20.md`.

```
docs/benchmark_results/data/draft-block-width-gb10-2026-09-20/harness/run_session.sh \
  /path/to/mlxcel-server
```

## Files

| file | what it is |
|---|---|
| `run_session.sh` | the whole session: identity, both families, both summaries |
| `identity.sh` | environment identity for one session |
| `sweep_server_widths.py` | served sweep, one `mlxcel-server` per arm |
| `summarize.py` | JSONL to the record's table, including the drift check |
| `prompt_retry.txt` | the fixed prompt every arm sends |

## Four things that are not stylistic

**The binary is renamed before it runs.** `run_session.sh` copies it to `mlxcel1797-server`. The imported host gate blocks on a foreign inference process by `/proc/<pid>/comm`, matching `mlxcel` and `mlxcel-server`, which is what keeps a peer session's model off the GPU during a timed run. A server started under the stock name matches that list and gates itself forever.

**`MLX_CUDA_ARCHITECTURES=121` is pinned for the build and for every server.** `build.rs` auto-detects `121a` on this host. The shipped release uses plain `121` (`.github/workflows/release.yml`), and so does every earlier GB10 record here. A block width is chosen by which CUDA kernel a verify block dispatches to, so the architecture the kernels were generated for is part of the measured configuration.

**The host gate is imported from `../../sdpa-plan-bucket-gb10-2026-09-12/harness/hostgate.py`, not copied.** It encodes three predicates that were each paid for with a lost measurement: contention means processes that can consume CPU now (match `/proc/<pid>/comm` exactly, drop state `T`, exclude the CI runner's permanent `RunnerService.js` and `Runner.Listener` daemons, never a `%cpu` threshold because `ps` reports a lifetime average); a foreign model process is a reason to wait; and the cumulative `NV_ERR_NO_MEMORY` count is the freeze precursor on this host and does not decay. A second copy would drift from the original.

**Classic runs first and last.** A served width cannot be interleaved the way `scripts/bench_block_width.sh` interleaves the offline path, because `draft_block_size` is worker-owned and changing it means restarting the server. The two classic arms bracket the widths instead: ranges that overlap say the session did not drift under them, ranges that separate say every arm in between is suspect. `summarize.py` prints that verdict before the table.

## Driver budget

`sweep_server_widths.py` records the `NV_ERR_NO_MEMORY` delta per arm and refuses to start an arm whose projected cumulative count would pass `--nvrm-budget` (400 by default), projecting from the worst arm measured so far. The window and cumulative ceilings themselves live in `hostgate.py` (`NVRM_WINDOW_MAX`, `NVRM_TOTAL_MAX`). Runs completing is not evidence the errors are benign; the two hard freezes this host took in July 2026 had exactly that shape.
