# Raw data for DRY penalty step cost on GB10 (2026-10-11)

This directory preserves timing and validation evidence for issue #2259.

## Timing run

`dry-step-cost.log` is the byte-preserved output of the ignored `sampling::tests::dry_step_cost_report` test. The test compares the exact pre-#2259 reference retained under `cfg(test)` with the new production path on period-3 windows of 1,024, 4,096, and 16,384 tokens, using one row of four f32 logits and no breakers. Reference iterations are 3, 3, and 1; the new path runs 20 iterations at each size. The returned MLX arrays remain lazy and are not evaluated, so the results measure host matching and MLX graph construction rather than GPU end-to-end latency.

`2259-quiet-timing.sh` is the byte-preserved inner driver. `outer-invocation.txt` records the exact outer `gpu-lock` command and the SHA-256 of the batch driver that issued it. The inner driver records the clean measurement checkout, source commit, MLX pin and fetched checkouts, host and GPU, start/end UTC, the measured test artifact hash, and its own hash in `environment.txt`.

The reused host gate has SHA-256 `69e6d3f468c42ba7e478578e3bde9efb9c5a30165866d8e9da19440a6eb9413f`. Its default 12 samples at five-second intervals require 55 continuous seconds of quiet because the first sample is immediate. `gate_wait_s=0` means no busy interval reset the quiet counter; it does not mean the sustained quiet check took zero seconds. `hostgate.log` also records that no CI job was active at the end of the gate.

## Validation logs

The raw final logs correspond to source commit `ecb89a308aecc468dd1886d9c9f4982f556f3c93`:

- `core-build.log`: CUDA core build, completed in 26.74 seconds.
- `clippy.log`: CUDA library/test clippy with warnings denied for Rust, completed in 9.08 seconds.
- `fmt.log`: formatting check.
- `sampling-tests.log`: 210 passed, 1 ignored, 0 failed.
- `engine-tests.log`: 63 passed, 0 failed.

The native build logs contain existing compiler diagnostics from the pinned dependency build; the Rust clippy command still exited successfully with `-D warnings`.

## Replay

After squash merge, rebuild and run the ignored test from the merge commit on a quiet CUDA host with `gpu-lock run --tag issue-2259-quiet-timing -- cargo test --profile test-fast --features cuda -p mlxcel-core --lib sampling::tests::dry_step_cost_report -- --ignored --nocapture --test-threads=1`. That direct cargo form may compile while holding the lock if the cache is cold. For a controlled measurement, first build the test artifact outside the timing window, then adapt the preserved inner driver to invoke that exact artifact while retaining the same provenance capture and gate. Compare production and test code against the measured source and keep any post-measurement difference explicit.

Validate the preserved files from this directory with:

```bash
sha256sum --check SHA256SUMS
```
