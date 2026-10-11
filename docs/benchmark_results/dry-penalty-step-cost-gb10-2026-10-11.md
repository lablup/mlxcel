# DRY penalty step cost on GB10 (2026-10-11)

Issue #2259 replaces the DRY sampler's repeated backward suffix scans with a reverse-Z pass, preprocesses request-supplied breakers with a reversed-token Aho-Corasick matcher, and scatters only touched token IDs. This record measures host matching and lazy MLX graph construction for the exact pre-change reference retained in the test module and the new production path on a period-3 stream. It does not measure GPU end-to-end latency because the returned arrays are not evaluated.

## Environment and provenance

- Host: NVIDIA GB10 on `spark-102`
- Backend: CUDA, pinned MLX `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`
- Measured source commit: `ecb89a308aecc468dd1886d9c9f4982f556f3c93`
- Measurement base commit: `73700aafa6a792f07830578c0c26137cea3a038b`
- Rebased publication checkout: `327b6bf13a31d0f98e4cce717443d03b4be2cb73` on `3ae626c2a4d67e7628a4050f01cc62b7419296fb`
- Source equivalence: both checkouts use `sampling.rs` blob `59fb6b02c6b54913aa549085fafc090e0b516d55`; their scoped diff across `src/lib/mlxcel-core`, `src/lib/mlx-cpp`, and `Cargo.lock` is empty
- Profile: `test-fast`
- Measured test artifact: `target/test-fast/deps/mlxcel_core-5feb86ac05c84f53`, SHA-256 `128d6b6c4e06c8a0a2c473d268c318110c643a583f8e233e1b3e35adf7aad9c8`
- Inner driver SHA-256: `06b48baeb28c191f550564075a65e7e1f6ee6c9dc594437a63a467319c1ab988`
- Outer driver SHA-256: `af9da7d10881230ecf0aa4cc5c7127921bb238a2a1e1cab7b832f85c773baed0`
- Host gate: `hostgate.py` SHA-256 `69e6d3f468c42ba7e478578e3bde9efb9c5a30165866d8e9da19440a6eb9413f`; 12 quiet samples at five-second intervals enforce 55 continuous seconds
- Host-gate result: no busy-process reset (`gate_wait_s=0`) and no active CI job after the sustained quiet interval
- Benchmark start/end UTC: `2026-10-11T10:14:59+00:00` / `2026-10-11T10:15:55+00:00`

The outer driver ran `gpu-lock run --tag issue-2259-quiet-timing -- bash /tmp/mlxcel-auto-plan/2259-quiet-timing.sh`. The raw package records that invocation, checkout state, test artifact provenance, host details, stdout/stderr, validation logs, and checksums.

The benchmark preceded the PR branch rebase. `implementation-73700aaf-to-ecb89a3.patch` preserves the original two implementation commits from the public measurement base, while `rebase-equivalence.txt` records the measured and publication tree and source identities plus the empty scoped diff. The original `environment.txt` remains byte-for-byte unchanged.

## Method

`sampling::tests::dry_step_cost_report` constructs one row of four f32 logits and period-3 token histories of 1,024, 4,096, and 16,384 tokens, with no configured breakers. For each window it measures the exact pre-#2259 reference and the new production path. Reference repetitions are 3, 3, and 1; production repetitions are 20 for each window. Values below are elapsed host microseconds per call as printed by the test. Since MLX is lazy and the arrays are only passed to `black_box`, the measurement includes host matching and graph construction but excludes device evaluation.

## Results

| Window | Reference µs/step | New µs/step | Speedup |
|---:|---:|---:|---:|
| 1,024 | 619.751 | 23.607 | 26.25× |
| 4,096 | 4889.370 | 29.997 | 163.00× |
| 16,384 | 77199.293 | 111.433 | 692.79× |

From 1,024 to 16,384 tokens, the reference cost grew 124.57× while the new path grew 4.72×. At 16,384 tokens the measured per-step speedup was 692.79×.

## Correctness and work bounds

The performance result is paired with exact logit parity rather than inferred from generated text. The CUDA sampling selection passed 210 tests with one ignored manual timing test, including 512 deterministic DRY reference cases and f32/f16/bf16 frequency/presence parity. The engine selection passed 63 tests. Instrumented tests bound the period-3 window pass below `8N`, the exact-breaker build and scan additively, and a shared-trunk failure-chain family below `8P` build and `3N` scan.

## Raw evidence

The package at `data/dry-penalty-step-cost-gb10-2026-10-11/` contains:

- `dry-step-cost.log`: byte-preserved benchmark command output.
- `2259-quiet-timing.sh`: byte-preserved inner timing driver.
- `outer-invocation.txt`: exact `gpu-lock` invocation and outer driver hash.
- `hostgate.log`: byte-preserved host-gate result.
- `environment.txt`: byte-preserved host, GPU, checkout, MLX, timing, and artifact output.
- `provenance.txt`: derived source, artifact, driver, host-gate, and raw-log hash mapping.
- `implementation-73700aaf-to-ecb89a3.patch`: exact two-commit format patch from the public measurement base to the measured source.
- `rebase-equivalence.txt`: measured-source, publication-checkout, source-blob, and empty scoped-diff proof.
- `core-build.log`, `clippy.log`, `fmt.log`, `sampling-tests.log`, `engine-tests.log`: byte-preserved final validation logs.
- `README.md`: artifact interpretation and replay instructions.
- `SHA256SUMS`: checksums for the package.

For an exact replay of the measured implementation, start at public commit `73700aafa6a792f07830578c0c26137cea3a038b`, apply the archived two-commit format patch with `git am`, and verify tree `58b9a04161eaf88048699d0930a69dd59ccdd7b3`. After squash merge, a fresh build from the merge commit is also valid when the scoped source-equivalence check remains empty; the raw measurement still belongs to the original measured commit and artifact.
