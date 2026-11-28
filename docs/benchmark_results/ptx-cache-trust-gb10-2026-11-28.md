# PTX cache under a container volume root, GB10 (2026-11-28)

Issue: [#3503](https://github.com/lablup/mlxcel/issues/3503). Question: with the two new trust rules, is a CUDA kernel cache on a volume root the process cannot chmod actually hit on the second start, and what does that save.

## Setup

- Host: NVIDIA GB10 (spark-101), 20 cores, driver 580.178.04, CUDA 13.0.3, kernel 7.0.0-1019-nvidia. Not quiet: 1-minute load between 2.0 and 6.0 during the runs (other users' jobs), recorded per arm below.
- Binaries: `mlxcel-server` built with `cargo build --release --features cuda`, `MLX_CUDA_ARCHITECTURES=121`. Branch arm at `5a9c96f2`; control arm from `main` at `7ab71f92`. Review changes after `5a9c96f2` (the device subdirectory opened from a kept descriptor, the sweep through the descriptor, no Rust-side directory creation under the opt-in) were re-checked with the child tests and one cold and warm pass, not with the full table.
- Model: `mlx-community/Qwen3-4B-4bit` (local `models/qwen3-4b-4bit`).
- Volume roots, created as root through a container so their owner and mode match what a nonroot pod gets: `vol-0777` is `root:root` mode `0777`; `vol-fsgroup` is `root:1000` mode `2775`, where 1000 is the test user's own group. The user is uid 1000, gid 1000; `/etc/nsswitch.conf` lists `files systemd` for `passwd` and `group`, `/etc/group` has no other member of gid 1000 and no other `/etc/passwd` account has it as primary group, so the group is provably private.
- Harness: `scripts/bench/ptx_cache_trust_ttft.py`, run under `gpu-lock run --tag issue-3503`. One run starts the server with `MLXCEL_CACHE_DIR` set to the volume root, waits for `/health`, sends one streaming chat request (`max_tokens` 16, temperature 0) and stops the server. A cold run deletes `<root>/cuda-ptx` first. Five runs per cell.

## Attribution

Timing is not the evidence that the cache was hit. Each run records two signals that move only one way: the `Compiling CUDA kernels` notice on stderr, and whether any file under `<root>/cuda-ptx` was created, replaced or touched (inode and mtime of every file before and after). The harness exits without a table when a cold run lacks either, when a warm run shows either, or when a refused arm does not print exactly one refusal line and leave the cache empty. All 30 timed runs and the two populating runs agreed with their arm.

## Results

Seconds, median (min, max), n = 5 per cell.

| Arm | Binary | Volume root | Environment | Cache | Start to first token | Start to ready | Request to first token | JIT notice | Runs that wrote cache files | Load (1 min) |
|---|---|---|---|---|---|---|---|---|---|---|
| A cold | branch | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | accepted | 18.77 (18.71, 19.12) | 5.72 (5.66, 7.29) | 13.09 (11.83, 13.13) | 5/5 | 5/5 | 3.86-5.90 |
| A warm | branch | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | accepted | 1.99 (1.93, 2.04) | 1.77 (1.72, 1.82) | 0.22 (0.21, 0.22) | 0/5 | 0/5 | 4.52-4.82 |
| B cold | branch | `root:1000 2775` | none (Rule 1) | accepted | 19.55 (17.91, 19.86) | 5.81 (5.57, 5.88) | 13.73 (12.34, 13.98) | 5/5 | 5/5 | 5.08-5.94 |
| B warm | branch | `root:1000 2775` | none (Rule 1) | accepted | 1.94 (1.93, 2.05) | 1.72 (1.72, 1.82) | 0.22 (0.21, 0.23) | 0/5 | 0/5 | 5.87-6.03 |
| C | branch | `root:root 0777` | no opt-in | refused | 18.53 (17.25, 19.83) | 5.72 (5.42, 5.85) | 12.68 (11.83, 14.00) | 5/5 | 0/5 | 3.55-5.32 |
| D | main | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | refused | 18.09 (17.59, 19.88) | 5.66 (5.55, 5.80) | 12.43 (11.90, 14.08) | 5/5 | 0/5 | 3.23-4.44 |
| E | main | `root:1000 2775` | none | refused | 17.06 (17.01, 19.14) | 5.42 (5.38, 6.47) | 11.66 (11.63, 12.68) | 5/5 | 0/5 | 1.98-3.87 |

Refused arms printed `mlxcel: CUDA kernel disk cache disabled: <root> is group- or other-writable` once per run. In the accepted arms `cuda-ptx` and `cuda-ptx/<mlx-commit>` were created `0700`, owned by the user.

## Reading

- Both rules turn a refused cache into a hit. On `main` (D, E), and on the branch without the opt-in on a `0777` root (C), every start compiles and nothing is written: every start is a cold start. With the rule that applies, the second start compiles nothing and writes nothing.
- A warm start reaches the first token in about 2.0 s against about 19 s cold, for this model on this host. Kernel compilation is paid in two places: about 4 s before `/health` answers and about 13 s inside the first request.
- The absolute numbers come from a loaded host and should not be quoted as GB10 start-up times. The cold-to-warm ratio (about 9x to 10x) is far outside the spread within either cell, and the hit signals do not depend on load at all.
- The refused arms are a little faster than the accepted cold arms (17 to 18.5 s against 18.8 to 19.6 s). They ran later, under lower load, and do not publish entries; the two effects were not separated.

## Not measured

- A real Kubernetes pod. The owner and mode that `emptyDir`, local-path and NFS volume roots get (expected `0777`) were not checked on a cluster (`stat -c '%a %u:%g' /cache` on kind). The directories here were shaped by hand to match.
- A minimal container image. `own_group_is_private()` fails closed without `/etc/nsswitch.conf` or a passwd entry for the uid, so Rule 1 is expected not to apply there and the opt-in to be required; that was not run.
- Kubernetes' recursive `fsGroup` ownership change. With the default `fsGroupChangePolicy: Always`, the kubelet is expected to add group write to everything on a volume that supports ownership changes at each mount, which would make the directories below a trusted root and every entry group-writable and so refused or missed after a pod restart. Not verified here; if it holds, the chart needs `fsGroupChangePolicy: OnRootMismatch` or no `fsGroup` on the cache volume.
- Any GPU other than GB10, and any model other than Qwen3-4B-4bit. The saving scales with how many kernels a model compiles.
- A quiet host.

## Correctness evidence

`tests/cuda_ptx_cache_trust.rs` (29 scenarios through a real kernel, test-fast profile, GB10, under the GPU lock): all pass, including a symlink planted at the cache's first component below the trusted root, a group-writable and a root-owned directory below it, a group-writable entry, and each unusable value of the variable. `scripts/mlxcelverse/cuda_runtime_dir_test.sh` runs the same rules against real directories without a GPU.
