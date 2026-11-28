# PTX cache under a container volume root, GB10 (2026-11-28)

Issue: [#3503](https://github.com/lablup/mlxcel/issues/3503). Question: with the two new trust rules, is a CUDA kernel cache on a volume root the process cannot chmod actually hit on the second start, and what does that save.

## Setup

- Host: NVIDIA GB10 (spark-101), 20 cores, driver 580.178.04, CUDA 13.0.3, kernel 7.0.0-1019-nvidia. Not quiet: 1-minute load between 3.7 and 8.0 during the runs (other users' jobs), recorded per arm below.
- Binaries: `mlxcel-server` built with `cargo build --release --features cuda`, `MLX_CUDA_ARCHITECTURES=121`. Branch arms at `2a62154c`, the last commit that changes compiled code; control arms from `main` at `7ab71f92`. An earlier pass at `5a9c96f2`, before the review changes, gave the same picture (warm 1.99 s and 1.94 s against cold 18.77 s and 19.55 s) and is not tabulated.
- Model: `mlx-community/Qwen3-4B-4bit` (local `models/qwen3-4b-4bit`).
- Volume roots, created as root through a container so their owner and mode match what a nonroot pod gets: `vol-0777` is `root:root` mode `0777`; `vol-fsgroup` is `root:1000` mode `2775`, where 1000 is the test user's own group. The user is uid 1000, gid 1000; `/etc/nsswitch.conf` lists `files systemd` for `passwd` and `group`, `/etc/group` has no other member of gid 1000 and no other `/etc/passwd` account has it as primary group, so the group is provably private.
- Harness: `scripts/bench/ptx_cache_trust_ttft.py`, run under `gpu-lock run --tag issue-3503`. One run starts the server with `MLXCEL_CACHE_DIR` set to the volume root, waits for `/health`, sends one streaming chat request (`max_tokens` 16, temperature 0) and stops the server. A cold run deletes `<root>/cuda-ptx` first. Five runs per cell.

## Attribution

Timing is not the evidence that the cache was hit. Each run records two signals that move only one way: the `Compiling CUDA kernels` notice on stderr, and whether any file under `<root>/cuda-ptx` was created, replaced or touched (inode and mtime of every file before and after). The harness exits without a table when a cold run lacks either, when a warm run shows either, or when a refused arm does not print exactly one refusal line and leave the cache empty. All 35 timed runs and the two populating runs agreed with their arm.

## Results

Seconds, median (min, max), n = 5 per cell.

| Arm | Binary | Volume root | Environment | Cache | Start to first token | Start to ready | Request to first token | JIT notice | Runs that wrote cache files | Load (1 min) |
|---|---|---|---|---|---|---|---|---|---|---|
| A cold | branch | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | accepted | 17.98 (17.63, 19.27) | 5.67 (5.57, 5.77) | 12.31 (12.05, 13.55) | 5/5 | 5/5 | 4.84-5.31 |
| A warm | branch | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | accepted | 2.10 (2.04, 2.10) | 1.87 (1.82, 1.88) | 0.23 (0.22, 0.23) | 0/5 | 0/5 | 5.10-5.12 |
| B cold | branch | `root:1000 2775` | none (Rule 1) | accepted | 17.96 (17.89, 19.31) | 5.65 (5.60, 5.81) | 12.36 (12.27, 13.54) | 5/5 | 5/5 | 5.06-6.16 |
| B warm | branch | `root:1000 2775` | none (Rule 1) | accepted | 1.99 (1.94, 2.05) | 1.77 (1.72, 1.82) | 0.22 (0.22, 0.24) | 0/5 | 0/5 | 5.51-5.81 |
| C | branch | `root:root 0777` | no opt-in | refused | 18.13 (17.03, 18.65) | 5.51 (5.42, 5.66) | 12.47 (11.61, 13.20) | 5/5 | 0/5 | 4.66-6.52 |
| D | main | `root:root 0777` | `MLXCEL_CACHE_TRUSTED_ROOT=<root>` | refused | 17.35 (16.92, 22.06) | 5.50 (5.42, 5.73) | 11.73 (11.50, 16.33) | 5/5 | 0/5 | 4.14-8.00 |
| E | main | `root:1000 2775` | none | refused | 19.51 (16.92, 22.01) | 5.67 (5.41, 8.03) | 13.63 (11.50, 15.23) | 5/5 | 0/5 | 3.66-6.36 |

Refused arms printed `mlxcel: CUDA kernel disk cache disabled: <root> is group- or other-writable` once per run. In the accepted arms `cuda-ptx` and `cuda-ptx/<mlx-commit>` were created `0700`, owned by the user.

## The recommended layout for a root that may be shared

Arm F puts mlxcel one level below the trusted root, `MLXCEL_CACHE_DIR=<root>/mlxcel` with `MLXCEL_CACHE_TRUSTED_ROOT=<root>` on the `0777` root, so that the model store does not sit beside the cache directly in the root. Two runs per cell, same binary, harness and attribution rule: cold 18.16 (17.15, 19.16) s with the JIT notice and new cache files in 2/2, warm 1.99 (1.99, 1.99) s with neither in 0/2, load 1.9 to 3.4.

## Reading

- Both rules turn a refused cache into a hit. On `main` (D, E), and on the branch without the opt-in on a `0777` root (C), every start compiles and nothing is written: every start is a cold start. With the rule that applies, the second start compiles nothing and writes nothing.
- A warm start reaches the first token in about 2.0 s against about 18 s cold, for this model on this host. Kernel compilation is paid in two places: about 4 s before `/health` answers and about 12 s inside the first request.
- The absolute numbers come from a loaded host and should not be quoted as GB10 start-up times. The cold-to-warm ratio (about 9x) is far outside the spread within either cell, and the hit signals do not depend on load at all.
- The refused arms and the accepted cold arms are not distinguishable in time (medians 17.4 to 19.5 s against 18.0 s, with a 5 s spread inside arms D and E under load up to 8). Publishing the entries on a cold start was not shown to cost anything measurable.

## Not measured

- A real Kubernetes pod. The owner and mode that `emptyDir`, local-path and NFS volume roots get (expected `0777`) were not checked on a cluster (`stat -c '%a %u:%g' /cache` on kind). The directories here were shaped by hand to match.
- A minimal container image. `own_group_is_private()` fails closed without `/etc/nsswitch.conf` or a passwd entry for the uid, so Rule 1 is expected not to apply there and the opt-in to be required; that was not run.
- Kubernetes' recursive `fsGroup` ownership change. With the default `fsGroupChangePolicy: Always`, the kubelet is expected to add group write to everything on a volume that supports ownership changes at each mount, which would make the directories below a trusted root and every entry group-writable and so refused or missed after a pod restart. Not verified here; if it holds, the chart needs `fsGroupChangePolicy: OnRootMismatch` or no `fsGroup` on the cache volume.
- Any GPU other than GB10, and any model other than Qwen3-4B-4bit. The saving scales with how many kernels a model compiles.
- A quiet host.

## Correctness evidence

`tests/cuda_ptx_cache_trust.rs` (30 scenarios through a real kernel, test-fast profile, GB10, under the GPU lock): all pass, including a symlink planted at the cache's first component below the trusted root, a group-writable and a root-owned directory below it, a group-writable entry, the stale-temporary sweep through the pinned descriptor, and each unusable value of the variable. `scripts/mlxcelverse/cuda_runtime_dir_test.sh` runs the same rules against real directories without a GPU.
