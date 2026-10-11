# Model-owned decode storage on GB10 (2026-10-11)

Issue #2230 required a paired/null serving sweep before deciding whether Gemma 3 and Llama 4 should continue to advertise paged decode storage. Both flags remain `true`. Across the six prompt/concurrency cells for each family, neither dense nor paged storage cleared the repeated-paged null spread under the preregistered rule.

This decision concerns storage allocation and the per-step model-owned KV mirror. These model families continue to compute attention from their dense rows; selecting paged storage does not imply that their attention reads the paged pool.

## Environment and provenance

- Host: `spark-102`, NVIDIA GB10, driver `580.178.04`, PCI `0000000F:01:00.0`
- Run interval: `2026-10-11T04:57:35Z` through `2026-10-11T07:12:28Z`
- Clean measurement checkout: `35da2dbe32c731ff6645b2b32a2117af2965f332`
- Release binary production source: `2cb92404b153897440c4906a5548dcfb4889a9ad` (post-run derived verification described below)
- Release binary SHA-256: `0cc49217021dab12b5add25dc2bb8de7e90c405c0b406607b244a8060d3ed800`
- Release binary size and mtime: `108106312` bytes, `2026-10-11T03:25:49Z`
- Configured and built MLX commit: `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`
- Models: `gemma-3-1b-it-4bit` and `llama-4-scout-17b-16e-instruct-4bit`
- Outer lock: `gpu-lock run --tag bench-2230-storage -- .../run.sh`

The committed [raw evidence](data/model-owned-storage-gb10-2026-10-11/README.md) includes the exact contemporaneous environment, measured script hashes, schedule, host-gate snapshots, server and client output, analyzer output, completion marker, checksums, and a separate derived binary-source proof.

`environment.txt` uses `commit` for the clean checkout that ran the sweep. The release binary does not embed a source revision, so its association with `2cb92404` was reconstructed after the run from the preserved release-build session context, the artifact hash and mtime, and Git history. The committed release log proves the successful 19m43s build but does not print HEAD. Commit `2cb92404` is an ancestor of `35da2dbe`, and the complete intervening diff contains only this run's `analyze.py`, `gate.py`, `run.sh`, and `test_protocol.py`. No Rust, C++, model, or build-definition path changed. The original environment and measured script bytes and hashes remain unchanged. [The companion provenance record](data/model-owned-storage-gb10-2026-10-11/binary-source-provenance.json) records the commands, exact paths, release-log hash, and limitation.

## Method

Each arm started a fresh `mlxcel serve --parallel 8` process on port 18080 with an explicit `--decode-storage-backend paged|dense`. The client requested 128 generated tokens for concurrency 1, 4, and 8, first with a 512-token prompt and then with a 4096-token prompt. Five rounds were run for each model and prompt length.

Every round contained `paged-a`, `dense`, and `paged-b`. Their order rotated across rounds to distribute position effects. This produced 60 server arms and 180 successful client cells:

```text
2 models x 2 prompt lengths x 5 rounds x 3 arms = 60 arms
60 arms x 3 concurrency levels = 180 cells
```

The driver ran inside one GPU lock. Before the run and before every arm, its fail-closed gate required port 18080 to be free, a sustained quiet-host interval, no active CI/build/foreign-model processes, acceptable memory pressure, and clean NVIDIA driver state. It repeated the checks after the quiet interval and recorded only accepted snapshots. The analyzer additionally required the completion marker, a clean source commit and artifact provenance, matching configured and built MLX commits, exact script hashes, the exact ordered 60-arm schedule, 61 ordered quiet-gate records, 60 nonempty backend-marked server logs, 60 client logs, and the exact 180 successful measurement keys.

`paged-a` is the paired baseline. For every round, the dense paired delta is `dense / paged-a - 1`, while the null delta is `paged-b / paged-a - 1`. A cell is a clear dense win only when the median paired delta is greater than `max(0, null_high)`. A cell is a clear paged win only when the median paired delta is less than `min(0, null_low)`. Per family, dense storage would replace paged storage only if at least one cell were a clear dense win and no cell were a clear paged win.

## Results

Rates are aggregate generated tokens per second. Parenthesized ranges cover the five round-level paired deltas; the null spread is the range of the five repeated-paged deltas.

| Model | Prompt | C | paged-a median | dense median | paged-b median | dense paired delta | null spread | Cell |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gemma3 | 512 | 1 | 154.4 | 154.0 | 154.0 | +0.52% (-2.76..+3.63) | -2.37..+3.84% | unresolved |
| gemma3 | 512 | 4 | 361.0 | 363.6 | 360.6 | +1.29% (-2.11..+2.49) | -1.35..+3.52% | unresolved |
| gemma3 | 512 | 8 | 194.9 | 197.2 | 194.4 | +1.16% (+0.25..+6.37) | -0.88..+3.80% | unresolved |
| gemma3 | 4096 | 1 | 114.6 | 114.4 | 113.0 | -0.18% (-1.82..+0.26) | -2.33..+1.31% | unresolved |
| gemma3 | 4096 | 4 | 219.4 | 216.6 | 218.6 | -0.60% (-2.74..+0.54) | -3.45..+0.50% | unresolved |
| gemma3 | 4096 | 8 | 184.4 | 185.8 | 184.0 | +0.49% (-0.53..+1.98) | -3.13..+1.91% | unresolved |
| llama4 | 512 | 1 | 11.7 | 11.8 | 11.8 | +0.85% (+0.00..+83.08) | -0.85..+81.54% | unresolved |
| llama4 | 512 | 4 | 31.0 | 30.8 | 30.7 | +0.65% (-3.53..+2.61) | -0.97..+3.53% | unresolved |
| llama4 | 512 | 8 | 17.8 | 17.9 | 17.7 | +0.00% (-1.69..+13.29) | -1.13..+10.76% | unresolved |
| llama4 | 4096 | 1 | 8.0 | 8.0 | 8.0 | +1.25% (+0.00..+1.27) | +0.00..+2.53% | unresolved |
| llama4 | 4096 | 4 | 22.9 | 23.1 | 23.0 | +0.44% (-0.43..+2.25) | -2.99..+4.05% | unresolved |
| llama4 | 4096 | 8 | 14.7 | 14.7 | 14.5 | +0.68% (-0.68..+3.55) | -2.03..+2.84% | unresolved |

No cell met either clear-win condition. `Gemma3Wrapper::supports_paged_decode_backend()` and `Llama4Wrapper::supports_paged_decode_backend()` therefore remain `true`. `auto` on a multi-slot server and an explicit `paged` request continue to select paged storage for both families.

## Observed first-arm anomaly

The first Llama 4 round at a 512-token prompt contained a large anomaly in its first arm, `paged-a`, at concurrency 1. That arm recorded 6.5 aggregate tok/s and 12690.4 ms mean TTFT. The following dense and `paged-b` arms recorded 11.9 and 11.8 aggregate tok/s with 4072.6 and 4128.2 ms TTFT. Rounds 2 through 5 recorded approximately 11.7 to 11.9 aggregate tok/s across all three arms at concurrency 1.

Consequently, round 1 contributes a +83.08% dense paired delta and a +81.54% repeated-paged null delta. All records are retained exactly as measured. The host gate accepted the arm, and the evidence does not establish the anomaly's cause. A cold or warm-up effect is one possible hypothesis, but it was not measured and is not asserted here. The preregistered paired/null rule absorbs the observed variation: the five-round paired median is only +0.85%, while the null upper bound is +81.54%, so the cell remains unresolved. The run was not repeated or filtered after observing the result.

## Reproduction

The driver contains absolute paths because those paths are part of the run provenance. On a matching checkout, model layout, release artifact, and MLX build tree, run it under the GPU lock:

```bash
gpu-lock run --tag bench-2230-storage -- bash docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/run.sh
python3 docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/analyze.py \
  --data-dir docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11
```

The committed `complete` marker prevents accidental overwrite. A deliberate rerun should use a new dated directory and preserve this run unchanged.
