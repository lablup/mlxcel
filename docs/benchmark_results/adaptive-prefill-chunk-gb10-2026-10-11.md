# Adaptive server prefill chunks (GB10, 2026-10-11)

Acceptance measurement for issue #2228 and PR #2268. The server now uses a
2048-token prefill chunk while no other sequence is decoding and a 512-token
chunk while a decode batch is live. The acceptance gates check both sides of
that policy: concurrent admission must preserve live-stream latency, and an
idle server must retain the TTFT of an explicit 2048-token chunk.

## Host and method

NVIDIA GB10, driver 580.178.04, host `spark-102`, release profile with
`--features cuda`. The run started at 2026-10-11 04:18 UTC under `gpu-lock`.
Every admission round passed the quiet-host gate before its server started;
every TTFT arm used the benchmark harness's `--hostgate`. No other build, test,
or GPU job ran during the measurement.

The release binaries contain production commit
`88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752`. The clean measurement worktree was
at `b2394566a71fc1f6db83ad40b44380b9c129b57c`; the driver verified that the only
differences were two test files (a prefill-plan type annotation and the source
audit fixtures), so the measured production code exactly matches the release
commit. The declared MLX pin and fetched MLX HEAD were both
`81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`. The fetched source carried the
project's recorded patch overlay; its complete status is in `environment.txt`.

Admission used Llama-3.2-1B-Instruct 4-bit, `--parallel 8 --ignore-eos`, four
4096-token live streams, and one admitted 8192-token request. It ran three
rounds through `scripts/bench_mixed_step_admission.py`; acceptance required
live-stream ITL p95 at or below 62.0 ms in every admission window.

Idle-server TTFT used Qwen3-1.7B 4-bit and Llama-3.2-1B-Instruct 4-bit with an
8192-token prompt. Five interleaved rounds compared the adaptive default
(`policy`) with an explicit `--prefill-chunk 2048` reference (`c2048`) and a
repeat of that reference (`c2048-null`). Acceptance required the paired policy
median to lie inside the paired null range, using unrounded measurements.

## Concurrent admission

| Round | Quiet ITL p95 | Admission ITL p95 | Admission ITL mean | Admitted TTFT |
|---|---:|---:|---:|---:|
| 1 | 15.1 ms | 54.1 ms | 13.1 ms | 3347 ms |
| 2 | 16.0 ms | 54.8 ms | 13.4 ms | 3331 ms |
| 3 | 18.8 ms | 55.2 ms | 13.3 ms | 3555 ms |

**Passed:** all three admission p95 values are below 62.0 ms. Each round
recorded 14 prefill chunks and 13 prefill grants while 4095 decode steps ran;
the admitted request produced its first token before the live streams ended.

## Idle-server TTFT

| Model | Explicit 2048 median | Policy median | Policy paired delta | Null paired range | Result |
|---|---:|---:|---:|---:|---|
| Qwen3-1.7B 4-bit | 688.76 ms | 663.41 ms | +0.014712% (-3.794710..+0.975952%) | -4.729402..+3.261029% | Pass |
| Llama-3.2-1B 4-bit | 416.15 ms | 413.29 ms | -0.202854% (-1.606240..+1.595997%) | -1.115018..-0.029672% | Pass |

Both policy medians fall inside their model's null range. On an idle server,
the adaptive default is therefore indistinguishable from explicitly fixing
the chunk to 2048 at the resolution of this run.

## Decision

The adaptive default passes both acceptance gates. Under live decode it keeps
admission ITL p95 at 54.1 to 55.2 ms, matching the 512-token behavior that
motivated the policy. With no live decode, its TTFT stays within the explicit
2048 noise floor on both tested model families. The shipped defaults remain
2048 alone and 512 contended; an explicit CLI, compatibility, or environment
override continues to pin both states.

## Integrated validation

After PR #2270 and PR #2269 merged, the complete issue #2228 series was
validated with both preceding changes present. Workspace test compilation
(`--no-run`), workspace all-targets Clippy, formatting, the kernel dtype-key,
kernel port-dispatch, and llama-server compatibility contract suites, and the
full single-threaded CUDA workspace run (`make verify-test-cuda`) all passed.
The CUDA run reported zero failed test summaries, including
`ssm_update_kernel_matches_graph_step_bf16` and
`ssm_update_kernel_matches_graph_step_f32`.

The validated pre-rebase tree was
`ab8ff77ade8d6b1b4fb494707320cebc95150c8e`; rebasing the series onto the
current `origin/main` preserved that tree exactly. The final edits after that
comparison update documentation and PR status only.

## Reproduce

The exact raw records, server logs, environment provenance, and machine-readable
summary are in
[`data/adaptive-prefill-chunk-gb10-2026-10-11/`](data/adaptive-prefill-chunk-gb10-2026-10-11/).
Run its `analyze.py` against that directory to recheck the archived acceptance
without rounding. `run.sh` reruns the full protocol from a clean worktree,
acquires `gpu-lock`, checks port 18080, validates the release artifact against
`MLXCEL_BUILD_COMMIT`, and writes a new result directory under `/tmp`.
`MLXCEL_BIN`, `MLXCEL_BENCH_BIN`, and `MLXCEL_MODELS` can point it at relocated
artifacts and checkpoints.

The default `MLXCEL_BUILD_COMMIT` is the archived production commit and is only
valid on an unsquashed PR lineage where that commit is an ancestor and every
later production source is unchanged. It is not expected to work unchanged from
a squash-merged checkout. For a safe replay of the current clean revision,
build both release binaries from that revision and identify it explicitly:

```bash
cargo build --release --features cuda --bin mlxcel --bin mlxcel-bench-engine
MLXCEL_BUILD_COMMIT="$(git rev-parse HEAD)" \
  docs/benchmark_results/data/adaptive-prefill-chunk-gb10-2026-10-11/run.sh
```

To replay with the archived release binaries instead, check out an unsquashed
PR revision that contains this driver and descends from
`88c2e6a0964d4a3a144c2ff1cc3c80dfd2893752`; keep the worktree clean and leave
`MLXCEL_BUILD_COMMIT` at its default. The script rejects unrelated production
changes in that mode rather than attributing them to the archived binary.

The archived data came from the original driver invocation:

```text
/tmp/mlxcel-auto-plan/2228-acceptance.sh /tmp/mlxcel-issue-2228-acceptance-20261011
python3 /tmp/mlxcel-auto-plan/2228-analyze.py /tmp/mlxcel-issue-2228-acceptance-20261011
```
