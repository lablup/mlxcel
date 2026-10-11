# Raw data for model-owned-storage-gb10-2026-10-11.md

This directory contains the complete accepted run for the Gemma 3 and Llama 4 model-owned decode-storage decision in issue #2230. The run started at `2026-10-11T04:57:35Z`, completed at `2026-10-11T07:12:28Z`, and produced 60 fresh-server arms and 180 successful client cells.

## Contents

- `run.sh`: fail-fast driver for the exact two-model, two-prompt, five-round, three-arm protocol. It must be invoked through the outer `gpu-lock` wrapper shown below.
- `gate.py`: per-arm fail-closed quiet-host gate. It checks the port before and after the wait, retries a post-wait race, and accepts only snapshots without active CI, build, foreign-model, memory-pressure, or NVIDIA-driver failures.
- `analyze.py`: protocol and provenance validator plus paired/null decision analyzer.
- `test_protocol.py`: lightweight synthetic fixtures for the valid protocol and rejected missing, dirty, reordered, nonmonotonic, incomplete, or backend-substituted evidence.
- `environment.txt`: contemporaneous start/end times, clean benchmark-checkout commit, release artifact identity, configured and built MLX commits, host/GPU identity, host-gate identity, and the measured gate/analyzer/driver hashes. Its `commit` key identifies the checkout used to run the sweep, not an embedded binary-source revision.
- `binary-source-provenance.json`: post-run derived verification that identifies the release binary's production source commit, proves it is an ancestor of the benchmark checkout, and records the exact four protocol-only paths in the intervening diff. It also records the preserved release-log reference and hash. This companion does not rewrite or claim to be part of the contemporaneous environment record.
- `schedule.tsv`: the exact ordered 60-arm schedule. The header plus 60 arms gives 61 lines.
- `hostgate.jsonl`: one accepted preflight snapshot followed by one accepted snapshot for each arm, in schedule order (61 records total).
- `server-*.log`: 60 nonempty server logs. Each begins with the requested dense or paged backend marker used by the analyzer.
- `client-*.txt`: 60 client logs, each with successful concurrency 1, 4, and 8 rows (180 measurement cells total).
- `run.log`: the full driver transcript, including every gate, server start, completion, and final `ALLDONE` marker.
- `analysis.txt`: analyzer output after validation, with all 12 per-model/prompt/concurrency decisions.
- `complete`: completion marker written only after all 60 arms finished successfully.
- `SHA256SUMS`: checksums for every other committed file in this directory.

The measured release binary was built from production commit `2cb92404b153897440c4906a5548dcfb4889a9ad`. The clean checkout recorded in `environment.txt` was `35da2dbe32c731ff6645b2b32a2117af2965f332`. The earlier commit is an ancestor, and the complete intervening diff contains only `analyze.py`, `gate.py`, `run.sh`, and `test_protocol.py` in this directory. Thus the production Rust/C++/model source used by the binary is unchanged at the measurement checkout. The binary does not embed a commit; this association is explicitly derived after the run and is documented with its limitation in `binary-source-provenance.json`.

The large Llama 4 `p512`/concurrency-1 range is traceable to the first round's first `paged-a` arm. Its client log and all surrounding arms remain unchanged in this directory. The dated report discusses the observation and its limitation without assigning an unmeasured cause.

## Replay and validation

`run.sh` records absolute build, model, and worktree paths because they are part of the original provenance. Replaying it requires those paths, the recorded release build dependencies, both model directories, and an idle GB10 host. The script refuses to overwrite this completed run. Use a new dated data directory for any new measurement.

The original outer command was:

```bash
gpu-lock run --tag bench-2230-storage -- bash \
  /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2230/docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/run.sh
```

Validate and reproduce the committed summary without a GPU:

```bash
python3 docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/analyze.py \
  --data-dir docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11
python3 -W error docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11/test_protocol.py
(cd docs/benchmark_results/data/model-owned-storage-gb10-2026-10-11 && \
  sha256sum --check SHA256SUMS)
```
