# Issue #2227 pre-change B=1 profile analysis

## Provenance and method

The profile used one release build from source `73700aafa6a792f07830578c0c26137cea3a038b` plus the temporary, environment-gated profiler patch `306b2664d8b69186cc896ec83c1f6323e534372fc1148fd75fc4df9e5e2fe0b6`. The profiled `mlxcel-bench-engine` SHA-256 was `eb208514dc7dfbb8dbf9314d9c6d6cee5fd0ef50139abbb546e48d5b42a57572`. The temporary instrumentation was removed before the production change.

Both paths used Qwen3-1.7B 4-bit, a 256-token prompt, 16 warm-up tokens, 128 measured decode tokens, dense storage, and `MLXCEL_SDPA_DETERMINISTIC=1`. The server and CLI executions came from the same binary. Host: `spark-102`, NVIDIA GB10, driver 580.178.04, CUDA 13.0, MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.

Linux had `perf_event_paranoid=4`, and Nsight Systems reported CPU sampling unavailable. The CUDA/OSRT trace therefore used Nsight Systems 2025.3.2 with `--sample=none --event-sample=none --backtrace=none --cpuctxsw=none`. Temporary `Instant` timers attributed host-side stages. The tick denominator starts at the normal scheduler loop entry and ends after `finalize_completed`; the profiler clears any startup and prefill samples when the first B=1 decode tick begins. `lookahead_submit` includes asynchronous GPU work that completes before the call returns and must not be read as pure host overhead.

The exact driver, parser, temporary patch, logs, and Nsight summaries are in the raw-data directory. Their source hashes are:

- `run-profile.sh`: `9c3cab236f478d46524c6e990a566167be24694d53fb17651d240172b9e5b8f1`
- `parse-profile.py`: `f36f9d38083f07c40002f86a101565dd4cb268f5ec9ae1fd94a1bb9f3b86bb79`
- `instrumentation.patch`: `306b2664d8b69186cc896ec83c1f6323e534372fc1148fd75fc4df9e5e2fe0b6`
- `server-host-breakdown.txt`: `a9e49debc092aead3b64f5674062bdfbb2b087758a0d9772c768a69280ba1728`
- `server-nsys-stats.txt`: `c2439b8e2c180c71066f7877852da37f247bc9ed655cac3e84b59d9f4b2c7ed4`
- `cli-nsys-stats.txt`: `7eccd23136832d2c55c8c16f5c2ce20cda4acb81bcf43d5ad2c4bef0ffca397b`

## Host timers

The measured request had 127 decode ticks averaging 5,318.203 microseconds.

| Stage | Calls | Total ns | Percent of full decode tick |
|---|---:|---:|---:|
| Metrics publication | 128 | 312,052 | 0.0462% |
| Row lookup | 127 | 17,824 | 0.0026% |
| Row build | 127 | 18,096 | 0.0027% |
| Lookahead collect | 126 | 564,546 | 0.0836% |
| Fused finish rows | 125 | 976,026 | 0.1445% |
| Provider channel send | 127 | 471,319 | 0.0698% |
| Lookahead submit | 127 | 656,315,740 | 97.1727% |
| Synchronous engine step | 2 | 13,668,204 | 2.0237% |

The provider measurement covers only the `mpsc` send. It does not cover token decoding or `String` allocation, so no broader claim is made about those operations. `StreamFilter` is downstream of the provider and outside `mlxcel-bench-engine`'s server path.

The issue requires leaving candidates below 0.1% alone. Row lookup plus row construction measured about 0.0053%, metrics 0.0462%, provider channel send 0.0698%, and lookahead collect 0.0836%; none was changed. `fused_finish_rows` exceeded 0.1%, but it performs the required shared sampling/finish state transition and is retained. The profiler did not identify an equivalent result-preserving call that could be removed from every tick.

## CUDA evidence and selected change

The Nsight Systems CUDA summaries recorded 4,067 `cudaGraphLaunch` calls on the server and 3,941 on the CLI, a difference of 126. `cudaGraphExecUpdate` was 4,019 versus 3,894, a difference of 125. These are aggregate CUDA graph API counts, not decode-token or model-forward counts. Each trace covers both the 16-token warm-up request and the 128-token measured request, and a model forward emits multiple graph calls, so this aggregate difference is used only as corroborating evidence of extra server-side GPU work. It is not divided into a per-token or per-forward estimate.

Source tracing and the scheduler forward-count test identify the request-level cause. `DirectEngine` checks whether the pending token will exhaust `max_tokens` before submitting another lookahead. The scheduler previously submitted token n+1 first, collected pending terminal token n, unwound both speculative KV appends, and synchronously replayed token n. At the length boundary this performs two complete, unused Qwen decode forwards: the n+1 lookahead and the replay of n. It happens once per fused-lookahead-eligible capped B=1 request, rather than once per decode token.

The production change applies the same known length-budget check to a B=1 pending lookahead. When that pending token must exhaust the budget, the scheduler skips n+1 and commits n through the shared `finish_rows` path. EOS and errors retain existing precedence and handling. B>1 keeps its existing teardown/replay path. Dense cycle-Llama tests first assert that lookahead is eligible and live, then count actual scheduler `submit`/`step` calls across max-token boundaries 0, 1, 2, 3, and 4; they pin B=1 at two forwards and unchanged B=2 at four for `max_tokens=3`, plus stochastic pending-draw commit and EOS precedence. A separate model-owned Muse fixture compares B=1 and B=2 generated token IDs, `FinishReason`, stream text, pool offset, and model-state offsets while explicitly documenting that Muse remains on its synchronous path.
