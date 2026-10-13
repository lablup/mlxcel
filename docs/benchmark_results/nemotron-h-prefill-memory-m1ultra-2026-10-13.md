# Nemotron-H chunked SSD prefill memory — M1 Ultra — 2026-10-13

## Decision

Ship the model-internal SSD chunking for Nemotron-H. The production server path no longer builds prompt-length `L × L` surrogate-attention matrices inside every Mamba layer: it carries recurrent state across the checkpoint's `chunk_size=128` blocks and evaluates each block before constructing the next one.

This record resolves the quadratic-prefill-memory part of issue #2281. It does **not** establish request-level rejection from a projected transient-memory budget. The current server has context-size and KV-budget admission plus an optional startup `--estimate-memory` check, but no per-request activation/transient-memory admission; a separate host/model combination can therefore still reach a fatal Metal driver OOM. The existing 8,319-token context rejection reported below is evidence only for context admission and is not treated as memory admission.

## Environment and provenance

- Host: Mac Studio `Mac13,2`, Apple M1 Ultra, 128 GB unified memory, 20 logical CPUs.
- OS: macOS 27.0.1 (`26A434`).
- Repository base: `61329ee2863afe1120a70f054fd39317fb49877a`; the candidate binary was built from the identical production source in this record's commit before committing the documentation.
- MLX pin: `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.
- Baseline binary SHA-256: `f89beb4d08739e82c5c99df01a7ee3c0d8c295fb8c5dc2ffda64c2044a054e82`.
- Candidate binary SHA-256: `5d7f9b65901785e6a71cf8c2b711e5d647beff2871d9655af0bdaad067452f31`.
- Build command for both arms: `RUSTC_WRAPPER= cargo build --release --features metal,accelerate --bin mlxcel-server`.
- Checkpoint: local `models/mlx/nvidia-nemotron-3-nano-30b-a3b-4bit`; config SHA-256 `bee8181e005ea1ca595d926f10ee3c327d65e1e5fd77596e42d1421c806b2b90`; affine 4-bit/group-64, 52 layers, 64 Mamba heads, head dimension 64, state dimension 128, `chunk_size=128`.
- Server command: `mlxcel-server --model models/mlx/nvidia-nemotron-3-nano-30b-a3b-4bit --ctx-size 8192 --jinja --host 127.0.0.1 --port <fresh-port> --alias repro --parallel 1 --offline`.

## Method

Each measurement launched a fresh server process, waited for `/v1/models`, submitted one non-streaming `/v1/chat/completions` request with `temperature=0`, `max_tokens=8`, and `chat_template_kwargs.enable_thinking=false`, then read the process lifetime high-water mark from `vmmap -summary <pid>` (`Physical footprint (peak)`). The deterministic filler was `benchmark ` repeated to the target length; API usage supplied the actual prompt-token count. There were three fresh-process repetitions per reported cell.

The exact harnesses and JSON outputs are archived under [`data/nemotron-h-prefill-memory-m1ultra-2026-10-13/`](data/nemotron-h-prefill-memory-m1ultra-2026-10-13/).

The harness refused to report a candidate result unless the log contained both the load-time `chunked SSD prefill dispatch enabled` marker and the production-forward `chunked SSD prefill forward` marker with `sequence_tokens`, `ssm_chunk_size`, and `ssd_chunks`. It also refused a baseline result if either marker appeared. For example, the 6,019-token candidate request logged `sequence_tokens=6013 ssm_chunk_size=128 ssd_chunks=47`; all 12 paired baseline/candidate cells passed the attribution guard.

`vmmap` prints GiB high-water values to one decimal place, so the zero ranges below mean that all three readings rounded to the same 0.1 GiB value, not that the underlying byte peaks were identical. The host was not quiet: baseline load-average samples ranged from 3.43 to 6.68 and candidate 1,519–6,019-token samples ranged from 3.03 to 5.03. Because the claim is about per-process high-water memory rather than latency, request wall times are retained in the raw run artifacts but are not used for a throughput claim.

## Results

| Prompt tokens | Baseline peak median (range), GiB | Candidate peak median (range), GiB | Peak reduction |
|---:|---:|---:|---:|
| 1,519 | 20.9 (20.9–20.9) | 17.6 (17.6–17.6) | 3.3 GiB / 15.8% |
| 3,019 | 31.4 (31.4–31.4) | 18.0 (18.0–18.0) | 13.4 GiB / 42.7% |
| 4,519 | 48.5 (48.5–48.5) | 18.4 (18.4–18.4) | 30.1 GiB / 62.1% |
| 6,019 | 72.2 (72.2–72.2) | 18.7 (18.7–18.7) | 53.5 GiB / 74.1% |
| 8,019 | not run (baseline projection approached the host wired limit) | 19.3 (19.3–19.3) | not computed |

Across 1,519–6,019 tokens, the baseline peak grew by 51.3 GiB while the candidate grew by 1.1 GiB. The candidate also completed three 8,019-token runs, within the configured 8,192-token context, at a 19.3 GiB peak. The baseline 8,019-token cell was deliberately left unmeasured because the observed quadratic curve made a driver-level OOM plausible; no result is inferred for that arm.

## Output correctness

The baseline and final candidate produced byte-identical greedy responses for three substantive prompts:

| Prompt | Prompt tokens | Completion tokens | Result |
|---|---:|---:|---|
| Three-sentence solar-eclipse explanation | 37 | 81 | exact response and finish-reason match |
| Warehouse arithmetic with shown work | 65 | 128 | exact response and length finish match |
| Long repeated archive context with a two-fact query | 2,810 | 17 | exact response and finish-reason match |

The numerical unit coverage separately compares 2+2+1 chunking with a full-step evaluation while carrying a nonzero prior recurrent state, and compares multi-batch/multi-group f16 and bf16 chunked outputs and final states with a direct scalar recurrence oracle.

## Existing rejection behavior and remaining safety gap

A final-candidate request containing 8,319 prompt tokens under `--ctx-size 8192` returned HTTP 400 with `type=exceed_context_size_error`; the server process remained alive and `/health` returned 200. This verifies the existing context guard only. No test in this record claims that a within-context request projected to exceed the wired-memory budget is rejected, because the scheduler currently has no such admission input. Arbitrary command-buffer OOM remains fatal and cannot be converted into an HTTP error after Metal terminates the process.

## Unmeasured cells

- The reporter's M4 Pro 48 GB host and original 2-bit checkpoint were unavailable; the checked-in decision uses the available M1 Ultra and the 4-bit Nemotron checkpoint above.
- Candidate 8,019 tokens has three repetitions; baseline 8,019 tokens is intentionally unmeasured to avoid a likely high-blast-radius OOM.
- M5 and CUDA behavior is unmeasured.
- No throughput or latency improvement is claimed because the host carried background load and this change targets memory.
- Memory-derived per-request admission is unimplemented and unmeasured; the issue should remain open or receive an explicitly split follow-up until that acceptance item is decided.
