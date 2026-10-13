# PR #2290: Bound Nemotron-H SSD prefill memory

**Date**: 2026-10-13
**Status**: Complete for the bounded SSD-prefill scope; implementation reviewed and focused validation passed
**Risk**: High at issue scope because request-level transient-memory admission remains unimplemented

## Summary

PR #2290 changes Nemotron-H Mamba prefill from one prompt-length quadratic surrogate-attention graph to checkpoint-sized SSD chunks. Each chunk carries the recurrent state forward and is evaluated before the next chunk is constructed, reducing the measured 6,019-token process peak from 72.2 GiB to 18.7 GiB on the tested M1 Ultra checkpoint while preserving substantive greedy outputs.

## Problem

The prior `nemotron_h` prefill path constructed an `L × L` surrogate-attention matrix in every Mamba layer. Long prompts therefore produced a transient memory peak that grew roughly quadratically and could terminate the process in the Metal driver. Issue #2281 also requires predictable oversized requests to be rejected, but the server does not yet estimate per-request activation or transient memory.

## Design and review

- `NemotronHConfig` now reads `chunk_size`, accepts the legacy `mamba_chunk_size` alias, defaults to 128, and rejects zero before model construction.
- `nemotron_h_ssm_attn` preserves float32 timestep, state, and matrix-product boundaries; computes each quadratic block independently; evaluates its output and next state; and uses that state as the prior state for the following block.
- Focused numerical coverage compares a 2+2+1 partial-tail partition with a full step carrying a nonzero prior state, and compares multi-batch/multi-group f16 and bf16 results with a scalar recurrence oracle.
- The implementation is wired through the production Nemotron-H model builder and is exercised by the real server benchmark. No code-level correctness defect remained within the declared partial scope.

## Benchmark evidence

Three fresh-process repetitions were recorded for each 1,519-, 3,019-, 4,519-, and 6,019-token cell. Arm-specific load and forward markers prevent the harness from reporting a comparison unless the candidate executes the new path and the baseline does not. Candidate peak memory remained between 17.6 and 18.7 GiB across those cells and reached 19.3 GiB for three 8,019-token runs. Three substantive greedy prompts, including a 2,810-token prompt, produced identical choices in both arms. The 8,019-token baseline, the reporter's M4 Pro and 2-bit checkpoint, M5, CUDA, and latency under a quiet host were explicitly left unmeasured.

## Verification

- Release Metal/Accelerate `mlxcel-server` build: passed.
- Nemotron-H module suite: 22 passed, 3 ignored.
- Dedicated serial Metal numerical tests: 2 passed.
- Formatting, diff hygiene, Python harness compilation, benchmark artifact consistency, and cross-repository reference review: passed.
- Current CI formatting, Clippy, CPU-only link, license, Python tooling, and repository contract checks observed during review: passed; remaining jobs were still running or queued.

## Remaining work

Request-level activation/transient-memory admission remains required to address arbitrary within-context Metal command-buffer OOM that can still terminate the process. The user approved tracking and implementing that work as a separate follow-up after this bounded SSD-prefill fix merges; no follow-up issue number exists yet. The PR body's non-closing wording was rewritten after GitHub interpreted the earlier negated phrase as a closing keyword, and `closingIssuesReferences` is now empty.
