# Technical Report: PR #1278 - Pricing the qmv_wide narrow pin's collateral

## Executive Summary

Issue #1261 asked what the rest of a server process pays when the MTP exactness gate disables `qmv_wide` for the whole process to buy back temperature-0 byte-identity. The issue was deliberately measurement-gated ("nothing below is worth building until the collateral cost is a number") and carried an explicit exit condition: if the tax is small on production shapes, document it and stop.

This PR takes that exit. It changes no Rust source. It adds two benchmark harnesses, records the measurement on the required Apple GPU generation 15 host, and corrects three documents whose `MLXCEL_MTP_ALLOW_INEXACT` recipe went stale when #1199 changed the gate's ordering.

The measured answer is that the batched-decode tax is at most 1%, and the reason is structural rather than numerical: neither MTP family dispatches the `M = B` projection the issue assumed. The real collateral cost is one prompt-cache-adopted suffix prefill per request, +15.4 ms per forward on the Gemma target and +12.6 ms on the Qwen target.

## 1. Problem Statement

PR #1199 gave the exactness gate a retry: when the multi-token verify block diverges from the single-token decode chain under `qmv_wide`, the gate turns `qmv_wide` off and re-probes, and keeps it off for the rest of the process when that restores byte-identity. The switch is deliberately never restored, because re-enabling it would break the very block the gate just approved.

But the switch is process-wide. It sits on the dispatch path of every quantized matmul in `dispatch_qmv`, so a server sharing the process pays it on work that never asked for byte-identity. #1199 said so explicitly and filed the scoping work as follow-up. #1261 is that follow-up, and its first step was to find out whether the collateral cost is large enough to justify the surgery.

The verify-side cost was already priced (17 to 20% on the Qwen verify forward, about 23% on the Gemma 4 verify forward). What the bystanders pay had never been measured.

## 2. What Was Measured

Two arms on a Mac Studio M3 Ultra (`applegpu_g15d`, generation 15, macOS 26.6.1), under `scripts/with_indexers_paused.sh` with Time Machine off.

**Arm 1, batched-decode B-sweep with no drafter.** Throughput at B = 1, 2, 4, 8 with `MLXCEL_QMV_WIDE=1` pinned against `MLXCEL_QMV_WIDE=0` pinned, on `models/gemma-4-31b-it-4bit`, confirmed on `models/qwen3.8-27b-4bit`. Eight boots in two balanced ABBA blocks, one discarded warm-up pass and two measured passes per boot, giving 8 samples per arm per cell (4 on the Qwen confirmation). Result: 0.0 to 0.9% on Gemma, 0.0 to 0.2% on Qwen, every spread at or under 1% against the harness's 4% trust limit.

**Arm 2, mixed workload.** One MTP stream holding the tick-slice speculative slot plus four classic streams on the Qwen MTP pairing, arm A being the default env where the gate's retry pins the process narrow and arm B being `MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1`. Only the classic streams' decode rates are read. Result: the classic streams lose 0.1% to 2.6%, and the loss tracks the MTP stream's own verify slowdown occupying the shared worker rather than the classic streams' own kernels.

## 3. Technical Decisions

### 3.1 Take the issue's exit rather than build Step 2

The issue's Step 2 offered three shapes for scoping the exact kernel to the verify forward, from bracketing with explicit synchronization to a dispatch-side stream predicate. None was built, which is the sanctioned outcome when the B-sweep delta is small.

What makes the exit defensible is not the smallness of the number but the mechanism behind it. `M = 1` always dispatches `qmv`; the pin only selects between `qmv` and `qmv_wide` at `M >= 2`. Both MTP families decode batches one row per forward, so batched decode in a pinned process never reaches the kernel the pin disables. Gemma 3 and Llama 4 do stack decode rows into one `M = B` forward, but they are not MTP families, so no process they run in is ever pinned by the gate.

The record names the boundary that would reopen the question: a real joint batched decode for an MTP family. The moment Gemma 4 or Qwen 3.5 stacks decode rows the way Gemma 3 already does, batched decode lands in the qmv window and the pin starts taxing it.

### 3.2 Pin both arms explicitly rather than let the gate choose one

`qmv_wide_pinned_by_operator()` reads `std::env::var("MLXCEL_QMV_WIDE").is_ok()`, so setting the variable to any value, including `0`, counts as an operator pin and skips the gate's retry in both directions. Setting it on both arms is what makes the B-sweep a clean A/B: the gate cannot flip an arm mid-run. With no drafter loaded there is no probe to run in the first place, and `mlxcel_core::set_qmv_wide` has exactly one caller, the gate, so nothing else can move the flag either.

### 3.3 Report the contaminated cell rather than average it in

The long-context B = 4 cell reads higher throughput on the narrow arm, stably across all 16 samples. That is not a kernel effect. At temperature 0 the two kernels legitimately generate different text, and on that prompt the divergence changes the generation length itself (130 tokens wide against 182 narrow), so the two columns compare different workloads. The cell is reported as text-divergence-contaminated and excluded from the tax, with its TTFT columns retained because prefill precedes generation and they serve as the chunked-prefill control.

The ladder cells avoid this by construction: every generation ran 59 tokens in both arms, verified from per-stream `usage` counts, so the rate comparison is length-matched even though the bytes differ mid-stream.

### 3.4 Correct the stale `MLXCEL_MTP_ALLOW_INEXACT` recipe

`mtp_exactness_gate` runs `retry_without_qmv_wide` before `allow_inexact()` is consulted (`let decision = exact || allow_inexact();`). On a host where the narrow retry passes, which is every generation 15+ host measured so far, the override alone is therefore inert: the retry pins the process narrow first and the flag never becomes load-bearing. Documents written before #1199 merged say the override alone reaches the fast kernel, which it no longer does.

The correction is verified three independent ways rather than asserted: the log lines (the override-alone run logs the retry's INFO line and never the loud warning), the bytes (the default run and the override-alone run produce byte-identical text, while the pinned-wide run diverges six words in), and the throughput (117 against 139 tok/s, reproducing the byte-identical and fast-kernel figures `docs/benchmarks.md` already carries). The working recipe is `MLXCEL_QMV_WIDE=1` together with `MLXCEL_MTP_ALLOW_INEXACT=1`.

### 3.5 Count `reasoning_content` deltas in the concurrency harness

Qwen 3.8 streams its reasoning channel as `reasoning_content` deltas. `bench_serving_concurrency.py` counted only `content`, so a request that spent its whole budget thinking reported no TTFT and no decode rate at all. Both channels are decoded tokens, so both now count. Previously recorded measurements on non-reasoning models are unaffected, since `reasoning_content` is simply absent there.

## 4. Review Findings and Corrections

The implementation review verified the two load-bearing code claims directly against the source, and one of them cited a route the measurement did not take.

**The Gemma 4 mechanism citation (corrected).** The record said Gemma 4 "does not override `forward_batched`, so it inherits the trait default". That is true of `src/models/gemma4.rs`, but `models/gemma-4-31b-it-4bit` carries `embed_vision.*` weights, so `gemma4_has_vision_weights` routes it to `LoadedModel::Gemma4VLM`. The scheduler's `execute_batched_decode` calls `forward_batched_with_context_and_ids` with the batch's sequence ids, and `Gemma4VLModel` overrides that entry point (`src/vision/gemma4_vl.rs:687`) in favour of `forward_batched_with_seq_ids_dispatch`. The trait default is never reached on the measured configuration.

The conclusion survives and in fact rests on firmer ground, because that dispatch helper is itself an explicit per-row loop over `forward_with_sequence_id`. But the record's whole value is that a future reader can re-verify it, and a reader who followed the citation would have found the override and concluded the claim was false. The record now describes both routes and names the one the measurement took.

The Qwen claim needed no correction. `src/models/qwen3_5.rs:3388` branches to a per-row loop whenever `shape[1] <= 1`, which is every decode step, and the `Qwen35VLModel` wrapper delegates straight through to the same function, so the citation is correct for both variants.

**Quantity naming (corrected).** `docs/benchmarks.md` carries two different 23% figures: a verify-forward cost for the Gemma 4 family and an end-to-end decode cost for the M5 Max code row, and it explicitly warns not to conflate them. The record quoted "~23% on Gemma 4" without saying which, and quoted a composite "17 to 23%" range in a section measuring the Qwen pairing alone. Both now name the quantity and the family.

**Fit tolerance (corrected).** The suffix-forward fit claimed agreement "to within 0.1 ms on every row"; the B = 8 row is 0.2 ms off (4.5 times 15.4 is 69.3 against a measured 69.5).

Verified and found accurate: the `MLXCEL_QMV_WIDE` documentation row against the C++ flag parsing and `qmv_wide_pinned_by_operator`, all four gate-recipe log lines against the `tracing` calls they quote, the claim that `set_qmv_wide` has exactly one caller, the qmv batch limit of 12 for every projection of the 31B target on an `applegpu_g15d` part, the `mtp_capable_target` family list, and the Gemma 3 and Llama 4 joint-decode counterexamples.

## 5. Change Summary

| File | Change |
| --- | --- |
| `scripts/bench_qmv_wide_pin.sh` | New. ABBA boot driver for both arms. `sweep` alternates pinned-wide and pinned-narrow boots without a drafter; `mixed` alternates gate recipes and greps each boot's exactness-gate line so arm identity is evidenced. |
| `scripts/bench_qmv_pin_mixed.py` | New. Mixed-workload client. Classic decode rates are the reported quantity, and a window is invalid unless the MTP stream decoded through at least 95% of it. |
| `scripts/bench_serving_concurrency.py` | Count `reasoning_content` deltas alongside `content`. |
| `docs/benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md` | New. The measurement record, including the excluded contaminated cell and the gate-recipe verification. |
| `docs/benchmarks.md` | Link the record; correct the fast-row reproduction recipe and the declining-probe sentence for the post-#1199 retry ordering. |
| `docs/environment-variables.md` | Correct the `MLXCEL_MTP_ALLOW_INEXACT` row; add the missing `MLXCEL_QMV_WIDE` row. |
| `docs/benchmark_results/mtp-drafter-step-profile-m5max-2026-08-17.md` | Dated note that its reproduction recipe predates the #1199 retry. |
| `.gitignore` | Ignore the raw `bench-results/` run directories the driver writes. |

No Rust source changed, so there is no runtime behaviour change to regress. Validation was `python3 -m py_compile` on both Python harnesses and `bash -n` on the driver, plus the measurement runs themselves.

## 6. Follow-up

Found on the way and recorded rather than resolved: on current `main` the 31B plus bf16 assistant pairing probes non-identical under both kernels on this host, so the default-env gate declines the batch-capable burst that #1217 enabled. #1217's 1.95x to 2.65x rows were measured at `9e2c6675`, which predates #1258's Gemma probe. Anyone rerunning those rows on current `main` will hit this, and deciding what to do about that default belongs to its own issue.

The scoping work itself stays unbuilt until an MTP family gains a real joint batched decode. At that point the B-sweep in this record should be rerun, a material number should be expected, and only then are the issue's Step 2 candidates worth weighing.
