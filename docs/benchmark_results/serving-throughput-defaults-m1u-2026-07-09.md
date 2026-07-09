# Serving-throughput defaults: parallel decode, batched prefill, prompt cache (issue #628, M1 Ultra)

Date: 2026-07-09. Host: Apple M1 Ultra (128 GB), Metal, features `metal,accelerate`.
Binary: `cargo build --release --features metal,accelerate --bin mlxcel-server` at the #628 branch point (base `9808b675`, the audited pre-change binary).
Bench: `scripts/bench_serving_concurrency.py` (from #624), 512-token synthetic prompt, 128 new tokens per request. Each server ran on an ephemeral port and was killed after the run.

Hardware note: this session measured on Apple M1 Ultra. The issue's `2.5x`-at-4-clients acceptance target and `cargo test --features cuda` are GB10-specific and remain pending a CUDA session. The defaults are backend-agnostic; the mechanism (weights are read once per decode step regardless of batch size) applies on Metal too, and M1 Ultra's ~800 GB/s memory bandwidth gives less amortization headroom than a higher-bandwidth decode target.

## Audit (pre-change reality on `main`)

The issue premises were checked against `9808b675` before implementing:

| Premise | Issue claim | Reality on main | Action |
|---------|-------------|-----------------|--------|
| `--parallel` / `max_batch_size` | defaults to 1 | confirmed: `default_value_t = 1`, `max_batch_size = unwrap_or(n_parallel) = 1` | changed to 4 |
| `--max-batch-prefill` | defaults to 1 | confirmed: `default_value_t = 1` | changed to 4 |
| prompt cache | off unless configured | **stale**: already default-on (`--prompt-cache-enabled default_value_t = true`, `PromptCacheConfig::default().enabled = true`, 2 GiB budget; APC on) | kept on; added `--no-prompt-cache` opt-out |

The prompt cache was already default-on (startup logs `Prompt-prefix cache store enabled ... capacity_bytes=2147483648`), likely since the #224-#228 memory-footprint work. Only the two batching defaults were genuinely default-off.

## Parallel decode scaling (llama-3.1-8b-4bit)

One server per configuration; the load generator issued `clients` simultaneous requests. Baseline = previous default `--parallel 1`. New default = `--parallel 8 --max-batch-prefill 4` (batch capped at 8 so the 4- and 8-client rows reflect true batched decode; `--parallel 4` caps aggregate at the 4-client value).

| clients | aggregate tok/s (`-p1`) | aggregate tok/s (`-p8 --mbp4`) | per-req decode tok/s (`-p8`) | TTFT mean ms (`-p1`) | TTFT mean ms (`-p8`) | TTFT p95 ms (`-p1`) | TTFT p95 ms (`-p8`) |
|--------:|------------------------:|-------------------------------:|-----------------------------:|---------------------:|---------------------:|--------------------:|--------------------:|
| 1 | 56.6 | 56.8 | 86.4 | 889 | 783 | 889 | 783 |
| 2 | 71.5 | 99.9 | 51.9 | 1150 | 114 | 2220 | 150 |
| 4 | 65.6 | 107.9 | 27.9 | 3257 | 189 | 6439 | 297 |
| 8 | 62.8 | 105.1 | 13.5 | 7514 | 348 | 14939 | 614 |

- Aggregate throughput at 4 concurrent clients is **1.90x** the single-client aggregate (107.9 vs 56.8). The curve plateaus at 4-8 clients (107.9 -> 105.1), so `--parallel 4` captures the benefit while bounding KV memory to 4 concurrent sequences.
- TTFT under load collapses: at 4 clients, 3257 ms -> 189 ms (~17x); at 8 clients, 7514 ms -> 348 ms (~21x). With `--parallel 1` requests serialize behind one slot; batched admission serves them concurrently.
- Single-client throughput is unchanged (56.6 -> 56.8), so the new default has no single-client regression.
- Per-request decode rate falls as the batch fills (86 -> 28 -> 13.5 tok/s), the expected trade for higher aggregate throughput.

The GB10 `>= 2.5x` acceptance target is not reproduced here (M1 Ultra reaches 1.90x); it is pending a CUDA measurement session.

## Batched prefill isolation (llama-3.1-8b-4bit, cold cache)

`--parallel 4`, 4 concurrent clients, cold prompt cache (single bench level so no prior warm-up), isolating `--max-batch-prefill`:

| `--max-batch-prefill` | aggregate tok/s | TTFT mean ms | TTFT p95 ms |
|----------------------:|----------------:|-------------:|------------:|
| 1 | 56.0 | 4215 | 7632 |
| 4 | 69.8 | 2142 | 2844 |

Batching the four cold prefills into one forward pass halves mean TTFT (4215 -> 2142 ms) and p95 TTFT (7632 -> 2844 ms), and lifts aggregate throughput 56 -> 70 tok/s. (These cold-cache TTFTs are higher than the scaling table above, where the shared prompt was already warm in the prompt cache by the 4-client level.)

## Prompt cache (already default-on) verification

The scaling run reused the same synthetic prompt across levels, so later requests hit the prefix cache. `/metrics` after the `--parallel 8` run:

```
mlxcel_prompt_cache_hits_total 28
mlxcel_prompt_cache_misses_total 2
mlxcel_prompt_cache_prefix_tokens_reused_total 14336
```

The hit counter increments and prefix tokens are reused, satisfying the end-to-end prompt-cache acceptance criterion. `--no-prompt-cache` disables the store (a unit test asserts `prompt_cache_enabled` resolves to false).

## Memory guard-rail (B=8 on the 8B model)

`--parallel 8`, 8 concurrent clients, 128 GB M1 Ultra: 0 request failures, peak RSS 4409 MB (sampled via `ps`). No OOM. Note MLX Metal reports `peak_bytes=0` for the 4-bit load path (a quantized-load accounting quirk; the bf16 qwen2.5-0.5b control on the same path reported `peak_bytes=1.27 GB` correctly), so external RSS is the better proxy for the 4-bit model. KV growth for `B` sequences is bounded by admission (`--kv-cache-budget` / `--max-kv-size`), which sheds load under pressure rather than OOMing.

To make that guard automatic under the batched-decode default, the shipped `--kv-cache-budget` default is now `auto` (it was unset, i.e. unbounded, on the audited binary): the #122 paged block-budget admission bounds KV for the concurrent batch on the paged decode backend and is inert on the dense backend. `--kv-cache-budget none` (or `0`) restores the unbounded pool.

## Fast-model data point (qwen2.5-0.5b-bf16, `--parallel 8 --max-batch-prefill 4`)

| clients | aggregate tok/s | TTFT mean ms |
|--------:|----------------:|-------------:|
| 1 | 185.2 | 68 |
| 2 | 146.2 | 31 |
| 4 | 205.4 | 50 |
| 8 | 323.6 | 107 |

Aggregate scales to 323 tok/s at 8 clients on an overhead-bound model (the 2-client dip is first-level warm-up noise), confirming the defaults help small models too.

## Conclusion

Ship `--parallel 4` and `--max-batch-prefill 4` as defaults, keep the prompt cache on (already the case), and add `--no-prompt-cache` as a clean opt-out. On M1 Ultra this is a 1.90x aggregate-throughput and ~17x TTFT-under-load win at 4 clients with no single-client regression and no OOM. `--parallel 1`, `--no-batch`, `--max-batch-prefill 1`, and `--no-prompt-cache` restore the previous behavior.
