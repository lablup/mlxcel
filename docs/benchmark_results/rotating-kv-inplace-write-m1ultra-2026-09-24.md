# In-place decode KV write for the rotating sliding-window cache

Follow-up to issue #1959, which made the dense FP16 `KVCache` write a decode row
in place. `RotatingKVCache` kept the copying `slice_update`. This change applies
the same ownership proof to its warmup phase, while the cache is still filling
its window: each decode token then lands on a slot that was never valid, so no
other holder of the buffer can see it change. Once the window is full a step
overwrites the oldest slot, which a prompt-cache snapshot may still hold, so
those writes keep copying.

- **Date:** 2026-09-24
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** one binary; the rotating in-place path disabled through a
  measurement-only switch (equivalent to `main` at `d3f6ee77`, where the dense
  path is already in place) against the change
- **Condition:** `mlxcel-bench-decode` with `bench_decode.sh`'s arguments
  (pp512, tg128, warmup 20, `--ignore-eos`), two runs per arm, interleaved
- **Machine load:** indexers suspended; load average 6 to 7.5, so single runs
  move about 2%

## Results (decode tok/s)

| Model | sliding window | main | this change |
|---|---|---|---|
| Gemma 3 4B IT 4-bit | 1024 | 100.57, 100.77 | 102.75, 104.79 (about +3%) |
| Gemma 4 12B IT 4-bit | 1024 | 36.08, 36.48 | 37.65, 37.06 (about +3%) |
| Gemma 4 E4B IT 4-bit | 512 | 78.40, 80.11 | 78.78, 80.10 (none) |
| gpt-oss 20B mxfp4-q4 | 128 | 100.86, 100.59 | 101.34, 98.08 (noise) |
| EXAONE 4.0 1.2B 4-bit | none | 250.53, 243.28 | 249.69, 248.49 (noise) |

The split follows the design. Gemma 3 and Gemma 4 12B have a 1024-token window,
so pp512 plus 128 generated tokens stays in the warmup phase. Gemma 4 E4B (512)
and gpt-oss (128) fill their windows during the prompt and decode entirely in
the steady-state phase, which still copies; their copies are small because the
buffer is only the window long. EXAONE 4.0 1.2B has no sliding window and was
already covered by #1959.

## Correctness

- Greedy output byte-identical with the in-place paths on and off
  (`MLXCEL_KV_INPLACE_WRITE=0`) for all five models, two prompts (short and
  512 tokens), 300 tokens each.
- Unit test: warmup rows written in place leave a shared snapshot's rows intact,
  and after the ring wraps the overwrite goes to a copy, so a snapshot taken at
  the full window keeps the evicted token. Allowing the in-place write in the
  steady state makes the test fail (the snapshot's slot 0 becomes the new token).

## Not measured

Contexts past the window (steady state unchanged by design), CUDA and ROCm.
