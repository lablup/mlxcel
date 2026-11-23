# Inkling 2 Mini Metal kernels on M3 Ultra (#3359)

Decode throughput and verify-forward latency of Inkling 2 Mini with the three ported mlx-vlm kernels (fused short convolution, attention head preparation, grouped NVFP4/MXFP8 expert matvec) against the ops path they replace. Arm A sets `MLXCEL_INKLING2_KERNELS=0` (every forward on the ops path, the code `b41c07b7` runs); arm B leaves the kernels on.

## Environment

| Item | Value |
|---|---|
| Host | Apple M3 Ultra, 512 GB, macOS 27.2.1, shared with other jobs |
| Build | `cargo build --release --features metal,accelerate --bin mlxcel` on `b41c07b7` plus this PR |
| Checkpoints | `models/inkling-2-mini` (`thinkingmachines/Inkling-2-Mini` b85c3cfa, native BF16 with NVFP4/MXFP8 experts) and `models/inkling-2-mini-4bit` (`mlx-community/Inkling-2-Mini-4bit` 9c0e2960, affine 4-bit dense, the same NVFP4/MXFP8 expert planes) |
| Decode command | `mlxcel generate -m models/<name> -p "Explain why the sky is blue." -n 128 --temp 0 --show-reasoning --profile` |
| Procedure | A, B, A, B, A, B per checkpoint, 20 s pause before each run, every run under `/usr/bin/lockf -k /Users/devops/Development/.mlxcel-gpu.lock` |

## Decode

Decode tok/s from `--profile` (128 tokens, 22 prompt tokens).

| Checkpoint | A runs | B runs | A mean | B mean | Change |
|---|---|---|---|---|---|
| inkling-2-mini | 32.28 / 32.97 / 30.95 | 32.80 / 36.61 / 36.47 | 32.07 | 35.29 | +10.1% |
| inkling-2-mini-4bit | 35.94 / 37.21 / 37.28 | 44.75 / 44.72 / 44.80 | 36.81 | 44.76 | +21.6% |

The first B run on the native checkpoint (32.80) is the slowest of the three; the other two agree within 0.4%. For reference, #3367 recorded mlx-vlm at 32.8 tok/s (native) and 38.5 tok/s (4-bit) on this host, so both checkpoints now decode faster than the reference.

Plain decode takes the short convolution and head preparation kernels only; the grouped expert kernel runs inside a verify forward.

## Greedy output

All six runs of a checkpoint print the same 128 tokens (md5 of the generated text: native `32e7b82d`, 4-bit `90ec71a7`), so the output is identical with and without `MLXCEL_INKLING2_KERNELS=0`.

## Verify forward (grouped expert kernel)

`models::inkling2::verify_bench_tests::verify_round_timing` (ignored test) on the native checkpoint: after a 96-token prefill, a `width`-token forward is run inside a `DecodeLookaheadAppendScope` (what `Engine::verify` opens, so the grouped kernel applies), its argmax evaluated, and the block trimmed off; 3 warm-up and 30 timed rounds. The same width is also run without the scope ("plain"), which takes the other two kernels but not the grouped one. Order A, B, A, B, each under the GPU lock. Median ms per forward, rounds 1 / 2:

| Width | A verify | B plain (no grouped kernel) | B verify | B verify vs A | Grouped kernel alone |
|---|---|---|---|---|---|
| 4 | 52.00 / 52.15 | 46.98 / 46.89 | 45.76 / 45.64 | -12.1% | -2.6% |
| 8 | 60.35 / 60.62 | 54.57 / 54.47 | 51.83 / 51.71 | -14.4% | -5.0% |

The argmax digest of every forward is the same in both arms and both modes (width 4 `17194762049552837459`, width 8 `3326155864557964828`).

An end-to-end `--prompt-lookup` run of the decode command (now allowed on Inkling 2, which rewinds a rejected verify block from its own state) printed the same text on both arms (121 rounds, 8 drafted, 5 of 35 proposals accepted). That text agrees with plain greedy decode for its first 213 characters and then rephrases a clause ("Atmosphere has molecules ... much smaller" against "Atmosphere molecules ... are much smaller"), on both arms alike, so the difference comes from the multi-token verify forward, not from the kernels. With so few proposals accepted on this prompt, prompt-lookup decode (27.36 tok/s A, 31.67 tok/s B) is slower than plain decode.

## Raw logs

The per-run logs were kept outside the repository; every value above is copied from them.
