# Raon-1 mini NVFP4 exports on M3 Ultra (2026-11-26)

Issue #3437. Compares the original checkpoint (`KRAFTON/Raon-1-mini`, bf16 linears with MXFP8 experts) against the two NVFP4 exports: the vendor MLX export (`KRAFTON/Raon-1-mini-mlx`, revision `3804c114`) and the NVIDIA ModelOpt export (`KRAFTON/Raon-1-mini-nvfp4`, revision `b485e3af`).

## Setup

- Host: Apple M3 Ultra, 512 GB, macOS, Metal backend.
- Build: `cargo build --release --features metal,accelerate`, branch `feature/issue-3437-raon-nvfp4-exports` (implementation commit `8c8f3906a`).
- Checkpoints on disk: original 30 GB, MLX export 16 GB (four model shards; the optional 198 MB `head_stage1_*` file and `dflash2/` are not loaded), ModelOpt export 16 GB.
- Every GPU command ran under the host GPU lock. The benchmark ran in one lock hold with `scripts/with_indexers_paused.sh`, interleaved original, MLX, ModelOpt, repeated three times (ABAB).
- `uptime` at start: `20:24 up 4 days, 22:19, load averages: 4.36 5.24 6.08`; at end: `20:29 up 4 days, 22:24, load averages: 6.60 5.66 5.98`.

## Teacher-forced traces against the original

`examples/logit_trace MODEL TEXT CHUNK 1 8` over one forward of 512 and of 4096 tokens (text: the repository README, `docs/architecture.md` and `docs/decisions.md` concatenated). The reference is mlxcel's own trace of the original checkpoint, which matches the Transformers reference (#3225, #3435). Default reasoning effort (`high`) in every arm. Compared with `scripts/compare_logit_traces.py REF CAND --decided 1`.

| Candidate | Tokens | Top-1 agreement | Disagreement on decided positions (gap >= 1) | Perplexity (ref / cand) | Perplexity delta | Logit delta on ref choice p50 / p99 / max |
|---|---|---|---|---|---|---|
| MLX export | 512 | 88.67% (454 / 512) | 2 / 299 (0.67%) | 14.2005 / 14.5533 | +2.48% | 0.25 / 1.56 / 3.75 |
| ModelOpt export | 512 | 86.91% (445 / 512) | 3 / 299 (1.00%) | 14.2005 / 14.5679 | +2.59% | 0.25 / 1.56 / 3.75 |
| MLX export | 4096 | 89.77% (3677 / 4096) | 12 / 2382 (0.50%) | 11.7933 / 12.2214 | +3.63% | 0.25 / 1.75 / 5.25 |
| ModelOpt export | 4096 | 89.92% (3683 / 4096) | 11 / 2382 (0.46%) | 11.7933 / 12.2045 | +3.49% | 0.25 / 1.75 / 5.13 |

Every disagreement is at a position where the original's top-two gap is at most 2.375 (1.125, 1.5, 1.875 and 2.375 in the four rows); above that, both exports agree everywhere. About three quarters of the disagreements put the original's choice at rank 2.

The two exports against each other (MLX as reference): 0 of 291 (512 tokens) and 0 of 2370 (4096 tokens) decided positions disagree, perplexity delta +0.10% and -0.14%, logit delta p99 0.375. They encode the same NVFP4 weights in two layouts, so this is the check that each layout's scales are applied with their own meaning: a per-expert or per-tensor scale read wrong in one of them would separate them far more than the quantization does.

## Load time, resident memory and decode

`mlxcel generate -p "Write a detailed essay about the history of the printing press and its effect on European society." -n 256 --temp 0 --reasoning-effort none`. Load time and resident memory are mlxcel's `Model loaded in ... (resident: ...)` line; tok/s is the `[Generated 256 tokens in ...]` line (256 generated tokens over a short prompt, so it is decode-dominated).

| Model | Load time (s), runs 1 / 2 / 3 | Resident (GB) | tok/s, runs 1 / 2 / 3 | tok/s mean |
|---|---|---|---|---|
| Original | 1.38 / 1.37 / 1.51 | 30.56 | 33.49 / 33.43 / 33.12 | 33.35 |
| MLX export | 0.69 / 0.68 / 0.73 | 16.34 | 39.93 / 39.98 / 39.72 | 39.88 |
| ModelOpt export | 67.4 / 70.2 / 72.2 | 16.39 | 38.89 / 38.57 / 38.82 | 38.76 |

Both exports take 53.5% of the original's memory and decode 16 to 20% faster. The load times are warm page-cache times. The ModelOpt export's load runs the shared ModelOpt NVFP4 repack, which reads each of its 9538 NVFP4 triplets to host and transcodes its E4M3 scales on the CPU; the MLX export is stored in MLX's layout and loads lazily. Speeding up that repack is shared with every ModelOpt NVFP4 family and is not part of #3437.

## Generation checks

- `mlxcel generate -m <export> -p "What is 17 + 25? Reply with only the number." -n 64 --temp 0 --reasoning-effort none` prints `42` and stops on both exports.
- The same prompt at the default effort with `--show-reasoning`, both exports: thinking `We need answer simple arithmetic 17+25=42. User asks reply only number. Final 42.`, answer `42` (24 tokens).

## Vendor runtime comparison (MLX export)

Vendor: `raon_mlx_model.py generate --no-dflash --temp 0 --presence-penalty 0 --max-tokens 64` with `RAON_EXACT_KERNELS=0`, mlx 0.32.4, mlx-lm 0.32.2, transformers 5.20.1 (the export's `requirements.txt`). mlxcel: `generate -n 64 --temp 0`. Prompt: `Explain why the sky is blue in a short paragraph.`

- Default effort (thinking on): the 64 tokens are identical (thinking `We need respond to user: Explain why sky is blue in short paragraph. Simple. Final concise.`, then the answer through `...absorbed by the upper atmosphere,`).
- Thinking off (vendor `--no-thinking`, mlxcel `--reasoning-effort none`, both the `none` alpha): the first 49 of 60 tokens are identical. Token 50 is `so we` (vendor) against `so the` (mlxcel), after `...absorbed by the upper atmosphere,`. A teacher-forced mlxcel trace over the vendor's text puts the two at 34.25 and 34.0 at that position: one bf16 step apart at that magnitude, and that wider forward agrees with the vendor. The divergence is a tie broken by rounding in the one-token decode step, not a layout error; both continuations end the paragraph with Rayleigh scattering.
