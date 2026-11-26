# Inkling 2 Mini 4-bit traces against mlx-vlm on M3 Ultra (#3378)

Teacher-forced traces of `mlx-community/Inkling-2-Mini-4bit` (9c0e2960) against mlx-vlm 466069f5, loaded through the committed remap `scripts/inkling2_mlx_community_remap.py` (`--inkling2-mlx-community` on both trace scripts). #3367 ran the 4-bit text trace through an uncommitted rename-only script and no 4-bit image trace.

## Environment

| Item | Value |
|---|---|
| Host | Apple M3 Ultra, 512 GB, macOS, shared with other jobs; every run under the GPU lock |
| mlxcel | `cargo build --profile test-fast --features metal,accelerate --bin mlxcel --example logit_trace` on `fd879f20` plus this PR |
| Reference | mlx-vlm 466069f5 (`uv run --python 3.12 --with-editable references/mlx-vlm`), remapped export in a temporary directory |
| Compare | `scripts/compare_logit_traces.py REF.tsv MLXCEL.tsv --decided 2` |

## Image trace

| Item | Value |
|---|---|
| Image | `tests/fixtures/sapiens2/person.jpg` (1280 x 720, the image #3367 used) |
| Prompt | `What is in this image?` |
| Continuation | the first 128 tokens of mlxcel's own greedy reply (`generate --image ... -n 128 --temp 0 --show-reasoning`), 127 scored rows |
| mlxcel arm | `MLXCEL_TRACE_IMAGE=... MLXCEL_TRACE_IMAGE_PROMPT="What is in this image?" logit_trace models/inkling-2-mini-4bit reply.txt 128 1 8` |
| Reference arm | `scripts/mlx_vlm_image_logit_trace.py models/inkling-2-mini-4bit IMAGE mlxcel.tsv --prompt "What is in this image?" --inkling2-mlx-community` |

| Check | Result |
|---|---|
| Prompt ids | 600 tokens, identical in both arms (no `--mlxcel-prompt` needed) |
| Top-1 disagreements | 7 / 127 (5.5%), all at positions where the reference top-two gap was under 1.4 |
| Decided positions (gap >= 2.0) | 0 / 70 disagreements |
| Largest gap at a disagreement | 1.371 |
| Perplexity | reference 1.9273, mlxcel 1.9074 (-1.03%) |

## Text trace through the committed remap

Corpus: bytes 4000 to 12000 of `docs/supported-models.md` as it stood on `fd879f20` (the #3367 corpus file was not recorded), `logit_trace ... 256 4 8`, 4 x 256 = 1024 positions.

| Check | Result |
|---|---|
| Top-1 disagreements | 33 / 1024 (3.2%), the largest reference gap at a disagreement 0.932 |
| Decided positions (gap >= 2.0) | 0 / 372 disagreements |
| Perplexity | reference 9.7653, mlxcel 9.7344 (-0.32%) |

#3367 recorded 0 / 494 decided and -0.17% on its own corpus. The decided count differs only because the corpus differs, and the result, no disagreement at a decided position, is the same.

## What the remap does

It renames the export's tensors to mlx-vlm's parameter names, splits the fused `gate_up_proj` planes with mlx-vlm's `_split_gate_up`, sets `weight_scale_2 = global_scale / 2688`, and declares each expert projection as NVFP4 (layers 0-19) or MXFP8 (layers 20-23) in `config.json`, inferred from the packed and scale widths. Dense tensors are not touched. The export has no `processor_config.json`, so the processor files come from `models/inkling-2-mini`. Writing the 17 GB remapped copy took a couple of minutes and later runs reuse it.
