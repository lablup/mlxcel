# Rope-folded MLA prefill for the remaining hand-masked decoders (issue #3099), M5 Max, 2026-11-15

Issue #3099: after #3083 moved the DeepSeek V3 family onto `mla::rope_folded_attention`, four MLA decoders still built an f32 `[L, L + offset]` causal mask per forward and either added it to a dense `[B, H, L, S]` rope-score tensor handed to SDPA as an additive mask (GLM4MoeLite and its pipeline stage, LongcatFlashNgram) or computed the nope, rope, summed and softmax score tensors by hand (KimiLinear's MLA layers, KimiK3). Every multi-token step of these four now goes through `rope_folded_attention` with MLX's causal mode and no mask array, in both the absorbed and the materializing branch. Single-token decode keeps its old form.

Host: Apple M5 Max, 128 GB, macOS 27. Main is `00cd2d69`; the branch is that commit plus this change; both release builds. Greedy, 48 tokens, reasoning included in the compared text. Peak is the `/usr/bin/time -l` footprint, prefill is `--profile`. Every run follows a 60 s idle period under the shared GPU lock. Rep 1 ran main before branch; rep 2 ran branch before main (the rep 2 main cells and three branch cells ran after a host pause). Raw rows: `mla-prefill-rope-fold-remaining-m5max-2026-11-15/runs.csv`, identity per cell: `output-identity.csv`, trace summary: `logit-trace.csv`.

## Results

| model | prefill | peak main | peak branch | prefill main (rep 1 / rep 2) | prefill branch (rep 1 / rep 2) | greedy output |
|---|---|---|---|---|---|---|
| glm-4.7-flash-4bit (22754 tokens, 28K prompt) | 2048 chunks | 28.82 GB | **20.51 GB** | 49.71 / 35.22 s | 12.56 / 12.30 s | identical |
| glm-4.7-flash-4bit | single pass | fails (82.76 GB at the failure) | **24.11 GB** | Metal GPU timeout / timeout | 11.25 / 15.84 s | main produces none |
| kimi-k3-2l-mla-synth (2831 tokens, 4K prompt) | 2048 chunks | 41.37 / 41.51 GB | **39.69 GB** | 0.74 / 0.47 s | 0.44 / 0.50 s | identical |
| kimi-k3-2l-mla-synth | single pass | 41.51 GB | **39.69 GB** | 0.51 / 0.50 s | 0.43 / 0.47 s | identical |

On GLM-4.7-Flash the old path built the f32 rope-score tensor `[20, L, S]` in all 47 layers before SDPA ran. At 2048 chunks the branch prefills 2.8 to 4.0 times faster and peaks 8.3 GB lower. In a single pass the old tensor is `[20, 22754, 22754]` per layer, and main fails both reps with `kIOGPUCommandBufferCallbackErrorTimeout` after 10 to 13 s. The branch completes in 11 to 16 s at 24.1 GB.

`kimi-k3-2l-mla-synth` is a two-layer truncation (one KDA layer, one MLA layer at 96 heads) whose output is degenerate text. It shows the MLA layer's change in isolation: about 1.8 GB less peak and identical output. A 4K prompt was used because the old manual score path at 96 heads needs four `[96, L, S]` tensors per step, too much at 28K on a host shared with other jobs.

Families measured by unit tests only, with no local checkpoint: KimiLinear (`kimi_linear`) and LongcatFlashNgram (`longcat_flash`, `longcat_flash_ngram`). The GLM4MoeLite pipeline stage is also checked only by unit tests: a two-stage split against the single-process model, and the #3081 last-row check.

## Teacher-forced logit trace (GLM-4.7-Flash)

The fold changes the order in which the nope and rope halves of the score are summed, and moves the score from a bf16 rope matmul plus an f32 mask into SDPA's own accumulation. Greedy output is identical at 2048 chunks. The single-pass output cannot be compared, because main produces none. `examples/logit_trace` from each arm over `docs/architecture.md`, `docs/benchmarks.md`, `docs/adding-models.md`, `docs/code-guidelines.md`, compared with `scripts/compare_logit_traces.py main.tsv branch.tsv --decided 2`:

| shape | positions | top-1 disagreement | on decided positions (gap >= 2) | largest gap at a disagreement | reference choice at candidate rank 2 | perplexity main / branch |
|---|---|---|---|---|---|---|
| width 2048, cold (`2048 4 8 0`) | 8192 | 495 (6.04%) | 2 / 3335 (0.060%) | 2.50 | 81.4% | 19.5353 / 19.5808 |
| width 4096 behind 2048 context (`4096 2 8 2048`) | 8192 | 474 (5.79%) | 3 / 3360 (0.089%) | 5.75 | 82.5% | 15.8950 / 15.9023 |

This checkpoint's logits are bf16 near 100 to 130, where one ulp is 0.5. The median logit delta on the reference's choice is exactly 0.5, so one ulp. Most disagreements are near-ties: 477 of 495 (cold) and 457 of 474 (behind context) sit where the reference's top two were within 1.0. Per-chunk mean NLL moves in both directions: on the cold trace, chunk 0 has 2.6896 on main and 2.6870 on the branch, and chunk 3 has 2.8239 and 2.8294. The largest single-position NLL changes also go both ways. Perplexity moves by +0.23% (cold) and +0.05% (behind context). This is the reassociation class. It is wider than the Kimi-VL trace in #3083 because the logits here are bf16 at a larger magnitude.

## Reproduce

```bash
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 28000 > prompt_28000.txt
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 4000 > prompt_4000.txt
MLXCEL_PREFILL_CHUNK=2048 /usr/bin/time -l ./target/release/mlxcel generate \
  -m models/glm-4.7-flash-4bit -p "$(cat prompt_28000.txt)" -n 48 --temp 0 --profile --show-reasoning
# MLXCEL_PREFILL_CHUNK=0 for single pass; models/kimi-k3-2l-mla-synth with prompt_4000.txt.
./target/release/examples/logit_trace models/glm-4.7-flash-4bit corpus.txt 2048 4 8 0 > arm.tsv
./target/release/examples/logit_trace models/glm-4.7-flash-4bit corpus.txt 4096 2 8 2048 > arm-ctx.tsv
python3 scripts/compare_logit_traces.py main.tsv branch.tsv --decided 2
```
