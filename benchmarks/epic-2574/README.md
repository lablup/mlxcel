# Epic #2574: batched decode measurements

Measured 2026-10-27 for [epic #2574](https://github.com/lablup/mlxcel/issues/2574), delivered through PRs #2670, #2672, #2673, #2678, #2686 and #2685. Final integrated workspace verification at `5bc57887e01fb9c49d8c85409730710f933241bf` passed with 14,251 tests, zero failures and 457 ignored tests (10,623 main-library tests passed). Workspace Clippy, formatting, 18 benchmark-tool CPU tests and the 792-artifact raw manifest also pass. A test-only device-lock correction resolved two parallel Metal test races; measured runtime paths are unchanged.

## Protocol and provenance

Apple Mac Studio, M3 Ultra, 32 CPU cores, 512 GB unified memory, macOS 26.6.2; release `metal,accelerate`. Serving uses `--parallel 4`, concurrency 1/2/4, a nominal ~512-token textual prompt requested by `--prompt-tokens 512`, and a 128-token generation cap. Actual tokenized prompt counts vary by tokenizer (Nemotron logged 448). Aggregate tokens/s includes serving overhead and differs from the separately recorded decode tokens/s. Client concurrency does not guarantee that every scheduler tick has that width.

Greedy parity uses `Hello`, temperature zero and identical `--show-reasoning` display flags, excluding timings/status text. Traces record four chunks of 64 teacher-forced positions, top-k 8, with maximum prefill/history budget 512. This is not a fixed 512-token context: chunks start at corpus position zero and history grows. “Decided” means reference top-two logit gap >=2. Zero decided mismatches does not imply identical logits, identical free-running text or general model quality.

Most baseline model code is `a29a2692`; per-model candidate revisions are in JSON. Nemotron uses baseline `b4cdf023` and candidate `1249c337` because main PR #2679 changed Metal sorted prefill and both arms needed that shared change. The other four Mamba2-family measurements use `00e33351bf5bb1ac63f9d0b6ce22f3845caf272a`, whose corresponding runtime remained unchanged by that merge. Old model-owned baseline traces use the trace-only scalar-reference patch to preserve sequence IDs; invalid original traces are excluded.

## Text-model results

Each throughput cell is concurrency **1 / 2 / 4**, aggregate tokens/s. Speedup compares candidate C4 with baseline C4. All 23 configuration pairs (21 checkpoints) have identical greedy text and recorded B1 traces (zero recorded B1 logit delta).

| Checkpoint | Baseline aggregate tok/s | Candidate aggregate tok/s | C4 speedup | B4 top-1 mismatch | Decided mismatch | Evidence |
|---|---|---|---:|---:|---:|---|
| GPT-OSS 20B MXFP4-Q4 | 109.8 / 110.1 / 110.1 | 108.3 / 155.5 / 209.0 | 1.90x | 8/256 | 0/96 | [JSON](gpt-oss-20b.json), [#2670](https://github.com/lablup/mlxcel/pull/2670) |
| ExaOne 4.0 1.2B 4-bit | 226.2 / 226.0 / 226.1 | 219.0 / 332.3 / 434.0 | 1.92x | 2/256 | 0/100 | [JSON](mixed-cache-families.json), [#2672](https://github.com/lablup/mlxcel/pull/2672) |
| Ministral 3 3B 4-bit, text view | 121.4 / 119.0 / 115.5 | 120.9 / 168.5 / 203.3 | 1.76x | 3/256 | 0/114 | [JSON](mixed-cache-families.json), [#2672](https://github.com/lablup/mlxcel/pull/2672) |
| Laguna XS.2 4-bit | 93.8 / 94.4 / 94.1 | 89.8 / 119.2 / 170.6 | 1.81x | 21/256 | 0/123 | [JSON](mixed-cache-families.json), [#2672](https://github.com/lablup/mlxcel/pull/2672) |
| Mellum2 12B A2.5B Base | 23.0 / 51.0 / 52.3 | 33.2 / 69.3 / 81.8 | 1.56x | 0/256 | 0/184 | [JSON](mixed-cache-families.json), [#2672](https://github.com/lablup/mlxcel/pull/2672) |
| Trinity Nano Preview 4-bit | 88.1 / 98.4 / 98.9 | 96.6 / 158.4 / 202.3 | 2.05x | 91/256 | 0/16 | [JSON](model-owned-attention.json), [#2673](https://github.com/lablup/mlxcel/pull/2673) |
| Inkling 0.6B | 152.9 / 164.8 / 177.4 | 158.1 / 200.6 / 340.9 | 1.92x | 0/256 | 0/96 | [JSON](model-owned-attention.json), [#2673](https://github.com/lablup/mlxcel/pull/2673) |
| DeepSeek-V4 Flash 4-bit | 6.8 / 7.3 / 6.4 | 7.9 / 11.5 / 15.4 | 2.41x | 5/256 | 0/129 | [JSON](model-owned-attention.json), [#2673](https://github.com/lablup/mlxcel/pull/2673) |
| Granite 4.0-H Tiny 4-bit | 97.3 / 109.1 / 129.9 | 120.0 / 173.3 / 260.2 | 2.00x | 10/256 | 0/105 | [JSON](mamba2-families.json), [#2678](https://github.com/lablup/mlxcel/pull/2678) |
| Nemotron 3 Nano 30B A3B 4-bit | 94.0 / 71.0 / 91.8 | 88.6 / 211.4 / 313.5 | 3.42x | 6/256 | 0/113 | [JSON](mamba2-families.json), [#2678](https://github.com/lablup/mlxcel/pull/2678) |
| Mamba2 1.3B 4-bit | 85.0 / 92.3 / 100.0 | 98.2 / 157.9 / 283.4 | 2.83x | 1/256 | 0/109 | [JSON](mamba2-families.json), [#2678](https://github.com/lablup/mlxcel/pull/2678) |
| Falcon-H1 Tiny 90M 4-bit | 180.8 / 194.4 / 165.5 | 208.0 / 324.7 / 567.2 | 3.43x | 4/256 | 0/92 | [JSON](mamba2-families.json), [#2678](https://github.com/lablup/mlxcel/pull/2678) |
| PLaMo2 1B | 88.3 / 95.2 / 103.8 | 99.3 / 129.2 / 273.5 | 2.63x | 0/256 | 0/97 | [JSON](mamba2-families.json), [#2678](https://github.com/lablup/mlxcel/pull/2678) |
| Jamba Reasoning 3B 4-bit | 128.6 / 167.0 / 167.4 | 144.5 / 290.3 / 475.9 | 2.84x | 1/256 | 0/136 | [JSON](recurrent-families.json), [#2685](https://github.com/lablup/mlxcel/pull/2685) |
| Falcon Mamba 7B 4-bit | 38.8 / 37.5 / 78.8 | 38.8 / 45.8 / 195.5 | 2.48x | 2/256 | 0/105 | [JSON](recurrent-families.json), [#2685](https://github.com/lablup/mlxcel/pull/2685) |
| LFM2 350M 8-bit | 399.6 / 405.9 / 434.1 | 430.0 / 712.1 / 1417.4 | 3.27x | 4/256 | 0/94 | [JSON](recurrent-families.json), [#2685](https://github.com/lablup/mlxcel/pull/2685) |
| LFM2 8B A1B 4-bit | 88.3 / 109.9 / 202.7 | 111.9 / 137.5 / 450.3 | 2.22x | 0/256 | 0/117 | [JSON](recurrent-families.json), [#2685](https://github.com/lablup/mlxcel/pull/2685) |
| Qwen3-Next 80B A3B 4-bit | 69.7 / 66.6 / 80.6 | 71.1 / 103.3 / 148.1 | 1.84x | 8/256 | 0/126 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |
| Ring-mini Linear 2.0 4-bit | 140.0 / 179.2 / 177.8 | 159.6 / 298.8 / 460.8 | 2.59x | 22/256 | 1/118 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |
| Qwen3.5 4B 4-bit, default | 109.5 / 188.8 / 168.6 | 112.6 / 225.3 / 357.9 | 2.12x | 2/256 | 0/117 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |
| Qwen3.5 35B A3B 4-bit, default | 87.4 / 138.8 / 122.5 | 89.3 / 165.7 / 262.9 | 2.15x | 3/256 | 0/113 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |
| Qwen3.5 4B 4-bit, dense | 110.9 / 191.8 / 170.3 | 112.7 / 225.3 / 361.8 | 2.12x | 2/256 | 0/117 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |
| Qwen3.5 35B A3B 4-bit, dense | 88.7 / 146.2 / 123.9 | 89.5 / 166.5 / 262.9 | 2.12x | 3/256 | 0/113 | [JSON](linear-attention.json), [#2686](https://github.com/lablup/mlxcel/pull/2686) |

Five warm B1 repetitions gave median decode tok/s 136.1→135.4 for GPT-OSS, 272.3→271.0 for ExaOne, 109.1→109.9 for Laguna, and 140.3→138.5 for the fair Nemotron pair (-1.3%). Observed ranges overlap; these small samples do not establish statistical equivalence. Nemotron repetitions finished at 09:41:27 UTC, before focused GPU tests began at 09:42:45 UTC.

## Limitations and VLM behavior

- Ministral's original VLM configuration selects a different tower. The measured text view reuses unchanged weights/tokenizer files, removes `vision_config`, and sets `architectures` to `["Mistral3ForCausalLM"]`, retaining nested `text_config` and the outer model type.
- Mellum's baseline architecture-document trace produced non-finite NLL. The reported default-precision comparison uses the same fixed Rust corpus from `git show a29a2692:src/models/mellum.rs` in both arms. A BF16 diagnostic passed but is not the measured production setting; the original corpus precision issue remains unresolved. The alternate corpus affects traces, not the serving benchmark prompt workload.
- Trinity's final native path retains the complete fused expert path per row when eligible. It still has 91/256 top-1 disagreements despite 0/16 decided disagreements. B1/B4 PPL is 69864.6970→69357.6777 (-0.726%); its already degenerate baseline text and high PPL preclude a general quality claim. The superseded whole-batch expert path had +15.440% PPL and is excluded from the final table. Inkling also has pre-existing high corpus PPL (~1.876 million).
- Final Nemotron Nano Omni matches all three solo/concurrent free-form requests (two text plus one image) and reports native batching without a clamp. Final Granite Vision also matches all three original requests with **per-row wrapper fallback**, not native wrapper decode; its text tower remains native and the server reports no clamp. Both were replayed successfully at `098b569a9aac3eac151d46f721a5b88cb1964478`, after main `cfa14e7b` changed vision loading. [Granite evidence](raw/vlm/granite/integrated-fallback/manifest.json) and [Nemotron evidence](raw/vlm/nemotron/integrated-native/manifest.json) include source/command provenance, route excerpts, and the observed server-binary and embedded Metal-library hashes. Earlier replay and diagnostic records retain their original source identities.
- LFM2-VL 450M matches all three solo/concurrent full messages, including reasoning fields, on both baseline and candidate. The [recurrent evidence](recurrent-families.json) identifies measured source `2bd2eff0` and subsequent non-ID wrapper/test fixes; exact requests, input image, driver hashes and response bodies are archived under [raw/vlm/lfm2](raw/vlm/lfm2/).
- Ring-mini has one decided-position mismatch out of 118 (22/256 total top-1 disagreements; reference gap 2.312), with PPL -2.052%. A separately archived `MLXCEL_FUSED_MOE=0` diagnostic has 0/114 decided mismatches, consistent with different quantized MoE arithmetic across batch widths. The production result remains 1/118; neither diagnostic nor PPL change establishes quality equivalence.
- Qwen3.5 VLM full solo/concurrent messages match 2/3 in both baseline and candidate; counting-prompt reasoning drifts in both versions. Across revisions, solo messages match 3/3 and concurrent messages match 2/3. Image-conditioned B1 is exact across 64 rows; B4 has 2/64 top-1 and 0/15 decided disagreements (PPL -0.898%). See [linear-attention evidence](linear-attention.json) for original responses, diagnostic inputs and source hashes.
- Earlier Granite native-wrapper diagnostics are retained: exact messages `[false, false, true]`; constrained prompts 3/3 exact; image-conditioned trace 4/64 top-1 mismatches, 0/11 decided, PPL +0.525%. These explain the fallback and are not final fallback measurements.
- GPT-OSS 120B, Step3p5/Step3p7, IQuestLoopCoder and Inkling's image path lack real-checkpoint validation in this record. Synthetic/delegation coverage does not substitute for those measurements. RecurrentGemma and RWKV7 have synthetic coverage only because suitable local checkpoints were unavailable. Kimi Linear and Kimi K3 also have synthetic coverage only.
- Nemotron's initial HTTP-500 serving run is excluded. A benchmark process exit code of zero alone is insufficient: the evidence tools require successful C1/C2/C4 rows, zero failed requests and no request-error log.

## Reproduction and durable evidence

Build each revision separately and retain its source identity. When copying binaries, also preserve the matching `mlx.metallib` alongside the executable (including the trace example); a binary may otherwise depend on the original build worktree. The runner requires `--backend metal --loaded-metallib /exact/loaded/mlx.metallib` and records that single library's SHA-256; identify the library actually loaded by all measured binaries, not possible search candidates. `--backend cpu`, `--backend cuda` or `--backend rocm` requires no Metal library. The backend option records provenance and does not select the binaries' runtime device. Use the same local checkpoint, corpus, precision and display settings; run GPU measurements sequentially.

```bash
cargo build --release --features metal,accelerate --bin mlxcel --bin mlxcel-server --example logit_trace
python3 scripts/bench_batched_decode.py --repo "$PWD" --binary-dir "$PWD/target/release" \
  --model "$MODEL" --corpus "$CORPUS" --output "$OUTPUT" --source-revision "$REVISION" \
  --backend metal --loaded-metallib "$LOADED_METALLIB" --rows 1,4
# Equivalent individual measurement commands:
./target/release/mlxcel generate -m "$MODEL" -p Hello -n 128 --temp 0 --show-reasoning
MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=0 ./target/release/examples/logit_trace "$MODEL" "$CORPUS" 64 4 8 512 > trace-b1.tsv
MLXCEL_TRACE_BATCH_ROWS=4 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=0 ./target/release/examples/logit_trace "$MODEL" "$CORPUS" 64 4 8 512 > trace-b4.tsv
python3 scripts/compare_logit_traces.py trace-b1.tsv trace-b4.tsv
./target/release/mlxcel-server -m "$MODEL" --parallel 4 --port 18574 --offline --metrics
# Once /health is ready, from another terminal:
python3 scripts/bench_serving_concurrency.py --port 18574 --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128 --metrics
```

For the older model-owned baseline only, use the instrumentation-patched trace binary with B1 and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`. Never silently change the history window or use dropped-ID baseline traces. The runner explicitly controls the scalar-reference flag and records relevant non-sensitive environment switches.

The [raw evidence manifest](raw/manifest.json) indexes selected outputs in `raw/<checkpoint>/<baseline|candidate>/`: trace TSV content, phase stdout/stderr and status/commands, generated text, ladder, warm repeats and short server-route excerpts. Archive mappings must explicitly declare `expected_trace_rows`: `[1]` for these baseline runs and `[1, 4]` for candidates. A missing expected B4 trace or phase stderr is rejected rather than silently omitted. Manifests record source revisions and file/config/corpus hashes. The selected text runs retain bounded plain-text outputs; plain text is small enough for direct version control. VLM solo/concurrent JSON and explicitly labeled native diagnostics should accompany them. Do not commit weights, binaries, credentials, complete environment dumps or oversized loader logs. Temporary paths are not durable evidence links.

Bilingual implementation reports are under `TECHNICAL_REPORTS/` with prefixes `2670-`, `2672-`, `2673-`, `2678-`, `2685-`, and `2686-`. The [benchmark runner](../../scripts/bench_batched_decode.py) captures local runs; the [archive helper](../../scripts/archive_batched_decode.py) validates and copies explicitly selected runs.

The [VLM replay runner](../../scripts/bench_batched_decode_vlm.py) captures two text requests and one image request, first alone and then concurrently, preserving full messages and exact reproducible inputs. Run its `--help` for required source, binary, checkpoint, output and backend provenance arguments.
