# Raon-1 mini serving performance follow-ups (M3 Ultra, 2026-11-26)

Issue #3458. Six follow-ups from the Raon-1 mini PRs, measured against `main` before them. Three are kept (prompt-cache reuse, the decode fast paths, the ModelOpt NVFP4 load), one is a validation (DFlash2 on the NVFP4 exports), and two were measured and not kept (one host read per sampled verify block, batched DFlash2), as was one of the two decode steps the issue proposed (fused RDA projections).

## Setup

- Host: Apple M3 Ultra, 512 GB, macOS, Metal backend.
- Build: `cargo build --release --features metal,accelerate`. Base is `main` at `477564e2e` with only the ModelOpt repack phase timers added; new is this branch at `c8eba6a90`, the same base plus the changes. The final trace and test runs were repeated after merging `main` at `d43915faf` (the #3457 correctness follow-ups), which this document says where it applies.
- Checkpoints: `KRAFTON/Raon-1-mini` (bf16 + MXFP8, `6691a076`), `KRAFTON/Raon-1-mini-mlx` (`3804c114`), `KRAFTON/Raon-1-mini-nvfp4` (`b485e3af`), `KRAFTON/Raon-1-mini-dflash2`.
- Every GPU command ran under the host GPU lock, one lock hold per table, arms interleaved inside it (base, new, base, new, or plain, width 3, 5, 7, repeated). Decode and load tables also ran under `scripts/with_indexers_paused.sh`. The runs started late on 2026-11-26 and ended at 01:30 on 2026-11-27 (KST); each table gives its own `uptime`. Another job was building on the host for part of the night, which the load averages show.
- Unless a table says otherwise: temperature 0, 256 generated tokens, chat template on, default reasoning effort.
- After the merge with `main` at `d43915faf` the tables were spot-checked, not rerun (01:42 to 01:45): decode with the fast paths off is 33.5 tok/s with the base's text and 54.2 tok/s with them on, drafting without `--draft-block-size` binds at width 3 with plain decode's text, the server text check of section 2 again finds 0 of 36 texts differing (the server now applies the checkpoint's presence penalty of 1.5 by default, #3457), and the TTFT hits repeat (0.04 s on a repeat, 0.08 and 0.17 s on the follow-up turns).

## 1. Single-stream decode

`mlxcel generate -m models/raon-1-mini -p "Write a detailed essay about the history of the printing press and its effect on European society." -n 256 --temp 0`, tok/s from the `[Generated ...]` line. `MLXCEL_RAON_DECODE_FUSION` selects the fast paths of the new binary. Start 01:00:16, load averages 6.04 8.85 11.75; end 01:02:30, load averages 2.02 6.34 10.32.

| Arm | Round 1 | Round 2 | Round 3 | Mean | Against base |
|---|---|---|---|---|---|
| Base (`477564e2e`) | 33.52 | 33.53 | 33.50 | 33.52 | |
| New, `off` (reference paths) | 33.50 | 33.55 | 33.51 | 33.52 | 1.00x |
| New, `mhc` (compiled site graphs) | 34.51 | 34.52 | 34.51 | 34.51 | 1.03x |
| New, `selector` (row-major one-row product) | 51.53 | 51.42 | 51.50 | 51.48 | 1.54x |
| New, default (both) | 54.04 | 54.22 | 54.21 | 54.16 | 1.62x |

The issue's target was 38 tok/s. The generated text is identical to the base in every arm and round on this prompt.

### Where the time was

The step was assumed to be bound by the number of small ops it dispatches. Timing a decode step phase by phase on the base code (an evaluation after each phase, one row against two rows on the same code path; taken while the host was busy, so read the ratios) said otherwise: the mHC route and aggregate of a site took 0.64 ms with one row and 0.30 ms with two, and a whole step took 31.4 ms with one row and 22.7 ms with two. The one-row case was slower because of one product.

Each site multiplies the flattened streams (`[1, 10240]`) by the selector plane. The plane was stored transposed (`[10240, 17]`), so with one row MLX ran its transposed gemv kernel. Microbenchmark of that product alone (bf16, 300 calls each, an input op and an evaluation included in every figure):

| Form | us per call | Less the 142 us of the input op and evaluation |
|---|---|---|
| `x @ selector_t`, transposed plane (before) | 397 | 255 |
| `x @ selector.T`, lazy transpose | 214 | 72 |
| `selector @ x^T`, row-major (now) | 175 | 33 |
| two identical rows through gemm | 194 | 52 |

At 50 sites per token that is about 11 ms of a 30 ms step. The row-major form is as accurate as the form it replaces: against the exact product of the bf16 inputs, over 200 random vectors, both have a mean error of 0.249 ulp and a maximum of 0.5 ulp (the old form misses correct rounding in 0.03% of outputs, the new one in none). The two sum in different orders, though, so they can differ in the last bf16 bit.

### Steps kept and dropped

- **Selector product, kept.** A product of exactly one row (a decode step, and each row of a DFlash2 verify block, which is formed per row) takes the row-major form. Products of several rows (prefill, batched decode) keep the plain matmul.
- **Compiled mHC site graphs, kept.** The route, aggregate and mix of a single-position step run as two compiled graphs that replay the eager ops in the same order. Bit-identical: a unit test compares them on random inputs in bf16 and f32 at the real width, and the teacher-forced decode trace below is byte-identical with and without them.
- **Fused RDA projections, dropped.** One product for the seven projections that read the layer input measured 33.57 against 33.38 tok/s alone (+0.6%) and 54.29 against 53.85 on top of the other two (+0.8%), three rounds each (00:23:50 to 00:27:01, load averages 11.17 8.36 9.75 to 6.59 7.04 8.86). It is not bit-identical at the real width: MLX picks another kernel configuration for a `[20640, 2560]` plane than for a `[5120, 2560]` one, and 1008 of 1024 prefill rows and 984 of 1024 decode rows of the trace changed at least one printed logit (decided-position agreement stayed 0 of 578). Under 1% was not worth moving the Transformers-validated prefill forward, so the projections stay separate.

### Correctness

Teacher-forced traces of `examples/logit_trace` on the #3441 corpus (4 x 256 tokens, `MLXCEL_TRACE_REASONING_EFFORT=max`) against the Transformers reference trace of #3441 (`scripts/raon1_mini_oracle_trace.py trace`, bf16, CPU), compared with `scripts/compare_logit_traces.py --decided 1`. The prefill trace is the #3441 bar; the decode trace (`MLXCEL_TRACE_BATCH_ROWS=1`, every position a one-token step) covers the path this change touches. Run on the merged tree (`main` at `d43915faf`).

| Trace | Setting | Decided positions disagreeing | Top-1 disagreement | Perplexity delta | Rows differing from `off` |
|---|---|---|---|---|---|
| Prefill | `off` | 0 / 578 | 40 / 1024 | +0.186% | |
| Prefill | default | 0 / 578 | 40 / 1024 | +0.186% | 0 of 1024 |
| Decode | `off` | 0 / 578 | 43 / 1024 | +0.035% | |
| Decode | `mhc` | 0 / 578 | 43 / 1024 | +0.035% | 0 of 1024 |
| Decode | `selector` | 0 / 578 | 31 / 1024 | +0.517% | 978 of 1024 |
| Decode | default | 0 / 578 | 31 / 1024 | +0.517% | 978 of 1024 |

The prefill path is byte-identical with the fast paths on (01:35:53 to 01:38:05; the same figures came out of the branch before the merge). On the decode path the compiled graphs change nothing and the selector form changes last bits, with no decided position flipped. Its top-1 agreement is a little higher and its perplexity delta a little larger than the reference form's. A rounding-only change moves these figures by that much on this corpus size (#3457 measured +0.34% to +0.61% on its long prompt for one), and the product itself is no less accurate (above).

Plain greedy text can therefore differ from the previous build at a near tie. On the essay prompt it did not on the original checkpoint; on both NVFP4 exports (with `--reasoning-effort none`) it diverged after 1189 and 666 characters into an equally plausible continuation. `MLXCEL_RAON_DECODE_FUSION=mhc` keeps the previous build's bytes at 34.5 tok/s. What must hold inside one build still does: drafted text equals plain text (sections 3 and 4) and a prompt-cache hit equals the cold run (section 2).

## 2. Prompt-cache reuse

`mlxcel-server -m models/raon-1-mini`, one streaming chat request (`temperature 0`, `max_tokens 32`, `enable_thinking false`): a system line plus the first 9000 characters of `README.md` and a summary instruction, 2132 prompt tokens. The same request four times, then two follow-up turns that append the reply and a new question. TTFT is the time to the first content chunk. Base server, new server, base, new, each freshly started. Start 01:13:01, load averages 2.33 2.67 5.98; end 01:13:58, load averages 4.31 3.19 5.95 (a release build was running).

| Request | Base TTFT (s), run 1 / 2 | Base cached tokens | New TTFT (s), run 1 / 2 | New cached tokens |
|---|---|---|---|---|
| Identical request 1 (cold server) | 1.178 / 1.372 | 0 | 1.717 / 1.408 | 0 |
| Identical request 2 | 1.002 / 1.004 | 0 | 1.140 / 1.141 | 0 |
| Identical request 3 | 1.002 / 1.004 | 0 | 0.039 / 0.039 | 2126 |
| Identical request 4 | 1.002 / 1.002 | 0 | 0.038 / 0.038 | 2126 |
| Turn 2 (2183 tokens) | 1.038 / 1.039 | 0 | 0.076 / 0.078 | 2169 |
| Turn 3 (2236 tokens) | 1.057 / 1.059 | 0 | 0.172 / 0.174 | 2220 |

Every reply is the same text on both servers, hit or miss.

- A hit prefills only the tokens after the stored prefix: TTFT falls from 1.0 s to 0.04 s on a repeat and to 0.08 to 0.17 s on a follow-up turn.
- The second identical request misses. After the first reply the scheduler's idle warm-up extends the stored boundary snapshot to the end of the reply, for the next turn, and that entry supersedes the boundary one. A regenerate then finds only the longer entry, which a recurrent state cannot rewind, so it prefills cold and stores the boundary again; no warm-up follows it, and the third request hits. This is the scheduler's policy for every snapshot family, not Raon's.
- A miss costs more than before: 1.14 s against 1.00 s. The prefill is split at the history boundary (the six generation-prompt tokens are a second forward) and the 67 MB state is copied at the boundary.
- A cold Raon chat request on the server therefore prefills in two pieces where earlier builds ran one pass, and a split prefill does not round exactly like one pass (the prompt-cache hit and the cold run of this build agree, since both split at the same place). Cold server text can differ from earlier builds, and from `mlxcel generate` (one pass), when two continuations are a near tie. The 36-text check below compares texts within this build only.

Penalty setting: these server runs (sections 2 and 7, 01:13 to 01:25) used the builds before the merge with `main` at `d43915faf`, which has no checkpoint `presence_penalty` overlay, and no `--presence-penalty` flag is recorded for them, so the server default of 0.0 applied. The 36-text check was repeated after the merge (01:42 to 01:45), with the checkpoint's 1.5 default in force and 0 of 36 texts differing.

Greedy text through the server, new binary, 96 tokens, three request shapes (a 30-token turn, a 1439-token three-message conversation, a code request) at the default and `low` efforts, three attempts each, on a plain server and on a server with `--model-draft models/raon-1-mini-dflash2` (01:23:26 to 01:24:37): 36 texts compared with each request's cold plain text, 0 differ. The plain server answered the third multi-turn attempt from a 1433-token hit; the drafter server ran 19 bursts, declined none and stored no snapshot (a burst keeps its caches outside the sequence, and its drafter's feature history is not snapshotted).

## 3. DFlash2 on the original checkpoint, after the decode change

`mlxcel generate ... --draft-model models/raon-1-mini-dflash2 --draft-kind dflash --draft-block-size N` against plain decode of the same (new) binary, the three prompts of `raon-dflash2-m3ultra-2026-11-26.md`. tok/s, round 1 / round 2; the speedup is the mean of the two rounds over plain. Start 01:14:11, load averages 3.80 3.12 5.89; end 01:17:30, load averages 3.27 3.27 5.38.

| Prompt | Plain | Width 3 | Width 5 | Width 7 |
|---|---|---|---|---|
| Short prose (19 tokens) | 53.91 / 54.06 | 55.34 / 59.76 (1.07x) | 52.10 / 52.67 (0.97x) | 44.11 / 43.99 (0.82x) |
| Code | 54.00 / 53.90 | 65.59 / 66.82 (1.23x) | 69.78 / 69.55 (1.29x) | 63.37 / 63.37 (1.17x) |
| Long prompt (3137 tokens) | 40.68 / 40.74 | 61.70 / 61.78 (1.52x) | 56.78 / 56.73 (1.39x) | 47.65 / 47.65 (1.17x) |

Every drafted run printed the text of plain decode (18 of 18). The startup exactness probe passed at every width.

Drafted decode is faster in absolute terms than before (width 5 on code: 58.96 to 69.67 tok/s) but its lead over plain decode is smaller, because plain decode gained more. A verify block forms every dense product one row at a time, the 260K-row head included, so it costs about 2.4 decode steps at width 5 and 1.9 at width 3 now (45.0 and 34.6 ms a round in these runs against an 18.5 ms step), where it cost 1.8 steps at width 5 before (54.8 ms against 29.8 ms). The draft adds about 2 ms a round. Width 5 fell below plain decode on the low-acceptance prose prompt. Over the nine prompt and checkpoint pairs of this section and the next, width 3 averages 1.18x and width 5 1.11x, so the default width without `--draft-block-size` is now 3 (`raon_dflash2::DEFAULT_VERIFY_WIDTH`).

## 4. DFlash2 on the NVFP4 exports

The same command and prompts on the two exports, new binary. Start 01:17:30, load averages 3.27 3.27 5.38; end 01:23:27, load averages between 3.3 and 7.4 (the other job was building; drafted arms are the more CPU-sensitive ones, and round 2 of the MLX code row shows it).

| Export | Prompt | Plain | Width 3 | Width 5 | Width 7 |
|---|---|---|---|---|---|
| MLX | Short prose | 72.17 / 72.26 | 72.26 / 72.66 (1.00x) | 70.98 / 70.56 (0.98x) | 57.40 / 56.49 (0.79x) |
| MLX | Code | 72.05 / 66.37 | 82.39 / 68.40 (1.09x) | 75.96 / 72.47 (1.07x) | 63.60 / 66.87 (0.94x) |
| MLX | Long prompt | 47.02 / 48.90 | 63.00 / 66.67 (1.35x) | 55.44 / 56.54 (1.17x) | 47.53 / 47.04 (0.99x) |
| ModelOpt | Short prose | 66.92 / 67.10 | 60.20 / 61.54 (0.91x) | 54.96 / 56.06 (0.83x) | 48.84 / 50.22 (0.74x) |
| ModelOpt | Code | 67.82 / 67.94 | 70.16 / 75.07 (1.07x) | 71.62 / 76.29 (1.09x) | 67.23 / 70.71 (1.02x) |
| ModelOpt | Long prompt | 48.19 / 48.27 | 64.05 / 63.86 (1.33x) | 57.14 / 57.09 (1.18x) | 48.45 / 48.88 (1.01x) |

Acceptance rate and tokens emitted per verify (identical in both rounds):

| Export | Prompt | Width 3 | Width 5 | Width 7 |
|---|---|---|---|---|
| MLX | Short prose | 0.578, 2.14 | 0.435, 2.74 | 0.304, 2.80 |
| MLX | Code | 0.762, 2.52 | 0.583, 3.31 | 0.477, 3.81 |
| MLX | Long prompt | 0.618, 2.24 | 0.395, 2.58 | 0.282, 2.68 |
| ModelOpt | Short prose | 0.594, 2.18 | 0.398, 2.58 | 0.299, 2.77 |
| ModelOpt | Code | 0.750, 2.50 | 0.572, 3.27 | 0.458, 3.70 |
| ModelOpt | Long prompt | 0.556, 2.11 | 0.367, 2.45 | 0.272, 2.60 |

- The gate allows drafting on both exports at widths 3, 5 and 7: the probe found every verify row byte-identical to its decode step. NVFP4 `gather_qmm` and the quantized dense products are per row at these widths, so nothing had to be rerouted.
- Greedy text equals plain decode in 36 of 36 drafted runs.
- With plain decode at 67 to 72 tok/s the drafter no longer pays on short prose (0.8x to 1.0x), is about even on code and helps on the long prompt (1.2x to 1.35x).

## 5. ModelOpt NVFP4 load

`mlxcel generate -m models/raon-1-mini-nvfp4 -p "<essay prompt>" -n 256 --temp 0 --reasoning-effort none` with `RUST_LOG=warn,mlxcel::models::sanitize=debug` for the repack profile, under `/usr/bin/time -l`; the MLX export as a control. Warm page cache. Start 01:02:36, load averages 1.94 6.25 10.26; end 01:07:16, load averages 1.78 3.37 7.79.

| Checkpoint | Binary | Load (s), runs 1 / 2 / 3 | Resident (GB) | Peak RSS (GB) | Wall (s) | Decode tok/s |
|---|---|---|---|---|---|---|
| ModelOpt | Base | 70.27 / 63.44 / 64.26 | 16.39 | 16.85 / 16.87 / 16.87 | 77.8 / 70.9 / 71.7 | 39.02 / 39.00 / 39.01 |
| ModelOpt | New | 2.24 / 2.25 / 2.24 | 16.39 | 16.83 / 16.81 / 16.81 | 6.88 / 6.88 / 6.87 | 68.83 / 68.78 / 68.75 |
| ModelOpt | New, `MLXCEL_RAON_DECODE_FUSION=off` | 2.24 / 2.23 | 16.39 | | | 39.01 / 39.02 |
| MLX | Base | 2.69 / 0.67 / 0.69 | 16.34 | 16.68 | 9.5 / 7.5 / 7.5 | 39.99 / 40.00 / 40.03 |
| MLX | New | 0.67 / 0.69 / 0.69 | 16.34 | 16.69 | 4.59 | 72.70 / 72.84 / 72.87 |

Profile of the base repack (9488 triplets, seconds, runs 1 / 2 / 3): total 69.7 / 62.9 / 63.7, of which copying the packed weights to host buffers 53.2 / 49.0 / 49.5, copying and decoding the E4M3 scales 12.4 / 9.9 / 10.1, and re-encoding them 3.2 / 3.1 / 3.2. The weight copy, not the scale round trip, was three quarters of it.

The new path does none of the three: the scale bytes are passed through as stored, the packed weight is viewed as `uint32` on the device, and the planes are evaluated in 1.7 s. With the decode fast paths off the new binary decodes at the base's speed and prints the base's text, so the planes and the decode path are unchanged; the 68.8 tok/s row is section 1's gain on this checkpoint. The route applies to every ModelOpt NVFP4 family whose block scales arrive as raw E4M3 bytes (MLX's safetensors reader); a loader that has already decoded them to a float dtype (the Gemma 4 selective loader) keeps the host transcode.

## 6. Sampled verify: one host read per block, not kept

Code prompt, `--temp 0.7 --top-p 0.95 --seed 7`, 200 tokens, plain against width 5, three rounds. The verify time per round is `verify_ms / rounds` from the `DFlash:` line, which does not depend on how many drafts a run happened to accept.

| Build | Plain tok/s | Width 5 tok/s | Speedup | Verify per round (ms) |
|---|---|---|---|---|
| Base | 32.65 / 32.64 / 32.64 | 48.35 / 50.35 / 48.33 | 1.48x / 1.54x / 1.48x | 59.0 / 59.3 / 59.1 |
| New (per-row read, as shipped) | 51.93 / 51.86 / 51.87 | 62.11 / 61.92 / 59.66 | 1.20x / 1.19x / 1.15x | 48.8 / 48.9 / 48.5 |
| New decode paths, one read per block | 51.88 / 51.78 / 51.85 | 59.03 / 56.80 / 60.04 | 1.14x / 1.10x / 1.16x | 49.7 / 49.3 / 49.6 |
| Reference decode paths, per-row read (base) | | 53.64 / 48.58 / 43.61 | | 59.4 / 58.7 / 58.5 |
| Reference decode paths, one read per block | 32.68 / 32.63 / 32.64 | 53.78 / 52.62 / 52.63 | | 60.2 / 60.4 / 60.4 |

Rows 1 and 2: 01:25:51 to 01:27:18, load averages 2.29 3.92 5.04 to 2.66 3.73 4.87. Row 3: 01:09:02 to 01:10:28. Rows 4 and 5: 01:10:48 to 01:11:57.

Reading the block's tokens back once needs every row's sampler graph built before any token is known, so the rows after the first mismatch are sampled and discarded. On a 260K-token vocabulary a top-p row costs more than a round trip saves: a round took 0.8 ms longer with the new decode paths and 1.5 ms longer with the reference ones. The per-row loop of #3455 is unchanged in this PR, and so is the behavior of sampled drafted runs.

Seeded sampled drafted runs were not reproducible run to run before this PR either (the three base width-5 runs printed three different texts; seeded plain runs repeat exactly).

## 7. Batched DFlash2 for concurrent requests, not enabled

Classic batched decode through the server (`mlxcel-server --parallel 4`, 128 tokens per request from a small client that sends N chat requests at once), aggregate tok/s at 1 / 2 / 4 concurrent requests, two rounds. Start 01:24:37, load averages 3.32 4.48 5.33; end 01:25:40.

| Binary | 1 | 2 | 4 |
|---|---|---|---|
| Base | 29.7 / 29.7 | 73.6 / 76.5 | 125.4 / 125.5 |
| New | 46.9 / 46.9 | 79.3 / 82.9 | 134.7 / 135.1 |

Whether an exact batched verify could beat that was measured before building the drafter and scheduler side. A probe (not in the tree) ran the verify forward of this PR at B rows with rectangular caches, which is the cheapest layout a batched verify could have, against the native batched decode step, on the real checkpoint, with a 64-token prompt per row. Two runs, 00:39 to 00:40, load averages 17.6 to 14.5 (the other job was building, so all figures are inflated together).

| Rows | Decode step (ms) | Aggregate decode (tok/s) | Verify, width 3 (ms) | Width 4 | Width 5 | Width 7 |
|---|---|---|---|---|---|---|
| 1 | 21.8 / 21.9 | 45.8 / 45.7 | 35.7 / 35.6 | 40.9 / 41.3 | 46.7 / 46.7 | 58.5 / 58.8 |
| 2 | 23.7 / 23.9 | 84.2 / 83.8 | 50.7 / 50.2 | 62.2 / 61.8 | 72.4 / 71.8 | 93.2 / 92.9 |
| 4 | 28.2 / 28.6 | 142.1 / 140.1 | 79.7 / 79.0 | 99.3 / 106.4 | over the 16-row limit | over the limit |

A batched decode step shares every dense product across rows, so two rows cost 9% more than one. An exact verify block cannot: each row's products are formed alone, so two rows cost 42 to 59% more than one. The ceiling on batched drafting is `rows * tokens emitted per verify / verify time`, taking section 3's solo acceptance and charging nothing for the draft:

| Rows | Width | Ceiling, short / code / long (tok/s) | Classic batched decode (tok/s) |
|---|---|---|---|
| 2 | 3 | 86 / 96 / 92 | 84 |
| 2 | 5 | 69 / 91 / 77 | 84 |
| 4 | 3 | 110 / 123 / 117 | 141 |

At four rows the ceiling is 13 to 22% below classic batched decode. At two rows it is 3 to 15% above at width 3 and mostly below at width 5, before three costs the ceiling leaves out: the draft itself (about 2 ms per row per round, which alone brings width 3 to between 5% below and 6% above), the batched round loop's uniform accept (every row commits the shortest row's accept each round, so acceptance is below the solo figures), and a per-row prefill (a `[B, L]` prefill does not round like B separate ones, so exactness needs the rows prefilled alone). The burst would also only ever collect requests whose prompts have the same length. So `supports_batched` stays `false` for Raon-1 mini and concurrent requests decode classically, as before; no B > 1 speculative code was added.

Penalty setting and the batched window. The probe above is a library-level forward with no sampler, so no presence penalty applies to it; the server table above ran before the merge with `main` at `d43915faf` and used 0.0 (see the note in section 2). Since PR #3459 the server applies the checkpoint's `presence_penalty` of 1.5 by default. A nonzero penalty makes `SamplingConfig::needs_token_history()` true, and `sampling_config_eq` in `src/server/batch/speculative_burst.rs` requires both configs to need no token history, so such requests stay out of batched speculative windows (the batched round loop passes an empty history). A batched DFlash2 burst would not be reached by default-setting concurrent Raon requests unless per-row token history is threaded through the window or the server runs with `--presence-penalty 0`; with the default they draft as single-request bursts.

## Reproduce

```bash
# 1. decode, and the fast paths one at a time
mlxcel generate -m models/raon-1-mini -p "<essay prompt>" -n 256 --temp 0
MLXCEL_RAON_DECODE_FUSION=off mlxcel generate -m models/raon-1-mini -p "<essay prompt>" -n 256 --temp 0

# traces (prefill, then the decode path)
MLXCEL_TRACE_REASONING_EFFORT=max cargo run --release --features metal,accelerate --example logit_trace -- \
  models/raon-1-mini corpus.txt 256 4 8 > prefill.tsv
MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_TRACE_REASONING_EFFORT=max cargo run --release --features metal,accelerate \
  --example logit_trace -- models/raon-1-mini corpus.txt 256 4 8 > decode.tsv
python3 scripts/compare_logit_traces.py reference.tsv prefill.tsv --decided 1

# 2. prompt cache: repeat one chat request against the server and read usage.prompt_tokens_details.cached_tokens
mlxcel-server -m models/raon-1-mini --port 8080

# 3 and 4. DFlash2
mlxcel generate -m models/raon-1-mini-nvfp4 -p "<prompt>" -n 256 --temp 0 \
  --draft-model models/raon-1-mini-dflash2 --draft-kind dflash --draft-block-size 5

# 5. load, with the repack profile
RUST_LOG=warn,mlxcel::models::sanitize=debug mlxcel generate -m models/raon-1-mini-nvfp4 \
  -p "<essay prompt>" -n 256 --temp 0 --reasoning-effort none

# 6. sampled drafting
mlxcel generate -m models/raon-1-mini -p "<code prompt>" -n 200 --temp 0.7 --top-p 0.95 --seed 7 \
  --draft-model models/raon-1-mini-dflash2 --draft-kind dflash --draft-block-size 5
```
