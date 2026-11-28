# Chunked SSD and blocked Mamba1 graph scan on M5 Max (#3073)

Peak footprint, prefill time and greedy output of a long single-pass and chunked prefill, before and after the Mamba2 SSD scans of Falcon-H1, GraniteMoeHybrid and PLaMo 2 moved to the shared chunked scan in `src/models/ssd_scan.rs`, and Jamba's Mamba1 graph-scan fallback started building its state increments in blocks of 256 timesteps.

## Environment

| Item | Value |
|---|---|
| Host | Apple M5 Max, 128 GB, macOS 27 |
| Build | `cargo build --release --features metal,accelerate --bin mlxcel --example logit_trace`; main is `9c5ea4df`, branch is `9c5ea4df` plus this PR |
| Prompt | `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000` (19,715 to 28,734 tokens depending on the tokenizer) |
| Command | `/usr/bin/time -l mlxcel generate -m models/<name> -p "$PROMPT" -n 48 --temp 0 --profile --show-reasoning`, `MLXCEL_PREFILL_CHUNK=0` (single pass) or `2048` |
| Procedure | 75 s cooldown before every run; round 1 ran main first in each pair, round 2 branch first; the GPU was shared with other jobs under a lock, one run at a time |

Raw values: [`runs.csv`](ssm-prefill-scan-chunking-m5max-2026-11-15/runs.csv), [`output_identity.csv`](ssm-prefill-scan-chunking-m5max-2026-11-15/output_identity.csv), [`logit_trace.csv`](ssm-prefill-scan-chunking-m5max-2026-11-15/logit_trace.csv).

## Results

Prefill is in tok/s, rounds 1 / 2. Peak is the process peak memory footprint in GB.

| Checkpoint | Mode | main peak | branch peak | main prefill | branch prefill | Greedy output |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny-4bit | single pass | 87.9 / 162.9, fails | 17.10 | fails | 3,735 / 5,756 | main fails |
| granite-4.0-h-tiny-4bit | 2048 chunks | 9.85 | 7.46 | 1,834 / 2,345 | 3,639 / 5,872 | identical |
| falcon-h1-tiny-90m-instruct-4bit | single pass | 167.9, fails | 9.99 | fails | 15,417 / 15,625 | main fails |
| falcon-h1-tiny-90m-instruct-4bit | 2048 chunks | 3.99 | 2.28 / 2.34 | 6,756 / 7,222 | 17,916 / 20,439 | differs at about the 38th generated character |
| plamo-2-1b | single pass | 75.0, aborts | 25.62 | fails | 4,902 / 5,583 | main fails |
| plamo-2-1b | 2048 chunks | 12.21 | 9.90 | 5,191 / 6,115 | 8,130 / 8,910 | identical |
| ai21-jamba-reasoning-3b-4bit | single pass | 6.42 / 6.45 | 6.45 | 5,213 / 6,849 / 6,849 / 6,830 | 4,736 / 6,339 / 6,906 / 6,813 | identical |
| ai21-jamba-reasoning-3b-4bit | 2048 chunks | 3.25 | 3.25 | 5,471 / 7,060 | 5,389 / 7,036 | identical |

On main every single-pass prefill of the three Mamba2 families fails at the first token with `Insufficient Memory` from Metal (one granite run reported a GPU timeout instead, at 87.9 GB). With the scan chunked they complete. At 2048 chunks the 2048-token surrogate attention was the remaining working set, so peaks drop and prefill runs 1.5x to 2.8x faster, because the quadratic work per piece shrinks by the chunk ratio.

Jamba on Metal takes the fused Mamba1 scan kernel, which this change does not touch, so its rows are a control. Rounds 1 and 2 of the single pass read the branch 7 to 9% slower; two further cooled pairs (rounds 3 and 4, opposite orders) read it within 1% either way, so that gap is run-to-run noise on the shared GPU. The graph-scan fallback that changed is covered by the unit tests (`jamba_graph_scan_blocks_do_not_change_the_result` and `jamba_forced_graph_scan_prefill_spans_blocks_and_matches_the_kernel`).

## Arithmetic order

The SSD now runs in 256-token chunks (mlx-lm's `ssm_attn` default) and carries the state between them, where main built one chunk per prefill piece, so the arithmetic order changes for any piece longer than 256 tokens. Teacher-forced trace, `examples/logit_trace MODEL prompt_28000.txt 2112 8 8 2048`, every position of eight 2112-wide forwards (the prefill width under test), main as reference, compared with `scripts/compare_logit_traces.py --decided 2`:

| Checkpoint | Top-1 disagreement | Decided (gap >= 2) disagreement | Largest gap at a disagreement | Perplexity main / branch |
|---|---|---|---|---|
| granite-4.0-h-tiny-4bit | 121 / 16,896 (0.716%) | 4 / 16,148 (0.025%) | 2.75 | 1.2957 / 1.2945 (-0.098%) |
| falcon-h1-tiny-90m-instruct-4bit | 190 / 16,896 (1.125%) | 0 / 12,168 (0.000%) | 1.17 | 2.2758 / 2.2752 (-0.024%) |
| plamo-2-1b | 19 / 16,896 (0.112%) | 0 / 9,400 (0.000%) | 0.01 | 2.7860 / 2.7862 (+0.007%) |

Falcon-H1 and PLaMo 2 differ only where the reference was undecided. Granite has four disagreements at decided positions, the largest at a gap of 2.75, with 72% of all its disagreements landing on the reference's second choice and perplexity slightly lower on the branch. In greedy decoding at 2048 chunks, granite and PLaMo 2 are byte-identical to main. Falcon-H1 tiny diverges in output that is already degenerate at this length on both arms.
