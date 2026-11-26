# Prompt lookup on first-generation Inkling (M3 Ultra, 2026-11-25)

Issue #3422. Host: Mac Studio M3 Ultra, 512 GB. Checkpoints: `models/inkling-0.6b-a0.6b` (bf16) and `models/inkling-small-mlx-4bit` (affine 4-bit). Every run held the shared GPU lock; other worktrees were building on the host throughout (load averages below).

## Exactness

`models::speculative_exactness_survey_tests::survey_block_vs_chain`, engine path (the forward the token-only verify round runs). Before this change the engine row was `NotRun` for both checkpoints, because the gate refused the family.

| Checkpoint | Widths | Prompts | Verdict |
|---|---|---|---|
| inkling-0.6b-a0.6b | 2, 4, 8 | 8, 600, 2100 | equal in all 9 cells (family rows also equal) |
| inkling-small-mlx-4bit | 2, 4, 8, 16 | 8, 600, 2100 | equal in all 12 cells, wide quantized kernel |

```text
EXACTNESS_SURVEY_MODEL=models/inkling-0.6b-a0.6b \
  target/test-fast/deps/mlxcel-<hash> survey_block_vs_chain --ignored --nocapture
EXACTNESS_SURVEY_WIDTHS=2,4,8,16 EXACTNESS_SURVEY_PATH=engine \
EXACTNESS_SURVEY_MODEL=models/inkling-small-mlx-4bit \
  target/test-fast/deps/mlxcel-<hash> survey_block_vs_chain --ignored --nocapture
```

## End to end

`mlxcel generate -m <dir> -p <prompt> -n <N> --temp 0` against the same command with `--prompt-lookup` (defaults: max draft 7, so verify blocks of at most 8 rows). Inkling-Small runs add `--show-reasoning`, since its reply starts on the reasoning channel. The generated text was identical in every pair below, on every cycle (SHA-1 of the output from `Generating...` on).

The tok/s figure is the one `mlxcel generate` prints, which times the whole call, prefill included.

### Inkling-Small 4-bit, copy prompt

The prompt asks for a 16-line Python function to be reprinted with `total` renamed to `running_sum`. With `-n 900` the reply is 550 tokens (reasoning, then the rewritten function). ABAB, 4 cycles, load average 10.2 at the start and 11.5 at the end.

| Arm | tok/s (4 runs) | Mean | Against classic |
|---|---|---|---|
| classic | 41.81, 42.25, 42.36, 42.08 | 42.12 | 1.00x |
| prompt lookup | 49.35, 46.34, 48.52, 47.80 | 48.00 | 1.14x |

Prompt lookup: 243 rounds, 99 drafted, 577 proposed, 307 accepted (0.53), 2.26 tokens per target forward. Prompt lookup was ahead in all 4 adjacent pairs.

With `-n 400` (the reply stops inside the reasoning, before the code): classic 42.59, 41.93, 42.33, 42.29 (mean 42.29), prompt lookup 43.45, 45.31, 42.71, 44.54 (mean 44.00, 1.04x), 1.89 tokens per forward.

Single prose run (`Explain in three short paragraphs why the sky is blue.`, 256 tokens): classic 44.64, prompt lookup 38.75 tok/s (0.87x, 1.05 tokens per forward), text identical.

### Inkling-0.6B bf16

This checkpoint does not follow instructions or continue code: on most prompts it emits the same GNU/Linux passage and stops, so the copy prompt gives prompt lookup nothing to draft. The copy-heavy prompt here is that passage once and a half, so the reply repeats the prompt. ABAB, 4 cycles, load average 6.9 to 6.2, 50 generated tokens.

| Arm | tok/s (4 runs) | Mean | Against classic |
|---|---|---|---|
| classic | 235.73, 235.10, 235.66, 235.58 | 235.52 | 1.00x |
| prompt lookup | 502.05, 498.50, 497.33, 503.78 | 500.41 | 2.12x |

Prompt lookup: 8 rounds, 6 drafted, 42 of 42 proposals accepted, 6.13 tokens per target forward.

A prompt whose reply does not repeat it (a five-week weekday list, 52 generated tokens, no proposal ever made) runs classic at 234.24 and prompt lookup at 190.85 tok/s (0.81x), text identical: the round loop's overhead on a 0.6B model, with no drafted round to pay for it.

## Notes

- With `MLXCEL_INKLING_ROW_INVARIANT=0` the CLI refuses `--prompt-lookup` on the family, as before this change.
- Verify blocks wider than 16 rows (`--prompt-lookup-max-draft` above 15) rewind correctly but are outside the row-invariant scope, so they are not byte-identical to the chain; none of the runs above used one.
- On Inkling-Small the gain is far below the 2.26 tokens per forward: an exact verify block runs its experts and attention per position, so it costs more than one decode step. The per-block cost was not measured separately here.
