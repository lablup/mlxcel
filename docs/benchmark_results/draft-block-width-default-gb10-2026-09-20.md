# Draft block width as a per-device default (GB10, 2026-09-20)

Issue #1797, follow-up to #1782 (PR #1795). PR #1795 removed the fixed per-round verify cost and left the effective DFlash default of 16 losing to classic decode on the very pairing it fixed, so the throughput it bought was unreachable without passing `--draft-block-size` by hand. This record is the measurement the new default is seeded from, the test of whether the mechanism behind the crossover transfers to a second quantization family, and the place where a byte-identity failure found along the way is written down rather than left in a log.

## Host

```
host: spark-101
kernel: 7.0.0-1019-nvidia
arch: aarch64
nvidia_driver: 580.178.04
gpu: NVIDIA GB10, compute capability 12.1
MLX_CUDA_ARCHITECTURES: 121
cuda_toolkit: 13.0
mlx_pin: 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a
rustc_used_for_build: 1.93.1 (01f6ddf75 2026-02-11)
mem_total_gib: 121.7
binary_sha256: e4e0cce124d9a72b087ff9c0700b6013b281452b10cb483c365f604a2812208a
```

Three of those are new to a record in this tree, and they are why this session cannot be compared to an earlier one by assumption. The kernel moved from `6.17.0-1029-nvidia` to `7.0.0-1019-nvidia` earlier today; the upgrade landed without its matching NVIDIA module, the GPU was dead on first boot, and the driver was reinstalled at `580.178.04`, up from `580.173.02`. No harness here recorded either value before today, so no earlier GB10 record can say which driver it ran on. `MLX_CUDA_ARCHITECTURES` is pinned to `121` rather than left to `build.rs`, which auto-detects `121a` here. Not because the suffix could move the kernel boundaries this record measures: it cannot. In the pinned MLX tree the only code keyed on `__CUDA_ARCH_SPECIFIC__` is the NVFP4 weight-quantization converter in `nvfp4_quantize.cuh`, reached only from the weight-quantization kernel, and `qmv`, `fp_qmv` and `qmm_sm80` compile identically under both targets. The reason is plainer: `.github/workflows/release.yml` ships `90a;100;121`, every earlier GB10 record pins `121`, and a measurement that seeds a shipped default should describe the binaries users run. The binary confirms it at startup: `CUDA compute capability 12.1 (sm_121); compiled for [121] (cubin)`.

Every arm below was measured in one session, on one binary, on an otherwise idle host, so the comparisons close inside this environment whatever an earlier session ran on.

## Method

One `mlxcel-server` per arm, `--ignore-eos --max-batch-size 1`, `RUST_LOG=info`, the same binary throughout, copied to a distinct name so the harness's foreign-model gate does not match the server it started itself. Client: streaming `POST /v1/completions`, `temperature 0`, `max_tokens 200`, a fixed 158-token Python source prompt (`harness/prompt_retry.txt`), one warm-up request discarded, then n = 3 measured requests. The reported rate is end-to-end wall time for the fixed 200-token budget, from request start to the last SSE chunk, because a DFlash burst delivers its chunks at the end of a round and an inter-chunk rate would measure the burst shape rather than the throughput.

Arms run in this order: classic, `default` (the drafter with no width flag, which is what this issue is about), each explicit width, the two override controls, classic again. Classic first and last is the drift control. A served width cannot be interleaved the way `scripts/bench_block_width.sh` interleaves the offline path, because `draft_block_size` is worker-owned in the settings API and changing it means restarting the server.

Before every arm the harness waits for 60 seconds of sustained host quiet, checks for a foreign model process, checks a memory floor, and reads the kernel's cumulative `NV_ERR_NO_MEMORY` count. The gate is the one hardened for #1820, imported rather than copied.

`MLX_ENABLE_TF32=1` on every arm of the main tables, which is MLX's own default (`get_var("MLX_ENABLE_TF32", 1)` in `mlx/utils.h`) and what #1782 ran. A width seeded into a shipped default has to be measured in the configuration users get. An earlier pass of this sweep forced it off; that pass is kept below as a control because it says something about where TF32 lands.

Differences from #1782 worth stating rather than glossing: a different prompt (158 tokens against 165), driver `580.178.04` against `580.173.02`, kernel `7.0.0-1019-nvidia` against `6.17.0-1029-nvidia`, and nine days of `main`.

## What the two kernel boundaries predict

Both predictions below are read off the pinned MLX tree (`81ba1c6a`) before any number was measured, so the NVFP4 arm is a test of them rather than a description of them.

`mlx/backend/cuda/quantized/quantized.cpp` picks the kernel family. `supports_qmm_sm90` requires `compute_capability_major() == 9`, so GB10 (12.1) never takes that branch and falls through to the `qmm_sm80` one, whose gate is `if (can_use_qmv && (M * B < 8))`. A verify block of 2 to 7 rows therefore goes to the vector kernel and 8 or more goes to `qmm_sm80`, which the #1782 attribution measured at about 3.5x the single-row `qmv` and flat from 8 to 16. That boundary is on both families.

Which vector kernel depends on the quantization. `can_use_fp_qmv` is `supports_fp_qmv`, which returns false for `compute_capability_major() <= 9`, false for `mode == QuantizationMode::Affine`, and false above 8 rows. So an affine target takes `qmv` and an NVFP4 target takes `fp_qmv`.

Only `qmv` has a second boundary. `dispatch_multirow_width<Cap = 8>` (`qmm/qmv.cu`) instantiates exactly three compile-time accumulator widths: 2 for `x_rows <= 2`, 4 for `x_rows <= 4`, and `Cap` for everything above. A 5, 6 or 7 row verify therefore takes the 8-wide instantiation and its register cost while using fewer rows. The in-tree comment measures that cost on a V100 with `cuobjdump -res-usage` at `elems_per_thread` 16 (`qmv_kernel` 61 registers against `qmv_multirow_kernel<..., 8>` 168) and notes it drops the occupancy from 4 resident blocks to 1. No equivalent measurement exists for sm_121, so the register figure stays the V100 datum it is.

`fp_qmv.cu` has no such dispatch. It instantiates `fp_qmv_single` and `fp_qmv_batched` only, and its `rows_per_block = 8` indexes the output dimension (`row = g_idx.y * rows_per_block + t_idx.y`, with `blocks_y` derived from `N`), not the input rows.

So, before measuring:

- The affine family should peak at 4 and lose ground at 5 to 7, then step again at 8.
- The NVFP4 family should have no step between 4 and 7, so it should keep improving up to 7 and step only at 8.

An NVFP4 optimum at 4 would falsify the second prediction, and the accumulator width would then not be what makes wider verify blocks lose.

### What the NVFP4 pairing can and cannot test

`laguna-xs-2.1-nvfp4` is a 40-layer MoE with 256 experts and `num_experts_per_tok` 8, and only layer 0 is dense (`mlp_layer_types`). Its attention projections are dense and do take `fp_qmv`. Its expert projections do not take `fp_qmv` at any width: they are a gathered matmul through `SwitchLinear`'s compressed-tensors NVFP4 path, and the gather makes the dispatcher's row count `tokens * top_k`, which clears the `M * B < 8` window at every verify width including 2.

So this pairing tests the `fp_qmv` prediction on the attention share of the verify round only, and the expert share, which is most of the weight traffic, is on a third kernel path that neither boundary describes. The sharpened prediction is therefore that the NVFP4 curve has no step at 4 to 5, because the only part of it that could produce one is the attention projections' accumulator width, and `fp_qmv` has none. A step at 4 to 5 would falsify that. A flat region followed by an acceptance-driven decline would confirm it without saying anything about where an NVFP4 optimum sits on a dense checkpoint, which this host has none of.

## Affine family: `qwen3.5-4b-4bit` with `qwen3.5-4b-dflash`

Classic bracket: opening 57.98 to 58.38, closing 58.31 to 58.71 tok/s. The two overlap, by 0.07 tok/s, so nothing drifted under the widths they enclose. The overlap is thin enough to be worth stating as a number rather than as a verdict.

| arm | n | e2e tok/s mean (min to max) | vs classic | separates from classic | acceptance | emitted per verify | round device sync ms |
|---|---:|---:|---:|---|---:|---:|---:|
| classic (opening) | 3 | 58.23 (57.98 to 58.38) | | | | | |
| no flag (resolves to 4) | 3 | 67.61 (67.22 to 67.91) | 1.16x | yes, above every classic run | 0.532 | 2.58 | 30.6 |
| 2 | 3 | 68.47 (68.14 to 68.85) | 1.17x | yes, above every classic run | 0.746 | 1.75 | 19.0 |
| 3 | 3 | 70.31 (69.78 to 70.72) | 1.20x | yes, above every classic run | 0.618 | 2.24 | 24.5 |
| 4 | 3 | 68.14 (68.04 to 68.29) | 1.17x | yes, above every classic run | 0.532 | 2.58 | 30.0 |
| 5 | 3 | 58.29 (58.15 to 58.54) | 1.00x | no, ranges overlap | 0.466 | 2.84 | 40.8 |
| 6 | 3 | 55.57 (55.40 to 55.77) | 0.95x | yes, below every classic run | 0.412 | 3.06 | 47.0 |
| 7 | 3 | 49.38 (49.36 to 49.40) | 0.85x | yes, below every classic run | 0.352 | 3.11 | 54.5 |
| 8 | 3 | 48.35 (48.25 to 48.40) | 0.83x | yes, below every classic run | 0.309 | 3.16 | 56.4 |
| 16 (today's default) | 3 | 41.91 (41.88 to 41.97) | 0.72x | yes, below every classic run | 0.159 | 3.32 | 69.0 |
| `MLXCEL_DRAFT_BLOCK_SIZE=16`, no flag | 3 | 42.17 (42.02 to 42.40) | 0.72x | yes, below every classic run | 0.159 | 3.32 | 68.3 |
| classic (closing) | 3 | 58.50 (58.31 to 58.71) | | | | | |

### The prediction held, and width 5 is where it shows

The per-round device sync steps 19.0, 24.5, 30.0, 40.8 across widths 2, 3, 4, 5. Adding the fourth row costs 22%; adding the fifth costs 36%. That is the `dispatch_multirow_width` boundary and nothing else: rows 2 to 4 get an accumulator sized to fit them, row 5 takes the 8-wide instantiation and pays for three rows it does not use. #1782 measured 4 and 6 but not 5, so its table showed the cliff as a range; this one puts it on a single row.

Above 5 the sync rises gently (40.8, 47.0, 54.5) as rows are added at a fixed accumulator width, and 7 to 8 costs almost nothing per round (54.5 to 56.4) because the family switch to `qmm_sm80` trades the register cost for its own. Throughput still falls from 7 to 8, but acceptance does that (0.352 to 0.309), not the round.

The per-round sync agrees with #1782 at every width the two records share: 19.0 against 18.6 at width 2, 24.5 against 24.4 at 3, 30.0 against 29.9 at 4, 54.5 against 53.9 at 7, 56.4 against 59.2 at 8, 69.0 against 64.1 at 16. The kernel, the driver and nine days of `main` all moved between the two sessions and the per-round device cost did not, which is the closest thing to a cross-session control this record has.

### Which width to seed, and why it is 4 rather than 3

Within this session width 3 is the peak and it separates from 4: 69.78 to 70.72 against 68.04 to 68.29. #1782 measured the opposite, 4 over 3 by 0.6% with its own disjoint ranges. Two sessions, each internally separating them, disagreeing about which way.

This session also carries its own estimate of how much that kind of comparison is worth. The `no flag` arm and the `4` arm are the same width reached by two different code paths, measured six arms apart from separate server processes, and they produce byte-identical text but non-overlapping rates: 67.22 to 67.91 against 68.04 to 68.29. So between-server variation on this host is around 1% even when the within-arm spread is under 0.5%, and an n = 3 range understates it.

Width 3's 3% lead is larger than that but not by much, and the plateau at 2, 3 and 4 is 1.17x, 1.20x and 1.17x with every arm clear of classic. The seeded value is 4: it is what the prior record measured as its own peak, it is what this issue proposed, and it sits inside the plateau either way. Seeding 3 would chase a 3% that has not reproduced across sessions.

## NVFP4 family: `laguna-xs-2.1-nvfp4` with `laguna-xs-2.1-dflash`

PR #1771 merged on 2026-09-11, a few hours after #1797 was filed, so the issue's statement that this pairing could not be measured yet is stale. Both checkpoints are on this host and Laguna has a server-side DFlash target (`LagunaWrapper` in `src/server/batch/dflash_target.rs`), not only the offline arm from #1351. It was measured.

The result is not a width. **On an unmodified server the pairing does not speculate at all, at any width.** Every arm declines to classic decode before the round loop starts, so its throughput is classic's:

| arm | n | e2e tok/s (min to max) | what happened |
|---|---:|---:|---|
| classic | 3 | 29.45 to 29.52 | |
| no flag (resolves to 16) | 3 | 29.01 to 29.95 | declined to classic |
| 2 | 3 | 29.59 to 29.94 | declined to classic |

The decline is the target's own exactness probe refusing, and the log says exactly why:

```
MTP declined: verify block position 0 differs from the single-token chain in
108477 of 200704 logit bytes. Disabling qmv_wide did not make it exact either.
Falling back to classic decode. block_size=2
```

At width 16 the same probe reports 107002 of 200704 bytes. Half the bytes sounds structural and is not: 200704 bytes is 50176 f32 logits, and every one of them differing in its low two bytes is already 50%. The measured 54% is what a small last-mantissa-bits difference across essentially the whole vector looks like, which the acceptance rate confirms below. The probe is strict by design, comparing bytes rather than argmax, and a difference this size is enough to fail it while leaving the verify's decisions almost entirely intact.

**No NVFP4 policy entry follows from this, and none is added.** A default resolves on a server that does not set `MLXCEL_MTP_ALLOW_INEXACT`, and that server declines whatever width the default names. Seeding a width there would be seeding a number nothing reads.

The contrast with the affine pairing is the useful part, and it is about the gate rather than about the defect. Both verify blocks disagree with the single-token chain in their last mantissa bits, but the consequences differ. Laguna's difference barely changes any decision, which its own 0.887 acceptance under the override shows; it fails only because the probe compares bytes rather than argmax. Qwen 3.5's is large enough to change served output, about one token in thirty. Laguna's probe catches the harmless one and declines; Qwen 3.5 has no probe, so the consequential one ships. The gate, not the defect, is what differs usefully between them.

### The mechanism arm, with the contract deliberately forfeited

The decline above answers the policy question and not the mechanism one, so the pairing was measured again with `MLXCEL_MTP_ALLOW_INEXACT=1`, which makes the gate engage anyway and log `MTP exactness probe FAILED but MLXCEL_MTP_ALLOW_INEXACT is set` instead of declining. These rows are not eligible to seed anything, because a default resolves on a server that does not set that flag and that server declines. They exist to test the prediction.

Read the ratios against this pairing's own classic arm and nothing else. Laguna's drafter accepts far more of its proposals than Qwen 3.5's (0.887 at width 2 against 0.746) and its classic decode is half the speed (29.9 tok/s against 58.2), so each accepted proposal buys more here. A 1.7x on this pairing and a 1.17x on the affine one are not comparable quantities.

| arm | n | e2e tok/s mean (min to max) | vs classic | acceptance | emitted per verify | round device sync ms |
|---|---:|---:|---:|---:|---:|---:|
| classic | 3 | 30.36 (29.88 to 30.66) | | | | |
| 2 | 3 | 35.03 (34.42 to 35.82) | 1.15x | 0.887 | 1.88 | 40.3 |
| 3 | 3 | 45.41 (45.32 to 45.53) | 1.50x | 0.750 | 2.49 | 41.8 |
| 4 | 3 | 51.83 (51.75 to 51.95) | 1.71x | 0.720 | 3.16 | 45.9 |
| 5 | 3 | 52.50 (52.43 to 52.55) | 1.73x | 0.643 | 3.55 | 51.2 |
| 6 | 3 | 52.44 (52.31 to 52.55) | 1.73x | 0.547 | 3.69 | 54.2 |
| 7 | 3 | 54.41 (54.27 to 54.68) | 1.79x | 0.531 | 4.15 | 57.4 |
| 8 | 3 | 47.64 (47.53 to 47.87) | 1.57x | 0.420 | 3.90 | 63.8 |

Every width's range is disjoint from classic's. This pass has no closing classic bracket, so it has no drift check of its own; it is a mechanism arm, not a seeding one, and the affine pass measured immediately before it bracketed cleanly.

**The prediction held.** The affine family loses 14% going from width 4 to width 5 (68.14 to 58.29 tok/s); this one gains 1.2% over the same step (51.85 to 52.49). That is the whole test in two number pairs. `dispatch_multirow_width` rounds a 5-row affine verify up to the 8-wide accumulator instantiation and charges it for three rows it does not use, and `fp_qmv` has no such dispatch to charge anything, so the step that dominates the affine curve is simply absent here.

The same rule, fixed before either number was seen, applied to both families' per-round sync. An increment above 1.5x the mean of its two neighbours is a step; the family switch is visible if its increment is at least the median of the increments below it.

| | affine (`qmv`) | NVFP4 (`fp_qmv`) |
|---|---|---|
| increments 2 to 8 (ms) | +5.5, +5.5, **+10.8**, +6.2, +7.4, +1.9 | +1.5, +4.1, **+5.2**, +3.0, +3.2, **+6.4** |
| 4 to 5 against neighbour mean | 10.8 against 5.9, ratio 1.84 | 5.2 against 3.6, ratio 1.47 |
| accumulator step at 4 to 5 | present, as predicted | absent, as predicted |
| 7 to 8 against the 2-to-7 median | 1.9 against 6.2 | 6.4 against 3.2 |
| family switch visible in sync | no | yes |

The two are mirror images, and each has exactly the boundary its kernel has. `dispatch_multirow_width` exists only on the affine path and only the affine path steps at 4 to 5. The `M * B < 8` switch is on both paths, and it is the NVFP4 one that shows it in the sync column while the affine one absorbs it (affine's 7-to-8 round is almost free; what costs it throughput there is acceptance falling 0.352 to 0.309).

The NVFP4 4-to-5 ratio of 1.47 sits just under the 1.5 threshold rather than far below it, so the sync column alone is a near miss rather than a clean null. The throughput column is not close: over the same step the affine family loses 14% and this one gains 1.2%.

At width 8 both the round and the drafter give way together here: the sync steps 6.4 ms and acceptance falls 0.531 to 0.420, with emitted per verify turning over from 4.15 to 3.90. Both contribute to the 12% throughput drop, and this record cannot apportion them.

Acceptance falls monotonically with width, 0.887 at 2 down to 0.420 at 8, while emitted per verify climbs to 4.15 at width 7 before turning over. So the curve rising all the way to 7 is the drafter still converting extra rows into extra tokens faster than the round costs them, not acceptance saturating. Width 7 is the highest width measured, and it is where the sweep stops because the family switch at 8 is the boundary this arm exists to test; whether anything above 8 recovers is a different question and was not asked.

None of this changes the policy. `measured_default_block_size` returns `None` for `(12, 1, Nvfp4)` and the `gb10_with_a_non_affine_target_takes_the_flat_fallback` test asserts it, and both stay as they are: the widths above win only with the byte-identity contract forfeited by hand, and an unmodified server declines at every one of them. If the Laguna probe is ever satisfied, whether by fixing the verify's numerics or by narrowing what the probe compares, this table is the measurement an entry would be seeded from.

## Greedy byte-identity: not met, at any width, including today's default

This issue's acceptance criteria require the greedy output at the new default to be byte-identical to classic decode, matching the property PR #1795 established. It is not, and the failure is not specific to the new default.

Grouping every affine arm by the sha256 of its completion text gives two groups: the two classic arms, which are byte-identical to each other, and all ten speculative arms, which are byte-identical to each other and different from classic. Width 16 is in that second group, and 16 is the width `main` resolves today, so this is the behavior an operator already gets and not something the new default introduces. The `no flag` arm and the explicit `4` arm produce identical text, so the resolution path itself is deterministic.

A representative divergence, about 30 tokens in: classic emits `for attempt in range(self.max_attempts):` where every speculative width emits `for i in range(self.max_attempts):`.

Nothing gates this. `DFlashTargetModel::exactness_allows` defaults to `true` (`src/server/batch/dflash_target.rs`), the LFM2 and Muse Glimmer targets override it with a measured block-versus-chain probe, and the Qwen 3.5 target does not. So the property #1782 reported for this pairing was asserted by measurement rather than enforced, and nothing re-checks it.

The forced-TF32-off pass splits the speculative arms further, into 2 through 7 against 8 and 16, which is exactly the `M * B < 8` boundary between `qmv_multirow_kernel` and `qmm_sm80_kernel`. A grouping that tracks the kernel dispatch boundary points at the failure class the Metal `MLXCEL_MTP_ALLOW_INEXACT` gate exists for: a quantized projection dispatching to a different kernel at `M = K` than at `M = 1` without being bit-equal to it.

<!-- MULTIROW -->

This is recorded here and filed separately; it is not a reason to keep a default that loses 28% against classic decode, and it is not fixed by choosing a different width, because every width has it.

## Control: the same affine sweep with TF32 forced off

The first pass of this sweep forced `MLX_ENABLE_TF32=0`. That is neither MLX's default (`get_var("MLX_ENABLE_TF32", 1)`, so on) nor what #1782 ran, and a width seeded into a shipped default has to be measured in the configuration users get, so the table above was re-measured. The forced-off pass is kept because it is a free control on the ordering, and because it shows the per-round cost is not where TF32 lands.

| arm | n | e2e tok/s mean (min to max) | vs classic | acceptance | emitted per verify | round device sync ms |
|---|---:|---:|---:|---:|---:|---:|
| classic (opening) | 3 | 57.37 (57.19 to 57.58) | | | | |
| 2 | 3 | 68.33 (67.83 to 68.87) | 1.19x | 0.761 | 1.76 | 19.2 |
| 3 | 3 | 69.71 (69.29 to 70.15) | 1.21x | 0.618 | 2.24 | 24.8 |
| 4 | 3 | 67.81 (67.54 to 67.97) | 1.18x | 0.532 | 2.58 | 30.3 |
| 5 | 3 | 58.49 (58.31 to 58.74) | 1.02x | 0.466 | 2.84 | 40.7 |
| 6 | 3 | 54.96 (54.82 to 55.09) | 0.96x | 0.422 | 3.11 | 47.8 |
| 7 | 3 | 48.35 (47.80 to 48.73) | 0.84x | 0.352 | 3.11 | 54.9 |
| 8 | 3 | 45.90 (45.80 to 46.04) | 0.80x | 0.298 | 3.02 | 56.6 |
| 16 | 3 | 40.99 (40.98 to 41.02) | 0.71x | 0.153 | 3.16 | 66.7 |
| classic (closing) | 3 | 57.46 (57.16 to 57.64) | | | | |

The classic bracket overlaps, so nothing drifted under the widths it encloses.

What TF32 costs, measured rather than assumed: the classic arm runs 58.23 tok/s with it on and 57.37 with it off, about 1.2%. Every speculative arm is inside its own spread either way (width 4: 68.14 against 67.81; width 3: 70.31 against 69.71). It does not reorder the widths. The guess that the linear-attention chunked scan's f32 matmuls would make the verify path pay far more for it is wrong.

It does change greedy output. The classic completion differs between the two passes, and so does the split of the speculative arms into identity groups, which is the second reason a record has to state the setting rather than leave it unset.

## Driver budget

Every arm in this record cost zero kernel `NV_ERR_NO_MEMORY` errors, and the cumulative count for the boot is still zero after all of them: 34 arms across four passes, each starting and tearing down its own server, including ten that held the 20.1 GB Laguna NVFP4 target with its 0.86 GB drafter.

That is worth recording because it contradicts the expectation this sweep was planned against. On driver `580.173.02` a single 16k-context run on this host cost 38 of these errors at about 21 GB resident, and a 13-run rung was abandoned at a 400 cumulative ceiling. The Laguna arms here sit at the same resident scale and delivered none. This is one session on `580.178.04`, not evidence that the driver is fixed; the trip wire stays where it is, and the next long sweep should still take its first run as calibration and project from a measured per-run delta rather than from this result.

The host gate earned its place once. During the mechanism sweep it reported `foreign_models: ['mlxcel']` and held, because another session's model process appeared between arms. It was gone before the next arm started. That is the predicate working as designed on a shared host, which is the situation the #1820 harness was hardened for.

<!-- POLICY -->
