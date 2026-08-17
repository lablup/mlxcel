# Quantizing the MTP drafter, Qwen 3.8 27B on M5 Max

Issue #1185, Phase 3. The drafter ships bf16 and is read once per drafted
token, so its cost is weight traffic. This measures what 4-bit affine
quantization does to that cost, to acceptance, and to output.

## Setup

| Field | Value |
|---|---|
| Host | Apple M5 Max, 40 GPU cores, 128 GB, macOS 26.6.1, Xcode 26.5 |
| Build | `cargo build --release --features metal,accelerate`, `main` at `1e7b1d13` |
| Target | `models/qwen3.8-27b-4bit` (affine, group 64, 4-bit) |
| Drafter | `models/qwen3.8-27b-mtp-bf16`, 810 MiB, 15 tensors |
| Command | `mlxcel generate --draft-kind mtp --draft-block-size 3 --temp 0` |
| Exactness gate | Passed with no override. The #1199 retry disables `qmv_wide` and the probe reports the verify block byte-identical to the single-token chain, so these are shipping-configuration numbers. |

Load averages stayed between 1.5 and 2.1 across the run, and the arms were
alternated, because this host loses throughput on repeated 27B work faster
than it loses it to anything else.

## What gets quantized

The drafter is one decoder layer plus an `fc` projection and seven norms. It
borrows `embed_tokens` and the LM head from the target, so those are not its
to convert. Eight 2-D projections are, all with the contraction axis last and
all divisible by the group size:

| tensor | shape |
|---|---|
| `fc.weight` | 5120 x 10240 |
| `layers.0.mlp.{gate,up}_proj.weight` | 17408 x 5120 |
| `layers.0.mlp.down_proj.weight` | 5120 x 17408 |
| `layers.0.self_attn.q_proj.weight` | 12288 x 5120 |
| `layers.0.self_attn.{k,v}_proj.weight` | 1024 x 5120 |
| `layers.0.self_attn.o_proj.weight` | 5120 x 6144 |

810.0 MiB becomes 227.9 MiB, 0.281x, including scales and biases. The seven
1-D norms stay bf16.

## Result

Two reps per arm at each length, alternated.

| | bf16 drafter | 4-bit drafter |
|---|---:|---:|
| `draft_block` | 10.40, 10.71 ms/round | **2.70, 2.69** |
| accept hook | 10.35, 10.60 ms/round | **2.74, 2.71** |
| verify forward | 40.08, 43.50 ms/round | 40.58, 40.05 |
| acceptance, n=120 | 0.6831 | 0.6601 |
| acceptance, n=300 | 0.6500 | 0.6589 |
| tok/s, n=120 | 37.90, 35.57 | **49.71, 50.33** |
| tok/s, n=300 | 35.24, 34.76 | **47.68, 45.68** |

**The drafter step is 3.9x cheaper and the verify forward is untouched**, which
is the shape the cost model predicted: the drafter is memory-bound and the
target is not affected by anything the drafter does.

**Acceptance does not move.** It falls 3.4% at 120 tokens and rises 1.4% at
300. Two lengths disagreeing in sign is what noise looks like; a real
degradation would not reverse.

Against classic decode measured on the same cooled host (32.33, 31.87, 30.79
tok/s, mean 31.66), MTP goes from **1.19x to about 1.5x**.

## Output is unchanged, and that is not a tolerance claim

Classic decode, MTP with the bf16 drafter, and MTP with the 4-bit drafter
produce **byte-identical** text at temperature 0. That is structural rather
than fortunate: the drafter only proposes and the target verifies every
proposal, so a worse draft is rejected and a better one is accepted, and
neither reaches the output. Quantization can cost acceptance and nothing else.

## Shipped as a load-time conversion, not a second checkpoint

Requiring a converted checkpoint would mean publishing and version-matching
one per drafter. Instead `Qwen35MtpDraftModel::from_path` quantizes its own
dense projections after sanitizing weights and before the bf16 to f16 pass,
so every existing bf16 drafter gets this without being re-downloaded.

A tensor whose `.scales` sibling already exists is left alone, so a
pre-converted checkpoint loads unchanged, and one whose contraction axis is
not a multiple of the group size stays dense rather than failing the load.

Verified equivalent to converting offline:

| arm | acceptance | `draft_block` |
|---|---:|---:|
| bf16 checkpoint, load-time conversion | 0.6601 | 2.74 ms/round |
| pre-converted 4-bit checkpoint | 0.6601 | 2.70 ms/round |
| bf16 checkpoint, `MLXCEL_MTP_QUANTIZE_DRAFTER=0` | 0.6831 | 10.43 ms/round |

`scripts/tools/quantize_mtp_drafter.py` still converts a checkpoint offline,
for publishing one or for A/B work that wants the two on disk side by side.

## What this leaves for the rest of Phase 2

The drafter side was 33.4% of the round before this and is about 11.8% after.
Phases 2a and 2b target the LM head, which is 21.4% of a drafter step that now
costs 2.7 ms per round rather than 10.5, so the absolute prize has shrunk by
roughly a factor of four.

The ceiling has also moved. With the #1199 `qmv_wide` retry the verify forward
costs 40.02 ms per round, so a round with a free drafter is 41.6 ms and emits
2.33 tokens: 56.0 tok/s, or 1.77x classic. The 1.93x in #1185 was computed
against the cheaper, non-exact verify. At about 1.5x today, most of what
remains is on the verify side, which is where tree drafting would act.
