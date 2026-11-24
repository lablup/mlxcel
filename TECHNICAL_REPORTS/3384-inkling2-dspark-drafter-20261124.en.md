# PR #3384: Inkling 2 bundled DSpark drafter

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3374). Narrow suites pass; validated on an M3 Ultra (512 GB) on `models/inkling-2-mini` against mlx-vlm 466069f5. Inkling 2 Small is deferred.

## Executive Summary

Both Inkling 2 checkpoints ship a DSpark drafter in `mtp.safetensors`. This PR ports it from mlx-vlm and runs it on the existing DFlash round loop, offline and in the server. `--draft-model` names the target directory itself. On this host the block-versus-chain exactness probe declines at every width, so by default the CLI refuses with a named error and the server serves classic decode. With `MLXCEL_MTP_ALLOW_INEXACT=1`, 256-token greedy decode on Mini goes from 35.5 to 38.6 tok/s on one prompt and from 35.9 to 55.0 tok/s on another. The drafter's proposals equal mlx-vlm's token for token in every round where the two contexts agree.

## 1. Problem Statement

#3367 left the bundled drafter out of scope, so mlxcel decoded Inkling 2 one token at a time while mlx-vlm could draft with the same checkpoint. The drafter is not a stock DSpark: it reads the partial block entering four of the 48 layer visits, caches fused target context per draft layer (two sliding layers, one global), and uses an `rnn_hidden` Markov head whose output projection is folded into the target LM head. None of the LFM2 DSpark code covers that head.

| Risk | Impact | Handling |
|---|---|---|
| Drafter numerics drift from upstream | Lower acceptance, no visible error | Host f64 reference tests, and a round-by-round comparison with mlx-vlm on the real checkpoint |
| A draft token is emitted without verification | Wrong output | Test that every emitted token is the verify argmax of its round |
| The multi-row verify is not byte-identical to decode | Greedy text differs from classic decode | Exactness probe gates the drafter; override is explicit |
| Wrong drafter paired with the target | MLX shape abort inside the forward | Own drafter family; bind compares model type, sizes, layer count and loop |

## 2. Design

### 2.1 Drafter (`mlxcel-core/src/drafter/inkling2_dspark/`)

| Part | Upstream | Notes |
|---|---|---|
| `config.rs` | `config.py` | Parses `mtp_config` from the top level or `text_config`, computes the visit count from `loop_config`, refuses each invalid field by name, and accepts verify widths 2..=`block_size` |
| `model.rs` | `DraftAttention`, `DraftMLP`, `DraftLayer`, `Model.mix` | A local layer keeps the newest 1024 rope'd context keys; a longer context skips its oldest rows and still advances the offset |
| `markov.rs` | `MarkovHead` | The fold of `hidden_out` into the LM head runs on the host in f64 over 4096-row tiles, multithreaded, then rounds to f32 and BF16 like upstream's numpy path |
| `drafter.rs` | `Model.bind`, `Model.draft_block` | Always runs the full trained block; draft logits come from the target's BF16 head, softcap 30, argmax over the unpadded vocabulary |

The target lends its embedding, `embed_norm` and LM head through a new `LanguageModel::inkling2_dspark_binding` hook. The dense head is kept in f32 for the target's own product, so the binding casts it back to the checkpoint dtype, which is exact.

### 2.2 Target (`src/models/inkling2_speculative.rs`)

The verify forward reuses the model's hidden pass with captures taken before each listed visit and after the final norm for id `visit_count`, matching `language.py`. It runs inside `DecodeLookaheadAppendScope`, which turns on the grouped expert kernel and a full block of convolution margin, as upstream's `speculative_verify=True` does. Every verify row gets the model's suppressed ids (the image placeholder) at `-inf`, the same bias classic decode composes, so the CLI can accept exactly that bias. Rollback trims every visit cache by `block_size - (accepted + 1)`.

### 2.3 Wiring

- CLI: `generate_dflash.rs` now serves Laguna and Inkling 2 through one shared tail. An image prompt prefills from the merged embeddings.
- Server: `DFlashTargetModel for Inkling2VlModel` (every prompt row seeds the drafter, B = 1 only, own family `Inkling2DSpark`), plus the dispatch, exactness, label and drive arms.
- `/v1/internal/mtp-policy`: a DFlash dispatch used to publish `no_mtp_dispatch` with empty pairing fields. It now publishes `unavailable` with reason `dflash_dispatch` and the pairing filled, or `exactness_declined` with the probe sentence. Reason labels may be added without a schema bump, per `docs/mtp-policy-api.md`.

## 3. Validation

All GPU runs ran under the shared lock on an M3 Ultra.

### 3.1 Exactness verdict

The probe declines at width 8 (verify position 0 differs from the chain in 464983 of 800232 logit bytes) and at width 2 (559076), also with `qmv_wide` disabled. mlx-vlm has no gate and shows the same effect: its drafter-on text leaves its own classic text after 213 and 214 characters on the two prompts.

### 3.2 Throughput (ABAB, 256 tokens, `MLXCEL_MTP_ALLOW_INEXACT=1`)

| Prompt | Classic tok/s | Drafter tok/s | Acceptance | Mean accepted per round | mlx-vlm classic / drafter |
|---|---|---|---|---|---|
| Sky is blue | 35.54 | 38.55 | 0.170 | 1.18 | 33.9 / 42.2 (1.27 accepted) |
| Fibonacci function | 35.93 | 54.95 | 0.307 | 2.12 | 33.1 / 65.9 (2.71 accepted) |

Load averages rose from 2.2 to 8.4 during the block because other worktrees were compiling, so the absolute numbers carry a few percent of noise.

### 3.3 Agreement with mlx-vlm

With the round transcript enabled, prompt 2's rounds 0 to 2 propose the same 7 token ids as mlx-vlm, and so do rounds 4 and 5, which start from the same bonus tokens. The runs split in round 2, where both accept all 7 proposals but the bonus row picks token 149899 here and a comma in mlx-vlm, a near tie in the 8-row verify. The acceptance gap in 3.2 therefore comes from different text paths, not from the drafter. Text agreement with mlx-vlm's drafter run is 213 and 71 characters.

### 3.4 Other paths

- An image prompt with the drafter emits the same 48 tokens as classic decode.
- `models/inkling-2-mini-4bit` has no sidecar and runs classic decode as before (43.05 tok/s).
- Server: one chat request ran as a DSpark burst (34 rounds, acceptance 0.272), and mtp-policy reported `dflash_dispatch` with the pairing. Without the override it reported `exactness_declined` and served classic decode.
- Unit tests: 13 drafter tests against an f64 host reference (mix, Markov step, fold, `draft_block` over three rounds that cross the window and narrow the block, bind refusals, config refusals), 5 target tests on the tiny looped model (captures, suppression, rollback, probe, emitted tokens equal to the verify argmax every round), and the existing `inkling`, `drafter::`, coverage, detection, dispatch and mtp_policy suites.

## 4. Limitations and follow-ups

- Inkling 2 Small is not validated: its download has no `.done` marker and #3373 is open. The code path has no Mini-specific constants.
- The drafter cannot engage on this host without the inexact override. Making it exact needs a verify forward whose rows match the decode step's kernels, which is the same problem #3369 recorded.
- The f64 fold takes about 8.8 s on the first bind. The server warm-up request absorbs it; the CLI pays it at load.
- The server serves image requests, B > 1 windows and adopted prompt-cache prefixes with classic decode.
