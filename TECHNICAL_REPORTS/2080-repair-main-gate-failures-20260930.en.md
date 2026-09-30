# Technical Report: PR #2080 - Repair gate failures left by #2034 and #2037

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Rust (tests, one constant), Python (fixture generator), PNG fixture

**Risk Level**: Low (two test files, one fixture, its generator and one line of `FAMILY_ORDER`; no model, kernel or runtime code changes)

## Executive Summary

Two feature PRs, #2034 (Nemotron-Parse port) and #2037 (Nemotron VoiceChat), were merged while hosted CI was down, and each left main failing a gate it would have failed in CI. #2034's page fixture is over the size budget that `make verify-binary-assets` enforces on `tests/fixtures/*.png`. #2037 added a model family without listing it in `FAMILY_ORDER`, which fails `tests::family_order_is_exhaustive`, and it added a gelu test with f32 values hardcoded from Metal, which fails on ROCm by one ulp. The orchestrator of the epic #1801 run found the two test failures in every local ROCm gate: they were two of the three baseline `verify-test-rocm` failures that every epic PR in the run had to be judged against.

This PR fixes all three without changing any runtime behavior:

1. The fixture is re-encoded as 8-bit grayscale, 41.1 KB down to 20.6 KB, and decodes to RGB pixels byte-identical to the old file.
2. `Speech` is added to `FAMILY_ORDER` after `Text-to-speech`.
3. The gelu test builds its reference at run time from the pinned MLX expression on the same device and compares bit for bit over 4097 points in three dtypes, with backend-independent f64 anchors so a formula that is wrong on both sides still fails.

## 1. Problem Statement

Hosted CI was unavailable while #2034 and #2037 merged, so neither PR ran the gates that would have caught these regressions. Each failure is independent.

**Fixture size.** `scripts/ci/check_binary_assets.py` gives `tests/fixtures/*.png` a 32 KB ceiling per file. #2034 added `tests/fixtures/nemotron_parse_page.png`, a rendered page of text for the Nemotron-Parse preprocessing and parity tests, at 41.1 KB saved as RGB. `make verify-binary-assets` failed on it.

**Family ordering.** `FAMILY_ORDER` in `src/main.rs` sets the section order of the `mlxcel arch` output. A family missing from it is still printed (appended alphabetically), but `family_order_is_exhaustive` in `src/main_tests.rs` turns the omission into a test failure so the display position is chosen on purpose. #2037 introduced the `Speech` family for Nemotron VoiceChat (speech-to-speech) and did not add it.

**Gelu reference values.** #2037 also added `gelu_approx_matches_mlx_nn_bit_for_bit` for the Gemma 3 backbone's `gelu_approx`. It asserted exact f32 and bf16 outputs at seven inputs, values measured with mlx 0.32 on Metal. ROCm's f32 result at x = 4.1 is 4.0999565, one ulp away from the hardcoded 4.099957. Both are correct for their backend: the port issues the same ops as upstream, and the backends' `tanh`/`power` kernels round the last bit differently. A test that pins one backend's kernel output cannot express "bit for bit with upstream" on another backend.

With these two tests red on every ROCm run, each epic PR's `verify-test-rocm` result had to be read against a known-failing baseline, which hides new failures in the same targets.

## 2. Change Summary

- **`tests/fixtures/nemotron_parse_page.png`**: re-encoded losslessly as 8-bit grayscale (`L`) with Pillow `optimize=True`, 42104 to 21077 bytes. The page is black text on white rendered with antialiasing, so every pixel already has R == G == B.
- **`tests/fixtures/generate_nemotron_parse_page.py`**: saves with `img.convert("L").save(path, optimize=True)`, and its docstring explains the encoding, the 32 KB ceiling, and why the decoded pixels are unchanged.
- **`src/main.rs`**: `"Speech"` added to `FAMILY_ORDER` directly after `"Text-to-speech"`.
- **`src/models/gemma3_backbone_tests.rs`**: `gelu_approx_matches_mlx_nn_bit_for_bit` rewritten, with two helpers:
  - `mlx_nn_gelu_approx_reference`, a transcription of the pinned `python/mlx/nn/layers/activations.py` line `0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))`.
  - `gelu_approx_f64`, the same formula on the host in f64.

No model, kernel, CLI behavior or documentation changes. `docs/supported-models.md` already lists the ASR, TTS and speech-to-speech sections in the new order.

## 3. Technical Decisions

### 3.1 Fixture: grayscale re-encode instead of a per-file size exception

`check_binary_assets.py` allows per-file rules, and raising the ceiling for this one file would have been a one-line change. It was not needed. Since every pixel is gray, the RGB file stores each sample three times, and an `L` encoding drops that redundancy without losing anything. The alternatives measured worse or were unavailable: an RGB `optimize=True` re-encode reached only 39.6 KB, and none of oxipng, optipng, pngcrush or zopflipng is installed on the host.

The property that matters is that consumers see the same pixels. The Rust side decodes through the `image` crate (0.25.10) and calls `to_rgb8()` in `NemotronParseImageProcessor::preprocess_to_vec`, and the reference side uses Pillow. Both decode paths were checked: Pillow `convert("RGB")` and `image`'s `to_rgb8()` of the new file give bytes identical to the old RGB file. The parity values that #2034 recorded against the checkpoint therefore stay valid without re-derivation. The generator writes the same encoding, so regenerating the fixture does not reintroduce the oversize file.

### 3.2 `Speech` after `Text-to-speech`

The position follows `docs/supported-models.md`, which orders the audio sections as ASR, TTS, then speech-to-speech. Matching it keeps `mlxcel arch` and the docs in the same order, so the docs need no change. `family_order_has_no_orphans` also still passes, since the family is in use.

### 3.3 Gelu: compute the reference, do not pin it

The test's purpose is that the Rust port equals `mlx.nn.gelu_approx` bit for bit. Hardcoded outputs tie that claim to one backend's kernels. The new test evaluates the upstream expression with MLX ops on the same device and stream and compares raw bits with `gelu_approx`, so any backend passes exactly when the port issues the same ops as upstream.

The transcription follows Python's semantics, not the port's source:

- `*` and `+` are left-associative, so the leading factor is `(0.5 * x)`, multiplied by the tanh term afterwards.
- `x**3` is `mx.power`, not repeated multiplication.
- Every Python float becomes a weak scalar, which the bindings build as `array(float(v), x.dtype)`: rounded to f32 first, then cast to the input dtype.

Upstream wraps the function in `@partial(mx.compile, shapeless=True)`. Both sides here are the uncompiled expression. On Metal the compiled kernel differs from it by one f32 ulp at 4.1 (as #2037 measured), while its bf16 outputs matched the uncompiled expression, so the test documents that it checks op-for-op equality with the uncompiled graph.

The input is a dense grid of 4097 points over [-8, 8] instead of seven points, run in f32, bf16 and f16. A changed op usually differs from the original at only a few inputs. On ROCm, `x * x * x` in place of `power` differs at 2 of the 4097 f32 inputs, so a short list of points can miss it. On failure the assertion prints the count and the first eight (x, got, want) triples.

### 3.4 Why the f64 anchors exist

Because the reference is a transcription of the same upstream line, it issues the same ops as the port by design. If both sides carried the same wrong constant, the bitwise comparison would pass. The second half of the test compares `gelu_approx` against `gelu_approx_f64` on the host at the original seven inputs, with tolerances that hold on any backend:

- f32: `1e-6 + 1e-6 * |want|`.
- bf16: `4e-3 + 1e-2 * |want|`. bf16 rounds after every op, which moves the value at -3.0 from -0.00364 to -0.00586 on every backend, so the bound is a few bf16 ulps of the largest intermediate.

These tolerances cover the values #2037 pinned on Metal, which is the basis for expecting the test to pass there.

### 3.5 Deliberate breaks

The PR author broke `gelu_approx` temporarily in four ways to confirm the test still catches real regressions. Each break failed the test:

| Break | Where | Caught by |
|---|---|---|
| `x * x * x` instead of `power(x, 3)` | port only | bitwise check, 2 of 4097 f32 elements differ |
| `0.0447` instead of `0.044715` | port only | bitwise check, 2011 of 4097 f32 elements differ |
| compute in f32 and round once to the input dtype | port only | bitwise check, 1270 of 4097 bf16 elements differ |
| `0.0447` instead of `0.044715` | port and reference | f32 f64-anchor check |

The last row is the case the anchors exist for: the bitwise comparison passes because both sides agree, and the host check still fails.

## 4. Validation

PR author (gfx1151, `--features rocm`):

- `cargo test -p mlxcel --lib models::gemma3_backbone::gemma3_backbone_tests`: 5 passed.
- `cargo test -p mlxcel --bin mlxcel tests::family_order`: 2 passed (`family_order_is_exhaustive` and `family_order_has_no_orphans`).
- `cargo test --test nemotron_parse_real_model -- --ignored`: builds and runs. All three tests skip, since no Nemotron-Parse checkpoint is on the host.
- `cargo test --test cli_help_consistency`: 27 passed.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt verify-binary-assets`: pass.
- `cargo clippy -p mlxcel --lib --bin mlxcel --tests -- -D warnings`: clean.
- The fixture's decoded pixels were compared with Pillow and with the `image` crate against the old file: identical.

Orchestrator:

- The full `make verify-rocm` suite was not rerun for this PR. The change is confined to two test files, one fixture, its generator and one line of `FAMILY_ORDER`, and the targeted runs above cover each of them. The epic run's final gate on a fresh build covers the full suite after merge.

## 5. Learning Points

- **A bit-exact test should compute its reference, not store it.** "Equal to upstream bit for bit" is a relation between two computations on the same backend. Storing one backend's outputs turns it into "equal to Metal", which is a different claim and fails on correct ports elsewhere.
- **A transcribed reference needs an independent anchor.** When the reference and the implementation come from the same source line, a bitwise match proves they issue the same ops, not that the ops are right. A low-precision, backend-independent check covers the gap. The four deliberate breaks show which check catches which class of error.
- **Transcribe the language's semantics, not the formula.** Python's left-to-right `*`, `**` as `mx.power`, and weak scalars rounded through f32 all change the last bit. A reference written from the mathematical formula would disagree with upstream at scattered points.
- **Lossless recompression must be verified through the consumer's decoder.** The grayscale fixture is only safe because both the Pillow and `image` decode paths return the old RGB bytes. Checking file size alone would not show that.
- **Gates skipped while CI is down come back as a baseline tax.** These regressions were cheap to fix, but while they stood, every PR in the epic run was evaluated against three known failures. Fixing the baseline early makes a new failure in the same targets visible again.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA** were not run (not available on this host). The gelu test is expected to pass there: the port and the reference issue the same MLX ops in the same order on the same stream, so each backend rounds them identically, and the anchor tolerances cover the values #2037 pinned on Metal. The fixture and `FAMILY_ORDER` changes are not backend-specific.
- **The compiled upstream kernel** is not what the test compares against. `mx.compile` can differ from the uncompiled graph by one f32 ulp, so a port that matched the compiled kernel exactly would fail this test. The port and the test both deliberately target the uncompiled expression.
- **Nemotron-Parse end-to-end parity** with the re-encoded fixture ran only to the skip point, since no checkpoint is on the host. Pixel identity after decode is what carries the parity values over.
- **The full `make verify-rocm` suite** was not rerun on this branch (see Validation).
- **The third baseline failure**, `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, is not addressed by this PR.

Refs: #1801, PR #2034, PR #2037, PR #2079.
