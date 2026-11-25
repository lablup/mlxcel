# Falcon-H1 on OpenXLA/IREE (#3346)

The raw Hugging Face layout uses parallel attention and Mamba2 branches from one normalized input. Their sum enters the first residual; a separately normalized SwiGLU MLP enters the second residual. This is distinct from the existing Cohere `parallel_block` (attention and MLP). The shared model-layer dispatcher applies the Falcon-H1 ordering to token prefill, single-sequence decode, and ragged decode. Every layer has both a compact KV row and a recurrent-state row. Existing recurrent-model guards disable prefix-hash prompt caches and default execution to f32.

The loader reads `model.embed_tokens`, `model.final_layernorm`, and per-layer `self_attn`, `mamba`, `feed_forward`, `input_layernorm`, and `pre_ff_layernorm`. `mamba_d_head` is the canonical SSM head width; `mamba_head_dim` is accepted as an alias when the canonical key is absent. Explicit `mamba_d_ssm` and `intermediate_size` take precedence over expansion factors. Attention and SSM head widths are independent.

Original HF weights have not folded muP. Embeddings and final logits receive their own factors; a tied head reads the original embedding matrix. Branch input factors precede their projections, the key factor precedes RoPE/cache writes, and branch output factors follow their output projections. The five SSM channel factors scale contiguous `[z,x,B,C,dt]` segments after the packed input projection. The MLP gate factor precedes SiLU and the down factor follows the down projection. Optional grouped SSM normalization supports both gating orders; absent normalization uses `y * silu(z)`. Input and output SSM biases use separate HF flags.

The first supported checkpoint is `tiiuae/Falcon-H1-Tiny-90M-Instruct` revision `e6389502a0b12cd8da894b395ba5bf7436873b16`. Its SSM width is 768, hidden width 512, attention head width 64, SSM head width 32, and SSM normalization is disabled. Sparse attention schedules, sliding windows, non-default RoPE scaling, and an explicit serialized `mup_vector` are rejected rather than approximated; HF derives its nonpersistent vector from `ssm_multipliers`.

## Reproducible independent gate

Generate three small checkpoints and fresh-prefix HF fp32 oracle trajectories using the pinned Transformers 5.12.1 CPU environment:

```sh
$MLXCEL_ORACLE_PYTHON spike/openxla/falcon_h1_oracle.py --out /tmp/falcon-h1-fixtures
MLXCEL_FALCON_H1_FIXTURES=/tmp/falcon-h1-fixtures MLXCEL_XLA_DEVICE=local-task \
  cargo test --profile test-fast --features xla-iree,test-utils --test xla_falcon_h1 \
  falcon_h1_hf_synthetic_parity -- --ignored --nocapture --test-threads=1
```

The ignored device test compares single-sequence tokens and four concurrent ragged trajectories with independent HF logits (absolute tolerance `3e-4`) and verifies slot re-admission. Fixtures use non-unit values for every multiplier, two SSM groups, unequal branch dimensions, both gating orders, no norm, and independent bias flags. On a GB10 host, wrap a CUDA invocation in `gpu-lock run --tag issue3346 --`; on ROCm use `scripts/rocm_gpu_guard.sh --hold --`.

These synthetic gates do not replace real-checkpoint acceptance: 40 oracle-exact CLI tokens, batch 4 / requests 8, four concurrent server completions, and the requested HIP `validate_arch.sh` gate remain mandatory before marking the issue complete.

## External ROCm acceptance handoff

The committed `falcon_h1_reference/hf-f32-oracle.json` contains the independent native-HF CPU fp32 40-token oracle for the pinned Tiny90M checkpoint; `provenance.json` records its revision, package versions, hashes, and regeneration command. Use the same original HF checkpoint, not an MLX conversion. Set the device explicitly: `setup-rocm.sh --env` supplies paths but does not select HIP.

```sh
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_DEVICE=hip MLXCEL_XLA_ROCM_TARGET=gfx1151 MLXCEL_XLA_PRECISION=f32
export MLXCEL_FALCON_H1_MODEL=/path/to/original-hf-falcon-h1
export MLXCEL_FALCON_H1_ORACLE="$PWD/spike/openxla/falcon_h1_reference/hf-f32-oracle.json"
export MLXCEL_FALCON_H1_FIXTURES=/tmp/falcon-h1-fixtures
$MLXCEL_ORACLE_PYTHON spike/openxla/falcon_h1_oracle.py --out "$MLXCEL_FALCON_H1_FIXTURES"
scripts/rocm_gpu_guard.sh --hold -- cargo test --profile test-fast \
  --features xla-iree,test-utils --test xla_falcon_h1 -- --ignored --nocapture --test-threads=1
scripts/rocm_gpu_guard.sh --hold -- scripts/xla/validate_arch.sh \
  --model "$MLXCEL_FALCON_H1_MODEL" --device hip --batch 4 --requests 8 --maxcap 24
```

The integration suite prints the selected device for real-checkpoint token parity. Keep the requested CLI and four concurrent server-completion checks as separate acceptance evidence; direct engine parity does not prove HTTP serving or chat-template/tokenization behavior.

## Measured results (2026-11-25)

At source head `205d0a3dd2d92eca2dbcba3dffcefba54e825eca`, pinned IREE `3.12.0rc20260721` passed the independent synthetic gate on both `local-task` and NVIDIA GB10 `cuda`: all three normalization modes, four unequal-length trajectories each, eight generated tokens each, per-step ragged logits within `3e-4` absolute error with matching argmax, and slot re-admission. Original Tiny90M engine generation matched all 40 committed HF fp32 oracle tokens on both devices. Focused emitter/configuration tests, all registered byte-exact goldens, every-family weight-layout agreement, 27 recurrent regression tests, five Nemotron-H tests, original-checkpoint shape validation, formatting, and narrow core-crate clippy passed. These are local results; HIP acceptance remains external.

The CUDA CLI also matched the independent 40-token oracle with `--no-chat-template`, temperature zero, and printed token IDs. Four simultaneous `/v1/completions` requests then matched the raw token decode of their separate 24-token CLI references and prompt/completion token counts. That original harness did not compare displayed stdout: it decoded the printed token IDs. The response intervals overlapped and the owned server process was reaped afterward. The HTTP API does not expose generated token IDs, so this HTTP result is text/count parity, not a direct HTTP token-ID observation. Command/environment and binary hashes are recorded in `falcon_h1_reference/cuda-cli-server-provenance.json`.

The CUDA `xla_batch_bench` gate passed with batch 4, requests 8, maximum per-request budget 24, and context capacity 256: all 170 emitted tokens across staggered requests and reused slots were reference-exact. The local test-fast run measured 102.21 tokens/s batched versus 67.58 tokens/s sequential (1.5x); these are bounded acceptance measurements rather than release-build performance claims. The exact output is in `falcon_h1_reference/cuda-batch-b4-n8.log`.

The final core-crate regression run passed 544 tests with 22 ignored and zero failures. The ignored real-checkpoint and synthetic IREE gates were executed separately as recorded above. `local-validation.json` summarizes the measured source and evidence boundaries.

## Displayed CLI whitespace follow-up

External ROCm acceptance at `852129eb1b4bf45d957f4c63b082d9d0276e4951` passed the numerical, 40-token CLI, batch/recycling, and release `validate_arch.sh` gates ([results](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147795531)). The stricter displayed CLI/server check passed only three of four prompts: the Germany completion ended in two spaces and a newline, which raw HTTP retained and CLI rendering removed ([diagnosis](https://github.com/lablup/mlxcel/pull/3414#issuecomment-7147960509)). All four server strings matched the raw decode of CLI token IDs. This is inherited numerical evidence for that exact head, not a HIP result for subsequent changes.

`StreamFilter::flush` intentionally trims chat and structured-message whitespace. Full CLI display now opts into preserving plain completion whitespace, restoring its documented byte-preserving behavior. Consumed reasoning/tool/Harmony/Muse delimiters and prompt-primed reasoning still select the existing cleanup policy; chat callers keep the default flush. Raw streaming CLI already forwards text unchanged. CPU regressions cover the exact `"  \n"` suffix, whitespace-only and Unicode replies, chunk accounting, and structured-message cleanup.

The portable replay compares actual displayed CLI stdout with four concurrent 40-token HTTP completions, including the Germany prompt. It removes only the CLI preamble and two framing newlines, without trimming model text, and separately verifies the original 40-token HF oracle. After rebuilding both binaries at the candidate head, external ROCm replay must explicitly select HIP:

```sh
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_DEVICE=hip MLXCEL_XLA_ROCM_TARGET=gfx1151 MLXCEL_XLA_PRECISION=f32
scripts/rocm_gpu_guard.sh --hold -- "$MLXCEL_ORACLE_PYTHON" \
  spike/openxla/falcon_h1_cli_server_check.py --device hip \
  --model "$MLXCEL_FALCON_H1_MODEL" --bin-dir "$PWD/target/release" \
  --binary-source-head "$(git rev-parse HEAD)" --out /tmp/falcon-h1-displayed-hip
```

The replay writes byte-preserved CLI stdout, token IDs, HTTP responses, exact comparisons, binary hashes, and owned-server termination evidence. Output-policy CPU tests and CUDA replay do not claim a new HIP run; the external replay remains a separate evidence boundary.
