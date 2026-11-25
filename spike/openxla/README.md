# OpenXLA export-route spike (issue #449, Phase 0 + Phase 1)

Standalone spike validating the **export-first** model-definition route from
ADR 0004: take Llama-3.2-1B, export `prefill` and a bucketed `decode-step` graph
to **StableHLO**, run greedy decode from the serialized artifact on **PJRT**
(and **IREE**), and check the continuation against an HF transformers temp-0
reference.

This directory is self-contained. It is **not** in the mlxcel Cargo workspace
and touches no Rust crate or build graph. Phase 2 (4-bit) and Phase 3 (mlxcel
integration) are out of scope here.

## Files

| File | Purpose |
|------|---------|
| `model_jax.py` | JAX Llama-3.2-1B: llama3 RoPE, RMSNorm, GQA attention, static-shape KV cache, `prefill` + `decode_step`. |
| `check_correctness.py` | Isolates the model math: JAX prefill last-token logits vs HF on identical ids. |
| `run_spike.py` | Exports both graphs to StableHLO, runs greedy from the **serialized** artifact on PJRT, compares to HF temp-0. |
| `analyze.py` | StableHLO op inventory + on-device argmax (host-copy) characterization. |
| `iree_run.py` | Bonus: runs one decode step through IREE, checks parity with PJRT. |
| `requirements.lock` | Pinned, resolved env (the reproducible Phase 0 spec). |
| `requirements-youtu.lock` | Separate pin (transformers 4.57.x) for the Youtu-VL oracle. |
| `FINDINGS.md` | The Phase 0/1 findings writeup. |

## Setup (Linux aarch64, CPU target)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv jax numpy safetensors huggingface-hub flatbuffers \
    transformers iree-base-compiler iree-base-runtime
# reproduce the exact pins instead with:  uv pip install --python .venv -r requirements.lock

# bf16 weights (ungated mirror, ~2.5 GB)
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('unsloth/Llama-3.2-1B-Instruct', \
allow_patterns=['*.json','*.safetensors','tokenizer*'], \
local_dir='models/Llama-3.2-1B-Instruct')"
```

## Setup (Linux x86_64, oracle venv for the SSM probe)

The SSM oracle (`ssm_oracle.py`, issue #3337) needs only CPU torch and
transformers from `requirements.lock`; it does not use the IREE Python
packages, because the XLA side runs through the Rust shim
(`examples/xla_ssm_probe.rs`) with the ROCm toolchain's `iree-compile`.
These steps produced the venv on the gfx1151 host (Python 3.12,
torch 2.12.1+cpu, transformers 5.12.1):

```bash
cd spike/openxla
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.lock \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -c "import torch, transformers.models.mamba2; print(torch.__version__)"
```

`requirements.lock` also pins `torchvision` (the LLaVA and Qwen-VL image processors import it). The Youtu-VL oracle (`youtu_vl_reference_oracle.py`) cannot use that environment: transformers 5.12.1 fails to import the Youtu remote code (`DefaultFastImageProcessorKwargs`), so it has its own lock with transformers 4.57.x (#3365):

```bash
uv venv --python 3.12 .venv-youtu
uv pip install --python .venv-youtu -r requirements-youtu.lock \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu
PYTHON=$PWD/.venv-youtu/bin/python scripts/xla/validate_youtu_vl_reference.sh ...   # the Youtu-VL gate reads PYTHON
```

A git worktree can symlink `spike/openxla/.venv` to an existing venv (it is
gitignored), or point the scripts at one with
`MLXCEL_ORACLE_PYTHON=/path/to/.venv/bin/python`.

## SSM scan probe (issue #3337)

```bash
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_ROCM_TARGET=gfx1151
PY=spike/openxla/.venv/bin/python
# A private scratch directory: the scripts write .npy files by fixed names and
# would follow a link planted at a predictable path in a shared /tmp.
WORK="$(mktemp -d -t ssm-probe.XXXXXX)"
scripts/rocm_gpu_guard.sh --hold -- cargo build --release --features xla-iree --example xla_ssm_probe
# one case by hand
$PY spike/openxla/ssm_oracle.py --emit-inputs "$WORK/inputs" --mode prefill --real-len 100
scripts/rocm_gpu_guard.sh --hold -- cargo run --release --features xla-iree \
    --example xla_ssm_probe -- --inputs "$WORK/inputs" --device hip --out "$WORK/hip"
$PY spike/openxla/ssm_oracle.py --check "$WORK/hip"
# every acceptance case on local-task and hip
scripts/rocm_gpu_guard.sh --hold -- spike/openxla/ssm_gate.sh "$WORK/gate"
# the prefill-lowering measurements behind FINDINGS_ssm.md
scripts/rocm_gpu_guard.sh --hold -- $PY spike/openxla/ssm_lowering_measure.py --out "$WORK/measure"
```

## Mamba2 whole-model check (issue #3338)

`ssm_model_check.py` runs the state-threaded token prefill of a whole Mamba2
checkpoint (production config reader and weight loader, through
`examples/xla_ssm_model_check.rs`) against HF `Mamba2ForCausalLM` in fp32.
Each run prefills the prompt once whole and once split in two calls that hand
the returned state on, and once token by token through the decode step, and
requires all three last-token logits within 3e-2 of HF with the same argmax.
The projection layout defaults to `--layout inout`, the one `IreeLlama` emits.
The config is cross-checked against the checkpoint's safetensors headers and
file sizes before any graph is emitted; see `FINDINGS_ssm.md`.

```bash
scripts/rocm_gpu_guard.sh --hold -- cargo build --release --features xla-iree --example xla_ssm_model_check
# a seeded synthetic model: 2 layers, n_groups 2 < 8 heads, biased projections, untied head
$PY spike/openxla/ssm_model_check.py --synthetic "$WORK/synth" --out "$WORK/synth-cpu" --device local-task
# AntonV/mamba2-130m-hf on hip, crossing a chunk boundary
scripts/rocm_gpu_guard.sh --hold -- $PY spike/openxla/ssm_model_check.py \
    --model models/hf/mamba2-130m --out "$WORK/130m-hip" --device hip --capacity 256 --tokens 200
# compile times of the prefill and decode graphs (uncached)
scripts/rocm_gpu_guard.sh --hold -- target/release/examples/xla_ssm_model_check \
    --model models/hf/mamba2-130m --device hip --capacity 256 --compile-times "$WORK/compile-hip"
```

### Near-tie rule for reduced-precision runs (issue #3343)

`oracle_continuation.py --top2` also records the oracle's runner-up id and its top-1 minus top-2 logit gap at every generated position (`ref_top2_ids`, `ref_top2_gaps`). `near_tie_check.py` judges a stream against that oracle: token-exact passes, and a stream that leaves the oracle at a position whose gap is below `--tolerance` (default 0.01) is reported as a near-tie (exit 3). Only the first divergence counts. The stream is given as `--ids`, `--ids-json`, or text (`--text`, `--text-file`, optionally `--mlxcel-cli` for captured `mlxcel generate` output); text needs `--model` as a local checkpoint directory and never downloads a tokenizer. `test_near_tie_check.py` covers the judge without a model (`python3 -m unittest spike/openxla/test_near_tie_check.py`). The IREE-Metal Mamba2 results that use it are in `FINDINGS_ssm.md`.

## Run

```bash
JAX_PLATFORMS=cpu .venv/bin/python check_correctness.py   # model math vs HF
JAX_PLATFORMS=cpu .venv/bin/python run_spike.py           # export + greedy + HF compare
JAX_PLATFORMS=cpu .venv/bin/python analyze.py             # op inventory + sampling

# bonus: compile the exported StableHLO with IREE and run one step
.venv/bin/iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu \
    artifacts/decode_step.stablehlo.mlir -o artifacts/decode_step.vmfb
JAX_PLATFORMS=cpu .venv/bin/python iree_run.py            # IREE prints benign nanobind teardown warnings
```

Artifacts land in `artifacts/`: `*.stablehlo.mlir` (text), `*.exported.bin`
(jax.export serialized), `decode_step.vmfb` (IREE), `results.json`.

## Result (this box: aarch64 Grace + GB10, CPU fp32)

`run_spike.py` greedy continuation is **token-exact (48/48)** with the HF
temp-0 reference. The same StableHLO runs on PJRT and IREE with matching
argmax. See `FINDINGS.md`.
