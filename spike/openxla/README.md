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

A git worktree can symlink `spike/openxla/.venv` to an existing venv (it is
gitignored), or point the scripts at one with
`MLXCEL_ORACLE_PYTHON=/path/to/.venv/bin/python`.

## SSM scan probe (issue #3337)

```bash
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_ROCM_TARGET=gfx1151
PY=spike/openxla/.venv/bin/python
scripts/rocm_gpu_guard.sh --hold -- cargo build --release --features xla-iree --example xla_ssm_probe
# one case by hand
$PY spike/openxla/ssm_oracle.py --emit-inputs /tmp/ssm --mode prefill --real-len 100
scripts/rocm_gpu_guard.sh --hold -- cargo run --release --features xla-iree \
    --example xla_ssm_probe -- --inputs /tmp/ssm --device hip --out /tmp/ssm-hip
$PY spike/openxla/ssm_oracle.py --check /tmp/ssm-hip
# every acceptance case on local-task and hip
scripts/rocm_gpu_guard.sh --hold -- spike/openxla/ssm_gate.sh /tmp/ssm-gate
# the prefill-lowering measurements behind FINDINGS_ssm.md
scripts/rocm_gpu_guard.sh --hold -- $PY spike/openxla/ssm_lowering_measure.py --out /tmp/ssm-measure
```

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
