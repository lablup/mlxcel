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
