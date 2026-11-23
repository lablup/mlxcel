"""Whole-model Mamba2 check against HF `Mamba2ForCausalLM` in fp32 (issue #3338).

Runs the mlxcel-xla state-threaded token prefill for a Mamba2 checkpoint
through the real IREE shim (`examples/xla_ssm_model_check.rs`, which loads the
checkpoint with the production config reader and weight loader) and compares
the last-token logits with HF's slow path (`torch_forward`, no `mamba_ssm`
kernels). The runner also prefills the same tokens split in two calls, handing
the returned state to the second, and feeds them one at a time through the
decode step (state kept in module globals), so the state-threading ABI and the
decode graph are checked against the same oracle.

    cargo build --release --features xla-iree --example xla_ssm_model_check
    # a seeded two-layer synthetic model (n_groups = 2 < num_heads, biased, untied):
    $PY spike/openxla/ssm_model_check.py --synthetic OUT/synthetic --device local-task
    # a real checkpoint:
    scripts/rocm_gpu_guard.sh --hold -- $PY spike/openxla/ssm_model_check.py \
        --model models/hf/mamba2-130m --device hip --tokens 200 --capacity 256

Passes when the max-abs logit error of all three runs is below `--tol` (default
3e-2) and every argmax matches HF.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "target" / "release" / "examples" / "xla_ssm_model_check"
TOL = 3e-2


def build_synthetic(out: Path, seed: int) -> Path:
    """Save a small seeded Mamba2ForCausalLM with every optional tensor."""
    from transformers import Mamba2Config, Mamba2ForCausalLM

    torch.manual_seed(seed)
    config = Mamba2Config(
        vocab_size=96,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=8,
        head_dim=16,
        expand=2,
        state_size=16,
        n_groups=2,
        conv_kernel=4,
        use_bias=True,
        use_conv_bias=True,
        tie_word_embeddings=False,
        chunk_size=256,
    )
    model = Mamba2ForCausalLM(config).float().eval()
    with torch.no_grad():
        # The default init leaves dt_bias / D near constant; spread them so a
        # mixed-up head order shows up in the logits.
        for layer in model.backbone.layers:
            mixer = layer.mixer
            mixer.dt_bias.uniform_(-3.0, 0.0)
            mixer.D.uniform_(0.5, 1.5)
            mixer.A_log.uniform_(0.0, 1.5)
            mixer.norm.weight.uniform_(0.5, 1.5)
            layer.norm.weight.uniform_(0.5, 1.5)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    return out


def hf_logits(model_dir: Path, tokens: list[int]) -> np.ndarray:
    from transformers import Mamba2ForCausalLM

    # safetensors only: a pickled `pytorch_model.bin` in an untrusted
    # checkpoint directory must not be unpickled.
    model = Mamba2ForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, use_safetensors=True
    ).eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens])).logits[0, -1]
    return logits.float().numpy()


def run_xla(args, model_dir: Path, tokens: list[int], out: Path) -> dict:
    runner = Path(args.runner)
    if not runner.is_file():
        sys.exit(f"{runner} is missing; build the xla_ssm_model_check example first")
    cmd = [
        str(runner),
        "--model", str(model_dir),
        "--device", args.device,
        "--capacity", str(args.capacity),
        "--precision", args.precision,
        "--layout", args.layout,
        "--tokens", ",".join(map(str, tokens)),
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True)
    return json.loads((out / "run.json").read_text())


def compare(name: str, got: np.ndarray, want: np.ndarray) -> dict:
    err = float(np.max(np.abs(got - want)))
    return {
        "name": name,
        "max_abs": err,
        "argmax": int(np.argmax(got)),
        "hf_argmax": int(np.argmax(want)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="a Mamba2 checkpoint directory")
    src.add_argument("--synthetic", metavar="DIR", help="build a seeded model here")
    ap.add_argument("--device", default="local-task")
    ap.add_argument("--precision", default="f32", choices=["f32", "f16", "bf16"])
    # `inout` is what the single-sequence runtime (`IreeLlama`) emits and
    # uploads; `outin` is the checkpoint layout the batching engine keeps.
    ap.add_argument("--layout", default="inout", choices=["inout", "outin"])
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=40, help="prompt length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--out", help="result directory (default: next to the model)")
    ap.add_argument("--runner", default=os.environ.get("MLXCEL_SSM_MODEL_RUNNER", RUNNER))
    ap.add_argument("--json", metavar="FILE", help="write the report here")
    args = ap.parse_args()

    model_dir = (
        build_synthetic(Path(args.synthetic), args.seed)
        if args.synthetic
        else Path(args.model)
    )
    if not 1 < args.tokens <= args.capacity:
        sys.exit(f"--tokens must be in 2..={args.capacity}")
    vocab = json.loads((model_dir / "config.json").read_text())["vocab_size"]
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, vocab, size=args.tokens).tolist()

    out = Path(args.out) if args.out else model_dir.parent / f"{model_dir.name}-check-{args.device}"
    run = run_xla(args, model_dir, tokens, out)
    want = hf_logits(model_dir, tokens)
    results = [
        compare("whole", np.load(out / "logits.npy", allow_pickle=False), want),
        compare("split", np.load(out / "logits_split.npy", allow_pickle=False), want),
        compare("decode", np.load(out / "logits_decode.npy", allow_pickle=False), want),
    ]
    passed = all(r["max_abs"] < args.tol and r["argmax"] == r["hf_argmax"] for r in results)
    report = {"model": str(model_dir), "run": run, "results": results, "passed": passed}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
