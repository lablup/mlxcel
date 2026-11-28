#!/usr/bin/env python3
"""Save tiny raw-HF Falcon-H1 checkpoints and independent fp32 logits for #3346.

Run with the pinned CPU Transformers environment. Then run the ignored
`falcon_h1_hf_synthetic_parity` Rust test with MLXCEL_FALCON_H1_FIXTURES
pointing at this script's output directory. No MLX weight conversion occurs.

With `--long-model`, also write `long_oracle.json` for the original checkpoint:
fresh-prefix fp32 logits at long positions (up to 3584 tokens), which the ignored
`falcon_h1_original_checkpoint_long_positions` test compares. RoPE scaling bugs
only change low-frequency bands, so they are invisible to short prompts.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import FalconH1Config, FalconH1ForCausalLM


def create_case(root: Path, norm: str, bias: bool) -> None:
    """Exercise non-unit factors, grouped norms, and independent branch widths."""
    torch.manual_seed(3346)
    config = FalconH1Config(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        mamba_d_ssm=24, mamba_n_heads=4, mamba_d_head=6,
        mamba_n_groups=2, mamba_d_state=4, mamba_d_conv=4,
        mamba_chunk_size=128, mamba_rms_norm=norm != "none",
        mamba_norm_before_gate=norm == "before", mamba_proj_bias=bias,
        projectors_bias=not bias, attention_bias=bias, mlp_bias=bias,
        tie_word_embeddings=True, embedding_multiplier=0.75,
        lm_head_multiplier=0.625, attention_in_multiplier=0.5,
        attention_out_multiplier=0.875, key_multiplier=1.25,
        ssm_in_multiplier=1.5, ssm_out_multiplier=0.375,
        ssm_multipliers=[0.5, 1.25, 0.75, 1.5, 0.875],
        mlp_multipliers=[1.125, 0.25],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = FalconH1ForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "layernorm.weight" in name or "mamba.norm.weight" in name:
                parameter.uniform_(0.5, 1.5)
            elif "dt_bias" in name:
                parameter.uniform_(-3.0, 0.0)
            elif name.endswith(".D"):
                parameter.uniform_(0.5, 1.5)
            elif name.endswith(".A_log"):
                parameter.uniform_(0.0, 1.0)
            elif name.endswith(".bias"):
                parameter.uniform_(-0.03, 0.03)
            else:
                parameter.uniform_(-0.15, 0.15)
    path = root / norm
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    prompts = [[3, 17, 5, 29, 11, 8], [21, 4, 4, 19, 2], [9], [14, 1, 27, 6, 6, 12, 25]]
    cases = []
    with torch.no_grad():
        for prompt in prompts:
            ids = list(prompt)
            logits = []
            generated = []
            for _ in range(8):
                # Full fresh prefix is independent of any mutable decode cache.
                scores = model(torch.tensor([ids]), use_cache=False).logits[0, -1]
                logits.append(scores.tolist())
                token = int(scores.argmax())
                generated.append(token)
                ids.append(token)
            cases.append({"prompt": prompt, "generated": generated, "logits": logits})
    (path / "oracle.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Wrote {norm}: {len(cases)} independent fp32 trajectories", flush=True)


LONG_PREFIXES = [32, 512, 2048, 3584]
LONG_STEPS = 4


def create_long_oracle(model_dir: Path, out: Path) -> None:
    """Teacher-forced fp32 logits after each long prefix and a few greedy steps.

    One seeded token stream supplies every prefix, so the cases differ only in
    position. Each step reruns the full prefix without a cache.
    """
    model = FalconH1ForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    generator = torch.Generator().manual_seed(3346)
    stream = torch.randint(100, model.config.vocab_size - 100, (max(LONG_PREFIXES),),
                           generator=generator).tolist()
    cases = []
    with torch.no_grad():
        for length in LONG_PREFIXES:
            ids = stream[:length]
            prompt = list(ids)
            logits, generated = [], []
            for _ in range(LONG_STEPS):
                scores = model(torch.tensor([ids]), use_cache=False).logits[0, -1]
                logits.append(scores.tolist())
                token = int(scores.argmax())
                generated.append(token)
                ids.append(token)
            cases.append({"prompt": prompt, "generated": generated, "logits": logits})
            print(f"long oracle: prefix {length} -> {generated}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "long_oracle.json").write_text(json.dumps(cases) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--long-model", type=Path,
                        help="original HF checkpoint for the long-position oracle")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for norm, bias in [("none", False), ("before", True), ("after", False)]:
        create_case(args.out, norm, bias)
    if args.long_model is not None:
        torch.set_num_threads(8)
        create_long_oracle(args.long_model, args.out)


if __name__ == "__main__":
    main()
