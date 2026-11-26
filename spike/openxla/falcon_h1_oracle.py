#!/usr/bin/env python3
"""Save tiny raw-HF Falcon-H1 checkpoints and independent fp32 logits for #3346.

Run with the pinned CPU Transformers environment. Then run the ignored
`falcon_h1_hf_synthetic_parity` Rust test with MLXCEL_FALCON_H1_FIXTURES
pointing at this script's output directory. No MLX weight conversion occurs.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for norm, bias in [("none", False), ("before", True), ("after", False)]:
        create_case(args.out, norm, bias)


if __name__ == "__main__":
    main()
