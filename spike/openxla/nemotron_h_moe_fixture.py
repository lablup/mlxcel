"""Generate the tiny tokenizer-backed Nemotron-H-MoE acceptance fixture.

The output contains the same seeded fp32 tensors in two layouts:

* ``hf/`` is the native Transformers checkpoint used only for the independent
  eager fp32 oracle.
* ``mlx/`` is the converted checkpoint consumed by mlxcel. Its tokenizer maps
  the default text prompt to the exact same token IDs as the HF tokenizer.

This is the full-model companion to ``moe_oracle.py``. It has ``M``, ``*``, and
``E`` blocks, eight routed experts, top-2, two groups, and an ungated relu2 shared
expert. It is intentionally tiny and does not establish 30B active-parameter,
memory-residency, or throughput acceptance.

Generate (outside the repository model tree):

    $PY spike/openxla/nemotron_h_moe_fixture.py --out /tmp/nemotron-h-moe-tiny

ROCm handoff (all commands under ``scripts/rocm_gpu_guard.sh --hold --``):

    cargo build --release --features xla-iree --bin mlxcel
    env MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=hip MLXCEL_XLA_ROCM_TARGET=gfx1151 \
        MLXCEL_XLA_PRECISION=f32 MLXCEL_PRINT_TOKEN_IDS=1 \
        target/release/mlxcel generate -m /tmp/nemotron-h-moe-tiny/mlx \
        -p 'tok4 tok17 tok5 tok29 tok11 tok8' -n 12 --temp 0 \
        --no-chat-template

Compare the printed continuation IDs with ``hf-f32-oracle.json``. The ignored
``xla_hybrid_nemotron_h`` integration test separately drives eight raw-ID requests
through batch four and compares every stream with the single-sequence engine.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    NemotronHConfig,
    NemotronHForCausalLM,
    PreTrainedTokenizerFast,
    __version__ as transformers_version,
)

PROMPT = "tok4 tok17 tok5 tok29 tok11 tok8"
SEED = 3345


def build_config() -> NemotronHConfig:
    return NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "moe", "mamba", "mlp"],
        tie_word_embeddings=True,
        use_cache=False,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        use_mamba_kernels=False,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        mamba_hidden_act="silu",
        n_groups=2,
        conv_kernel=4,
        use_conv_bias=True,
        chunk_size=128,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=12,
        moe_shared_expert_intermediate_size=20,
        num_experts_per_tok=2,
        routed_scaling_factor=1.25,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        use_bias=False,
        layer_norm_epsilon=1e-5,
        eos_token_id=2,
        bos_token_id=1,
        pad_token_id=0,
    )


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"tok{index}": index for index in range(32)}
    vocab.update({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )


def randomize_model(model: NemotronHForCausalLM) -> None:
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("A_log"):
                parameter.uniform_(-1.0, 0.5, generator=generator)
            elif name.endswith("dt_bias"):
                parameter.uniform_(-2.0, 0.0, generator=generator)
            elif name.endswith(".D"):
                parameter.uniform_(0.5, 1.5, generator=generator)
            elif name.endswith("norm.weight") or name.endswith("norm_f.weight"):
                parameter.uniform_(0.6, 1.4, generator=generator)
            else:
                parameter.normal_(mean=0.0, std=0.1, generator=generator)
        model.model.layers[2].mixer.gate.e_score_correction_bias.copy_(
            torch.tensor([0.07, -0.03, 0.02, -0.06, -0.01, 0.05, -0.04, 0.03])
        )


def mlx_name(name: str) -> str | None:
    if name == "lm_head.weight":
        return None  # tied to backbone.embeddings.weight
    if not name.startswith("model."):
        raise ValueError(f"unexpected HF tensor {name}")
    name = "backbone." + name.removeprefix("model.")
    name = name.replace(".mixer.experts.up_proj", ".mixer.switch_mlp.fc1.weight")
    name = name.replace(".mixer.experts.down_proj", ".mixer.switch_mlp.fc2.weight")
    return name


def greedy_oracle(
    model: NemotronHForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new: int,
) -> dict:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    continuation = []
    with torch.inference_mode():
        for _ in range(max_new):
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
            token = int(torch.argmax(logits, dim=-1).item())
            continuation.append(token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token]], dtype=torch.long)], dim=1
            )
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "ref_token_ids": continuation,
        "max_new_tokens": max_new,
        "dtype": "float32",
        "oracle": "transformers NemotronHForCausalLM eager CPU",
    }


def generate(out: Path, prompt: str, max_new: int) -> None:
    hf_dir = out / "hf"
    mlx_dir = out / "mlx"
    hf_dir.mkdir(parents=True, exist_ok=True)
    mlx_dir.mkdir(parents=True, exist_ok=True)

    config = build_config()
    model = NemotronHForCausalLM(config).float().eval()
    randomize_model(model)
    tokenizer = build_tokenizer()

    model.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)

    converted = {}
    for name, tensor in model.state_dict().items():
        target = mlx_name(name)
        if target is not None:
            converted[target] = tensor.detach().cpu().contiguous().clone()
    save_file(converted, mlx_dir / "model.safetensors")
    config.save_pretrained(mlx_dir)
    tokenizer.save_pretrained(mlx_dir)

    oracle = greedy_oracle(model, tokenizer, prompt, max_new)
    (out / "hf-f32-oracle.json").write_text(
        json.dumps(oracle, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "seed": SEED,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "pattern": "M*EM-",
        "experts": 8,
        "top_k": 2,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.25,
        "tokenizer": "WordLevel, shared by hf/ and mlx/",
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote HF fixture:  {hf_dir}")
    print(f"wrote mlx fixture: {mlx_dir}")
    print(f"prompt ids:        {oracle['prompt_ids']}")
    print(f"continuation ids:  {oracle['ref_token_ids']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new", type=int, default=12)
    args = parser.parse_args()
    if args.max_new < 1:
        parser.error("--max-new must be positive")
    generate(args.out, args.prompt, args.max_new)


if __name__ == "__main__":
    main()
