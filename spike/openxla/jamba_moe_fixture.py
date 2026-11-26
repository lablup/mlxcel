"""Generate the tiny seeded Jamba MoE acceptance fixture (issue #3349).

The real AI21-Jamba-Reasoning-3B checkpoint has ``num_experts = 1``, so every
feed-forward is dense. This fixture is the only MoE evidence for the OpenXLA
Jamba path. Its four blocks cover every block shape Jamba builds:

* block 0: Mamba1 + MoE (``attn_layer_period=2, offset=1``;
  ``expert_layer_period=3, offset=0``)
* block 1: attention + dense MLP
* block 2: Mamba1 + dense MLP
* block 3: attention + MoE

Four experts, top-2, Hugging Face routing (softmax over every expert, then
top-k, no renormalization). Every parameter is randomized, including the
``dt_layernorm`` / ``b_layernorm`` / ``c_layernorm`` weights (HF initializes
them to ones, which would hide the weighted norm) and a wide router so the
top-k is decisive and varies by token. The generator asserts that the
reference trajectories select more than one expert set in every MoE block and
that dropping the learned dt/B/C norm weights, or renormalizing the top-k
weights, changes the logits.

The output contains the same fp32 tensors in two layouts:

* ``hf/`` is the native Transformers checkpoint used only by the oracle.
* ``mlx/`` is the stacked ``switch_mlp`` layout mlxcel loads
  (``feed_forward.router`` plus ``feed_forward.switch_mlp.{gate,up,down}_proj``).

Generate (outside the repository model tree):

    $PY spike/openxla/jamba_moe_fixture.py --out /tmp/jamba-moe-tiny

Files:

* ``hf-f32-oracle.json``: the greedy 12-token continuation of the text prompt
  (CLI, ``xla_oracle_check``, and ``xla_batch_bench`` input).
* ``oracle.json``: four raw-ID prompts, eight steps each, with the full fp32
  logits of every step from a fresh full-prefix forward (the ignored
  ``xla_jamba`` parity test reads it).
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    JambaConfig,
    JambaForCausalLM,
    PreTrainedTokenizerFast,
    __version__ as transformers_version,
)

PROMPT = "tok4 tok17 tok5 tok29 tok11 tok8"
SEED = 3352
VOCAB_SIZE = 32
PAD_ID = 0
BOS_ID = 1
UNK_ID = 2
EOS_ID = 31
RAW_PROMPTS = [[3, 17, 5, 29, 11, 8], [21, 4, 4, 19, 26], [9], [14, 10, 27, 6, 6, 12, 25]]
STEPS = 8


def build_config() -> JambaConfig:
    config = JambaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_layer_period=2,
        attn_layer_offset=1,
        expert_layer_period=3,
        expert_layer_offset=0,
        num_experts=4,
        num_experts_per_tok=2,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=2,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        use_mamba_kernels=False,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_cache=False,
        eos_token_id=EOS_ID,
        bos_token_id=BOS_ID,
        pad_token_id=PAD_ID,
    )
    config._attn_implementation = "eager"
    if config.layers_block_type != ["mamba", "attention", "mamba", "attention"]:
        raise AssertionError(f"block schedule drifted: {config.layers_block_type}")
    if config.layers_num_experts != [4, 1, 1, 4]:
        raise AssertionError(f"expert schedule drifted: {config.layers_num_experts}")
    return config


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"tok{index}": index for index in range(3, EOS_ID)}
    vocab.update({"<pad>": PAD_ID, "<bos>": BOS_ID, "<unk>": UNK_ID, "<eos>": EOS_ID})
    if len(vocab) != VOCAB_SIZE or len(set(vocab.values())) != VOCAB_SIZE:
        raise AssertionError("fixture tokenizer IDs must be unique and fill the vocabulary")
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    if len(fast) != VOCAB_SIZE:
        raise AssertionError(f"saved tokenizer contract drifted: len={len(fast)}")
    return fast


def randomize_model(model: JambaForCausalLM) -> None:
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("A_log"):
                parameter.uniform_(-1.0, 1.0, generator=generator)
            elif name.endswith("dt_proj.bias"):
                parameter.uniform_(-2.5, -0.5, generator=generator)
            elif name.endswith(".D"):
                parameter.uniform_(0.5, 1.5, generator=generator)
            elif name.endswith("conv1d.bias"):
                parameter.uniform_(-0.1, 0.1, generator=generator)
            elif "layernorm.weight" in name:
                # Includes the learned dt/B/C norms the OpenXLA path must apply.
                parameter.uniform_(0.4, 1.6, generator=generator)
            elif name.endswith("router.weight"):
                parameter.normal_(mean=0.0, std=1.0, generator=generator)
            else:
                parameter.normal_(mean=0.0, std=0.2, generator=generator)


def mlx_state_dict(model: JambaForCausalLM) -> dict[str, torch.Tensor]:
    """Stack the experts into mlx-lm `switch_mlp` tensors; keep every other name."""
    out = {}
    intermediate = model.config.intermediate_size
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().float().contiguous()
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        if name.endswith(".feed_forward.experts.gate_up_proj"):
            stem = name.removesuffix(".experts.gate_up_proj")
            out[f"{stem}.switch_mlp.gate_proj.weight"] = tensor[:, :intermediate].clone()
            out[f"{stem}.switch_mlp.up_proj.weight"] = tensor[:, intermediate:].clone()
        elif name.endswith(".feed_forward.experts.down_proj"):
            stem = name.removesuffix(".experts.down_proj")
            out[f"{stem}.switch_mlp.down_proj.weight"] = tensor.clone()
        else:
            out[name] = tensor.clone()
    return out


class RouterRecorder:
    """Record the top-k expert set of every token at every MoE block."""

    def __init__(self, model: JambaForCausalLM):
        self.sets: dict[int, set[tuple[int, ...]]] = {}
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            router = getattr(layer.feed_forward, "router", None)
            if router is not None:
                self.sets[index] = set()
                self.handles.append(router.register_forward_hook(self._hook(index)))

    def _hook(self, index: int):
        top_k = 2

        def record(_module, _inputs, logits):
            for row in torch.topk(logits.reshape(-1, logits.shape[-1]), top_k, dim=-1).indices:
                self.sets[index].add(tuple(sorted(row.tolist())))

        return record

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def last_logits(model: JambaForCausalLM, ids: list[int]) -> torch.Tensor:
    return model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False).logits[0, -1]


def raw_cases(model: JambaForCausalLM) -> list[dict]:
    cases = []
    with torch.inference_mode():
        for prompt in RAW_PROMPTS:
            ids = list(prompt)
            logits, generated = [], []
            for _ in range(STEPS):
                # A fresh full-prefix forward is independent of any decode cache.
                scores = last_logits(model, ids)
                logits.append(scores.tolist())
                token = int(scores.argmax())
                generated.append(token)
                ids.append(token)
            cases.append({"prompt": prompt, "generated": generated, "logits": logits})
    return cases


def greedy_oracle(model, tokenizer, prompt: str, max_new: int) -> dict:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = list(prompt_ids)
    continuation = []
    with torch.inference_mode():
        for _ in range(max_new):
            token = int(last_logits(model, ids).argmax())
            if token in (PAD_ID, BOS_ID, UNK_ID, EOS_ID):
                raise AssertionError(
                    f"fixture emitted special token {token} at step {len(continuation)}"
                )
            continuation.append(token)
            ids.append(token)
        generated = model.generate(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=False,
        )
    if generated[0, len(prompt_ids):].tolist() != continuation:
        raise AssertionError("manual eager oracle and Transformers generate disagree")
    return {
        "prompt_text": prompt,
        "prompt_ids": prompt_ids,
        "ref_token_ids": continuation,
        "max_new_tokens": max_new,
        "eos_token_id": EOS_ID,
        "dtype": "float32",
        "oracle": "transformers JambaForCausalLM eager CPU, fresh prefix per step",
    }


def check_norm_weights_matter(model: JambaForCausalLM) -> float:
    """The learned dt/B/C norms must change the output, or the fixture proves nothing."""
    prompt = RAW_PROMPTS[0]
    with torch.inference_mode():
        weighted = last_logits(model, prompt)
        saved = {}
        for name, parameter in model.named_parameters():
            if any(f"{n}_layernorm.weight" in name for n in ("dt", "b", "c")):
                saved[name] = parameter.detach().clone()
                parameter.fill_(1.0)
        unit = last_logits(model, prompt)
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.copy_(saved[name])
    delta = float((weighted - unit).abs().max())
    if delta < 1e-2:
        raise AssertionError(f"dt/B/C norm weights barely change the logits ({delta})")
    return delta


def check_renormalization_matters(model: JambaForCausalLM) -> float:
    """HF does not renormalize the top-k weights; a renormalizing router must differ."""
    prompt = RAW_PROMPTS[0]
    blocks = [layer.feed_forward for layer in model.model.layers if hasattr(layer.feed_forward, "router")]
    with torch.inference_mode():
        reference = last_logits(model, prompt)
        originals = [block.route_tokens_to_experts for block in blocks]
        for block, original in zip(blocks, originals):

            def renormalized(hidden, logits, original=original):
                index, weights = original(hidden, logits)
                return index, weights / weights.sum(dim=-1, keepdim=True)

            block.route_tokens_to_experts = renormalized
        changed = last_logits(model, prompt)
        for block, original in zip(blocks, originals):
            block.route_tokens_to_experts = original
    delta = float((reference - changed).abs().max())
    if delta < 1e-2:
        raise AssertionError(f"top-k renormalization barely changes the logits ({delta})")
    return delta


def generate(out: Path, prompt: str, max_new: int) -> None:
    hf_dir, mlx_dir = out / "hf", out / "mlx"
    hf_dir.mkdir(parents=True, exist_ok=True)
    mlx_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    config = build_config()
    model = JambaForCausalLM(config).float().eval()
    randomize_model(model)
    tokenizer = build_tokenizer()
    norm_delta = check_norm_weights_matter(model)
    renorm_delta = check_renormalization_matters(model)

    model.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
    save_file(mlx_state_dict(model), mlx_dir / "model.safetensors")
    config.save_pretrained(mlx_dir)
    tokenizer.save_pretrained(mlx_dir)

    recorder = RouterRecorder(model)
    cases = raw_cases(model)
    oracle = greedy_oracle(model, tokenizer, prompt, max_new)
    recorder.close()
    routing = {str(layer): sorted(sets) for layer, sets in recorder.sets.items()}
    for layer, sets in recorder.sets.items():
        if len(sets) < 2:
            raise AssertionError(f"MoE block {layer} always selects {sets}")

    (out / "oracle.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    (out / "hf-f32-oracle.json").write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "seed": SEED,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "layers_block_type": config.layers_block_type,
        "layers_num_experts": config.layers_num_experts,
        "experts": 4,
        "top_k": 2,
        "routing": "softmax over all experts, top-k, no renormalization (HF JambaSparseMoeBlock)",
        "selected_expert_sets_per_moe_block": routing,
        "dt_b_c_norm_weight_logit_delta": norm_delta,
        "topk_renormalization_logit_delta": renorm_delta,
        "tie_word_embeddings": True,
        "raw_prompts": RAW_PROMPTS,
        "steps_per_raw_prompt": STEPS,
        "tokenizer": "WordLevel with unique IDs, shared by hf/ and mlx/",
        "eos_absent_from_reference": True,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote HF fixture:  {hf_dir}")
    print(f"wrote mlx fixture: {mlx_dir}")
    print(f"prompt ids:        {oracle['prompt_ids']}")
    print(f"continuation ids:  {oracle['ref_token_ids']}")
    print(f"expert sets:       {routing}")
    print(f"norm-weight delta: {norm_delta:.4f}")
    print(f"renorm delta:      {renorm_delta:.4f}")


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
