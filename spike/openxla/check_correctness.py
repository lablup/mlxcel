"""Isolate the model math: compare JAX prefill last-token logits vs HF on the
same input ids. Validates RoPE (llama3), causal mask, GQA repeat, tied head."""
import numpy as np
import jax, jax.numpy as jnp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import model_jax as M

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
BUCKETS = [32, 64, 128, 256]


def pick_bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"prompt {n} exceeds max bucket {BUCKETS[-1]}")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "Give me three short tips for staying focused while working."}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]  # template emits BOS
    n = int(ids.shape[0])
    print(f"prompt tokens: {n}")

    # --- HF reference (fp32) ---
    hf = AutoModelForCausalLM.from_pretrained(MODEL).float().eval()
    with torch.no_grad():
        hf_logits = hf(ids[None])[0][0, -1].numpy()
    hf_arg = int(hf_logits.argmax())

    # --- JAX prefill ---
    cfg = M.load_config(MODEL)
    params = M.load_params(MODEL, cfg)
    prefill, _ = M.make_fns(cfg, MAX_SEQ)
    Lp = pick_bucket(n)
    toks = np.zeros(Lp, np.int32)
    toks[:n] = ids.numpy().astype(np.int32)
    pos = np.arange(Lp, dtype=np.int32)
    jx_logits, _, _ = jax.jit(prefill)(params, jnp.asarray(toks), jnp.asarray(pos), jnp.int32(n))
    jx_logits = np.asarray(jx_logits)
    jx_arg = int(jx_logits.argmax())

    # --- compare ---
    print(f"HF argmax token : {hf_arg!r:>8}  {tok.decode([hf_arg])!r}")
    print(f"JAX argmax token: {jx_arg!r:>8}  {tok.decode([jx_arg])!r}")
    print(f"argmax match    : {hf_arg == jx_arg}")
    d = np.abs(hf_logits - jx_logits)
    print(f"logit abs diff  : max={d.max():.4e}  mean={d.mean():.4e}")
    # top-5 agreement
    hf5 = hf_logits.argsort()[-5:][::-1]
    jx5 = jx_logits.argsort()[-5:][::-1]
    print("HF  top5:", [(int(i), round(float(hf_logits[i]), 3)) for i in hf5])
    print("JAX top5:", [(int(i), round(float(jx_logits[i]), 3)) for i in jx5])


if __name__ == "__main__":
    main()
