#!/usr/bin/env python3
"""Greedy HF reference for Qwen2.5-0.5B: the external oracle for the OpenXLA
Stage B token-exactness gate. Loads the bf16 checkpoint as fp32 (the exact
widening the XLA path does on its weights), encodes a prompt, and records the
pure next-token-argmax trajectory for N steps WITHOUT stopping on EOS, so the
Rust side can reproduce exactly N tokens and diff. First generated token is the
argmax after the FULL prompt, matching XlaReferenceEngine's prefill_first."""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
out_path = sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
n_new = int(sys.argv[4]) if len(sys.argv) > 4 else 40

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
model.eval()

prompt_ids = tok(prompt, return_tensors="pt").input_ids  # [1, L]
ids = prompt_ids.clone()
ref = []
with torch.no_grad():
    for _ in range(n_new):
        logits = model(ids).logits[:, -1, :]  # [1, V]
        nxt = int(torch.argmax(logits, dim=-1).item())
        ref.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

out = {
    "prompt_text": prompt,
    "prompt_ids": prompt_ids[0].tolist(),
    "ref_token_ids": ref,
}
with open(out_path, "w") as f:
    json.dump(out, f)
print(f"prompt_ids ({len(out['prompt_ids'])}):", out["prompt_ids"])
print(f"ref_token_ids ({len(ref)}):", ref)
print("decoded continuation:", tok.decode(ref))
