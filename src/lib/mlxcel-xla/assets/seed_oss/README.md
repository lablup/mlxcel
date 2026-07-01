# Seed-OSS config fixture (#499 dense arch pack)

`config.json` is a **small synthetic** Seed-OSS config (real architecture switches,
tiny dims) registered as a byte-exact structural fixture in `validation.rs`. Real
Seed-OSS is 36B / 64 layers, so its full graphs would be multi-MB goldens; the real
`config.json` parsing is covered separately by the `config::tests` parse assertion.

Seed-OSS (`model_type` `seed_oss`) is the proven **Qwen2** forward with standard
names: a **q/k/v projection bias** (from `attention_bias = true`; `o_proj` and the
MLP have none, and `attention_out_bias = false`), **untied** embeddings, and
`rope_type = "default"` served as plain RoPE. Verified from the HF `SeedOssModel`
source to use the emitter's half-split RoPE, `cat(freqs, freqs)` table,
`head_dim^-0.5` scaling, RMSNorm and SwiGLU.

Validation: `validation::dense_pack_families_reuse_proven_graphs` asserts it emits
byte-for-byte identical StableHLO to an untied Qwen2 reference, and
`spike/openxla/synthetic_arch_check.py` matches a tiny random `SeedOssForCausalLM`'s
last-token logits to ~1e-9 (forward parity). The full token-exact / serve gate runs
post-merge via `scripts/xla/validate_arch.sh` on the real checkpoint.
