# InternLM3 config fixture (#499 dense arch pack)

`config.json` is a **small synthetic** InternLM3 config (real switches, tiny dims)
registered as a byte-exact structural fixture in `validation.rs`; the real
InternLM3-8B `config.json` parsing is covered by the `config::tests` parse assertion.

InternLM3 (`model_type` `internlm3`) is the proven **Llama** forward with standard
names: **untied** embeddings, no bias (`qkv_bias = false`; a `qkv_bias = true`
checkpoint turns the q/k/v bias on), and `rope_type = "dynamic"` served as plain
RoPE. Dynamic NTK RoPE is identity within the original context and only rescales
beyond it, so short / in-context generation is served with plain RoPE (`rope_theta`);
the long-context NTK rescale is a follow-up. Verified from the checkpoint's
`modeling_internlm3.py` to use the emitter's half-split RoPE, `cat(freqs, freqs)`
table and `head_dim^-0.5` scaling.

Validation: `validation::dense_pack_families_reuse_proven_graphs` asserts it emits
byte-for-byte identical StableHLO to an untied plain-RoPE Llama reference (the
execution-proven forward). The full token-exact / serve gate runs post-merge on the
real checkpoint via `scripts/xla/validate_arch.sh`.
