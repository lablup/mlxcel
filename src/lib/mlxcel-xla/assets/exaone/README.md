# ExaOne 3.x config fixture (#499 dense arch pack)

`config.json` is a **small synthetic** ExaOne config (real switches, tiny dims)
registered as a byte-exact structural fixture in `validation.rs`; the real
ExaOne-3.5 `config.json` parsing is covered by the `config::tests` parse assertion.
It deliberately uses the alternate field names ExaOne ships (`num_layers`,
`layer_norm_epsilon`, `activation_function`) so the fixture also exercises that
parsing.

ExaOne 3.x (`model_type` `exaone`) is the proven **Llama** forward (llama3 RoPE,
tied embeddings, RMSNorm, SwiGLU) with a different tensor-naming scheme: the
GPT-2-style `transformer.h.{i}...` layout (the `Exaone` `WeightScheme`). Verified
from the checkpoint's `modeling_exaone.py`: half-split RoPE, `cat(freqs, freqs)`
table, `head_dim^-0.5` scaling, and a gated MLP `c_proj(act(c_fc_0(x)) * c_fc_1(x))`
(so `c_fc_0` is the gate, `c_fc_1` the up projection), attention under
`attn.attention.*` with `out_proj` as `o_proj`.

The naming remap lives in `weight_names.rs` (unit-tested,
`exaone_scheme_maps_gpt2_style_names`); the emitted graph is loader-independent, so
`validation::dense_pack_families_reuse_proven_graphs` asserts ExaOne emits
byte-for-byte identical StableHLO to a tied llama3-RoPE Llama reference. The full
token-exact / serve gate runs post-merge on the real checkpoint via
`scripts/xla/validate_arch.sh`.
