# MiMo config fixture (#499 dense arch pack)

`config.json` is a **small synthetic** MiMo config (real switches, tiny dims)
registered as a byte-exact structural fixture in `validation.rs`; the real MiMo-7B
`config.json` parsing is covered by the `config::tests` parse assertion.

MiMo (`model_type` `mimo`) IS the **Qwen2** architecture: `MiMoForCausalLM` subclasses
`Qwen2ForCausalLM` and reuses `Qwen2Attention` / `Qwen2MLP` / `Qwen2RMSNorm` verbatim,
so it has a **q/k/v projection bias** and **untied** embeddings with plain RoPE
(half-split, `head_dim^-0.5` scaling). The extra multi-token-prediction (`MiMoMTPLayers`,
`num_nextn_predict_layers`) heads are not part of the base transformer and are not
loaded. Its config `sliding_window` is served globally (as for Qwen2), so it parses
to `sliding_window = None`.

Validation: `validation::dense_pack_families_reuse_proven_graphs` asserts it emits
byte-for-byte identical StableHLO to an untied Qwen2 reference; since MiMo's decoder
is Qwen2 verbatim, the existing Qwen2 token-exact gate covers its forward. The full
gate runs post-merge on the real checkpoint via `scripts/xla/validate_arch.sh`.
