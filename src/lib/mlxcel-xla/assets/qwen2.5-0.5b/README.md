# Qwen2.5-0.5B-Instruct config fixture (#449 M3 Stage 2d, Stage B)

`config.json` is the real Qwen2.5-0.5B-Instruct config, kept as the fixture for
the `Config::from_json` parse test (`emitter/mod.rs`). It is the Qwen2 architecture
the OpenXLA backend serves: `model_type` `qwen2`, **plain RoPE** (no `rope_scaling`,
`rope_theta = 1e6`), a **q/k/v projection bias** (`o_proj` and the MLP have none),
and tied word embeddings.

Unlike the Llama-3.2-1B directory, no `.mlir` graphs are committed here: the engine
emits the prefill / decode / ragged-decode graphs from this config at load (the
Stage A emit-at-load path), so the Qwen graphs are generated, not bundled. The
emitter's Qwen-specific surface (the QKV-bias adds and the plain-RoPE table) is
covered by the structural tests in `emitter/mod.rs`, and the end-to-end token
exactness is validated on GB10 against the model's own greedy reference.
