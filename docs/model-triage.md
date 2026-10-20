# Model coverage triage

Dispositions for the model families tracked in `lablup/mlxcel#548`, checked against the repository tree and public Hugging Face checkpoint headers. Each entry records the concrete evidence it rests on.

Last checked against `main` at `cf1354f5`.

## Already supported

| Family | Checkpoint / Issues | In-tree evidence |
|---|---|---|
| `falcon_ocr` | `tiiuae/Falcon-OCR` (`lablup/mlxcel#848`, PR `lablup/mlxcel#1075`) | `src/models/detection.rs` (`"falcon_ocr" => FalconOcrVL`), `src/models/falcon_ocr.rs`, `src/models/falcon_ocr_rope.rs`, `src/vision/falcon_ocr.rs`, `src/vision/processors/falcon_ocr.rs`, `src/loading/vlm_falcon_ocr.rs`, `src/commands/generate_falcon_ocr.rs`, [supported-models.md](supported-models.md) |
| `jina_vlm` | `jinaai/jina-vlm` (`lablup/mlxcel#849`, PR `lablup/mlxcel#1076`) | `src/models/detection.rs` (`"jvlm" \| "jina_vlm" => JinaVLM`), `src/models/jina_vlm.rs`, `src/vision/jina_vlm.rs`, `src/vision/encoders/jina_vlm.rs`, `src/vision/processors/jina_vlm.rs`, `src/loading/vlm_jina_vlm.rs`, [supported-models.md](supported-models.md) |
| `laguna` | Poolside Laguna XS/S (`lablup/mlxcel#1347`, PR `lablup/mlxcel#1739`; DFlash drafter `lablup/mlxcel#1351`, PR `lablup/mlxcel#1771`) | `src/models/detection.rs` (`"laguna" => Laguna`), `src/models/laguna.rs`, `src/models/laguna_layers.rs`, `src/models/laguna_sanitize.rs`, `src/models/laguna_speculative.rs`, [supported-models.md](supported-models.md) |

## Re-scoped to a dedicated epic and port issue

| Family | Checkpoint | Tracking |
|---|---|---|
| `bonsai` | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` (3.9 GB, Apache-2.0) | Marked out of scope for the text-generation runtime in `lablup/mlxcel#548` and re-scoped under the image-generation epic `lablup/mlxcel#2348` (shared runtime landed in `lablup/mlxcel#2341` / PR `lablup/mlxcel#2497`) with dedicated port issue `lablup/mlxcel#2428` (reuses the `lablup/mlxcel#2424` FLUX.2 Klein 4B transformer and loads the `transformer-packed-mflux/` 2-bit `mlx-packed-affine` group-128 weights through `UnifiedLinear`). |

## Remaining candidates in `lablup/mlxcel#548`

None of `falcon_perception`, `zaya1_vl`, `hrm_text`, or `multi_modality` has a `model_type` arm in `src/models/detection.rs` today. Their public checkpoints, `config.json` schemas, and safetensors headers are recorded below.

---

## `falcon_perception`

Two public checkpoints, both Apache-2.0 with `model_type: "falcon_perception"` and fp32 weights:

| | `tiiuae/Falcon-Perception` | `tiiuae/Falcon-Perception-300M` |
|---|---|---|
| Size | 2.53 GB, 231 tensors | 1.27 GB, 123 tensors |
| Decoder | `dim` 1024, `n_layers` 28, `n_heads` 16, `n_kv_heads` 8, `head_dim` 128, `ffn_dim` 3072, `vocab_size` 65536 | `dim` 768, `n_layers` 22, `n_heads` 16, `n_kv_heads` 8, `head_dim` 64, `ffn_dim` 2304, `vocab_size` 65536 |
| `freqs_cis_golden` | `[16, 32, 2]` | `[16, 16, 2]` |
| `do_segmentation` | `true` (`segm_out_dim` 256, `num_segm_layers` 3) | `false` (detection only) |

### Decoder and in-tree `falcon_ocr` compatibility

- `Attention`, `FeedForward`, and `TransformerBlock` in `modeling_falcon_perception.py` were diffed against `modeling_falcon_ocr.py`: apart from comments and imports they are identical.
- `modeling_falcon_perception.py` on `tiiuae/Falcon-Perception-300M` was diffed against `tiiuae/Falcon-Perception`: all model and layer classes (`Attention`, `FeedForward`, `TransformerBlock`, `FourierEncoder`, `MLPDecoder`, `FalconPerceptionModel`) are identical; the 300M file only adds a `task: str | None = None` argument (`"detection"` vs `"segmentation"`) in `generate()` / `_postprocess_aux()` and guards `upsample_img_features` and `proj_segm` behind `self.config.do_segmentation`.
- Both checkpoints use the same 5 per-layer tensor names as `falcon_ocr`: `layers.{i}.attention.{wqkv.weight, wo.weight, sinks}` and `layers.{i}.feed_forward.{w13.weight, w2.weight}`, plus `tok_embeddings.weight`, `img_projector.weight`, `freqs_cis_golden`, `norm.weight`, and `output.weight`.
- `src/models/falcon_ocr.rs`, `src/models/falcon_ocr_rope.rs`, `src/vision/falcon_ocr.rs`, and `src/loading/vlm_falcon_ocr.rs` were audited against both configs: `FalconOcrConfig` derives `head_dim` (`self.head_dim.unwrap_or(self.dim / self.n_heads)`), `n_kv_heads`, `patch_dim`, and `temporal_inv_freq(cfg.head_dim() / 2, cfg.rope_theta)` from `config.json`, and `golden_cos_sin` reads the `[n_heads, head_dim / 4, 2]` shape of `freqs_cis_golden` at runtime. Nothing in the `falcon_ocr` decoder hardcodes `dim` or `head_dim`.

### What a port adds on top of `falcon_ocr`

- Detection heads (present in both checkpoints, 8 tensors total): `coord_encoder.{embed,transform}.weight` (`[256, 2]`, `[dim, 512]`), `size_encoder.{embed,transform}.weight` (`[256, 2]`, `[dim, 512]`), `coord_decoder.{w1,w2}.weight` (`[8192, dim]`, `[2048, 8192]`), and `size_decoder.{w1,w2}.weight` (`[8192, dim]`, `[2048, 8192]`). The encoders map continuous `(x, y)` / `(h, w)` predictions back into embeddings at `coord_token_id` (240) and `size_token_id` (241); the decoders are two-layer squared-ReLU MLPs over 2048 coordinate/size bins.
- A box decoding loop: coordinates and sizes are decoded as bin argmaxes at `coord_token_id` / `size_token_id`, and a box prediction within 1% of an earlier box is rejected and resampled (up to 100 attempts).
- Large checkpoint only (78 additional tensors, omitted in the 300M safetensors file): segmentation on `seg_token_id` (262) via `proj_segm.{layers.0, layers.1, pixel_layer}`, `conv_segm.{weight,bias}` (`conv_segm.weight` stored as `[256, 1024, 3, 3]`, requiring the standard `(0, 2, 3, 1)` channels-last permutation at load), and `itok_upsampler.*` (the 479-line AnyUp cross-attention feature upsampler in `anyup.py` with `image_encoder`, `query_encoder`, `cross_decode`, `aggregation`, and `rope.freqs`, whose 4-D `1x1` and `3x3` conv weights also require channels-last permutation).
- Prompt contract: `<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>`.

### Suggested split and open decisions

1. Port `tiiuae/Falcon-Perception-300M` (detection only) first: reuse `FalconOcrTextModel` / `FalconOcrProcessor` and add the 8 Fourier/MLP head tensors and the box decoding loop, surfacing boxes through `mlxcel detect` (`src/commands/detect.rs`).
2. Scope the `tiiuae/Falcon-Perception` segmentation path (`conv_segm`, `proj_segm`, `itok_upsampler`, RLE masks) separately once the instance-mask output contract from epic `lablup/mlxcel#2382` (`lablup/mlxcel#2429`) is in place.

---

## `zaya1_vl`

- **Checkpoint**: `Zyphra/ZAYA1-VL-8B` (19.4 GB across 4 bf16 safetensors shards, 5,833 tensors).
- **Confirmed `config.json` keys**: top-level `model_type: "zaya1_vl"`, `architectures: ["Zaya1VLForConditionalGeneration"]`, `hidden_size: 2048`, `num_hidden_layers: 40`, `num_attention_heads: 8`, `num_key_value_heads: 2`, `num_query_groups: 2`, `head_dim: 128`, `ffn_hidden_size: 4096`, `num_experts: 16`, `moe_router_topk: 1`, `zaya_mlp_expansion: 256`, `zaya_use_eda: true`, `zaya_use_mod: true`, `cca: true`, `rope_pct: 0.5`, `rotary_base: 1000000`, `scale_residual_merge: true`, `tie_word_embeddings: true`, `vocab_size: 262272`, `image_token_id: 262147`, `vision_start_token_id: 255999`, `vision_end_token_id: 256000`, `vision_lora: true`, `vision_lora_rank_attn: 8`, `vision_lora_rank_mlp: 32`, and nested `vision_config` with `model_type: "qwen2_5_vl"` (`hidden_size: 1280`, `out_hidden_size: 2048`, `spatial_patch_size: 14`, `temporal_patch_size: 1`).
- **Confirmed weight layout**:
  - Vision tower (`vision_tower.patch_embed.proj.weight` `[1280, 3, 1, 14, 14]`, `vision_tower.blocks.{0..31}.*`, `vision_tower.merger.{ln_q, mlp.0, mlp.2}`) matches `src/vision/encoders/qwen2_5_vl.rs`.
  - Text decoder (`model.layers.{0..39}.*`, 133 tensors per layer):
    - Convolution-augmented attention (`attn.self_attn.qkv.*`): `linear_q.weight` `[1024, 2048]`, `linear_k.weight` `[256, 2048]`, causal 1-D convolutions `conv_qk.0.{weight,bias}` (`[1280, 1, 2]`) and `conv_qk.1.{weight,bias}` (`[1280, 128, 2]`, requiring `(out, in/groups, k)` to `(out, k, in/groups)` permutation), dual value projections `val_proj1.weight` and `val_proj2.weight` (`[128, 2048]` each) for time-shifted hidden states, learned per-KV-head temperature `qkv.temp` `[2]`, output projection `o_proj.weight` `[2048, 1024]`, and rank-8 vision LoRA adapters (`lora_linear_{q,k,o}.{0,1}.weight`, `lora_val_proj{1,2}.{0,1}.weight`).
    - MoE + Mixture-of-Depths block (`mlp.zaya_block.*`): router (`router.down_proj.{weight,bias}` `[256, 2048]`, `router.rmsnorm_eda.weight` `[256]`, `router.router_mlp.{0,2,4}` projecting `256 -> 256 -> 256 -> 17`, `router.balancing_biases` `[17]` in fp32) where the 17th slot is the MoD skip expert, plus 16 SwiGLU experts `experts.local_experts.{0..15}.{linear_fc1, linear_fc2}.weight` (`[4096, 2048]`, `[2048, 2048]`) each carrying rank-32 vision LoRA adapters (`lora_fc1.{0,1}.weight`, `lora_fc2.{0,1}.weight`).
    - Per-sublayer residual scaling (`attn.res_scale.{hidden_states_scale, hidden_states_bias}` and `mlp.res_scale.{hidden_states_scale, hidden_states_bias, residual_scale, residual_bias}`), plus top-level `model.res_scale.*`.
- **Port delta**: largest on the `lablup/mlxcel#548` list. While the vision tower reuses `qwen2_5_vl.rs`, the text backbone requires a custom hybrid per-layer cache (KV cache + 2-step causal conv/token-shift state), cross-layer router-state carryover (`zaya_use_eda`), MoD skip-expert routing (`zaya_use_mod`), and image-token-masked LoRA branches in both attention and experts.

---

## `hrm_text`

- **Checkpoints**: `sapientinc/HRM-Text-1B` (131 bf16 tensors) and `Xiaoye08/HRM-Text-0.6B` (99 bf16 tensors, ships `configuration_hrm_text.py` and `modeling_hrm_text.py`); `sapientinc/HRM-Text-1B-v3.2` is gated.
- **Confirmed `config.json` keys**:
  - `sapientinc/HRM-Text-1B`: `model_type: "hrm_text"`, `architectures: ["HrmTextForCausalLM"]`, `vocab_size: 65536` (not 151808 as initially noted in `lablup/mlxcel#548`), `hidden_size: 1536`, `intermediate_size: 4096`, `num_hidden_layers: 16`, `num_attention_heads: 12`, `num_key_value_heads: 12`, `head_dim: 128`, `H_cycles: 2`, `L_cycles: 3`, `L_bp_cycles: [0, 3]`, `embedding_scale: 39.191835884530846`, `prefix_lm: true`, `rms_norm_eps: 1e-6`, `rope_theta: 10000.0`, `tie_word_embeddings: false`.
  - `Xiaoye08/HRM-Text-0.6B`: same schema with `hidden_size: 1280`, `intermediate_size: 3584`, `num_hidden_layers: 12`, `num_attention_heads: 10`, `num_key_value_heads: 10`, `head_dim: 128`, and `L_bp_steps: [0, 3]` as the key alias for `L_bp_cycles`.
- **Confirmed weight layout**: `model.embed_tokens.weight` (`[65536, hidden_size]`), `model.z_L_init` (`[hidden_size]`), `lm_head.weight` (`[65536, hidden_size]`), and two separate `num_hidden_layers`-deep stacks `model.H_module.layers.{0..N-1}.*` and `model.L_module.layers.{0..N-1}.*`. Each layer has 4 weight tensors (norms are non-parametric): `attn.gqkv_proj.weight` (`[4 * hidden_size, hidden_size]`, fused gate + Q + K + V), `attn.o_proj.weight` (`[hidden_size, hidden_size]`), `mlp.gate_up_proj.weight` (`[2 * intermediate_size, hidden_size]`), and `mlp.down_proj.weight` (`[hidden_size, intermediate_size]`).
- **Port delta**: individual layer math is compact (fused `gqkv_proj` with sigmoid attention output gate, fused `gate_up_proj` SwiGLU, weightless RMSNorm), but every token step replays `H_cycles * (L_cycles + 1) = 8` module evaluations (`128` effective layer evaluations on 1B) with recurrent state addition (`z_L`, `z_H`) and prefix-LM bidirectional prompt masking (`prefix_lm: true`), requiring dedicated recurrent cache management.

---

## `multi_modality` (DeepSeek-VL v1)

- **Checkpoints**: `deepseek-ai/deepseek-vl-1.3b-chat` (529 fp16 tensors; `mlx-community/deepseek-vl-1.3b-chat-4bit`) and `deepseek-ai/deepseek-vl-7b-chat` (775 fp16 tensors across 3 shards, 14.7 GB; `mlx-community/deepseek-vl-7b-chat-4bit`).
- **Confirmed `config.json` and weight layout**:
  - Both checkpoints ship top-level `model_type: "multi_modality"`, `architectures: ["MultiModalityCausalLM"]`, and `language_config` with `model_type: "llama"` and `vocab_size: 102400` (24 layers × 2048 on 1.3B; 30 layers × 4096 on 7B). Text weights live under `language_model.model.*` and `language_model.lm_head.weight` and map directly onto `src/models/llama3.rs`.
  - **1.3B (`CLIPVisionTower` + `mlp_gelu`)**: single 384px SigLIP-L/16 tower (`vision_config.params.model_name: "siglip_large_patch16_384"`) under `vision_model.vision_tower.{patch_embed.proj, pos_embed, blocks.0..23, norm, attn_pool}` (`patch_embed.proj.weight` `[1024, 3, 16, 16]`, `pos_embed` `[1, 576, 1024]`, fused `attn.qkv`) plus a 2-layer GELU MLP projector `aligner.layers.{0,2}.{weight,bias}` (`1024 -> 2048 -> 2048`). This vision tower has the exact key and block structure already implemented for DeepSeek-VL2 (`lablup/mlxcel#533`) in `src/vision/encoders/deepseek_vl2.rs`.
  - **7B (`HybridVisionTower` + `low_high_hybrid_split_mlp_gelu`)**: pairs the 384px SigLIP tower (`vision_model.vision_tower_low.vision_tower.*`) with a 1024px SAM-B high-resolution tower (`vision_model.vision_tower_high.vision_tower.{patch_embed, pos_embed, blocks.0..11, neck, neck_hd, downsamples, hd_alpha_downsamples}` using windowed relative-position attention `rel_pos_h` / `rel_pos_w` and conv downsamplers) and a split projector (`vision_model.{low,high}_layer_norm.{weight,bias}`, `aligner.{low,high}_up_proj.{weight,bias}`, `aligner.layers.1.{weight,bias}`).
- **Disposition context**: DeepSeek-VL2 (`model_type: "deepseek_vl_v2"`, `lablup/mlxcel#533`) is already supported in `src/vision/deepseek_vl2.rs` and `src/loading/vlm_deepseek_vl2.rs` and supersedes v1. If v1 support is ever desired, the 1.3B variant is a small wrapper over `src/models/llama3.rs` and `src/vision/encoders/deepseek_vl2.rs`, whereas the 7B variant requires porting the `sam_b_downsample` encoder and `low_high_hybrid_split_mlp_gelu` aligner.
