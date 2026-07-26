// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Molmo2 Vision Encoder + Adapter
//!
//! Architecture:
//! - ViT: 25 transformer blocks, Linear patch embedding (not Conv2d),
//!   positional embedding with bicubic interpolation
//! - Adapter: Attention pooling 2D + SwiGLU image projector
//! - Layer selection: [-3, -9] over the declared 27 layers = [24, 18],
//!   then concatenate → pool_dim = 2*1152
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo2/vision.py

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
const MOLMO2_VIT_PROBE_LAYER: usize = 24;
#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
// The pinned actual failure was flat index 591490 at hidden width 1152:
// 591490 = 513 * 1152 + 514. Snapshot the whole row at producer boundaries.
const MOLMO2_VIT_PROBE_FLAT_ROW: usize = 513;

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn diagnostic_flat_row_snapshot(
    value: &MlxArray,
    tokens_per_crop: usize,
    flat_row: usize,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(value);
    assert_eq!(
        shape.len(),
        3,
        "Molmo2 ViT probe requires [crop, token, hidden]"
    );
    let crop = flat_row / tokens_per_crop;
    let token = flat_row % tokens_per_crop;
    assert!(
        crop < usize::try_from(shape[0]).expect("non-negative Molmo2 crop count"),
        "Molmo2 ViT probe row is outside the active crops"
    );
    let hidden = shape[2];
    let row = mlxcel_core::slice(
        value,
        &[crop as i32, token as i32, 0],
        &[crop as i32 + 1, token as i32 + 1, hidden],
    );
    let row = mlxcel_core::reshape(&row, &[1, hidden]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    let raw = mlxcel_core::array_to_raw_bytes(&row);
    let values = raw
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 probe value")))
        .collect::<Vec<_>>();
    assert_eq!(
        values.len(),
        usize::try_from(hidden).expect("non-negative Molmo2 hidden width")
    );
    mlxcel_core::from_slice_f32(&values, &[1, hidden])
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
fn diagnostic_probe_stage(
    stage: &'static str,
    value: &MlxArray,
    tokens_per_crop: usize,
) -> (&'static str, UniquePtr<MlxArray>) {
    (
        stage,
        diagnostic_flat_row_snapshot(value, tokens_per_crop, MOLMO2_VIT_PROBE_FLAT_ROW),
    )
}

/// Hugging Face's `gelu_pytorch_tanh`, evaluated in F32 so the eager Molmo2
/// reference matches both the checkpoint's declared activation and StableHLO.
fn gelu_pytorch_tanh(x: &MlxArray) -> UniquePtr<MlxArray> {
    let output_dtype = mlxcel_core::array_dtype(x);
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    let sqrt_two_over_pi = mlxcel_core::full_f32(&[1], 0.797_884_6, mlxcel_core::dtype::FLOAT32);
    let cubic_coefficient = mlxcel_core::full_f32(&[1], 0.044_715, mlxcel_core::dtype::FLOAT32);
    let squared = mlxcel_core::multiply(&x, &x);
    let cubed = mlxcel_core::multiply(&squared, &x);
    let cubic = mlxcel_core::multiply(&cubic_coefficient, &cubed);
    let inner = mlxcel_core::multiply(&sqrt_two_over_pi, &mlxcel_core::add(&x, &cubic));
    let cdf = mlxcel_core::multiply(&half, &mlxcel_core::add(&one, &mlxcel_core::tanh(&inner)));
    let activated = mlxcel_core::multiply(&x, &cdf);
    if output_dtype == mlxcel_core::dtype::FLOAT32 {
        activated
    } else {
        mlxcel_core::astype(&activated, output_dtype)
    }
}

// ViT MLP.
pub(crate) struct ViTMLP {
    w1: Linear,
    w2: Linear,
}

impl ViTMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.w1.forward(x);
        let h = gelu_pytorch_tanh(&h);
        self.w2.forward(&h)
    }

    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let w1 = Linear::from_weights(weights, &format!("{}.w1", prefix))?;
        let w2 = Linear::from_weights(weights, &format!("{}.w2", prefix))?;
        Ok(Self { w1, w2 })
    }
}

// ViT Multi-Head Dot Product Attention (supports cross-attention).
// Used by: Molmo2, MolmoPoint
pub(crate) struct ViTAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    float32_attention: bool,
}

impl ViTAttention {
    fn forward(
        &self,
        inputs_q: &MlxArray,
        inputs_kv: Option<&MlxArray>,
        attn_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let inputs_k = inputs_kv.unwrap_or(inputs_q);
        let inputs_v = inputs_kv.unwrap_or(inputs_q);

        let xq = self.wq.forward(inputs_q);
        let xk = self.wk.forward(inputs_k);
        let xv = self.wv.forward(inputs_v);

        let q_shape = mlxcel_core::array_shape(&xq);
        let bsz = q_shape[0];
        let q_len = q_shape[1];
        let k_shape = mlxcel_core::array_shape(&xk);
        let kv_len = k_shape[1];

        let xq = mlxcel_core::reshape(&xq, &[bsz, q_len, self.num_heads, self.head_dim]);
        let mut xk = mlxcel_core::reshape(&xk, &[bsz, kv_len, self.num_kv_heads, self.head_dim]);
        let mut xv = mlxcel_core::reshape(&xv, &[bsz, kv_len, self.num_kv_heads, self.head_dim]);

        // Repeat KV heads if GQA
        if self.num_heads != self.num_kv_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            xk = mlxcel_core::repeat(&xk, n_rep, 2);
            xv = mlxcel_core::repeat(&xv, n_rep, 2);
        }

        // Transpose to [B, heads, L, head_dim]
        let q = mlxcel_core::transpose_axes(&xq, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&xk, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&xv, &[0, 2, 1, 3]);

        // Float32 attention for stability (only convert when needed)
        let dtype = mlxcel_core::array_dtype(inputs_q);
        let (q, k, v) = if self.float32_attention {
            (
                mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32),
                mlxcel_core::astype(&k, mlxcel_core::dtype::FLOAT32),
                mlxcel_core::astype(&v, mlxcel_core::dtype::FLOAT32),
            )
        } else {
            (q, k, v)
        };

        // Use fast SDPA kernel instead of manual matmul+softmax
        let mask_ptr = attn_mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        // Cast back to input dtype if needed
        let out = if self.float32_attention {
            mlxcel_core::astype(&out, dtype)
        } else {
            out
        };

        // Transpose and reshape back
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[bsz, q_len, self.num_heads * self.head_dim]);
        self.wo.forward(&out)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let wq = Linear::from_weights(weights, &format!("{}.wq", prefix))?;
        let wk = Linear::from_weights(weights, &format!("{}.wk", prefix))?;
        let wv = Linear::from_weights(weights, &format!("{}.wv", prefix))?;
        let wo = Linear::from_weights(weights, &format!("{}.wo", prefix))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            float32_attention,
        })
    }
}

// ViT Block.
// Used by: Molmo2, MolmoPoint
pub(crate) struct Molmo2VisionBlock {
    attention: ViTAttention,
    feed_forward: ViTMLP,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl Molmo2VisionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Pre-norm attention + residual
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, None, None);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm MLP + residual
        let normed = self.ffn_norm.forward(&h);
        let mlp_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    fn forward_probe(
        &self,
        x: &MlxArray,
        tokens_per_crop: usize,
    ) -> (
        UniquePtr<MlxArray>,
        Vec<(&'static str, UniquePtr<MlxArray>)>,
    ) {
        let mut stages = vec![diagnostic_probe_stage("input", x, tokens_per_crop)];
        let normed = self.attention_norm.forward(x);
        stages.push(diagnostic_probe_stage(
            "attention_norm",
            &normed,
            tokens_per_crop,
        ));
        let attn_out = self.attention.forward(&normed, None, None);
        stages.push(diagnostic_probe_stage(
            "attention",
            &attn_out,
            tokens_per_crop,
        ));
        let residual = mlxcel_core::add(x, &attn_out);
        stages.push(diagnostic_probe_stage(
            "post_attention_residual",
            &residual,
            tokens_per_crop,
        ));
        let normed = self.ffn_norm.forward(&residual);
        stages.push(diagnostic_probe_stage("ffn_norm", &normed, tokens_per_crop));
        let mlp_out = self.feed_forward.forward(&normed);
        stages.push(diagnostic_probe_stage("mlp", &mlp_out, tokens_per_crop));
        let output = mlxcel_core::add(&residual, &mlp_out);
        stages.push(diagnostic_probe_stage("output", &output, tokens_per_crop));
        (output, stages)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _hidden_size: i32,
        _intermediate_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        layer_norm_eps: f32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let attention = ViTAttention::from_weights(
            weights,
            &format!("{}.attention", prefix),
            _hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            float32_attention,
        )?;
        let feed_forward = ViTMLP::from_weights(weights, &format!("{}.feed_forward", prefix))?;

        let attn_norm_w = get_weight_copy(weights, &format!("{}.attention_norm.weight", prefix))?;
        let attn_norm_b = weights
            .get(&format!("{}.attention_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let ffn_norm_w = get_weight_copy(weights, &format!("{}.ffn_norm.weight", prefix))?;
        let ffn_norm_b = weights
            .get(&format!("{}.ffn_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let attention_norm = LayerNorm::new(attn_norm_w, attn_norm_b, layer_norm_eps);
        let ffn_norm = LayerNorm::new(ffn_norm_w, ffn_norm_b, layer_norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

// Vision Transformer (returns all hidden states).
// Used by: Molmo2, MolmoPoint
pub(crate) struct Molmo2VisionTransformer {
    patch_embedding: Linear, // Linear, not Conv2d (patches already flattened)
    positional_embedding: UniquePtr<MlxArray>, // [image_num_pos, hidden_size]
    blocks: Vec<Molmo2VisionBlock>,
    image_num_pos: usize,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
struct Molmo2VitDiagnostics {
    patch_embedding: UniquePtr<MlxArray>,
    position_embedding: UniquePtr<MlxArray>,
    positioned_embedding: UniquePtr<MlxArray>,
    probe_rows: Vec<(&'static str, UniquePtr<MlxArray>)>,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
type Molmo2VitCapture = Option<Molmo2VitDiagnostics>;
#[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
type Molmo2VitCapture = ();

impl Molmo2VisionTransformer {
    fn position_embedding(&self, x: &MlxArray, patch_h: i32, patch_w: i32) -> UniquePtr<MlxArray> {
        let num_pos = self.image_num_pos as i32;
        let hidden_size = mlxcel_core::array_shape(&self.positional_embedding)[1];

        // For default size, use positional embedding directly
        // For non-default sizes, truncate/extend (bicubic interpolation would
        // require additional FFI but default 378x378 crops are always used)
        let num_patches = patch_h * patch_w;
        let pos_emb = if num_patches == num_pos {
            mlxcel_core::copy(&self.positional_embedding)
        } else if num_patches < num_pos {
            // Truncate
            let indices: Vec<i32> = (0..num_patches).collect();
            let idx = mlxcel_core::from_slice_i32(&indices, &[num_patches]);
            mlxcel_core::take(&self.positional_embedding, &idx, 0)
        } else {
            // For larger sizes, repeat last position (rare case)
            mlxcel_core::copy(&self.positional_embedding)
        };

        // x + pos_emb[None, :, :]
        let pos_emb = mlxcel_core::reshape(&pos_emb, &[1, num_patches.min(num_pos), hidden_size]);
        mlxcel_core::astype(&pos_emb, mlxcel_core::array_dtype(x))
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        patch_num: Option<(i32, i32)>,
    ) -> Vec<UniquePtr<MlxArray>> {
        self.forward_inner::<false>(x, patch_num).0
    }

    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    fn forward_diagnostics(
        &self,
        x: &MlxArray,
        patch_num: Option<(i32, i32)>,
    ) -> (Vec<UniquePtr<MlxArray>>, Molmo2VitDiagnostics) {
        let (hidden_states, capture) = self.forward_inner::<true>(x, patch_num);
        (
            hidden_states,
            capture.expect("Molmo2 ViT diagnostics requested a capture"),
        )
    }

    fn forward_inner<const CAPTURE: bool>(
        &self,
        x: &MlxArray,
        patch_num: Option<(i32, i32)>,
    ) -> (Vec<UniquePtr<MlxArray>>, Molmo2VitCapture) {
        let default_patch_size = (self.image_num_pos as f64).sqrt() as i32;
        let (patch_h, patch_w) = patch_num.unwrap_or((default_patch_size, default_patch_size));

        let patch_embedding = self.patch_embedding.forward(x);
        let position_embedding = self.position_embedding(&patch_embedding, patch_h, patch_w);
        let mut x = mlxcel_core::add(&patch_embedding, &position_embedding);
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let mut capture = CAPTURE.then(|| Molmo2VitDiagnostics {
            patch_embedding: mlxcel_core::copy(
                patch_embedding
                    .as_ref()
                    .expect("Molmo2 patch embedding must be materialized"),
            ),
            position_embedding: mlxcel_core::copy(
                mlxcel_core::reshape(
                    &position_embedding,
                    &[
                        -1,
                        mlxcel_core::array_shape(&position_embedding)
                            .last()
                            .copied()
                            .expect("Molmo2 position embedding has a hidden dimension"),
                    ],
                )
                .as_ref()
                .expect("Molmo2 position embedding must be materialized"),
            ),
            positioned_embedding: mlxcel_core::copy(
                x.as_ref()
                    .expect("Molmo2 positioned embedding must be materialized"),
            ),
            probe_rows: Vec::new(),
        });

        let mut hidden_states = Vec::with_capacity(self.blocks.len());
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let tokens_per_crop =
            usize::try_from(patch_h * patch_w).expect("positive Molmo2 patch grid");
        for (layer, block) in self.blocks.iter().enumerate() {
            #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
            let _ = layer;
            #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
            if CAPTURE
                && layer == MOLMO2_VIT_PROBE_LAYER
                && usize::try_from(mlxcel_core::array_shape(&x)[0])
                    .expect("non-negative Molmo2 crop count")
                    * tokens_per_crop
                    > MOLMO2_VIT_PROBE_FLAT_ROW
            {
                let (output, probe_rows) = block.forward_probe(&x, tokens_per_crop);
                x = output;
                capture
                    .as_mut()
                    .expect("Molmo2 probe requires diagnostics capture")
                    .probe_rows = probe_rows;
            } else {
                x = block.forward(&x);
            }
            #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
            {
                x = block.forward(&x);
            }
            hidden_states.push(mlxcel_core::copy(&x));
        }
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        {
            (hidden_states, capture)
        }
        #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
        {
            (hidden_states, ())
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_layers: usize,
        hidden_size: i32,
        intermediate_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        image_num_pos: usize,
        layer_norm_eps: f32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let patch_embedding =
            Linear::from_weights(weights, &format!("{}.patch_embedding", prefix))?;
        let positional_embedding =
            get_weight_copy(weights, &format!("{}.positional_embedding", prefix))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = Molmo2VisionBlock::from_weights(
                weights,
                &format!("{}.transformer.{}", prefix, i),
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                layer_norm_eps,
                float32_attention,
            )?;
            blocks.push(block);
        }

        Ok(Self {
            patch_embedding,
            positional_embedding,
            blocks,
            image_num_pos,
        })
    }
}

// Image Projector MLP (SwiGLU).
// Used by: Molmo2, MolmoPoint
pub(crate) struct ImageProjectorMLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
struct ImageProjectorDiagnostics {
    gate_linear: UniquePtr<MlxArray>,
    gate_activation: UniquePtr<MlxArray>,
    up_linear: UniquePtr<MlxArray>,
    product: UniquePtr<MlxArray>,
    output: UniquePtr<MlxArray>,
}

impl ImageProjectorMLP {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // silu(w1(x)) * w3(x) → w2(...)
        let gate = self.w1.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.w3.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.w2.forward(&h)
    }

    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    fn forward_diagnostics(
        &self,
        x: &MlxArray,
    ) -> (UniquePtr<MlxArray>, ImageProjectorDiagnostics) {
        let gate_linear = self.w1.forward(x);
        let gate_activation = mlxcel_core::silu(&gate_linear);
        let up_linear = self.w3.forward(x);
        let product = mlxcel_core::multiply(&gate_activation, &up_linear);
        let output = self.w2.forward(&product);
        let projector_width = mlxcel_core::array_shape(&gate_linear)
            .last()
            .copied()
            .expect("Molmo2 projector gate must have a feature dimension");
        let output_width = mlxcel_core::array_shape(&output)
            .last()
            .copied()
            .expect("Molmo2 projector output must have a feature dimension");
        let captured_output = mlxcel_core::reshape(&output, &[-1, output_width]);
        (
            output,
            ImageProjectorDiagnostics {
                gate_linear: mlxcel_core::reshape(&gate_linear, &[-1, projector_width]),
                gate_activation: mlxcel_core::reshape(&gate_activation, &[-1, projector_width]),
                up_linear: mlxcel_core::reshape(&up_linear, &[-1, projector_width]),
                product: mlxcel_core::reshape(&product, &[-1, projector_width]),
                output: captured_output,
            },
        )
    }

    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let w1 = Linear::from_weights(weights, &format!("{}.w1", prefix))?;
        let w2 = Linear::from_weights(weights, &format!("{}.w2", prefix))?;
        let w3 = Linear::from_weights(weights, &format!("{}.w3", prefix))?;
        Ok(Self { w1, w2, w3 })
    }
}

// Molmo2 Vision Model (ViT + Adapter).
pub struct Molmo2VisionModel {
    image_vit: Molmo2VisionTransformer,
    image_pooling_2d: ViTAttention,
    image_projector: ImageProjectorMLP,
    vit_layers: Vec<usize>, // Which ViT layers to extract features from
    pooling_attention_mask: bool,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
pub struct Molmo2VisionDiagnosticTensor {
    pub name: String,
    pub tensor: UniquePtr<MlxArray>,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
pub struct Molmo2VisionDiagnostics {
    pub stages: Vec<Molmo2VisionDiagnosticTensor>,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
struct Molmo2EncodeDiagnostics {
    vit: Molmo2VitDiagnostics,
    early_block: UniquePtr<MlxArray>,
    selected_layers: Vec<(usize, UniquePtr<MlxArray>)>,
    concatenated_features: UniquePtr<MlxArray>,
}

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
type Molmo2EncodeCapture = Option<Molmo2EncodeDiagnostics>;
#[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
type Molmo2EncodeCapture = ();

#[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
type Molmo2ForwardCapture = Option<Molmo2VisionDiagnostics>;
#[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
type Molmo2ForwardCapture = ();

impl Molmo2VisionModel {
    fn encode_image_inner<const CAPTURE: bool>(
        &self,
        images: &MlxArray,
    ) -> (UniquePtr<MlxArray>, Molmo2EncodeCapture) {
        let shape = mlxcel_core::array_shape(images);
        let batch_size = shape[0];
        let num_crops = shape[1];
        let num_patch = shape[2];
        let patch_dim = shape[3];

        // Reshape to [B*num_crops, num_patch, patch_dim]
        let flat = mlxcel_core::reshape(images, &[batch_size * num_crops, num_patch, patch_dim]);
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let (hidden_states, vit_capture) = if CAPTURE {
            let (hidden_states, capture) = self.image_vit.forward_diagnostics(&flat, None);
            (hidden_states, Some(capture))
        } else {
            (self.image_vit.forward(&flat, None), None)
        };
        #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
        let hidden_states = self.image_vit.forward(&flat, None);

        // Select and concatenate features from specified layers
        let features: Vec<&MlxArray> = self
            .vit_layers
            .iter()
            .map(|&layer| hidden_states[layer].as_ref().unwrap())
            .collect();

        let image_features = if features.len() == 1 {
            mlxcel_core::copy(features[0])
        } else {
            // Concatenate along last dimension
            let mut result = mlxcel_core::copy(features[0]);
            for &feat in &features[1..] {
                result = mlxcel_core::concatenate(&result, feat, -1);
            }
            result
        };

        // Reshape back to [B, num_crops, num_patch, features_dim]
        let feat_dim = mlxcel_core::array_shape(&image_features);
        let last_dim = feat_dim[feat_dim.len() - 1];
        let output = mlxcel_core::reshape(
            &image_features,
            &[batch_size, num_crops, num_patch, last_dim],
        );
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        if CAPTURE {
            let early_block = hidden_states
                .first()
                .and_then(|state| state.as_ref())
                .map(mlxcel_core::copy)
                .expect("Molmo2 diagnostics require at least one ViT block");
            let selected_layers = self
                .vit_layers
                .iter()
                .map(|&layer| {
                    (
                        layer,
                        mlxcel_core::copy(
                            hidden_states[layer]
                                .as_ref()
                                .expect("Molmo2 selected layer must be materialized"),
                        ),
                    )
                })
                .collect();
            let concatenated_features = mlxcel_core::reshape(
                &image_features,
                &[batch_size * num_crops * num_patch, last_dim],
            );
            return (
                output,
                Some(Molmo2EncodeDiagnostics {
                    vit: vit_capture.expect("Molmo2 ViT capture must exist"),
                    early_block,
                    selected_layers,
                    concatenated_features,
                }),
            );
        }
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        {
            (output, None)
        }
        #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
        {
            (output, ())
        }
    }

    /// Full forward: encode → pool → project
    pub fn forward(&self, images: &MlxArray, pooled_patches_idx: &MlxArray) -> UniquePtr<MlxArray> {
        self.forward_inner::<false>(images, pooled_patches_idx).0
    }

    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    pub fn forward_diagnostics(
        &self,
        images: &MlxArray,
        pooled_patches_idx: &MlxArray,
    ) -> (UniquePtr<MlxArray>, Molmo2VisionDiagnostics) {
        let (projected, capture) = self.forward_inner::<true>(images, pooled_patches_idx);
        (
            projected,
            capture.expect("Molmo2 vision diagnostics requested a capture"),
        )
    }

    fn forward_inner<const CAPTURE: bool>(
        &self,
        images: &MlxArray,
        pooled_patches_idx: &MlxArray,
    ) -> (UniquePtr<MlxArray>, Molmo2ForwardCapture) {
        let shape = mlxcel_core::array_shape(images);
        let batch_size = shape[0];

        let (image_features, _encode_capture) = self.encode_image_inner::<CAPTURE>(images);
        let feat_shape = mlxcel_core::array_shape(&image_features);
        let dim = feat_shape[feat_shape.len() - 1];

        // Flatten features: [B, num_crops * num_patch, dim]
        let flat_features = mlxcel_core::reshape(&image_features, &[batch_size, -1, dim]);

        // Build valid mask from pooling indices
        let pool_shape = mlxcel_core::array_shape(pooled_patches_idx);
        // pooled_patches_idx shape: [batch, num_pooled, pool_size]

        // valid = pooled_patches_idx >= 0
        let zeros = mlxcel_core::zeros_like(pooled_patches_idx);
        let valid = mlxcel_core::greater_equal(pooled_patches_idx, &zeros);
        // valid_token = any(valid, axis=-1)
        let valid_i32 = mlxcel_core::astype(&valid, mlxcel_core::dtype::INT32);
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let valid_counts = CAPTURE.then(|| {
            let valid_f32 = mlxcel_core::astype(&valid, mlxcel_core::dtype::FLOAT32);
            mlxcel_core::reshape(&mlxcel_core::sum_axis(&valid_f32, -1, false), &[-1])
        });

        // Clip indices to >= 0
        let idx = mlxcel_core::maximum(pooled_patches_idx, &zeros);

        // Gather features at pooling indices
        // idx shape: [B, num_pooled, pool_size]
        // flat_features shape: [B, total_patches, dim]
        // We need to do batched gather: for each batch, gather from flat_features using idx
        let to_pool = self.batched_gather(&flat_features, &idx, batch_size);
        // to_pool shape: [B, num_pooled, pool_size, dim]

        // Mask invalid positions
        let valid_4d =
            mlxcel_core::reshape(&valid, &[pool_shape[0], pool_shape[1], pool_shape[2], 1]);
        let valid_f = mlxcel_core::astype(&valid_4d, mlxcel_core::array_dtype(&to_pool));
        let to_pool = mlxcel_core::multiply(&to_pool, &valid_f);

        // Reshape for attention: [B * num_pooled, pool_size, dim]
        let to_pool = mlxcel_core::reshape(&to_pool, &[-1, pool_shape[2], dim]);
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let gathered_masked = CAPTURE.then(|| {
            mlxcel_core::copy(
                to_pool
                    .as_ref()
                    .expect("Molmo2 gathered features must be materialized"),
            )
        });

        // Build query: mean of valid patches per pooled position
        let (query, attn_mask) = if self.pooling_attention_mask {
            let valid_flat = mlxcel_core::reshape(&valid, &[-1, 1, 1, pool_shape[2]]);
            let valid_for_sum = mlxcel_core::reshape(&valid, &[-1, pool_shape[2]]);
            let valid_f32 = mlxcel_core::astype(&valid_for_sum, mlxcel_core::dtype::FLOAT32);
            let denom = mlxcel_core::sum_axis(&valid_f32, -1, true);
            // Clamp denom to at least 1
            let ones = mlxcel_core::ones(&[1, 1], mlxcel_core::dtype::FLOAT32);
            let denom = mlxcel_core::maximum(&denom, &ones);
            let denom = mlxcel_core::astype(&denom, mlxcel_core::array_dtype(&to_pool));
            let denom = mlxcel_core::reshape(&denom, &[-1, 1, 1]);

            // sum along pool_size axis (axis=-2 = axis=1 in 3D)
            let sum_pool = mlxcel_core::sum_axis(&to_pool, -2, true);
            let query = mlxcel_core::divide(&sum_pool, &denom);
            (query, Some(valid_flat))
        } else {
            let query = mlxcel_core::mean_axis(&to_pool, -2, true);
            (query, None)
        };
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let pooling_query = CAPTURE.then(|| mlxcel_core::reshape(&query, &[-1, dim]));

        // Cross-attention pooling
        let pooled = self.image_pooling_2d.forward(
            &query,
            Some(&to_pool),
            attn_mask.as_ref().map(|m| m.as_ref().unwrap()),
        );

        // Reshape: [B, num_pooled, hidden_size]
        let pooled_shape = mlxcel_core::array_shape(&pooled);
        let pooled_dim = pooled_shape[pooled_shape.len() - 1];
        let pooled = mlxcel_core::reshape(&pooled, &[batch_size, -1, pooled_dim]);
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let pooling_output = CAPTURE.then(|| mlxcel_core::reshape(&pooled, &[-1, pooled_dim]));

        // Project through SwiGLU MLP.
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        let (projected, projector_capture) = if CAPTURE {
            let (projected, diagnostics) = self.image_projector.forward_diagnostics(&pooled);
            (projected, Some(diagnostics))
        } else {
            (self.image_projector.forward(&pooled), None)
        };
        #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
        let projected = self.image_projector.forward(&pooled);

        // Flatten to [total_valid_tokens, output_dim]
        let proj_shape = mlxcel_core::array_shape(&projected);
        let out_dim = proj_shape[proj_shape.len() - 1];
        let projected = mlxcel_core::reshape(&projected, &[-1, out_dim]);

        // Filter valid tokens: valid_token = any(valid, axis=-1)
        // sum valid along pool_size axis, then check > 0
        let valid_sum = mlxcel_core::sum_axis(&valid_i32, -1, false);
        let zero_scalar = mlxcel_core::from_slice_i32(&[0], &[1]);
        let valid_token = mlxcel_core::greater(&valid_sum, &zero_scalar);
        let valid_flat = mlxcel_core::reshape(&valid_token, &[-1]);

        // Eval and extract valid indices on host
        mlxcel_core::eval(&valid_flat);
        let total_pooled = mlxcel_core::array_shape(&valid_flat)[0];
        let mut valid_indices: Vec<i32> = Vec::new();
        for i in 0..total_pooled {
            let idx_arr = mlxcel_core::from_slice_i32(&[i], &[1]);
            let val = mlxcel_core::take(&valid_flat, &idx_arr, 0);
            mlxcel_core::eval(&val);
            if mlxcel_core::item_bool(&val) {
                valid_indices.push(i);
            }
        }

        let active_projected = if valid_indices.is_empty() {
            mlxcel_core::zeros(&[0, out_dim], mlxcel_core::array_dtype(&projected))
        } else {
            let indices =
                mlxcel_core::from_slice_i32(&valid_indices, &[valid_indices.len() as i32]);
            mlxcel_core::take(&projected, &indices, 0)
        };

        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        if CAPTURE {
            let encode = _encode_capture.expect("Molmo2 encode capture must exist");
            let projector =
                projector_capture.expect("Molmo2 projector diagnostics must be captured");
            let mut stages = vec![
                Molmo2VisionDiagnosticTensor {
                    name: "vit.patch_embedding".to_string(),
                    tensor: encode.vit.patch_embedding,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "vit.position_embedding".to_string(),
                    tensor: encode.vit.position_embedding,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "vit.positioned_embedding".to_string(),
                    tensor: encode.vit.positioned_embedding,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "vit.block.0".to_string(),
                    tensor: encode.early_block,
                },
            ];
            let mut selected_layers = encode.selected_layers;
            selected_layers.sort_by_key(|(layer, _)| *layer);
            let probe_split =
                selected_layers.partition_point(|(layer, _)| *layer < MOLMO2_VIT_PROBE_LAYER);
            stages.extend(selected_layers.drain(..probe_split).map(|(layer, tensor)| {
                Molmo2VisionDiagnosticTensor {
                    name: format!("vit.selected.{layer}"),
                    tensor,
                }
            }));
            stages.extend(encode.vit.probe_rows.into_iter().map(|(stage, tensor)| {
                Molmo2VisionDiagnosticTensor {
                    name: format!(
                        "vit.probe.{}.row.{}.{}",
                        MOLMO2_VIT_PROBE_LAYER, MOLMO2_VIT_PROBE_FLAT_ROW, stage
                    ),
                    tensor,
                }
            }));
            stages.extend(selected_layers.into_iter().map(|(layer, tensor)| {
                Molmo2VisionDiagnosticTensor {
                    name: format!("vit.selected.{layer}"),
                    tensor,
                }
            }));
            stages.extend([
                Molmo2VisionDiagnosticTensor {
                    name: "vit.concatenated".to_string(),
                    tensor: encode.concatenated_features,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "pool.gathered_masked".to_string(),
                    tensor: gathered_masked.expect("Molmo2 gather capture must exist"),
                },
                Molmo2VisionDiagnosticTensor {
                    name: "pool.valid_counts".to_string(),
                    tensor: valid_counts.expect("Molmo2 valid-count capture must exist"),
                },
                Molmo2VisionDiagnosticTensor {
                    name: "pool.query".to_string(),
                    tensor: pooling_query.expect("Molmo2 query capture must exist"),
                },
                Molmo2VisionDiagnosticTensor {
                    name: "pool.output".to_string(),
                    tensor: pooling_output.expect("Molmo2 pool output capture must exist"),
                },
                Molmo2VisionDiagnosticTensor {
                    name: "projector.w1".to_string(),
                    tensor: projector.gate_linear,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "projector.silu".to_string(),
                    tensor: projector.gate_activation,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "projector.w3".to_string(),
                    tensor: projector.up_linear,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "projector.product".to_string(),
                    tensor: projector.product,
                },
                Molmo2VisionDiagnosticTensor {
                    name: "projector.output_all".to_string(),
                    tensor: projector.output,
                },
            ]);
            return (active_projected, Some(Molmo2VisionDiagnostics { stages }));
        }
        #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
        {
            (active_projected, None)
        }
        #[cfg(not(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu")))]
        {
            (active_projected, ())
        }
    }

    /// Batched gather: for each batch, gather from features using indices
    fn batched_gather(
        &self,
        features: &MlxArray,
        indices: &MlxArray,
        batch_size: i32,
    ) -> UniquePtr<MlxArray> {
        let idx_shape = mlxcel_core::array_shape(indices);
        let num_pooled = idx_shape[1];
        let pool_size = idx_shape[2];
        let feat_shape = mlxcel_core::array_shape(features);
        let dim = feat_shape[2];

        // Flatten indices: [B, num_pooled * pool_size]
        let flat_idx = mlxcel_core::reshape(indices, &[batch_size, num_pooled * pool_size]);

        // Build batch indices
        let mut batch_idx_data = Vec::with_capacity((batch_size * num_pooled * pool_size) as usize);
        for b in 0..batch_size {
            for _ in 0..(num_pooled * pool_size) {
                batch_idx_data.push(b);
            }
        }
        let batch_idx =
            mlxcel_core::from_slice_i32(&batch_idx_data, &[batch_size, num_pooled * pool_size]);

        // Flatten for gather
        let batch_idx_flat = mlxcel_core::reshape(&batch_idx, &[-1]);
        let flat_idx_flat = mlxcel_core::reshape(&flat_idx, &[-1]);

        // Gather: features[batch_idx, flat_idx]
        // Use advanced indexing via take
        let features_2d = mlxcel_core::reshape(features, &[batch_size * feat_shape[1], dim]);

        // Compute linear indices: batch_idx * num_patches + flat_idx
        let num_patches = mlxcel_core::from_slice_i32(&[feat_shape[1]], &[1]);
        let offset = mlxcel_core::multiply(&batch_idx_flat, &num_patches);
        let linear_idx = mlxcel_core::add(&offset, &flat_idx_flat);

        let gathered = mlxcel_core::take(&features_2d, &linear_idx, 0);

        // Reshape to [B, num_pooled, pool_size, dim]
        mlxcel_core::reshape(&gathered, &[batch_size, num_pooled, pool_size, dim])
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vit_num_layers: usize,
        vit_hidden_size: i32,
        vit_intermediate_size: i32,
        vit_num_heads: i32,
        vit_num_kv_heads: i32,
        vit_head_dim: i32,
        vit_image_num_pos: usize,
        vit_layer_norm_eps: f32,
        vit_float32_attention: bool,
        adapter_hidden_size: i32,
        _adapter_intermediate_size: i32,
        _adapter_text_hidden_size: i32,
        adapter_num_heads: i32,
        adapter_num_kv_heads: i32,
        adapter_head_dim: i32,
        adapter_float32_attention: bool,
        vit_layers: &[usize],
        pooling_attention_mask: bool,
    ) -> Result<Self, String> {
        if vit_layers.is_empty() {
            return Err("Molmo2 adapter must select at least one ViT layer".to_string());
        }
        if let Some(&layer) = vit_layers.iter().find(|&&layer| layer >= vit_num_layers) {
            return Err(format!(
                "Molmo2 selected ViT layer {layer} is outside execution depth {vit_num_layers}"
            ));
        }
        let image_vit = Molmo2VisionTransformer::from_weights(
            weights,
            &format!("{}.image_vit", prefix),
            vit_num_layers,
            vit_hidden_size,
            vit_intermediate_size,
            vit_num_heads,
            vit_num_kv_heads,
            vit_head_dim,
            vit_image_num_pos,
            vit_layer_norm_eps,
            vit_float32_attention,
        )?;

        // Pool dim = hidden_size * len(vit_layers)
        let _pool_dim = vit_hidden_size * vit_layers.len() as i32;

        let image_pooling_2d = ViTAttention::from_weights(
            weights,
            &format!("{}.image_pooling_2d", prefix),
            adapter_hidden_size,
            adapter_num_heads,
            adapter_num_kv_heads,
            adapter_head_dim,
            adapter_float32_attention,
        )?;

        let image_projector =
            ImageProjectorMLP::from_weights(weights, &format!("{}.image_projector", prefix))?;

        Ok(Self {
            image_vit,
            image_pooling_2d,
            image_projector,
            vit_layers: vit_layers.to_vec(),
            pooling_attention_mask,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

#[cfg(test)]
mod tests {
    use super::gelu_pytorch_tanh;
    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    use super::{MOLMO2_VIT_PROBE_FLAT_ROW, diagnostic_flat_row_snapshot};

    #[test]
    fn molmo2_uses_the_checkpoint_pytorch_tanh_gelu() {
        let input = mlxcel_core::from_slice_f32(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
        let output = gelu_pytorch_tanh(&input);
        mlxcel_core::eval(&output);
        let expected = [-0.003_637_433, -0.158_808, 0.0, 0.841_192, 2.996_362_7];
        for (index, expected) in expected.into_iter().enumerate() {
            let value = mlxcel_core::slice(&output, &[index as i32], &[index as i32 + 1]);
            assert!(
                (mlxcel_core::item_f32(&value) - expected).abs() <= 2.0e-6,
                "Molmo2 GELU mismatch at {index}"
            );
        }
    }

    #[cfg(any(feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
    #[test]
    fn diagnostic_probe_snapshots_the_row_containing_the_actual_failure() {
        let values = (0..(MOLMO2_VIT_PROBE_FLAT_ROW + 1) * 2)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let input =
            mlxcel_core::from_slice_f32(&values, &[1, (MOLMO2_VIT_PROBE_FLAT_ROW + 1) as i32, 2]);
        let row = diagnostic_flat_row_snapshot(
            &input,
            MOLMO2_VIT_PROBE_FLAT_ROW + 1,
            MOLMO2_VIT_PROBE_FLAT_ROW,
        );
        assert_eq!(mlxcel_core::array_shape(&row), vec![1, 2]);
        let bytes = mlxcel_core::array_to_raw_bytes(&row);
        let actual = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                (MOLMO2_VIT_PROBE_FLAT_ROW * 2) as f32,
                (MOLMO2_VIT_PROBE_FLAT_ROW * 2 + 1) as f32,
            ]
        );
    }
}
