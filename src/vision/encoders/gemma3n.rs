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

//! Gemma 3n MobileNetV5 vision encoder
//!
//! MobileNetV5-based vision encoder with Multi-Scale Fusion Adapter (MSFA).
//! Unlike other VLMs (ViT-based), Gemma 3n uses a convolutional architecture
//! with 4 stages of mixed Edge Residual, Universal Inverted Residual, and
//! Multi-Query Attention blocks.
//!
//! All spatial operations use NHWC layout (channels-last).
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3n/vision.py

#[path = "gemma3n_helpers.rs"]
mod helpers;

#[cfg(test)]
#[path = "gemma3n_helpers_tests.rs"]
mod helper_tests;

use self::helpers::{
    get_same_padding, get_static_padding, get_weight, is_static_pad, make_divisible,
    nearest_upsample_nchw, num_groups, sanitize_conv_weight, split_symmetric_padding,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// RMSNormAct2d — RMS normalization on channel axis + optional GELU.
/// RMS normalization operating on channel dimension (NHWC layout).
/// Internally transposes to NCHW for channel-wise normalization.
pub struct RMSNormAct2d {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
    pub apply_act: bool,
}

impl RMSNormAct2d {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // NHWC → NCHW for channel-wise normalization
        let x = mlxcel_core::transpose_axes(x, &[0, 3, 1, 2]);

        // RMS norm on channel axis (axis=1 in NCHW)
        // v = mean(x^2, axis=1, keepdims=true)
        let x_sq = mlxcel_core::square(&x);
        let v = mlxcel_core::mean_axis(&x_sq, 1, true);
        let eps_arr = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::dtype::FLOAT32);
        let v_eps = mlxcel_core::add(&v, &eps_arr);
        let rsqrt = mlxcel_core::rsqrt(&v_eps);
        let x = mlxcel_core::multiply(&x, &rsqrt);

        // Apply weight: reshape to [1, C, 1, 1] for broadcast
        let shape = mlxcel_core::array_shape(&x);
        let c = shape[1];
        let w = mlxcel_core::reshape(&self.weight, &[1, c, 1, 1]);
        let x = mlxcel_core::multiply(&x, &w);

        // Apply GELU activation if enabled
        let x = if self.apply_act {
            mlxcel_core::gelu_approx(&x)
        } else {
            x
        };

        // NCHW → NHWC
        mlxcel_core::transpose_axes(&x, &[0, 2, 3, 1])
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        apply_act: bool,
    ) -> Result<Self, String> {
        let weight = get_weight(weights, &format!("{}.weight", prefix))?;
        Ok(Self {
            weight,
            eps: 1e-5,
            apply_act,
        })
    }

    pub fn from_weights_eps(
        weights: &WeightMap,
        prefix: &str,
        apply_act: bool,
        eps: f32,
    ) -> Result<Self, String> {
        let weight = get_weight(weights, &format!("{}.weight", prefix))?;
        Ok(Self {
            weight,
            eps,
            apply_act,
        })
    }
}

// Conv2d operations (regular and "same" padding).
/// Regular Conv2d layer with stored weights
pub struct Conv2dLayer {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    pub stride_h: i32,
    pub stride_w: i32,
    pub padding_h: i32,
    pub padding_w: i32,
    pub dilation_h: i32,
    pub dilation_w: i32,
    pub groups: i32,
}

impl Conv2dLayer {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            self.groups,
        );
        if let Some(ref bias) = self.bias {
            mlxcel_core::add(&y, bias)
        } else {
            y
        }
    }
}

/// Conv2d with dynamic "same" padding
pub struct Conv2dSame {
    pub weight: UniquePtr<MlxArray>,
    pub bias: Option<UniquePtr<MlxArray>>,
    pub kernel_h: i32,
    pub kernel_w: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    pub dilation_h: i32,
    pub dilation_w: i32,
    pub groups: i32,
    // If stride==1, use static padding instead of dynamic
    pub use_static_pad: bool,
    pub static_pad_h: i32,
    pub static_pad_w: i32,
}

impl Conv2dSame {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = if self.use_static_pad {
            // Static padding via conv2d padding params (handled in conv2d call)
            mlxcel_core::conv2d(
                x,
                &self.weight,
                self.stride_h,
                self.stride_w,
                self.static_pad_h,
                self.static_pad_w,
                self.dilation_h,
                self.dilation_w,
                self.groups,
            )
        } else {
            // Dynamic "same" padding: compute pad sizes from input shape
            let shape = mlxcel_core::array_shape(x);
            let ih = shape[1]; // NHWC
            let iw = shape[2];
            let pad_h = get_same_padding(ih, self.kernel_h, self.stride_h, self.dilation_h);
            let pad_w = get_same_padding(iw, self.kernel_w, self.stride_w, self.dilation_w);
            let (pad_h_before, pad_h_after) = split_symmetric_padding(pad_h);
            let (pad_w_before, pad_w_after) = split_symmetric_padding(pad_w);

            // Pad: [batch_before, batch_after, h_before, h_after, w_before, w_after, c_before, c_after]
            let pad_width = [
                0,
                0, // batch
                pad_h_before,
                pad_h_after, // height
                pad_w_before,
                pad_w_after, // width
                0,
                0, // channels
            ];
            let padded = mlxcel_core::pad(x, &pad_width, 0.0);

            mlxcel_core::conv2d(
                &padded,
                &self.weight,
                self.stride_h,
                self.stride_w,
                0,
                0,
                self.dilation_h,
                self.dilation_w,
                self.groups,
            )
        };

        if let Some(ref bias) = self.bias {
            mlxcel_core::add(&x, bias)
        } else {
            x
        }
    }
}

// ConvNormAct — Conv2d + RMSNormAct2d.
pub enum ConvType {
    Regular(Conv2dLayer),
    Same(Conv2dSame),
}

pub struct ConvNormAct {
    pub conv: ConvType,
    pub bn: RMSNormAct2d,
}

impl ConvNormAct {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let c = match &self.conv {
            ConvType::Regular(conv) => conv.forward(x),
            ConvType::Same(conv) => conv.forward(x),
        };
        self.bn.forward(&c)
    }
}

// LayerScale2d — Element-wise scale.
pub struct LayerScale2d {
    pub gamma: UniquePtr<MlxArray>,
}

impl LayerScale2d {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::multiply(x, &self.gamma)
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let gamma = get_weight(weights, &format!("{}.gamma", prefix))?;
        Ok(Self { gamma })
    }
}

// EdgeResidual.
pub struct EdgeResidual {
    pub conv_exp: Conv2dSame,
    pub bn1: RMSNormAct2d,
    pub conv_pwl: Conv2dLayer,
    pub bn2: RMSNormAct2d,
    pub has_skip: bool,
}

impl EdgeResidual {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shortcut = if self.has_skip {
            Some(mlxcel_core::copy(x))
        } else {
            None
        };
        let x = self.conv_exp.forward(x);
        let x = self.bn1.forward(&x);
        let x = self.conv_pwl.forward(&x);
        let x = self.bn2.forward(&x);
        if let Some(shortcut) = shortcut {
            mlxcel_core::add(&x, &shortcut)
        } else {
            x
        }
    }
}

// UniversalInvertedResidual (UIR).
pub struct UniversalInvertedResidual {
    pub dw_start: Option<ConvNormAct>,
    pub pw_exp: ConvNormAct,
    pub dw_mid: Option<ConvNormAct>,
    pub pw_proj: ConvNormAct,
    pub layer_scale: Option<LayerScale2d>,
    pub has_skip: bool,
}

impl UniversalInvertedResidual {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shortcut = if self.has_skip {
            Some(mlxcel_core::copy(x))
        } else {
            None
        };
        let mut out = if let Some(ref dw) = self.dw_start {
            dw.forward(x)
        } else {
            mlxcel_core::copy(x)
        };
        out = self.pw_exp.forward(&out);
        if let Some(ref dw) = self.dw_mid {
            out = dw.forward(&out);
        }
        out = self.pw_proj.forward(&out);
        if let Some(ref ls) = self.layer_scale {
            out = ls.forward(&out);
        }
        if let Some(shortcut) = shortcut {
            mlxcel_core::add(&out, &shortcut)
        } else {
            out
        }
    }
}

// MultiQueryAttention2d.
pub struct MultiQueryAttention2d {
    // Query projection: 1x1 conv
    pub query_proj: Conv2dLayer,
    // Key pipeline: optional downsampling conv+norm, then 1x1 proj
    pub key_down_conv: Option<Conv2dLayer>,
    pub key_down_norm: Option<RMSNormAct2d>,
    pub key_proj: Conv2dLayer,
    // Value pipeline: optional downsampling conv+norm, then 1x1 proj
    pub value_down_conv: Option<Conv2dLayer>,
    pub value_down_norm: Option<RMSNormAct2d>,
    pub value_proj: Conv2dLayer,
    // Output projection: 1x1 conv
    pub output_proj: Conv2dLayer,
    pub num_heads: i32,
    pub key_dim: i32,
    pub value_dim: i32,
}

impl MultiQueryAttention2d {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let h = shape[1];
        let w = shape[2];

        // Query: [B,H,W,C] → conv1x1 → [B,H,W, num_heads*key_dim]
        let q = self.query_proj.forward(x);
        // Reshape to [B, H*W, num_heads, key_dim] then transpose to [B, num_heads, H*W, key_dim]
        let q = mlxcel_core::reshape(&q, &[b, h * w, self.num_heads, self.key_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Key: optionally downsample, then 1x1 proj → [B, 1, H'*W', key_dim]
        let mut k = mlxcel_core::copy(x);
        if let Some(ref dc) = self.key_down_conv {
            k = dc.forward(&k);
            if let Some(ref dn) = self.key_down_norm {
                k = dn.forward(&k);
            }
        }
        let k = self.key_proj.forward(&k);
        // Reshape: [B,H',W',key_dim] → [B,1,H'*W',key_dim]
        let k_shape = mlxcel_core::array_shape(&k);
        let k = mlxcel_core::reshape(&k, &[b, 1, k_shape[1] * k_shape[2], self.key_dim]);

        // Value: same downsample path → [B, 1, H'*W', value_dim]
        let mut v = mlxcel_core::copy(x);
        if let Some(ref dc) = self.value_down_conv {
            v = dc.forward(&v);
            if let Some(ref dn) = self.value_down_norm {
                v = dn.forward(&v);
            }
        }
        let v = self.value_proj.forward(&v);
        let v_shape = mlxcel_core::array_shape(&v);
        let v = mlxcel_core::reshape(&v, &[b, 1, v_shape[1] * v_shape[2], self.value_dim]);

        // Scaled dot-product attention
        let scale = 1.0 / (self.key_dim as f32).sqrt();
        // Use the unsafe SDPA with no mask (vision attention is non-causal)
        let o = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, scale, std::ptr::null(), 0.0, 0)
        };

        // Reshape output: [B, num_heads, H*W, value_dim] → [B, H, W, num_heads*value_dim]
        let o = mlxcel_core::transpose_axes(&o, &[0, 2, 1, 3]);
        let o = mlxcel_core::reshape(&o, &[b, h, w, self.num_heads * self.value_dim]);

        self.output_proj.forward(&o)
    }
}

// MobileAttention — norm → attention → layer_scale + skip.
pub struct MobileAttention {
    pub norm: RMSNormAct2d,
    pub attn: MultiQueryAttention2d,
    pub layer_scale: Option<LayerScale2d>,
    pub has_skip: bool,
}

impl MobileAttention {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shortcut = if self.has_skip {
            Some(mlxcel_core::copy(x))
        } else {
            None
        };
        let mut out = self.norm.forward(x);
        out = self.attn.forward(&out);
        if let Some(ref ls) = self.layer_scale {
            out = ls.forward(&out);
        }
        if let Some(shortcut) = shortcut {
            mlxcel_core::add(&out, &shortcut)
        } else {
            out
        }
    }
}

// Block enum (replaces dyn trait).
pub enum MobileNetBlock {
    EdgeRes(EdgeResidual),
    UIR(UniversalInvertedResidual),
    MobileAttn(MobileAttention),
}

impl MobileNetBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::EdgeRes(b) => b.forward(x),
            Self::UIR(b) => b.forward(x),
            Self::MobileAttn(b) => b.forward(x),
        }
    }
}

// Multi-Scale Fusion Adapter (MSFA).
pub struct MSFA {
    pub ffn: UniversalInvertedResidual,
    pub norm: RMSNormAct2d,
    pub output_h: i32,
    pub output_w: i32,
}

impl MSFA {
    /// Fuses features from two stages by upsampling smaller one and concatenating.
    ///
    /// Input: [stage2_feat, stage3_feat] where stage2 is higher resolution.
    /// 1. Upsample stage3 to match stage2 resolution (nearest-neighbor)
    /// 2. Concatenate channels
    /// 3. UIR block
    /// 4. RMSNorm (no activation)
    pub fn forward(&self, inputs: &[&MlxArray]) -> UniquePtr<MlxArray> {
        // inputs[0]: higher res (stage 2), inputs[1]: lower res (stage 3)
        // Transpose all to NCHW for interpolation
        let mut resized = Vec::new();
        let first = mlxcel_core::transpose_axes(inputs[0], &[0, 3, 1, 2]);
        let first_shape = mlxcel_core::array_shape(&first);
        let high_h = first_shape[2];
        let high_w = first_shape[3];
        resized.push(first);

        for inp in inputs.iter().skip(1) {
            let t = mlxcel_core::transpose_axes(inp, &[0, 3, 1, 2]);
            let t_shape = mlxcel_core::array_shape(&t);
            if t_shape[2] < high_h || t_shape[3] < high_w {
                // Nearest-neighbor upsample using reshape+broadcast
                let upsampled = nearest_upsample_nchw(&t, high_h, high_w);
                resized.push(upsampled);
            } else {
                resized.push(t);
            }
        }

        // Concatenate on channel axis (axis=1 in NCHW)
        let mut cat = mlxcel_core::copy(&resized[0]);
        for r in resized.iter().skip(1) {
            cat = mlxcel_core::concatenate(cat.as_ref().unwrap(), r.as_ref().unwrap(), 1);
        }

        // Back to NHWC for UIR block (which operates in NHWC)
        let cat_nhwc = mlxcel_core::transpose_axes(&cat, &[0, 2, 3, 1]);
        let img = self.ffn.forward(&cat_nhwc);

        // Downsample if needed via average pooling
        let img_shape = mlxcel_core::array_shape(&img);
        let cur_h = img_shape[1];
        let cur_w = img_shape[2];

        if cur_h != self.output_h || cur_w != self.output_w {
            if cur_h % self.output_h == 0 && cur_w % self.output_w == 0 {
                let stride_h = cur_h / self.output_h;
                let stride_w = cur_w / self.output_w;
                // AvgPool2d expects NHWC
                let pooled =
                    mlxcel_core::avg_pool2d(&img, stride_h, stride_w, stride_h, stride_w, 0, 0);
                self.norm.forward(&pooled)
            } else {
                // Fallback: just apply norm
                self.norm.forward(&img)
            }
        } else {
            self.norm.forward(&img)
        }
    }
}

// VisionTower.
pub struct VisionTower {
    pub conv_stem: ConvNormAct,
    pub blocks: Vec<Vec<MobileNetBlock>>, // 4 stages
    pub msfa: MSFA,
    pub msfa_indices: (usize, usize), // (3, 4)
}

impl VisionTower {
    /// Forward pass: pixel_values [B, C, H, W] → features [B, H', W', 2048]
    ///
    /// Input is NCHW, internally processes in NHWC.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // NCHW → NHWC
        let mut x = mlxcel_core::transpose_axes(x, &[0, 2, 3, 1]);
        x = self.conv_stem.forward(&x);

        let mut feat_idx = 0usize;
        let mut intermediates: Vec<UniquePtr<MlxArray>> = Vec::new();

        if feat_idx == self.msfa_indices.0 || feat_idx == self.msfa_indices.1 {
            intermediates.push(mlxcel_core::copy(&x));
        }

        for block_group in &self.blocks {
            feat_idx += 1;
            for block in block_group {
                x = block.forward(&x);
            }
            if feat_idx == self.msfa_indices.0 || feat_idx == self.msfa_indices.1 {
                intermediates.push(mlxcel_core::copy(&x));
            }
        }

        // MSFA: fuse intermediates
        let refs: Vec<&MlxArray> = intermediates.iter().map(|a| a.as_ref().unwrap()).collect();
        self.msfa.forward(&refs)
    }
}

// Gemma3nVisionModel — wraps VisionTower.
pub struct Gemma3nVisionModel {
    pub tower: VisionTower,
}

impl Gemma3nVisionModel {
    /// Forward: pixel_values [B, C, H, W] → [B, output_h, output_w, 2048]
    pub fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        self.tower.forward(pixel_values)
    }
}

// Architecture definition (gemma3n_mobilenet_def).
#[derive(Clone)]
enum BlockDef {
    ER {
        kernel_size: i32,
        filters: i32,
        stride: i32,
        expand_ratio: f32,
    },
    UIR {
        start_dw_kernel: i32,
        mid_dw_kernel: i32,
        filters: i32,
        stride: i32,
        expand_ratio: f32,
    },
    MMQA {
        num_heads: i32,
        kv_dim: i32,
        kv_stride: i32,
    },
}

fn gemma3n_mobilenet_def() -> Vec<Vec<BlockDef>> {
    let er = |k, f, s| BlockDef::ER {
        kernel_size: k,
        filters: f,
        stride: s,
        expand_ratio: 4.0,
    };
    let uir = |sk, mk, f, s, er| BlockDef::UIR {
        start_dw_kernel: sk,
        mid_dw_kernel: mk,
        filters: f,
        stride: s,
        expand_ratio: er,
    };
    let mmqa = |nh, kd, kvs| BlockDef::MMQA {
        num_heads: nh,
        kv_dim: kd,
        kv_stride: kvs,
    };

    vec![
        // Stage 0: Edge Residuals
        vec![er(3, 128, 2), er(3, 128, 1), er(3, 128, 1)],
        // Stage 1: UIR blocks
        vec![
            uir(3, 5, 256, 2, 6.0),
            uir(5, 0, 256, 1, 4.0),
            uir(3, 0, 256, 1, 4.0),
            uir(5, 0, 256, 1, 4.0),
            uir(3, 0, 256, 1, 4.0),
        ],
        // Stage 2: UIR + MMQA (640 filters, 12 heads, kv_dim=64, kv_stride=2)
        {
            let mut stage = vec![uir(5, 5, 640, 2, 6.0)];
            // 7 plain UIR blocks
            for _ in 0..7 {
                stage.push(uir(5, 0, 640, 1, 4.0));
            }
            // 1 UIR with exp_ratio=1.0 and no dw kernels
            stage.push(uir(0, 0, 640, 1, 1.0));
            // 13 pairs of MMQA + UIR
            for _ in 0..13 {
                stage.push(mmqa(12, 64, 2));
                stage.push(uir(0, 0, 640, 1, 2.0));
            }
            // Final pair (same as above)
            stage.push(mmqa(12, 64, 2));
            stage.push(uir(0, 0, 640, 1, 2.0));
            stage
        },
        // Stage 3: UIR + MMQA (1280 filters, 16 heads, kv_dim=96, kv_stride=1)
        {
            let mut stage = vec![uir(5, 5, 1280, 2, 6.0)];
            // 18 pairs of MMQA + UIR
            for _ in 0..18 {
                stage.push(mmqa(16, 96, 1));
                stage.push(uir(0, 0, 1280, 1, 2.0));
            }
            // Final pair
            stage.push(mmqa(16, 96, 1));
            stage.push(uir(0, 0, 1280, 1, 2.0));
            stage
        },
    ]
}

// Weight loading.
fn load_conv2d_regular(
    weights: &WeightMap,
    prefix: &str,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    bias: bool,
) -> Result<Conv2dLayer, String> {
    let weight = sanitize_conv_weight(get_weight(weights, &format!("{}.weight", prefix))?);
    let bias_arr = if bias {
        Some(get_weight(weights, &format!("{}.bias", prefix))?)
    } else {
        None
    };
    Ok(Conv2dLayer {
        weight,
        bias: bias_arr,
        stride_h: stride,
        stride_w: stride,
        padding_h: padding,
        padding_w: padding,
        dilation_h: dilation,
        dilation_w: dilation,
        groups,
    })
}

fn load_conv2d_same(
    weights: &WeightMap,
    prefix: &str,
    kernel_size: i32,
    stride: i32,
    dilation: i32,
    groups: i32,
    bias: bool,
) -> Result<Conv2dSame, String> {
    let weight = sanitize_conv_weight(get_weight(weights, &format!("{}.weight", prefix))?);
    let bias_arr = if bias {
        Some(get_weight(weights, &format!("{}.bias", prefix))?)
    } else {
        None
    };

    let use_static = is_static_pad(stride);
    let static_pad = if use_static {
        get_static_padding(kernel_size, dilation)
    } else {
        0
    };

    Ok(Conv2dSame {
        weight,
        bias: bias_arr,
        kernel_h: kernel_size,
        kernel_w: kernel_size,
        stride_h: stride,
        stride_w: stride,
        dilation_h: dilation,
        dilation_w: dilation,
        groups,
        use_static_pad: use_static,
        static_pad_h: static_pad,
        static_pad_w: static_pad,
    })
}

fn load_conv_norm_act_regular(
    weights: &WeightMap,
    prefix: &str,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    bias: bool,
    apply_act: bool,
) -> Result<ConvNormAct, String> {
    let conv = load_conv2d_regular(
        weights,
        &format!("{}.conv", prefix),
        stride,
        padding,
        dilation,
        groups,
        bias,
    )?;
    let bn = RMSNormAct2d::from_weights(weights, &format!("{}.bn", prefix), apply_act)?;
    Ok(ConvNormAct {
        conv: ConvType::Regular(conv),
        bn,
    })
}

fn load_conv_norm_act_same(
    weights: &WeightMap,
    prefix: &str,
    kernel_size: i32,
    stride: i32,
    dilation: i32,
    groups: i32,
    bias: bool,
    apply_act: bool,
) -> Result<ConvNormAct, String> {
    let conv = load_conv2d_same(
        weights,
        &format!("{}.conv", prefix),
        kernel_size,
        stride,
        dilation,
        groups,
        bias,
    )?;
    let bn = RMSNormAct2d::from_weights(weights, &format!("{}.bn", prefix), apply_act)?;
    Ok(ConvNormAct {
        conv: ConvType::Same(conv),
        bn,
    })
}

fn load_edge_residual(
    weights: &WeightMap,
    prefix: &str,
    in_chs: i32,
    out_chs: i32,
    kernel_size: i32,
    stride: i32,
    expand_ratio: f32,
) -> Result<EdgeResidual, String> {
    let _mid_chs = make_divisible(in_chs as f32 * expand_ratio, 8);
    let has_skip = in_chs == out_chs && stride == 1;

    // conv_exp: Conv2dSame(in_chs → mid_chs, kernel_size, stride)
    let conv_exp = load_conv2d_same(
        weights,
        &format!("{}.conv_exp", prefix),
        kernel_size,
        stride,
        1,
        1,
        false,
    )?;

    // bn1: RMSNormAct2d(mid_chs, apply_act=true)
    let bn1 = RMSNormAct2d::from_weights(weights, &format!("{}.bn1", prefix), true)?;

    // conv_pwl: Conv2d(mid_chs → out_chs, 1x1)
    let conv_pwl =
        load_conv2d_regular(weights, &format!("{}.conv_pwl", prefix), 1, 0, 1, 1, false)?;

    // bn2: RMSNormAct2d(out_chs, apply_act=false)
    let bn2 = RMSNormAct2d::from_weights(weights, &format!("{}.bn2", prefix), false)?;

    Ok(EdgeResidual {
        conv_exp,
        bn1,
        conv_pwl,
        bn2,
        has_skip,
    })
}

fn load_uir(
    weights: &WeightMap,
    prefix: &str,
    in_chs: i32,
    out_chs: i32,
    start_dw_kernel: i32,
    mid_dw_kernel: i32,
    stride: i32,
    expand_ratio: f32,
) -> Result<UniversalInvertedResidual, String> {
    let has_skip = in_chs == out_chs && stride == 1;
    let mid_chs = make_divisible(in_chs as f32 * expand_ratio, 8);

    // dw_start: optional depthwise conv
    let dw_start = if start_dw_kernel > 0 {
        let dw_stride = if mid_dw_kernel > 0 { 1 } else { stride };
        let groups = num_groups(1, in_chs); // group_size=1 → depthwise
        let padding = (start_dw_kernel - 1) / 2;
        Some(load_conv_norm_act_regular(
            weights,
            &format!("{}.dw_start", prefix),
            dw_stride,
            padding,
            1,
            groups,
            false,
            false, // apply_act=false
        )?)
    } else {
        None
    };

    // pw_exp: 1x1 pointwise expansion (in_chs → mid_chs)
    let pw_exp = load_conv_norm_act_regular(
        weights,
        &format!("{}.pw_exp", prefix),
        1,
        0,
        1,
        1,
        false,
        true, // apply_act=true
    )?;

    // dw_mid: optional depthwise conv with "same" padding
    let dw_mid = if mid_dw_kernel > 0 {
        let groups = num_groups(1, mid_chs); // depthwise
        Some(load_conv_norm_act_same(
            weights,
            &format!("{}.dw_mid", prefix),
            mid_dw_kernel,
            stride,
            1,
            groups,
            false,
            true, // apply_act=true
        )?)
    } else {
        None
    };

    // pw_proj: 1x1 pointwise projection (mid_chs → out_chs, no activation)
    let pw_proj = load_conv_norm_act_regular(
        weights,
        &format!("{}.pw_proj", prefix),
        1,
        0,
        1,
        1,
        false,
        false, // apply_act=false
    )?;

    // layer_scale: optional
    let layer_scale = LayerScale2d::from_weights(weights, &format!("{}.layer_scale", prefix)).ok();

    Ok(UniversalInvertedResidual {
        dw_start,
        pw_exp,
        dw_mid,
        pw_proj,
        layer_scale,
        has_skip,
    })
}

fn load_multi_query_attention2d(
    weights: &WeightMap,
    prefix: &str,
    in_chs: i32,
    num_heads: i32,
    key_dim: i32,
    value_dim: i32,
    kv_stride: i32,
    dw_kernel_size: i32,
) -> Result<MultiQueryAttention2d, String> {
    // Query: 1x1 conv
    let query_proj = load_conv2d_regular(
        weights,
        &format!("{}.query.proj", prefix),
        1,
        0,
        1,
        1,
        false,
    )?;

    // Key: optional downsampling + 1x1 proj
    let (key_down_conv, key_down_norm) = if kv_stride > 1 {
        let pad = (dw_kernel_size - 1) / 2;
        let dc = load_conv2d_regular(
            weights,
            &format!("{}.key.down_conv", prefix),
            kv_stride,
            pad,
            1,
            in_chs, // depthwise
            false,
        )?;
        let dn =
            RMSNormAct2d::from_weights_eps(weights, &format!("{}.key.norm", prefix), false, 1e-6)?;
        (Some(dc), Some(dn))
    } else {
        (None, None)
    };
    let key_proj =
        load_conv2d_regular(weights, &format!("{}.key.proj", prefix), 1, 0, 1, 1, false)?;

    // Value: optional downsampling + 1x1 proj
    let (value_down_conv, value_down_norm) = if kv_stride > 1 {
        let pad = (dw_kernel_size - 1) / 2;
        let dc = load_conv2d_regular(
            weights,
            &format!("{}.value.down_conv", prefix),
            kv_stride,
            pad,
            1,
            in_chs, // depthwise
            false,
        )?;
        let dn = RMSNormAct2d::from_weights_eps(
            weights,
            &format!("{}.value.norm", prefix),
            false,
            1e-6,
        )?;
        (Some(dc), Some(dn))
    } else {
        (None, None)
    };
    let value_proj = load_conv2d_regular(
        weights,
        &format!("{}.value.proj", prefix),
        1,
        0,
        1,
        1,
        false,
    )?;

    // Output: 1x1 conv
    let output_proj = load_conv2d_regular(
        weights,
        &format!("{}.output.proj", prefix),
        1,
        0,
        1,
        1,
        false,
    )?;

    Ok(MultiQueryAttention2d {
        query_proj,
        key_down_conv,
        key_down_norm,
        key_proj,
        value_down_conv,
        value_down_norm,
        value_proj,
        output_proj,
        num_heads,
        key_dim,
        value_dim,
    })
}

fn load_mobile_attention(
    weights: &WeightMap,
    prefix: &str,
    in_chs: i32,
    num_heads: i32,
    key_dim: i32,
    value_dim: i32,
    kv_stride: i32,
) -> Result<MobileAttention, String> {
    let has_skip = true; // stride=1 and in_chs==out_chs for attention blocks

    let norm = RMSNormAct2d::from_weights(weights, &format!("{}.norm", prefix), false)?;

    let attn = load_multi_query_attention2d(
        weights,
        &format!("{}.attn", prefix),
        in_chs,
        num_heads,
        key_dim,
        value_dim,
        kv_stride,
        3, // dw_kernel_size
    )?;

    let layer_scale = LayerScale2d::from_weights(weights, &format!("{}.layer_scale", prefix)).ok();

    Ok(MobileAttention {
        norm,
        attn,
        layer_scale,
        has_skip,
    })
}

/// Load the full Gemma3n vision model from weights
pub fn load_gemma3n_vision(
    weights: &WeightMap,
    prefix: &str, // "vision_tower.timm_model"
) -> Result<Gemma3nVisionModel, String> {
    // conv_stem: ConvNormAct with Conv2dSame(3→64, k=3, s=2)
    let conv_stem = load_conv_norm_act_same(
        weights,
        &format!("{}.conv_stem", prefix),
        3,    // kernel_size
        2,    // stride
        1,    // dilation
        1,    // groups
        true, // bias (conv_stem has bias)
        true, // apply_act
    )?;

    // Build blocks from architecture definition
    let arch = gemma3n_mobilenet_def();
    let mut blocks: Vec<Vec<MobileNetBlock>> = Vec::new();
    let mut in_chs = 64; // After conv_stem

    for (stage_idx, stage_def) in arch.iter().enumerate() {
        let mut stage_blocks = Vec::new();

        for (block_idx, block_def) in stage_def.iter().enumerate() {
            let block_prefix = format!("{}.blocks.{}.{}", prefix, stage_idx, block_idx);

            match block_def {
                BlockDef::ER {
                    kernel_size,
                    filters,
                    stride,
                    expand_ratio,
                } => {
                    let er = load_edge_residual(
                        weights,
                        &block_prefix,
                        in_chs,
                        *filters,
                        *kernel_size,
                        *stride,
                        *expand_ratio,
                    )?;
                    in_chs = *filters;
                    stage_blocks.push(MobileNetBlock::EdgeRes(er));
                }
                BlockDef::UIR {
                    start_dw_kernel,
                    mid_dw_kernel,
                    filters,
                    stride,
                    expand_ratio,
                } => {
                    let uir_block = load_uir(
                        weights,
                        &block_prefix,
                        in_chs,
                        *filters,
                        *start_dw_kernel,
                        *mid_dw_kernel,
                        *stride,
                        *expand_ratio,
                    )?;
                    in_chs = *filters;
                    stage_blocks.push(MobileNetBlock::UIR(uir_block));
                }
                BlockDef::MMQA {
                    num_heads,
                    kv_dim,
                    kv_stride,
                } => {
                    let attn = load_mobile_attention(
                        weights,
                        &block_prefix,
                        in_chs,
                        *num_heads,
                        *kv_dim,
                        *kv_dim, // value_dim == key_dim
                        *kv_stride,
                    )?;
                    stage_blocks.push(MobileNetBlock::MobileAttn(attn));
                }
            }
        }
        blocks.push(stage_blocks);
    }

    // MSFA: fuses stages 2 and 3 (640 + 1280 = 1920 → 2048)
    let msfa_in_chs = 640 + 1280; // 1920
    let msfa_out_chs = 2048;

    let msfa_ffn = load_uir(
        weights,
        &format!("{}.msfa.ffn", prefix),
        msfa_in_chs,
        msfa_out_chs,
        0,   // no dw_start
        0,   // no dw_mid
        1,   // stride
        2.0, // expansion_ratio
    )?;

    let msfa_norm = RMSNormAct2d::from_weights_eps(
        weights,
        &format!("{}.msfa.norm", prefix),
        false, // no activation
        1e-6,
    )?;

    let msfa = MSFA {
        ffn: msfa_ffn,
        norm: msfa_norm,
        output_h: 16,
        output_w: 16,
    };

    let tower = VisionTower {
        conv_stem,
        blocks,
        msfa,
        msfa_indices: (3, 4), // stage indices for MSFA
    };

    Ok(Gemma3nVisionModel { tower })
}
