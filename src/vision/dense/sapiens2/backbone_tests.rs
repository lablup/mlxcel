// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

// Preserve the exact decimal literals captured from the authentic BF16 oracle.
#![allow(clippy::excessive_precision)]

use std::collections::BTreeMap;

use super::*;

impl Sapiens2Backbone {
    pub(crate) fn first_block_stages(
        &self,
        pixel_values: &MlxArray,
    ) -> BTreeMap<String, UniquePtr<MlxArray>> {
        let input_shape = mlxcel_core::array_shape(pixel_values);
        let batch = input_shape[0];
        let input = mlxcel_core::astype(pixel_values, mlxcel_core::array_dtype(&self.patch_weight));
        let patch_map = mlxcel_core::add(
            &mlxcel_core::conv2d(
                &input,
                &self.patch_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            ),
            &self.patch_bias,
        );
        let patch_shape = mlxcel_core::array_shape(&patch_map);
        let (height, width) = (patch_shape[1] as usize, patch_shape[2] as usize);
        let patch = mlxcel_core::reshape(
            &patch_map,
            &[batch, (height * width) as i32, self.hidden_size as i32],
        );
        let cls = mlxcel_core::broadcast_to(&self.cls_token, &[batch, 1, self.hidden_size as i32]);
        let tokens = if let Some(registers) = &self.register_tokens {
            let registers = mlxcel_core::broadcast_to(
                registers,
                &[
                    batch,
                    (self.num_prefix_tokens - 1) as i32,
                    self.hidden_size as i32,
                ],
            );
            let prefix = mlxcel_core::concatenate(&cls, &registers, 1);
            mlxcel_core::concatenate(&prefix, &patch, 1)
        } else {
            mlxcel_core::concatenate(&cls, &patch, 1)
        };
        let rope = RopeTables::new(
            height,
            width,
            self.num_prefix_tokens,
            self.hidden_size / self.num_heads,
            &self.rope_periods,
        );
        let block = &self.blocks[0];
        let attention = &block.attention;
        let ln1 = block.norm1.forward(&tokens);
        let qkv = weights::linear_forward(&attention.qkv, &ln1);
        let shape = mlxcel_core::array_shape(&tokens);
        let (batch, token_count) = (shape[0], shape[1]);
        let hidden = (attention.heads * attention.head_dim) as i32;
        let kv_size = (attention.kv_heads * attention.head_dim) as i32;
        let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[batch, token_count, hidden]);
        let k = mlxcel_core::slice(
            &qkv,
            &[0, 0, hidden],
            &[batch, token_count, hidden + kv_size],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[0, 0, hidden + kv_size],
            &[batch, token_count, hidden + 2 * kv_size],
        );
        let q = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &q,
                &[
                    batch,
                    token_count,
                    attention.heads as i32,
                    attention.head_dim as i32,
                ],
            ),
            &[0, 2, 1, 3],
        );
        let k = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &k,
                &[
                    batch,
                    token_count,
                    attention.kv_heads as i32,
                    attention.head_dim as i32,
                ],
            ),
            &[0, 2, 1, 3],
        );
        let v = mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(
                &v,
                &[
                    batch,
                    token_count,
                    attention.kv_heads as i32,
                    attention.head_dim as i32,
                ],
            ),
            &[0, 2, 1, 3],
        );
        let q_norm = attention.q_norm.forward(&q);
        let k_norm = attention.k_norm.forward(&k);
        let q_rope = rope.apply(&q_norm);
        let k_rope = rope.apply(&k_norm);
        let sdpa = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q_rope,
                &k_rope,
                &v,
                (attention.head_dim as f32).powf(-0.5),
                std::ptr::null(),
            )
        };
        let attended = mlxcel_core::transpose_axes(&sdpa, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(&attended, &[batch, token_count, hidden]);
        let attended = mlxcel_core::multiply(
            &weights::linear_forward(&attention.projection, &attended),
            &attention.gamma,
        );
        let residual = mlxcel_core::add(&tokens, &attended);
        let block0 = mlxcel_core::add(
            &residual,
            &block.feed_forward.forward(&block.norm2.forward(&residual)),
        );

        BTreeMap::from([
            ("patch".into(), mlxcel_core::copy(&patch)),
            ("tokens".into(), mlxcel_core::copy(&tokens)),
            ("ln1".into(), mlxcel_core::copy(&ln1)),
            ("qkv".into(), mlxcel_core::copy(&qkv)),
            ("q_norm".into(), mlxcel_core::copy(&q_norm)),
            ("k_norm".into(), mlxcel_core::copy(&k_norm)),
            ("q_rope".into(), mlxcel_core::copy(&q_rope)),
            ("k_rope".into(), mlxcel_core::copy(&k_rope)),
            ("v".into(), mlxcel_core::copy(&v)),
            ("sdpa".into(), mlxcel_core::copy(&sdpa)),
            ("block0".into(), block0),
            (
                "sin".into(),
                mlxcel_core::reshape(
                    &rope.sin,
                    &[token_count, 2, (attention.head_dim / 2) as i32],
                ),
            ),
            (
                "cos".into(),
                mlxcel_core::reshape(
                    &rope.cos,
                    &[token_count, 1, (attention.head_dim / 2) as i32],
                ),
            ),
            ("periods".into(), mlxcel_core::copy(&self.rope_periods)),
        ])
    }
}

#[test]
fn sapiens2_bf16_rope_matches_upstream_compiled_graph() {
    // Exact MLX 0.32.5 / bff10220 ROCm oracle sampled from block zero of
    // facebook/sapiens2-pose-0.4b. On ROCm the second element distinguishes
    // the upstream compiled graph from the previous eager slice/concatenate
    // path, because ROCm's fused kernel rounds `x * cos + reversed * sin`
    // differently from the eager ops. Metal's fused kernel and the CPU stream
    // do not: upstream `mlx` 0.32.4 on an Apple M1 Ultra gives
    // -4.589557647705078e-6 for both graphs on GPU and CPU, and
    // `tests/fixtures/sapiens2/generate_rope_oracle.py` refuses to write a
    // fixture on that host because the two agree. So the compiled route is
    // only observable by value on ROCm, and elsewhere the oracle is upstream's
    // value for that backend. CUDA has no capture and is not keyed.
    let _device_lock = mlxcel_core::streams::lock_default_device();
    let eager_value = -4.589557647705078e-6;
    let compiled_value = if mlxcel_core::default_device_is_gpu()
        && mlxcel_core::hardware::gpu_backend_kind() == mlxcel_core::hardware::GpuBackendKind::Rocm
    {
        -4.559755325317383e-6
    } else {
        eager_value
    };
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[-0.134765625, -0.07470703125], &[1, 1, 1, 2]),
        mlxcel_core::dtype::BFLOAT16,
    );
    let rope = RopeTables {
        sin: mlxcel_core::from_slice_f32(
            &[0.48480960726737976, -0.48480960726737976],
            &[1, 1, 1, 2, 1],
        ),
        cos: mlxcel_core::from_slice_f32(&[0.874619722366333], &[1, 1, 1, 1, 1]),
        head_dim: 2,
    };

    let compiled = rope.apply(&x);
    let actual = mlxcel_core::utils::array_to_vec_f32(&compiled);
    assert_eq!(actual, [-0.154296875, compiled_value]);

    let xf = mlxcel_core::reshape(
        &mlxcel_core::astype(&x, mlxcel_core::dtype::FLOAT32),
        &[1, 1, 1, 2, 1],
    );
    let first = mlxcel_core::slice(&xf, &[0, 0, 0, 0, 0], &[1, 1, 1, 1, 1]);
    let second = mlxcel_core::slice(&xf, &[0, 0, 0, 1, 0], &[1, 1, 1, 2, 1]);
    let reversed = mlxcel_core::concatenate(&second, &first, 3);
    let eager = mlxcel_core::astype(
        &mlxcel_core::reshape(
            &mlxcel_core::add(
                &mlxcel_core::multiply(&xf, &rope.cos),
                &mlxcel_core::multiply(&reversed, &rope.sin),
            ),
            &[1, 1, 1, 2],
        ),
        mlxcel_core::dtype::BFLOAT16,
    );
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&eager),
        [-0.154296875, eager_value]
    );
}

impl Sapiens2Backbone {
    pub(crate) fn all_stage_metrics(
        &self,
        pixel_values: &MlxArray,
        reference_dir: &std::path::Path,
    ) -> (BackboneOutput, Vec<super::super::diagnostic::StageMetric>) {
        let input_shape = mlxcel_core::array_shape(pixel_values);
        let batch = input_shape[0];
        let input = mlxcel_core::astype(pixel_values, mlxcel_core::array_dtype(&self.patch_weight));
        let patch_map = mlxcel_core::add(
            &mlxcel_core::conv2d(
                &input,
                &self.patch_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            ),
            &self.patch_bias,
        );
        let patch_shape = mlxcel_core::array_shape(&patch_map);
        let (height, width) = (patch_shape[1] as usize, patch_shape[2] as usize);
        let patches = mlxcel_core::reshape(
            &patch_map,
            &[batch, (height * width) as i32, self.hidden_size as i32],
        );
        let cls = mlxcel_core::broadcast_to(&self.cls_token, &[batch, 1, self.hidden_size as i32]);
        let mut hidden = if let Some(registers) = &self.register_tokens {
            let registers = mlxcel_core::broadcast_to(
                registers,
                &[
                    batch,
                    (self.num_prefix_tokens - 1) as i32,
                    self.hidden_size as i32,
                ],
            );
            let prefix = mlxcel_core::concatenate(&cls, &registers, 1);
            mlxcel_core::concatenate(&prefix, &patches, 1)
        } else {
            mlxcel_core::concatenate(&cls, &patches, 1)
        };
        let rope = RopeTables::new(
            height,
            width,
            self.num_prefix_tokens,
            self.hidden_size / self.num_heads,
            &self.rope_periods,
        );
        let mut metrics = Vec::new();

        for (index, block) in self.blocks.iter().enumerate() {
            let input = hidden;
            let attention = &block.attention;
            let shape = mlxcel_core::array_shape(&input);
            let (batch, token_count) = (shape[0], shape[1]);
            let hidden_size = (attention.heads * attention.head_dim) as i32;
            let kv_size = (attention.kv_heads * attention.head_dim) as i32;
            let ln1 = block.norm1.forward(&input);
            let qkv = weights::linear_forward(&attention.qkv, &ln1);
            let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[batch, token_count, hidden_size]);
            let k = mlxcel_core::slice(
                &qkv,
                &[0, 0, hidden_size],
                &[batch, token_count, hidden_size + kv_size],
            );
            let v = mlxcel_core::slice(
                &qkv,
                &[0, 0, hidden_size + kv_size],
                &[batch, token_count, hidden_size + 2 * kv_size],
            );
            let q = mlxcel_core::transpose_axes(
                &mlxcel_core::reshape(
                    &q,
                    &[
                        batch,
                        token_count,
                        attention.heads as i32,
                        attention.head_dim as i32,
                    ],
                ),
                &[0, 2, 1, 3],
            );
            let k = mlxcel_core::transpose_axes(
                &mlxcel_core::reshape(
                    &k,
                    &[
                        batch,
                        token_count,
                        attention.kv_heads as i32,
                        attention.head_dim as i32,
                    ],
                ),
                &[0, 2, 1, 3],
            );
            let v = mlxcel_core::transpose_axes(
                &mlxcel_core::reshape(
                    &v,
                    &[
                        batch,
                        token_count,
                        attention.kv_heads as i32,
                        attention.head_dim as i32,
                    ],
                ),
                &[0, 2, 1, 3],
            );
            let q_norm = attention.q_norm.forward(&q);
            let k_norm = attention.k_norm.forward(&k);
            let q_rope = rope.apply(&q_norm);
            let k_rope = rope.apply(&k_norm);
            // SAFETY: these are live rank-four tensors and the null mask selects
            // the upstream unmasked bidirectional attention path.
            let sdpa = unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &q_rope,
                    &k_rope,
                    &v,
                    (attention.head_dim as f32).powf(-0.5),
                    std::ptr::null(),
                )
            };
            let attended = mlxcel_core::reshape(
                &mlxcel_core::transpose_axes(&sdpa, &[0, 2, 1, 3]),
                &[batch, token_count, hidden_size],
            );
            let projection = weights::linear_forward(&attention.projection, &attended);
            let residual = mlxcel_core::add(
                &input,
                &mlxcel_core::multiply(&projection, &attention.gamma),
            );
            let ln2 = block.norm2.forward(&residual);
            let w12 = weights::linear_forward(&block.feed_forward.w12, &ln2);
            let w12_shape = mlxcel_core::array_shape(&w12);
            let split = block.feed_forward.intermediate_size as i32;
            let first = mlxcel_core::slice(&w12, &[0, 0, 0], &[w12_shape[0], w12_shape[1], split]);
            let second = mlxcel_core::slice(
                &w12,
                &[0, 0, split],
                &[w12_shape[0], w12_shape[1], w12_shape[2]],
            );
            let gated = mlxcel_core::multiply(&mlxcel_core::utils::silu(&first), &second);
            let ffn = weights::linear_forward(&block.feed_forward.w3, &gated);
            // Match the upstream diagnostic capture: `output` is a separate
            // production block call, while the manual graph above exists only
            // to localize a divergence within the block.
            let output = block.forward(&input, &rope);

            let reference = mlxcel_core::weights::load_safetensors(
                reference_dir.join(format!("block{index:02}.safetensors")),
            )
            .expect("load exact-pin Sapiens2 block reference");
            let stages: [(&str, &MlxArray); 16] = [
                ("input", &input),
                ("ln1", &ln1),
                ("qkv", &qkv),
                ("q_norm", &q_norm),
                ("k_norm", &k_norm),
                ("q_rope", &q_rope),
                ("k_rope", &k_rope),
                ("v", &v),
                ("sdpa", &sdpa),
                ("proj", &projection),
                ("residual", &residual),
                ("ln2", &ln2),
                ("w12", &w12),
                ("gated", &gated),
                ("ffn", &ffn),
                ("output", &output),
            ];
            let expected_keys: std::collections::BTreeSet<_> =
                stages.iter().map(|(name, _)| (*name).to_owned()).collect();
            let reference_keys: std::collections::BTreeSet<_> = reference.keys().cloned().collect();
            assert_eq!(
                reference_keys, expected_keys,
                "unexpected exact-pin block reference schema"
            );
            for (name, actual) in stages {
                let (max_abs, relative_rms) =
                    super::super::diagnostic::parity_metrics(actual, &reference[name]);
                metrics.push((format!("block{index:02}.{name}"), max_abs, relative_rms));
            }
            hidden = output;
        }
        if let Some(norm) = &self.final_norm {
            hidden = norm.forward(&hidden);
        }
        (
            BackboneOutput {
                tokens: hidden,
                height,
                width,
            },
            metrics,
        )
    }
}
