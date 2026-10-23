// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Body-only DINOv3 → ray conditioning → six pose-feedback decoder layers.
use super::{
    backbone::Backbone,
    camera,
    config::Config,
    decoder::{Decoder, Ffn},
    layers::{Array, Linear, Norm, array},
    mhr_head::{BodyOutput, MhrHead},
    ops::{cat, evaluate, slice_axis},
    preprocess::InputGeometry,
    prompt_encoder::PromptEncoder,
};
use anyhow::Result;
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
pub struct Model {
    backbone: Backbone,
    decoder: Decoder,
    pub head: MhrHead,
    camera: Ffn,
    prompt: PromptEncoder,
    init_token: Linear,
    prev_token: Linear,
    prompt_token: Linear,
    kp_features: Linear,
    kp_position: Ffn,
    kp3_position: Ffn,
    init_pose: Array,
    init_camera: Array,
    kp_embed: Array,
    kp3_embed: Array,
    hand_embed: Array,
    ray_weight: Array,
    ray_norm: Norm,
    config: Config,
}
impl Model {
    pub fn from_weights(w: &WeightMap, c: &Config) -> Result<Self> {
        Ok(Self {
            backbone: Backbone::from_weights(w, c)?,
            decoder: Decoder::from_weights(w, c)?,
            head: MhrHead::from_weights(w, c)?,
            camera: Ffn::from_weights(w, "head_camera.proj")?,
            prompt: PromptEncoder::from_weights(w, c.prompt_embed_dim)?,
            init_token: Linear::from_weights(w, "init_to_token_mhr")?,
            prev_token: Linear::from_weights(w, "prev_to_token_mhr")?,
            prompt_token: Linear::from_weights(w, "prompt_to_token")?,
            kp_features: Linear::from_weights(w, "keypoint_feat_linear")?,
            kp_position: Ffn::from_weights(w, "keypoint_posemb_linear")?,
            kp3_position: Ffn::from_weights(w, "keypoint3d_posemb_linear")?,
            init_pose: array(w, "init_pose")?,
            init_camera: array(w, "init_camera")?,
            kp_embed: array(w, "keypoint_embedding")?,
            kp3_embed: array(w, "keypoint3d_embedding")?,
            hand_embed: array(w, "hand_box_embedding")?,
            ray_weight: array(w, "ray_cond_emb.conv.weight")?,
            ray_norm: Norm::from_weights(w, "ray_cond_emb.norm", 1e-5, false)?,
            config: c.clone(),
        })
    }
    pub fn forward(&self, image: &MlxArray, g: &InputGeometry) -> Result<(BodyOutput, Array)> {
        let c = &self.config;
        let d = c.decoder_dim as i32;
        let [h, w] = c.image_size.map(|v| (v / c.patch_size) as i32);
        let features = self.backbone.forward(image)?;
        let combined = cat(&[&features, &camera::encoded_rays(c, g)], -1);
        let features = self.ray_norm.forward(&*mx::try_conv2d(
            &combined,
            &self.ray_weight,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
        )?);
        let estimate = cat(&[&self.init_pose, &self.init_camera], 1);
        let cliff = mx::from_slice_f32(&g.cliff(), &[1, 3]);
        let init = self
            .init_token
            .forward(&mx::reshape(&cat(&[&cliff, &estimate], 1), &[1, 1, 525]));
        let prev = self
            .prev_token
            .forward(&mx::reshape(&estimate, &[1, 1, 522]));
        let prompt = self.prompt_token.forward(&self.prompt.invalid());
        let hand = mx::reshape(&self.hand_embed, &[1, 2, d]);
        let kp = mx::reshape(&self.kp_embed, &[1, 70, d]);
        let kp3 = mx::reshape(&self.kp3_embed, &[1, 70, d]);
        let mut tokens = cat(&[&init, &prev, &prompt, &hand, &kp, &kp3], 1);
        let mut pe = cat(
            &[
                &mx::zeros(&[1, 1, d], mx::dtype::FLOAT32),
                &prev,
                &prompt,
                &mx::zeros(&[1, 142, d], mx::dtype::FLOAT32),
            ],
            1,
        );
        let context = mx::reshape(&features, &[1, h * w, c.embed_dim as i32]);
        let context_pe = mx::reshape(&self.prompt.dense_pe(h, w), &[1, h * w, c.embed_dim as i32]);
        let mut final_result = None;
        for (i, layer) in self.decoder.layers.iter().enumerate() {
            tokens = layer.forward(&tokens, &context, &pe, &context_pe);
            let norm = self.decoder.norm_final.forward(&tokens);
            let pose_token = mx::reshape(&slice_axis(&norm, 1, 0, 1), &[1, d]);
            let body = self.head.forward(&pose_token, Some(&self.init_pose))?;
            let camera = mx::add(&self.camera.forward(&pose_token), &self.init_camera);
            if i + 1 < c.decoder_depth {
                let projected = camera::project(&body.keypoints_3d, &camera, g);
                let kp_pe = self.kp_position.forward(&projected);
                let sampled = self
                    .kp_features
                    .forward(&camera::sample(&features, &projected));
                let updated = mx::add(&slice_axis(&tokens, 1, 5, 75), &sampled);
                tokens = cat(
                    &[
                        &slice_axis(&tokens, 1, 0, 5),
                        &updated,
                        &slice_axis(&tokens, 1, 75, 145),
                    ],
                    1,
                );
                let pelvis = mx::divide_scalar(
                    &mx::add(
                        &slice_axis(&body.keypoints_3d, 1, 9, 10),
                        &slice_axis(&body.keypoints_3d, 1, 10, 11),
                    ),
                    2.0,
                );
                let kp3_pe = self
                    .kp3_position
                    .forward(&mx::subtract(&body.keypoints_3d, &pelvis));
                pe = cat(&[&slice_axis(&pe, 1, 0, 5), &kp_pe, &kp3_pe], 1);
                // Each head contains ordered host scatter barriers. Materialize only
                // the state consumed by the next layer before releasing this body.
                evaluate(&[&tokens, &pe]);
            } else {
                final_result = Some((body, camera));
            }
        }
        let (body, camera) =
            final_result.ok_or_else(|| anyhow::anyhow!("Body decoder produced no result"))?;
        evaluate(&[
            &body.vertices,
            &body.keypoints_3d,
            &body.joint_coords,
            &camera,
        ]);
        Ok((body, camera))
    }
}
