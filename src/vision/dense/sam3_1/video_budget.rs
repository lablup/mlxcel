// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, ensure};

use super::config::TrackerConfig;

const MAX_TRACKING_MASK_WORKING_SET_BYTES: u64 = 512 * 1024 * 1024;

pub(super) fn validate_video_mask_budget(
    width: u32,
    height: u32,
    frames: usize,
    objects: usize,
    config: &TrackerConfig,
) -> Result<()> {
    let checked_product = |values: &[u64], label: &str| {
        values
            .iter()
            .try_fold(1_u64, |product, value| product.checked_mul(*value))
            .ok_or_else(|| anyhow::anyhow!("SAM 3.1 {label} size overflowed"))
    };
    let source_pixels = checked_product(&[u64::from(width), u64::from(height)], "source mask")?;
    let output_bytes = checked_product(
        &[objects as u64, frames as u64, source_pixels],
        "mask output",
    )?;
    let retained_output = output_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("SAM 3.1 retained mask size overflowed"))?;
    let tracker_pixels = checked_product(
        &[
            objects as u64,
            config.image_size as u64,
            config.image_size as u64,
        ],
        "tracker mask",
    )?;
    let tracker_work = tracker_pixels
        .checked_mul(8)
        .ok_or_else(|| anyhow::anyhow!("SAM 3.1 tracker mask size overflowed"))?;
    let source_work = checked_product(&[objects as u64, source_pixels, 8], "resize mask")?;
    let lookback = config.max_object_pointers_in_encoder.max(
        2 + config
            .num_maskmem
            .saturating_sub(2)
            .saturating_mul(config.memory_temporal_stride_for_eval),
    );
    let retained_frames = 1_u64 + (frames.saturating_sub(1).min(lookback.saturating_add(1))) as u64;
    let buckets = objects.div_ceil(config.multiplex_count) as u64;
    let feature_side = config.image_size / config.vision_config.backbone_config.patch_size;
    let feature_values = checked_product(
        &[
            feature_side as u64,
            feature_side as u64,
            config.memory_attention_hidden_size as u64,
        ],
        "retained tracker feature",
    )?;
    // Each retained frame owns one image feature plus one memory feature per
    // bucket. Position grids are deterministic and recomputed when selected.
    let retained_state = checked_product(
        &[
            retained_frames,
            buckets + 1,
            feature_values,
            std::mem::size_of::<f32>() as u64,
        ],
        "retained tracker state",
    )?;
    let pointer_state = checked_product(
        &[
            retained_frames,
            buckets,
            config.multiplex_count as u64,
            config.memory_attention_hidden_size as u64,
            std::mem::size_of::<f32>() as u64,
        ],
        "retained object pointers",
    )?;
    let selected_memories = frames.saturating_sub(1).min(
        config
            .max_cond_frame_num
            .saturating_add(config.num_maskmem.saturating_sub(1)),
    ) as u64;
    // Conditioning materializes mask/image memories and their positions, then
    // concatenates them and projects keys/values. Count two complete sets to
    // cover the concatenated inputs plus the active attention projections.
    let conditioning_graph = checked_product(
        &[
            selected_memories,
            2 * buckets + 2,
            feature_values,
            std::mem::size_of::<f32>() as u64,
            2,
        ],
        "conditioning attention graph",
    )?;
    // The current source keeps self-attention Q/K/V and the expanded feed-
    // forward activation live within a layer until its output is evaluated.
    let current_attention_factor = 3_u64
        + (config.memory_attention_feed_forward_hidden_size / config.memory_attention_hidden_size)
            as u64;
    let current_attention = checked_product(
        &[
            buckets,
            feature_values,
            std::mem::size_of::<f32>() as u64,
            current_attention_factor,
        ],
        "current-frame attention graph",
    )?;
    let working_set = retained_output
        .checked_add(tracker_work)
        .and_then(|value| value.checked_add(source_work))
        .and_then(|value| value.checked_add(retained_state))
        .and_then(|value| value.checked_add(pointer_state))
        .and_then(|value| value.checked_add(conditioning_graph))
        .and_then(|value| value.checked_add(current_attention))
        .ok_or_else(|| anyhow::anyhow!("SAM 3.1 mask working-set size overflowed"))?;
    ensure!(
        working_set <= MAX_TRACKING_MASK_WORKING_SET_BYTES,
        "SAM 3.1 mask working set requires {working_set} bytes, exceeding the {MAX_TRACKING_MASK_WORKING_SET_BYTES}-byte limit"
    );
    Ok(())
}
