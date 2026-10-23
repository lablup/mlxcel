// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Checked geometry and workset limits for PP-DocLayoutV3 configs.

use super::config::PpDocLayoutV3Config;

const MAX_IMAGE_SIZE: usize = 1024;
const MAX_WORKSET_ELEMENTS: usize = 256 * 1024 * 1024;

pub(super) fn validate(config: &PpDocLayoutV3Config) -> Result<(), String> {
    let pixels = checked_product("input pixels", &[config.image_size, config.image_size])?;
    if config.image_size > MAX_IMAGE_SIZE || pixels > MAX_IMAGE_SIZE * MAX_IMAGE_SIZE {
        return Err("image_size exceeds the supported 1024x1024 input".into());
    }
    if config.image_size < 32 || !config.image_size.is_multiple_of(32) {
        return Err("image_size must be at least 32 and divisible by the stride-32 pyramid".into());
    }
    if config.num_queries > 512 || config.num_labels > 512 || config.num_feature_levels > 4 {
        return Err("queries, labels, or feature levels exceed inference limits".into());
    }
    if config.encoder_layers > 12 || config.decoder_layers > 12 {
        return Err("encoder_layers and decoder_layers must not exceed 12".into());
    }
    if config.encoder_attention_heads > 32 || config.decoder_attention_heads > 32 {
        return Err("attention head counts must not exceed 32".into());
    }
    if config.decoder_n_points > 64 {
        return Err("decoder_n_points must not exceed 64".into());
    }
    if config.encoder_ffn_dim > 8192 || config.decoder_ffn_dim > 8192 {
        return Err("feed-forward dimensions must not exceed 8192".into());
    }
    let channel_vectors: [&[usize]; 8] = [
        &config.encoder_in_channels,
        &config.decoder_in_channels,
        &config.mask_feature_channels,
        &config.backbone_config.stem_channels,
        &config.backbone_config.hidden_sizes,
        &config.backbone_config.stage_in_channels,
        &config.backbone_config.stage_mid_channels,
        &config.backbone_config.stage_out_channels,
    ];
    if [
        config.d_model,
        config.encoder_hidden_dim,
        config.x4_feat_dim,
        config.backbone_config.embedding_size,
    ]
    .iter()
    .any(|&value| value > 4096)
        || channel_vectors
            .iter()
            .flat_map(|values| values.iter())
            .any(|&value| value > 4096)
    {
        return Err("model and channel dimensions must not exceed 4096".into());
    }
    if config.global_pointer_head_size > 512 || config.num_prototypes > 512 {
        return Err("prototype and global-pointer head sizes must not exceed 512".into());
    }
    let csp_hidden = (config.d_model as f32 * config.hidden_expansion) as usize;
    if csp_hidden == 0 || csp_hidden > 4096 {
        return Err("hidden_expansion produces an invalid CSP width".into());
    }

    let pyramid_tokens = pyramid_tokens(config)?;
    if config.num_queries > pyramid_tokens {
        return Err(format!(
            "num_queries {} exceeds the {pyramid_tokens} available pyramid tokens",
            config.num_queries
        ));
    }
    let workset = estimated_workset(config, pixels, pyramid_tokens)?;
    if workset > MAX_WORKSET_ELEMENTS {
        return Err(format!(
            "PP-DocLayoutV3 estimated workset has {workset} elements; limit is {MAX_WORKSET_ELEMENTS}"
        ));
    }
    Ok(())
}

fn pyramid_tokens(config: &PpDocLayoutV3Config) -> Result<usize, String> {
    config
        .feat_strides
        .iter()
        .try_fold(0usize, |total, &stride| {
            let side = config.image_size / stride;
            checked_product("feature tokens", &[side, side])?
                .checked_add(total)
                .ok_or_else(|| "PP-DocLayoutV3 feature token count overflow".into())
        })
}

fn estimated_workset(
    config: &PpDocLayoutV3Config,
    pixels: usize,
    pyramid_tokens: usize,
) -> Result<usize, String> {
    let mut parts = vec![checked_product("input workset", &[pixels, 3])?];
    let backbone = [4usize, 8, 16, 32]
        .into_iter()
        .zip(&config.backbone_config.stage_out_channels)
        .try_fold(0usize, |total, (stride, &channels)| {
            let side = config.image_size / stride;
            checked_product("backbone workset", &[side, side, channels])?
                .checked_add(total)
                .ok_or_else(|| "PP-DocLayoutV3 backbone workset overflow".to_string())
        })?;
    parts.push(backbone);
    // Projected inputs, PAN outputs, decoder inputs, and flattened memory can
    // coexist at the encoder/decoder seam.
    parts.push(checked_product(
        "feature pyramid workset",
        &[pyramid_tokens, config.d_model, 4],
    )?);
    let encoded_side = config.image_size / config.feat_strides[config.encode_proj_layers[0]];
    let encoded_tokens = checked_product("encoded tokens", &[encoded_side, encoded_side])?;
    parts.push(checked_product(
        "encoder attention workset",
        &[
            encoded_tokens,
            encoded_tokens,
            config.encoder_attention_heads,
            config.encoder_layers,
        ],
    )?);
    parts.push(checked_product(
        "mask workset",
        &[
            config.num_queries,
            config.image_size / 4,
            config.image_size / 4,
        ],
    )?);
    parts.push(checked_product(
        "classification workset",
        &[pyramid_tokens, config.num_labels],
    )?);
    parts.push(checked_product(
        "reading-order workset",
        &[config.num_queries, config.num_queries],
    )?);
    let deformable = checked_product(
        "deformable attention workset",
        &[
            config.decoder_attention_heads,
            config.num_feature_levels,
            config.decoder_n_points,
            4,
        ],
    )?;
    let decoder_per_query = config
        .d_model
        .checked_add(config.decoder_ffn_dim)
        .and_then(|value| value.checked_add(deformable))
        .ok_or_else(|| "PP-DocLayoutV3 decoder workset overflow".to_string())?;
    parts.push(checked_product(
        "decoder workset",
        &[config.decoder_layers, config.num_queries, decoder_per_query],
    )?);
    parts.into_iter().try_fold(0usize, |total, part| {
        total
            .checked_add(part)
            .ok_or_else(|| "PP-DocLayoutV3 aggregate workset overflow".into())
    })
}

fn checked_product(context: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| format!("PP-DocLayoutV3 {context} overflow"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_geometry_is_within_the_documented_workset() {
        let config = PpDocLayoutV3Config::default();
        let pixels = config.image_size * config.image_size;
        let workset = estimated_workset(&config, pixels, pyramid_tokens(&config).unwrap()).unwrap();
        assert_eq!(workset, 47_978_325);
        assert!(workset < MAX_WORKSET_ELEMENTS);
    }

    #[test]
    fn rejects_too_few_tokens_and_aggregate_amplification() {
        let tiny = PpDocLayoutV3Config {
            image_size: 32,
            ..PpDocLayoutV3Config::default()
        };
        assert!(validate(&tiny).unwrap_err().contains("pyramid tokens"));

        let amplified = PpDocLayoutV3Config {
            image_size: 1024,
            encoder_layers: 12,
            encoder_attention_heads: 32,
            ..PpDocLayoutV3Config::default()
        };
        assert!(validate(&amplified).unwrap_err().contains("workset"));
    }
}
