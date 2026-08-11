use super::*;
use crate::models::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID, DEFAULT_IMAGE_TOKEN_ID,
};

fn tensor(values: &[f32], shape: &[i32]) -> mlxcel_core::UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn row_major_embeddings(seq_len: usize, hidden: usize) -> mlxcel_core::UniquePtr<MlxArray> {
    let values: Vec<f32> = (0..seq_len * hidden).map(|v| v as f32).collect();
    tensor(&values, &[1, seq_len as i32, hidden as i32])
}

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn muse_glimmer_one_image_scatter_matches_patch_order() {
    let prompt = vec![
        10,
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
        11,
    ];
    let inputs = row_major_embeddings(prompt.len(), 2);
    let features = tensor(&[100.0, 101.0, 200.0, 201.0], &[2, 2]);

    let merged = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &prompt,
        &[(1, 2, 4)],
        2,
        &features,
        &inputs,
    ) {
        Ok(merged) => merged,
        Err(err) => panic!("Muse Glimmer scatter failed: {err}"),
    };

    assert_eq!(
        mlxcel_core::array_shape(&merged.inputs_embeds),
        vec![1, 6, 2]
    );
    assert_eq!(
        to_vec_f32(&merged.inputs_embeds),
        vec![
            0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 200.0, 201.0, 8.0, 9.0, 10.0, 11.0
        ]
    );
    assert!(merged.attention_mask_4d.is_none());
}

#[test]
fn muse_glimmer_two_image_scatter_preserves_image_order() {
    let prompt = vec![
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
        5,
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
    ];
    let inputs = row_major_embeddings(prompt.len(), 2);
    let features = tensor(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2]);

    let merged = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &prompt,
        &[(1, 2, 2), (1, 2, 4)],
        2,
        &features,
        &inputs,
    ) {
        Ok(merged) => merged,
        Err(err) => panic!("Muse Glimmer two-image scatter failed: {err}"),
    };

    assert_eq!(
        to_vec_f32(&merged.inputs_embeds),
        vec![
            0.0, 1.0, 10.0, 11.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0, 21.0, 30.0, 31.0, 14.0, 15.0,
        ]
    );
}

#[test]
fn muse_glimmer_scatter_rejects_feature_row_mismatch() {
    let prompt = vec![
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
    ];
    let inputs = row_major_embeddings(prompt.len(), 2);
    let features = tensor(&[100.0, 101.0], &[1, 2]);

    let err = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &prompt,
        &[(1, 2, 4)],
        2,
        &features,
        &inputs,
    ) {
        Ok(_) => panic!("Muse Glimmer scatter accepted too few feature rows"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("projected feature rows 1"));
}

#[test]
fn muse_glimmer_scatter_rejects_prompt_patch_count_mismatch() {
    let prompt = vec![
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
    ];
    let inputs = row_major_embeddings(prompt.len(), 2);
    let features = tensor(&[100.0, 101.0, 200.0, 201.0], &[2, 2]);

    let err = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &prompt,
        &[(1, 2, 4)],
        2,
        &features,
        &inputs,
    ) {
        Ok(_) => panic!("Muse Glimmer scatter accepted too few patch tokens"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("expanded prompt contains 1 patch token"));
}

#[test]
fn muse_glimmer_scatter_rejects_non_divisible_grid() {
    let prompt = vec![DEFAULT_IMAGE_TOKEN_ID];
    let inputs = row_major_embeddings(prompt.len(), 2);
    let features = tensor(&[100.0, 101.0], &[1, 2]);

    let err = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &prompt,
        &[(1, 3, 4)],
        2,
        &features,
        &inputs,
    ) {
        Ok(_) => panic!("Muse Glimmer scatter accepted a non-divisible grid"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("not divisible"));
}

#[test]
fn muse_glimmer_differing_length_requests_do_not_share_scatter_state() {
    let short_prompt = vec![
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
    ];
    let long_prompt = vec![
        7,
        DEFAULT_IMAGE_START_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_TOKEN_ID,
        DEFAULT_IMAGE_END_TOKEN_ID,
        8,
    ];
    let short_inputs = row_major_embeddings(short_prompt.len(), 2);
    let long_inputs = row_major_embeddings(long_prompt.len(), 2);
    let short_features = tensor(&[90.0, 91.0], &[1, 2]);
    let long_features = tensor(&[100.0, 101.0, 200.0, 201.0], &[2, 2]);

    let short = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &short_prompt,
        &[(1, 2, 2)],
        2,
        &short_features,
        &short_inputs,
    ) {
        Ok(merged) => to_vec_f32(&merged.inputs_embeds),
        Err(err) => panic!("short Muse Glimmer scatter failed: {err}"),
    };
    let long = match merge_muse_glimmer_features(
        DEFAULT_IMAGE_TOKEN_ID,
        &long_prompt,
        &[(1, 2, 4)],
        2,
        &long_features,
        &long_inputs,
    ) {
        Ok(merged) => to_vec_f32(&merged.inputs_embeds),
        Err(err) => panic!("long Muse Glimmer scatter failed: {err}"),
    };

    assert_eq!(short, vec![0.0, 1.0, 90.0, 91.0, 4.0, 5.0]);
    assert_eq!(
        long,
        vec![
            0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 200.0, 201.0, 8.0, 9.0, 10.0, 11.0
        ]
    );
}
