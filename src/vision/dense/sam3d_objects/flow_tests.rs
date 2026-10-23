// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Constructor rejection before a malformed latent can reach indexing or MLX.
use super::*;

fn latent_weights() -> WeightMap {
    let mut w = WeightMap::new();
    for (name, shape) in [
        ("latent.input_layer.weight", vec![4, 2]),
        ("latent.input_layer.bias", vec![4]),
        ("latent.out_layer.weight", vec![2, 4]),
        ("latent.out_layer.bias", vec![2]),
        ("latent.pos_emb", vec![8, 4]),
    ] {
        w.insert(name.into(), mx::zeros(&shape, mx::dtype::BFLOAT16));
    }
    w
}

#[test]
fn latent_constructor_rejects_malformed_position_rank_and_dimensions() -> Result<()> {
    let mut w = latent_weights();
    assert!(Latent::from_weights(&w, "latent", 2, 4, 8).is_ok());
    for shape in [
        vec![],
        vec![32],
        vec![1, 8, 4],
        vec![4, 8],
        vec![8, 3],
        vec![7, 4],
    ] {
        w.insert(
            "latent.pos_emb".into(),
            mx::zeros(&shape, mx::dtype::BFLOAT16),
        );
        let result = Latent::from_weights(&w, "latent", 2, 4, 8);
        let error = result
            .err()
            .ok_or_else(|| anyhow::anyhow!("accepted invalid position shape {shape:?}"))?;
        assert!(error.to_string().contains("latent.pos_emb"), "{error}");
    }
    Ok(())
}

#[test]
fn latent_constructor_checks_configured_projection_widths() {
    let mut w = latent_weights();
    w.insert(
        "latent.input_layer.weight".into(),
        mx::zeros(&[4, 3], mx::dtype::BFLOAT16),
    );
    assert!(Latent::from_weights(&w, "latent", 2, 4, 8).is_err());
    let mut w = latent_weights();
    w.insert(
        "latent.out_layer.weight".into(),
        mx::zeros(&[2, 3], mx::dtype::BFLOAT16),
    );
    assert!(Latent::from_weights(&w, "latent", 2, 4, 8).is_err());
}
