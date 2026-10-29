// Copyright 2025-2026 Lablup Inc.
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

//! Teacher-forced per-stage check of the denoiser: every stage gets
//! upstream's own input to that stage and only its output is compared.
//! It located the one non-exact op of the first prediction (6.0e-2 on Turbo
//! text): the image RoPE tables, whose inverse frequencies were computed
//! with the host `powf`, which rounds one of the 40 frequencies differently
//! from upstream's device `pow`, so 2.8% of the table entries differed. With [`MRoPE::compute_freqs_on_device`] every
//! stage, and the whole Turbo text pass, is bit-identical; VQ 512 keeps one
//! GEMM-kernel residual in the final projection (see the bound below).
//!
//! Reads `<MLXCEL_LLADA_IMAGE_ORACLE_DIR>/turbo_text_1024/denoiser_stages.safetensors`,
//! written by `tests/fixtures/generate_llada_image_denoiser_stages.py` from
//! upstream's first Turbo text pass on that oracle case's inputs.

use std::path::PathBuf;

use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::testing::{rel_rms, values};

/// `(rel-RMS, fraction of elements not bit-identical)`.
fn compare(a: &MlxArray, b: &MlxArray) -> (f64, f64) {
    assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
    let (va, vb) = (values(a), values(b));
    let differ = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
    (rel_rms(&va, &vb), differ as f64 / vb.len() as f64)
}

#[test]
#[ignore = "needs models/mlx/llada-image-turbo, MLXCEL_LLADA_IMAGE_ORACLE_DIR stages and a GPU"]
fn denoiser_stages_on_upstream_inputs() {
    let Some(dir) = std::env::var_os("MLXCEL_LLADA_IMAGE_ORACLE_DIR").map(PathBuf::from) else {
        eprintln!("skipping: MLXCEL_LLADA_IMAGE_ORACLE_DIR is unset");
        return;
    };
    let root = std::env::var_os("MLXCEL_LLADA_IMAGE_TURBO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mlx/llada-image-turbo")
        });
    if !root.join("transformer").exists() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    let tr = crate::imagegen::llada_image::weights::load_transformer(&root).unwrap();
    for case in ["turbo_text_1024", "turbo_vq_512"] {
        let path = dir.join(case).join("denoiser_stages.safetensors");
        match load_safetensors(&path) {
            Ok(up) => check_case(&tr, case, &up),
            Err(_) => eprintln!("skipping {case}: {} is absent", path.display()),
        }
    }
}

fn check_case(tr: &LladaImageTransformer, case: &str, up: &WeightMap) {
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    let mut check = |name: &str, ours: &MlxArray, key: &str| {
        let (rel, differ) = compare(ours, &up[key]);
        eprintln!(
            "{case}: stage {name:<20} rel-RMS {rel:.3e}, {:.4}% of elements differ",
            differ * 100.0
        );
        rows.push((name.to_string(), rel, differ));
    };

    let x = &up["in.x"];
    let s = mlxcel_core::array_shape(x);
    let patches = mlxcel_core::reshape(
        &mlxcel_core::transpose_axes(x, &[0, 2, 3, 1]),
        &[s[0], s[2] * s[3], s[1]],
    );
    check("x_embedder", &tr.x_embedder.forward(&patches), "x_emb");
    let eps = tr.config.norm_eps;
    check(
        "cap_embedder.norm",
        &plain_rms_norm(&up["in.caption"], eps),
        "cap_norm",
    );
    check(
        "cap_embedder.linear",
        &tr.cap_embedder.dense.forward(&up["cap_norm"]),
        "cap_emb",
    );
    let cap_len = mlxcel_core::array_shape(&up["in.caption"])[1];
    let mut cap_pos = sequence_positions(cap_len, 1);
    let mut joint_caption = "cap_tokens";
    if let Some(semantic) = up.get("in.semantic") {
        check("semantic.norm", &plain_rms_norm(semantic, eps), "sem_norm");
        check(
            "semantic.linear",
            &tr.semantic_embedder.dense.forward(&up["sem_norm"]),
            "sem_emb",
        );
        let n = mlxcel_core::array_shape(semantic)[1];
        cap_pos.extend(sequence_positions(
            n,
            mlxcel_core::array_shape(&up["cap_tokens"])[1] + 1,
        ));
        joint_caption = "cap_tokens_sem";
    }
    let image_start = mlxcel_core::array_shape(&up[joint_caption])[1] + 1;
    let image_pos = image_positions(s[2], s[3], image_start);
    let (cc, cs) = tr
        .rope
        .compute_freqs_on_device(&positions_array(&cap_pos))
        .unwrap();
    let (ic, is) = tr
        .rope
        .compute_freqs_on_device(&positions_array(&image_pos))
        .unwrap();
    check("rope.cap_cos", &cc, "cap_cos");
    check("rope.cap_sin", &cs, "cap_sin");
    check("rope.image_cos", &ic, "image_cos");
    check("rope.image_sin", &is, "image_sin");
    check("t_embedder", &tr.timestep(&up["in.t"]), "t.out");

    let time = &up["t.out"];
    let (icos, isin) = (&up["image_cos"], &up["image_sin"]);
    let (ccos, csin) = (&up["cap_cos"], &up["cap_sin"]);
    let cos = concat(&[icos, ccos]);
    let sin = concat(&[isin, csin]);
    let groups: [(&str, &Vec<ZImageTransformerBlock>, bool); 3] = [
        ("noise_refiner", &tr.noise_refiner, true),
        ("context_refiner", &tr.context_refiner, false),
        ("layers", &tr.layers, true),
    ];
    for (name, blocks, modulated) in groups {
        for (i, block) in blocks.iter().enumerate() {
            let tag = format!("{name}.{i}");
            let (c, sn): (&MlxArray, &MlxArray) = match name {
                "noise_refiner" => (icos, isin),
                "context_refiner" => (ccos, csin),
                _ => (&*cos, &*sin),
            };
            let out = block
                .forward(
                    &up[&format!("{tag}.in")],
                    c,
                    sn,
                    modulated.then_some(time),
                    None,
                )
                .unwrap();
            check(&tag, &out, &format!("{tag}.out"));
        }
    }
    check(
        "final_layer",
        &tr.final_layer.forward(&up["final.in"], time),
        "final.out",
    );
    let whole = tr
        .forward(
            x,
            &up["in.t"],
            &up["in.caption"],
            up.get("in.semantic").map(|v| &**v),
            None,
        )
        .unwrap();
    check("whole forward", &whole, "ref.out");
    for (name, rel, differ) in &rows {
        // Every stage runs upstream's ops in upstream's order, so it is
        // bit-identical. The one exception is the final layer's output
        // projection on VQ 512 (`[1024, 3840] x [3840, 128]` bf16 `addmm`):
        // its LayerNorm, modulation and input are bit-identical, but mlxcel's
        // MLX pin and upstream's mlx 0.32.4 pick different GEMM kernels for
        // that shape, so 38.6% of its outputs differ by a rounding (rel-RMS
        // 3.0e-3). The same projection on Turbo text's 4096 rows is exact.
        if name == "final_layer" || name == "whole forward" {
            assert!(*rel < 5e-3, "{case}: {name}: rel-RMS {rel}");
        } else {
            assert_eq!(*differ, 0.0, "{case}: {name}: rel-RMS {rel}");
        }
    }
}
