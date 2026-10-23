// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Opt-in tensor captures for authentic parity diagnosis.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

const DIAGNOSTIC_DIR: &str = "MLXCEL_TEST_PP_DOCLAYOUT_DIAGNOSTIC_DIR";

struct CaptureState {
    path: PathBuf,
    tensors: Vec<(String, UniquePtr<MlxArray>)>,
}

thread_local! {
    static ACTIVE: RefCell<Option<CaptureState>> = const { RefCell::new(None) };
}

pub(super) struct CaptureScope {
    active: bool,
}

impl CaptureScope {
    pub(super) fn begin(name: &str) -> Self {
        let Some(root) = std::env::var_os(DIAGNOSTIC_DIR).map(PathBuf::from) else {
            return Self { active: false };
        };
        assert!(
            root.is_dir(),
            "{DIAGNOSTIC_DIR} must be an existing directory"
        );
        let path = root.join(format!("{name}.safetensors"));
        assert!(!path.exists(), "refusing to overwrite {}", path.display());
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            assert!(
                active.is_none(),
                "PP diagnostic capture scope is already active"
            );
            *active = Some(CaptureState {
                path,
                tensors: Vec::new(),
            });
        });
        Self { active: true }
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let state = ACTIVE.with(|active| active.borrow_mut().take());
        let state = state.expect("active PP diagnostic capture scope");
        write_bundle(&state.path, state.tensors);
    }
}

pub(super) fn capture(name: &str, value: &MlxArray) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(state) = active.as_mut() else {
            return;
        };
        assert!(
            !state.tensors.iter().any(|(current, _)| current == name),
            "duplicate PP diagnostic tensor {name}"
        );
        state
            .tensors
            .push((name.to_string(), mlxcel_core::copy(value)));
    });
}

fn write_bundle(path: &std::path::Path, tensors: Vec<(String, UniquePtr<MlxArray>)>) {
    let values: Vec<_> = tensors
        .into_iter()
        .map(|(name, value)| {
            assert_eq!(
                mlxcel_core::array_dtype(&value),
                mlxcel_core::dtype::FLOAT32,
                "diagnostic tensor {name} must preserve F32"
            );
            let shape: Vec<usize> = mlxcel_core::array_shape(&value)
                .into_iter()
                .map(|dimension| usize::try_from(dimension).unwrap())
                .collect();
            let bytes: Vec<u8> = mlxcel_core::utils::array_to_vec_f32(&value)
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            (name, shape, bytes)
        })
        .collect();
    let views: BTreeMap<_, _> = values
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(&views, None, path).unwrap();
}
