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

//! Binary-side drafter factory.
//!
//! [`mlxcel_core::drafter::load_drafter`] builds every drafter whose model
//! lives in `mlxcel-core`. Two MTP drafters reuse decoder blocks that live in
//! this crate, so core cannot construct them and refuses them by name: the
//! GLM-4.7-Flash head
//! ([`crate::models::glm4_moe_lite_mtp_drafter::Glm4MoeLiteMtpDraftModel`],
//! issue #1326) and the Qwen3.8-Flash-Next head
//! ([`crate::models::qwen4_exp_mtp_drafter::Qwen4ExpMtpDraftModel`], issue
//! #2880), which loads from a split `qwen4_exp_mtp` directory or from a
//! `qwen4_exp` target checkpoint that keeps its `mtp.*` head (issue #2917).
//! This wrapper is the one entry point the binary's callers (the
//! offline CLI, the server's drafter slot, the speculative bench) go through:
//! it builds those two itself and delegates everything else to core
//! unchanged.

use std::path::Path;

use mlxcel_core::drafter::{
    DrafterError, DrafterKind, GLM4_MOE_LITE_MTP_MODEL_TYPE, LoadedDrafter, QWEN4_EXP_MODEL_TYPE,
    QWEN4_EXP_MTP_MODEL_TYPE, peek_drafter_model_type, resolve_drafter_kind,
};

use crate::models::glm4_moe_lite_mtp_drafter::Glm4MoeLiteMtpDraftModel;
use crate::models::qwen4_exp_mtp_drafter::Qwen4ExpMtpDraftModel;

/// Load a drafter from `path`, reconciling `kind` with the directory's
/// `config.json` exactly as [`mlxcel_core::drafter::load_drafter`] does.
///
/// Used by: `commands::generate::run_offline_mtp`,
/// `server::batch::speculative_burst::WorkerDrafterSlot::ensure_loaded`,
/// `speculative_bench`.
pub fn load_drafter(path: &Path, kind: Option<DrafterKind>) -> Result<LoadedDrafter, DrafterError> {
    let resolved = resolve_drafter_kind(path, kind)?;
    if resolved == DrafterKind::Mtp {
        match peek_drafter_model_type(path)?.as_deref() {
            Some(GLM4_MOE_LITE_MTP_MODEL_TYPE) => {
                let model = Glm4MoeLiteMtpDraftModel::from_path(path)?;
                return Ok((Box::new(model), resolved));
            }
            Some(QWEN4_EXP_MTP_MODEL_TYPE) => {
                let model = Qwen4ExpMtpDraftModel::from_path(path)?;
                return Ok((Box::new(model), resolved));
            }
            // The target checkpoint passed again as its own drafter: its
            // embedded `mtp.*` head (issue #2917). `resolve_drafter_kind`
            // only resolves `Mtp` here when the tensors exist or the user
            // asked for it explicitly.
            Some(QWEN4_EXP_MODEL_TYPE) => {
                let model = Qwen4ExpMtpDraftModel::from_target_checkpoint(path)?;
                return Ok((Box::new(model), resolved));
            }
            _ => {}
        }
    }
    mlxcel_core::drafter::load_drafter(path, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_drafter_routes_to_the_binary_loader_not_the_core_refusal() {
        // A `glm4_moe_lite_mtp` directory with no weights must fail inside
        // the GLM drafter's own loader (a config error naming the family),
        // not with core's `BinaryCrateDrafter` refusal.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "glm4_moe_lite_mtp"}"#,
        )
        .unwrap();
        let err = load_drafter(dir.path(), None).expect_err("no weights");
        assert!(
            !matches!(err, DrafterError::BinaryCrateDrafter { .. }),
            "the wrapper must build the GLM drafter itself: {err}"
        );
        assert!(err.to_string().contains("text_config"), "{err}");
    }

    #[test]
    fn qwen4_exp_mtp_drafter_routes_to_the_binary_loader_not_the_core_refusal() {
        // Auto-detected from `model_type` with no `--draft-kind`, and built
        // here: the failure on an empty directory is the Qwen4-Exp config
        // error, not core's refusal and not the Gemma 4 assistant's inventory.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen4_exp_mtp"}"#,
        )
        .unwrap();
        let err = load_drafter(dir.path(), None).expect_err("no text_config");
        assert!(
            !matches!(err, DrafterError::BinaryCrateDrafter { .. }),
            "the wrapper must build the Qwen4-Exp drafter itself: {err}"
        );
        assert!(err.to_string().contains("text_config"), "{err}");
    }

    #[test]
    fn other_model_types_delegate_to_core() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen3_5_mtp"}"#,
        )
        .unwrap();
        let err = load_drafter(dir.path(), None).expect_err("no weights");
        assert!(err.to_string().contains("text_config"), "{err}");
    }
}
