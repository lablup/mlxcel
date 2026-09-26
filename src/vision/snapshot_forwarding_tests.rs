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

//! Every VLM wrapper around a snapshot-only text backbone must forward the
//! snapshot capability.
//!
//! The scheduler reuses a model-owned family's prompt only through
//! `supports_snapshot_reuse` / `snapshot_sequence_state` /
//! `restore_sequence_state`. A wrapper that leaves them at the trait defaults
//! answers "no snapshots", so the server stores nothing and every turn of a
//! conversation re-prefills from zero, with no error anywhere. MiniCPM-V 4.6
//! (Qwen 3.5 backbone) shipped that way and measured `cached=0` on every turn.

/// Text backbones whose server-side prompt reuse is snapshot-only.
const SNAPSHOT_BACKBONES: &[&str] = &[
    "Qwen35Model",
    "Gemma4Wrapper",
    "Lfm2Model",
    "GraniteMoeHybridModel",
    "NemotronHModel",
];

/// Wrappers allowed to skip the forward, each with the reason.
const EXEMPT: &[(&str, &str)] = &[];

/// `include_str!` needs literal paths, so this list is maintained by hand. It
/// catches a listed wrapper losing its forwards; a NEW wrapper around one of
/// `SNAPSHOT_BACKBONES` must be added here to be covered.
const SOURCES: &[(&str, &str)] = &[
    ("gemma4_unified.rs", include_str!("gemma4_unified.rs")),
    ("gemma4_vl.rs", include_str!("gemma4_vl.rs")),
    ("granite4_vision.rs", include_str!("granite4_vision.rs")),
    ("lfm2_vl.rs", include_str!("lfm2_vl.rs")),
    ("minicpmv4_6_vl.rs", include_str!("minicpmv4_6_vl.rs")),
    (
        "nemotron_h_nano_omni_vl.rs",
        include_str!("nemotron_h_nano_omni_vl.rs"),
    ),
    ("qwen3_5_vl.rs", include_str!("qwen3_5_vl.rs")),
];

fn wraps_snapshot_backbone(src: &str) -> bool {
    src.lines().any(|line| {
        let line = line.trim();
        line.contains("text_model: ")
            && !line.starts_with("//")
            && SNAPSHOT_BACKBONES
                .iter()
                .any(|ty| line.trim_end_matches(',').ends_with(ty))
    })
}

#[test]
fn every_snapshot_backbone_wrapper_forwards_snapshot_reuse() {
    let mut missing = Vec::new();
    for (file, src) in SOURCES {
        assert!(
            wraps_snapshot_backbone(src),
            "{file} is listed here but no longer wraps a snapshot backbone; update SOURCES"
        );
        if EXEMPT.iter().any(|(f, _)| f == file) {
            continue;
        }
        for method in [
            "fn supports_snapshot_reuse(",
            "fn snapshot_sequence_state(",
            "fn restore_sequence_state(",
        ] {
            if !src.contains(method) {
                missing.push(format!("{file}: {method}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "VLM wrappers around a snapshot-only backbone must forward the snapshot \
         methods or the server never reuses a prompt:\n  {}",
        missing.join("\n  ")
    );
}
