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

//! Failure-only Gemma 4 startup probe capture. The thread-local scope ends
//! before model loading returns, so serving never retains diagnostic tensors.
use std::cell::RefCell;

use mlxcel_core::{MlxArray, utils::slice_axis};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Stage {
    pub layer: usize,
    pub kind: String,
    pub op: &'static str,
    pub rows: Vec<Vec<u8>>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<Stage>>> = const { RefCell::new(None) };
}

pub(super) fn capture(layer: usize, kind: &str, op: &'static str, tensor: &MlxArray) {
    CAPTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(stages) = slot.as_mut() else { return };
        let width = mlxcel_core::array_shape(tensor)[1];
        let rows = (0..width)
            .map(|row| mlxcel_core::array_to_raw_bytes(&slice_axis(tensor, 1, row, row + 1)))
            .collect();
        stages.push(Stage {
            layer,
            kind: kind.to_owned(),
            op,
            rows,
        });
    });
}

pub(super) fn with_capture<T>(run: impl FnOnce() -> T) -> (T, Vec<Stage>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CAPTURE.with(|slot| *slot.borrow_mut() = None);
        }
    }
    CAPTURE.with(|slot| {
        assert!(slot.borrow().is_none(), "probe capture must not nest");
        *slot.borrow_mut() = Some(Vec::new());
    });
    let _reset = Reset;
    let result = run();
    let stages = CAPTURE.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (result, stages)
}

/// Scan in execution order, then token order: a later token at an earlier
/// layer is more useful than an earlier token whose error has propagated.
pub(super) fn first_divergence(block: &[Stage], chain: &[Vec<Stage>]) -> Option<String> {
    if block.is_empty() || chain.is_empty() || chain.iter().any(|token| token.len() != block.len())
    {
        return Some("probe trace stage counts differ or are empty".to_owned());
    }
    for (index, stage) in block.iter().enumerate() {
        if stage.rows.len() != chain.len() || chain.iter().any(|token| token[index].rows.len() != 1)
        {
            return Some("probe trace row counts differ".to_owned());
        }
        for (position, token) in chain.iter().enumerate() {
            let Some(reference) = token.get(index) else {
                return Some("probe trace stage counts differ".to_owned());
            };
            if (stage.layer, &stage.kind, stage.op)
                != (reference.layer, &reference.kind, reference.op)
            {
                return Some("probe trace stage order differs".to_owned());
            }
            if stage.rows.get(position) != reference.rows.first() {
                return Some(format!(
                    "first divergence: layer {} ({}) {} at position {}",
                    stage.layer, stage.kind, stage.op, position
                ));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "gemma4_probe_tests.rs"]
mod tests;
