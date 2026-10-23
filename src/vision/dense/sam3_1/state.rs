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

//! Bounded inference state for a SAM 3.1 multiplex session.

use std::collections::BTreeMap;

use mlxcel_core::{MlxArray, UniquePtr};

use super::multiplex::MultiplexState;

pub(crate) struct BranchFeatures {
    pub vision: UniquePtr<MlxArray>,
    pub high_resolution: [UniquePtr<MlxArray>; 2],
    pub position: Option<UniquePtr<MlxArray>>,
}

pub(crate) struct FrameFeatures {
    pub interactive: Option<BranchFeatures>,
    pub propagation: BranchFeatures,
}

pub(crate) struct TrackPrediction {
    pub masks: UniquePtr<MlxArray>,
    pub object_score_logits: UniquePtr<MlxArray>,
}

pub(crate) struct FrameOutput {
    pub object_pointers: UniquePtr<MlxArray>,
    pub memory_features: Option<UniquePtr<MlxArray>>,
    pub image_features: UniquePtr<MlxArray>,
}

pub(crate) struct TrackerState {
    pub multiplex: MultiplexState,
    pub conditioning: BTreeMap<usize, FrameOutput>,
    pub non_conditioning: BTreeMap<usize, FrameOutput>,
}

impl TrackerState {
    pub(crate) fn new(num_objects: usize) -> Result<Self, String> {
        Ok(Self {
            multiplex: MultiplexState::new(num_objects)?,
            conditioning: BTreeMap::new(),
            non_conditioning: BTreeMap::new(),
        })
    }

    #[must_use]
    pub(crate) fn num_objects(&self) -> usize {
        self.multiplex.num_objects()
    }

    pub(crate) fn insert_conditioning(&mut self, frame: usize, output: FrameOutput) {
        self.conditioning.insert(frame, output);
    }

    pub(crate) fn insert_propagated(
        &mut self,
        frame: usize,
        output: FrameOutput,
        max_lookback: usize,
    ) {
        self.non_conditioning.insert(frame, output);
        let cutoff = frame.saturating_sub(max_lookback);
        self.non_conditioning.retain(|&index, _| index >= cutoff);
    }
}

pub(crate) fn select_closest_conditioning(
    frame: usize,
    frames: impl IntoIterator<Item = usize>,
    limit: usize,
) -> Vec<usize> {
    let all: Vec<usize> = frames.into_iter().collect();
    if all.len() <= limit {
        return all;
    }
    let mut selected = Vec::with_capacity(limit);
    if let Some(before) = all.iter().copied().filter(|&index| index < frame).max() {
        selected.push(before);
    }
    if let Some(after) = all.iter().copied().filter(|&index| index >= frame).min()
        && !selected.contains(&after)
    {
        selected.push(after);
    }
    let mut remaining: Vec<usize> = all
        .into_iter()
        .filter(|index| !selected.contains(index))
        .collect();
    remaining.sort_by_key(|&index| frame.abs_diff(index));
    selected.extend(
        remaining
            .into_iter()
            .take(limit.saturating_sub(selected.len())),
    );
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_conditioning_keeps_nearest_before_after_then_distance() {
        assert_eq!(
            select_closest_conditioning(10, [1, 7, 12, 20, 30], 4),
            [7, 12, 1, 20]
        );
    }

    #[test]
    fn closest_conditioning_preserves_all_when_under_limit() {
        assert_eq!(select_closest_conditioning(3, [0, 2], 4), [0, 2]);
    }

    #[test]
    fn tracker_state_spills_seventeen_objects_into_two_buckets() {
        let state = TrackerState::new(17).unwrap();
        assert_eq!(state.num_objects(), 17);
        assert_eq!(state.multiplex.num_buckets(), 2);
    }
}
