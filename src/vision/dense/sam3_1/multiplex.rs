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

//! Exact gather-based object multiplexing for SAM 3.1.

use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) const MULTIPLEX_COUNT: usize = 16;
const PADDING: i32 = -1;

#[derive(Debug, Clone)]
pub(crate) struct MultiplexState {
    assignments: Vec<i32>,
    object_to_slot: Vec<i32>,
    object_ids: Vec<i64>,
    num_buckets: usize,
}

impl MultiplexState {
    pub(crate) fn new(num_objects: usize, object_ids: Option<Vec<i64>>) -> Result<Self, String> {
        if num_objects == 0 {
            return Err("SAM 3.1 cannot create an empty multiplex state".into());
        }
        let num_buckets = num_objects.div_ceil(MULTIPLEX_COUNT);
        let mut assignments = Vec::with_capacity(num_buckets * MULTIPLEX_COUNT);
        assignments.extend((0..num_objects).map(|index| index as i32));
        assignments.resize(num_buckets * MULTIPLEX_COUNT, PADDING);
        let object_ids = object_ids.unwrap_or_else(|| (1..=num_objects as i64).collect());
        if object_ids.len() != num_objects {
            return Err(format!(
                "SAM 3.1 got {} object IDs for {num_objects} objects",
                object_ids.len()
            ));
        }
        let mut object_to_slot = vec![0; num_objects];
        for (slot, &object) in assignments.iter().enumerate() {
            if object >= 0 {
                object_to_slot[object as usize] = slot as i32;
            }
        }
        Ok(Self {
            assignments,
            object_to_slot,
            object_ids,
            num_buckets,
        })
    }

    #[must_use]
    pub(crate) fn num_objects(&self) -> usize {
        self.object_to_slot.len()
    }

    #[must_use]
    pub(crate) fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    #[must_use]
    pub(crate) fn object_ids(&self) -> &[i64] {
        &self.object_ids
    }

    pub(crate) fn mux(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let input_shape = mlxcel_core::array_shape(input);
        if input_shape.first().copied() != Some(self.num_objects() as i32) {
            return Err(format!(
                "SAM 3.1 mux expected {} objects, got shape {input_shape:?}",
                self.num_objects()
            ));
        }
        let gather: Vec<i32> = self.assignments.iter().map(|&index| index.max(0)).collect();
        let gather = mlxcel_core::from_slice_i32(&gather, &[gather.len() as i32]);
        let gathered = mlxcel_core::take(input, &gather, 0);
        let mut mask = mlxcel_core::from_bytes(
            &self
                .assignments
                .iter()
                .map(|&index| u8::from(index >= 0))
                .collect::<Vec<_>>(),
            &[self.assignments.len() as i32],
            mlxcel_core::dtype::BOOL,
        );
        for _ in 1..input_shape.len() {
            mask = mlxcel_core::expand_dims(&mask, -1);
        }
        let gathered_shape = mlxcel_core::array_shape(&gathered);
        let mask = mlxcel_core::broadcast_to(&mask, &gathered_shape);
        let padded = mlxcel_core::where_cond(&mask, &gathered, &mlxcel_core::zeros_like(&gathered));
        let mut output_shape = vec![self.num_buckets as i32, MULTIPLEX_COUNT as i32];
        output_shape.extend_from_slice(&input_shape[1..]);
        Ok(mlxcel_core::reshape(&padded, &output_shape))
    }

    pub(crate) fn demux(&self, input: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let input_shape = mlxcel_core::array_shape(input);
        if input_shape.len() < 2
            || input_shape[0] != self.num_buckets as i32
            || input_shape[1] != MULTIPLEX_COUNT as i32
        {
            return Err(format!(
                "SAM 3.1 demux expected [{},16,...], got {input_shape:?}",
                self.num_buckets
            ));
        }
        let mut flat_shape = vec![(self.num_buckets * MULTIPLEX_COUNT) as i32];
        flat_shape.extend_from_slice(&input_shape[2..]);
        let flat = mlxcel_core::reshape(input, &flat_shape);
        let indices =
            mlxcel_core::from_slice_i32(&self.object_to_slot, &[self.object_to_slot.len() as i32]);
        Ok(mlxcel_core::take(&flat, &indices, 0))
    }

    #[must_use]
    pub(crate) fn valid_slots(&self) -> UniquePtr<MlxArray> {
        let bytes: Vec<u8> = self
            .assignments
            .iter()
            .map(|&index| u8::from(index >= 0))
            .collect();
        mlxcel_core::from_bytes(
            &bytes,
            &[self.num_buckets as i32, MULTIPLEX_COUNT as i32],
            mlxcel_core::dtype::BOOL,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(count: usize) {
        let state = MultiplexState::new(count, None).unwrap();
        let values: Vec<f32> = (0..count * 2).map(|value| value as f32).collect();
        let input = mlxcel_core::from_slice_f32(&values, &[count as i32, 2]);
        let muxed = state.mux(&input).unwrap();
        let output = state.demux(&muxed).unwrap();
        mlxcel_core::eval(&output);
        assert_eq!(
            mlxcel_core::array_shape(&muxed),
            [count.div_ceil(16) as i32, 16, 2]
        );
        assert_eq!(mlxcel_core::utils::array_to_vec_f32(&output), values);
    }

    #[test]
    fn mux_demux_preserves_three_objects_and_padding() {
        let state = MultiplexState::new(3, None).unwrap();
        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0], &[3, 1]);
        let muxed = state.mux(&input).unwrap();
        let valid = state.valid_slots();
        mlxcel_core::eval(&muxed);
        mlxcel_core::eval(&valid);

        let values = mlxcel_core::utils::array_to_vec_f32(&muxed);
        assert_eq!(&values[..3], &[1.0, 2.0, 3.0]);
        assert!(values[3..].iter().all(|value| *value == 0.0));
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(&valid),
            [vec![1_u8; 3], vec![0_u8; 13]].concat()
        );
        roundtrip(3);
    }

    #[test]
    fn mux_demux_preserves_full_bucket() {
        roundtrip(16);
    }

    #[test]
    fn mux_demux_preserves_two_bucket_order() {
        let state = MultiplexState::new(17, None).unwrap();
        let valid = state.valid_slots();
        mlxcel_core::eval(&valid);
        let bytes = mlxcel_core::array_to_raw_bytes(&valid);
        assert!(bytes[..17].iter().all(|value| *value == 1));
        assert!(bytes[17..].iter().all(|value| *value == 0));
        roundtrip(17);
    }
}
