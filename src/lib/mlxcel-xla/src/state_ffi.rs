// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! The threaded recurrent-state layout as the C shim receives it (epic #564,
//! issue #3339).
//!
//! `xla_llama_create` takes `has_kv`, `n_state` and three flat arrays: one rank
//! and one element code per state tensor, and every tensor's dims concatenated
//! in [`Config::state_tensors`](crate::emitter::Config::state_tensors) order.
//! The shim repeats every bound checked here; this side refuses a layout it
//! could not describe before any graph is compiled.

use std::os::raw::c_int;

use crate::emitter::{StateElem, StateTensorSpec};

/// `XLA_MAX_STATE` in `csrc/xla_iree.c`.
pub(crate) const MAX_STATE_TENSORS: usize = 8;
/// `XLA_MAX_STATE_RANK` in `csrc/xla_iree.c`.
pub(crate) const MAX_STATE_RANK: usize = 8;

/// The state element code the shim reads (the weight dtype codes: 0 is f32).
fn elem_code(elem: StateElem) -> c_int {
    match elem {
        StateElem::F32 => 0,
    }
}

/// The flattened state layout passed to `xla_llama_create`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StateFfi {
    pub(crate) has_kv: c_int,
    pub(crate) n_state: c_int,
    pub(crate) ranks: Vec<c_int>,
    pub(crate) dims: Vec<i64>,
    pub(crate) elems: Vec<c_int>,
}

impl StateFfi {
    /// Flatten `specs` for the C ABI. A bundle must have a KV cache, state, or
    /// both; every dim must be positive and fit the shim's `int64_t`.
    pub(crate) fn new(has_kv: bool, specs: &[StateTensorSpec]) -> Result<Self, String> {
        if specs.len() > MAX_STATE_TENSORS {
            return Err(format!(
                "{} recurrent state tensors exceed the shim limit of {MAX_STATE_TENSORS}",
                specs.len()
            ));
        }
        if !has_kv && specs.is_empty() {
            return Err("a model without a KV cache must thread recurrent state".to_string());
        }
        let mut out = Self {
            has_kv: c_int::from(has_kv),
            n_state: c_int::try_from(specs.len()).map_err(|_| "state count overflow")?,
            ..Self::default()
        };
        for spec in specs {
            if spec.dims.is_empty() || spec.dims.len() > MAX_STATE_RANK {
                return Err(format!(
                    "state tensor {} has rank {}, outside 1..={MAX_STATE_RANK}",
                    spec.name,
                    spec.dims.len()
                ));
            }
            out.ranks
                .push(c_int::try_from(spec.dims.len()).map_err(|_| "state rank overflow")?);
            for &dim in &spec.dims {
                if dim == 0 {
                    return Err(format!("state tensor {} has a zero dim", spec.name));
                }
                out.dims.push(i64::try_from(dim).map_err(|_| {
                    format!("state tensor {} dim {dim} does not fit i64", spec.name)
                })?);
            }
            out.elems.push(elem_code(spec.elem));
        }
        Ok(out)
    }

    /// Pointer to `values`, or null when there is no state (the shim reads
    /// none of the arrays when `n_state == 0`).
    fn ptr_or_null<T>(values: &[T]) -> *const T {
        if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        }
    }

    pub(crate) fn ranks_ptr(&self) -> *const c_int {
        Self::ptr_or_null(&self.ranks)
    }

    pub(crate) fn dims_ptr(&self) -> *const i64 {
        Self::ptr_or_null(&self.dims)
    }

    pub(crate) fn elems_ptr(&self) -> *const c_int {
        Self::ptr_or_null(&self.elems)
    }

    /// The layout as part of the compiled-artifact identity: a bundle compiled
    /// for one state layout can never be loaded against another.
    pub(crate) fn identity(&self) -> String {
        format!(
            "state-abi-v1:has_kv={}:ranks={:?}:dims={:?}:elems={:?}",
            self.has_kv, self.ranks, self.dims, self.elems
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mamba2_specs() -> Vec<StateTensorSpec> {
        vec![
            StateTensorSpec {
                name: "ssm_h",
                dims: vec![24, 24, 64, 128],
                elem: StateElem::F32,
            },
            StateTensorSpec {
                name: "ssm_conv",
                dims: vec![24, 3, 1792],
                elem: StateElem::F32,
            },
        ]
    }

    #[test]
    fn attention_only_layout_is_empty() {
        let ffi = StateFfi::new(true, &[]).unwrap();
        assert_eq!(ffi.has_kv, 1);
        assert_eq!(ffi.n_state, 0);
        assert!(ffi.ranks_ptr().is_null());
        assert!(ffi.dims_ptr().is_null());
        assert!(ffi.elems_ptr().is_null());
    }

    #[test]
    fn mamba2_layout_flattens_in_spec_order() {
        let ffi = StateFfi::new(false, &mamba2_specs()).unwrap();
        assert_eq!(ffi.has_kv, 0);
        assert_eq!(ffi.n_state, 2);
        assert_eq!(ffi.ranks, vec![4, 3]);
        assert_eq!(ffi.dims, vec![24, 24, 64, 128, 24, 3, 1792]);
        assert_eq!(ffi.elems, vec![0, 0]);
    }

    #[test]
    fn identity_distinguishes_layouts() {
        let a = StateFfi::new(false, &mamba2_specs()).unwrap();
        let mut wider = mamba2_specs();
        wider[1].dims[2] += 1;
        let b = StateFfi::new(false, &wider).unwrap();
        let hybrid = StateFfi::new(true, &mamba2_specs()).unwrap();
        assert_ne!(a.identity(), b.identity());
        assert_ne!(a.identity(), hybrid.identity());
        assert_ne!(a.identity(), StateFfi::new(true, &[]).unwrap().identity());
    }

    #[test]
    fn rejects_layouts_the_shim_cannot_describe() {
        assert!(StateFfi::new(false, &[]).is_err(), "no KV and no state");
        let mut zero = mamba2_specs();
        zero[0].dims[1] = 0;
        assert!(StateFfi::new(false, &zero).is_err());
        let mut scalar = mamba2_specs();
        scalar[0].dims.clear();
        assert!(StateFfi::new(false, &scalar).is_err());
        let mut deep = mamba2_specs();
        deep[0].dims = vec![1; MAX_STATE_RANK + 1];
        assert!(StateFfi::new(false, &deep).is_err());
        let many = vec![mamba2_specs()[0].clone(); MAX_STATE_TENSORS + 1];
        assert!(StateFfi::new(false, &many).is_err());
    }
}
