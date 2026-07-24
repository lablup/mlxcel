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

//! Generic resident-weight IREE module for compiler-side media front-ends.
//!
//! This lifecycle is intentionally independent of the language-model context:
//! creating, invoking, and dropping an auxiliary module cannot perturb the
//! established prefill/decode ABI. Vision and audio use typed descriptor arrays
//! so F32 features, I32 lengths, and bool masks share one fail-closed boundary.

use std::ffi::{CString, c_int, c_void};
use std::path::Path;

use crate::aux_manifest::{AuxiliaryArtifactContract, verify_auxiliary_manifest};

#[repr(C)]
struct XlaAuxCtx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XlaTensorDesc {
    data: *const c_void,
    byte_length: usize,
    dtype: c_int,
    rank: c_int,
    dims: [i64; 4],
}

#[repr(C)]
struct XlaMutTensorDesc {
    data: *mut c_void,
    byte_length: usize,
    dtype: c_int,
    rank: c_int,
    dims: [i64; 4],
}

unsafe extern "C" {
    fn xla_aux_create(
        device_uri: *const std::ffi::c_char,
        module_vmfb: *const std::ffi::c_char,
        entry_name: *const std::ffi::c_char,
        compatibility_fingerprint: u64,
        n_weights: c_int,
        weight_data: *const *const c_void,
        weight_dtypes: *const c_int,
        weight_ranks: *const c_int,
        weight_dims: *const i64,
    ) -> *mut XlaAuxCtx;
    fn xla_aux_invoke(
        context: *mut XlaAuxCtx,
        n_inputs: c_int,
        inputs: *const XlaTensorDesc,
        n_outputs: c_int,
        outputs: *mut XlaMutTensorDesc,
    ) -> c_int;
    fn xla_aux_free(context: *mut XlaAuxCtx);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryTensorDType {
    Float32,
    Int32,
    Bool,
}

impl AuxiliaryTensorDType {
    fn ffi_code(self) -> c_int {
        match self {
            Self::Float32 => 0,
            Self::Int32 => 1,
            Self::Bool => 2,
        }
    }

    fn size_bytes(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Bool => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryWeightDType {
    Float32,
    Float16,
    Uint32,
}

impl AuxiliaryWeightDType {
    fn ffi_code(self) -> c_int {
        match self {
            Self::Float32 => 0,
            Self::Float16 => 1,
            Self::Uint32 => 2,
        }
    }

    fn size_bytes(self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 | Self::Uint32 => 4,
        }
    }
}

pub(crate) struct AuxiliaryWeight {
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dtype: AuxiliaryWeightDType,
    pub(crate) shape: Vec<usize>,
}

pub(crate) struct AuxiliaryInput<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) dtype: AuxiliaryTensorDType,
    pub(crate) shape: &'a [usize],
}

pub(crate) struct AuxiliaryOutput<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) dtype: AuxiliaryTensorDType,
    pub(crate) shape: &'a [usize],
}

fn checked_i32(value: usize, label: &str) -> Result<c_int, String> {
    c_int::try_from(value).map_err(|_| format!("{label}={value} does not fit the native ABI"))
}

fn checked_shape(shape: &[usize], element_size: usize, label: &str) -> Result<[i64; 4], String> {
    if shape.len() > 4 {
        return Err(format!("{label} rank {} exceeds 4", shape.len()));
    }
    let mut dimensions = [0i64; 4];
    let mut elements = 1usize;
    for (axis, &dimension) in shape.iter().enumerate() {
        if dimension == 0 {
            return Err(format!("{label} axis {axis} must be positive"));
        }
        dimensions[axis] = i64::try_from(dimension)
            .map_err(|_| format!("{label} axis {axis} does not fit i64"))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| format!("{label} element count overflowed"))?;
    }
    elements
        .checked_mul(element_size)
        .ok_or_else(|| format!("{label} byte count overflowed"))?;
    Ok(dimensions)
}

fn expected_bytes(shape: &[usize], size: usize, label: &str) -> Result<usize, String> {
    shape.iter().try_fold(size, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{label} byte count overflowed"))
    })
}

fn c_path(path: &Path) -> Result<CString, String> {
    let path_text = path
        .to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))?;
    CString::new(path_text.as_bytes())
        .map_err(|_| format!("path has an interior nul byte: {}", path.display()))
}

pub(crate) struct IreeAuxiliaryModule {
    context: *mut XlaAuxCtx,
    fingerprint: u64,
}

impl IreeAuxiliaryModule {
    pub(crate) fn load(
        device: &str,
        vmfb: &Path,
        contract: &AuxiliaryArtifactContract,
        weights: Vec<AuxiliaryWeight>,
    ) -> Result<Self, String> {
        let fingerprint = verify_auxiliary_manifest(vmfb, contract, &weights)?;
        if weights.is_empty() {
            return Err("auxiliary module requires resident weights".to_string());
        }
        let mut dtypes = Vec::with_capacity(weights.len());
        let mut ranks = Vec::with_capacity(weights.len());
        let mut dimensions = Vec::with_capacity(weights.len() * 4);
        for (index, weight) in weights.iter().enumerate() {
            let label = format!("auxiliary weight {index}");
            let dims = checked_shape(&weight.shape, weight.dtype.size_bytes(), &label)?;
            let expected = expected_bytes(&weight.shape, weight.dtype.size_bytes(), &label)?;
            if weight.bytes.len() != expected {
                return Err(format!(
                    "{label} has {} bytes, expected {expected} for {:?} {:?}",
                    weight.bytes.len(),
                    weight.dtype,
                    weight.shape
                ));
            }
            dtypes.push(weight.dtype.ffi_code());
            ranks.push(checked_i32(weight.shape.len(), &format!("{label} rank"))?);
            dimensions.extend_from_slice(&dims);
        }
        let pointers: Vec<*const c_void> = weights
            .iter()
            .map(|weight| weight.bytes.as_ptr().cast())
            .collect();
        let device =
            CString::new(device).map_err(|_| "device URI has an interior nul byte".to_string())?;
        let vmfb = c_path(vmfb)?;
        let entry = CString::new(contract.entry_name.as_str())
            .map_err(|_| "auxiliary entry name has an interior nul byte".to_string())?;
        let count = checked_i32(weights.len(), "auxiliary weight count")?;
        // Safety: every pointer references immutable storage that remains live
        // through the call. The C context copies weights before returning.
        let context = unsafe {
            xla_aux_create(
                device.as_ptr(),
                vmfb.as_ptr(),
                entry.as_ptr(),
                fingerprint,
                count,
                pointers.as_ptr(),
                dtypes.as_ptr(),
                ranks.as_ptr(),
                dimensions.as_ptr(),
            )
        };
        if context.is_null() {
            return Err("xla_aux_create failed; see the IREE diagnostic above".to_string());
        }
        Ok(Self {
            context,
            fingerprint,
        })
    }

    #[must_use]
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub(crate) fn invoke(
        &mut self,
        inputs: &[AuxiliaryInput<'_>],
        outputs: &mut [AuxiliaryOutput<'_>],
    ) -> Result<(), String> {
        let input_descriptors = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let label = format!("auxiliary input {index}");
                let dims = checked_shape(input.shape, input.dtype.size_bytes(), &label)?;
                let expected = expected_bytes(input.shape, input.dtype.size_bytes(), &label)?;
                if input.bytes.len() != expected {
                    return Err(format!(
                        "{label} has {} bytes, expected {expected}",
                        input.bytes.len()
                    ));
                }
                Ok(XlaTensorDesc {
                    data: input.bytes.as_ptr().cast(),
                    byte_length: input.bytes.len(),
                    dtype: input.dtype.ffi_code(),
                    rank: checked_i32(input.shape.len(), &format!("{label} rank"))?,
                    dims,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut output_descriptors = outputs
            .iter_mut()
            .enumerate()
            .map(|(index, output)| {
                let label = format!("auxiliary output {index}");
                let dims = checked_shape(output.shape, output.dtype.size_bytes(), &label)?;
                let expected = expected_bytes(output.shape, output.dtype.size_bytes(), &label)?;
                if output.bytes.len() != expected {
                    return Err(format!(
                        "{label} has {} bytes, expected {expected}",
                        output.bytes.len()
                    ));
                }
                Ok(XlaMutTensorDesc {
                    data: output.bytes.as_mut_ptr().cast(),
                    byte_length: output.bytes.len(),
                    dtype: output.dtype.ffi_code(),
                    rank: checked_i32(output.shape.len(), &format!("{label} rank"))?,
                    dims,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let input_count = checked_i32(input_descriptors.len(), "auxiliary input count")?;
        let output_count = checked_i32(output_descriptors.len(), "auxiliary output count")?;
        // Safety: descriptors reference caller-owned slices for the synchronous
        // invocation only; C validates dtype/rank/shape/bytes before use.
        let status = unsafe {
            xla_aux_invoke(
                self.context,
                input_count,
                input_descriptors.as_ptr(),
                output_count,
                output_descriptors.as_mut_ptr(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format!("xla_aux_invoke failed (status {status})"))
        }
    }
}

impl Drop for IreeAuxiliaryModule {
    fn drop(&mut self) {
        if !self.context.is_null() {
            // Safety: context was returned by xla_aux_create and is freed once.
            unsafe { xla_aux_free(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}
