// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Bundled Objects precision policy; standalone MoGe retains its established graph.
use anyhow::Result;
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Policy {
    #[default]
    Standard,
    Objects,
}

impl Policy {
    pub(super) fn resize(
        self,
        x: &MlxArray,
        size: (usize, usize),
        antialias: bool,
    ) -> Result<UniquePtr<MlxArray>> {
        use crate::vision::dense::interpolate::{
            resize_bilinear_nhwc, resize_bilinear_objects_nhwc_with_antialias,
        };
        match self {
            Self::Standard => resize_bilinear_nhwc(x, size, false, antialias),
            Self::Objects => resize_bilinear_objects_nhwc_with_antialias(x, size, antialias),
        }
    }

    pub(super) fn uv(
        self,
        width: usize,
        height: usize,
        aspect: f64,
    ) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::Standard => {
                super::geometry::normalized_view_plane_uv(width, height, Some(aspect as f32))
            }
            Self::Objects => super::geometry::normalized_view_plane_uv_with_policy(
                width,
                height,
                Some(aspect),
                self,
            ),
        }
    }

    pub(super) fn recover(
        self,
        points: &MlxArray,
        mask: Option<&MlxArray>,
        focal: Option<&MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        match self {
            Self::Standard => super::geometry::recover_focal_shift(points, mask, focal),
            Self::Objects => {
                super::geometry::recover_focal_shift_with_policy(points, mask, focal, self)
            }
        }
    }

    /// Python computes scalar spans/endpoints in double precision before MLX
    /// converts linspace's scalar arguments to F32. Preserve that rounding once.
    pub(super) fn endpoints(self, width: usize, height: usize, aspect: Option<f64>) -> [f32; 4] {
        match self {
            Self::Objects => {
                let aspect = aspect.unwrap_or(width as f64 / height as f64);
                let diagonal = (1.0 + aspect * aspect).sqrt();
                let span_x = aspect / diagonal;
                let span_y = 1.0 / diagonal;
                let u = span_x * (width - 1) as f64 / width as f64;
                let v = span_y * (height - 1) as f64 / height as f64;
                [(-u) as f32, u as f32, (-v) as f32, v as f32]
            }
            Self::Standard => {
                let aspect = aspect
                    .map(|v| v as f32)
                    .unwrap_or(width as f32 / height as f32);
                let span_x = aspect / (1.0 + aspect * aspect).sqrt();
                let span_y = 1.0 / (1.0 + aspect * aspect).sqrt();
                let u = span_x * (width - 1) as f32 / width as f32;
                let v = span_y * (height - 1) as f32 / height as f32;
                [-u, u, -v, v]
            }
        }
    }
}

#[test]
fn moge3_objects_uv_endpoints_match_pinned_python_scalar_rounding() {
    // struct.unpack("<I", struct.pack("<f", endpoint)) from the unchanged
    // geometry.py::_view_plane_axes Python scalar expressions, without MLX.
    // Level 2/4 use original image aspect 1.5, not their rounded grid aspect.
    for (width, height, aspect, upstream, legacy) in [
        (
            292,
            196,
            Some(1.5),
            [0xbf544681, 0x3f544681, 0xbf0d475c, 0x3f0d475c],
            [0xbf544682, 0x3f544682, 0xbf0d475b, 0x3f0d475b],
        ),
        (
            1168,
            784,
            Some(1.5),
            [0xbf54d290, 0x3f54d290, 0xbf0dd277, 0x3f0dd277],
            [0xbf54d291, 0x3f54d291, 0xbf0dd277, 0x3f0dd277],
        ),
        // The focal solver infers aspect from the original output image.
        (
            672,
            448,
            None,
            [0xbf54b01a, 0x3f54b01a, 0xbf0dafb0, 0x3f0dafb0],
            [0xbf54b01b, 0x3f54b01b, 0xbf0dafb0, 0x3f0dafb0],
        ),
    ] {
        assert_eq!(
            Policy::Objects
                .endpoints(width, height, aspect)
                .map(f32::to_bits),
            upstream
        );
        assert_eq!(
            Policy::default()
                .endpoints(width, height, aspect)
                .map(f32::to_bits),
            legacy
        );
    }
}
