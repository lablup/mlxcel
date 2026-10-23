// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Test-only point conditioning boundaries; production evaluation is unchanged.
use super::*;

impl ConditionEncoder {
    /// Observe both point branches using the existing production point/projection
    /// forwards, including the current 512-patch attention batching policy.
    pub(in crate::vision::dense::sam3d_objects) fn point_stage_diagnostics(
        &self,
        inputs: &Inputs,
    ) -> Result<BTreeMap<String, UniquePtr<MlxArray>>> {
        let point = self
            .point
            .as_ref()
            .context("point diagnostics need a point encoder")?;
        let projection = self
            .projections
            .get(2)
            .context("point projection missing")?;
        let mut values = BTreeMap::new();
        for name in ["pointmap", "rgb_pointmap"] {
            let input = inputs
                .get(name)
                .with_context(|| format!("missing point input {name}"))?;
            let resized = resize_nearest_nhwc(input, (point.size, point.size))?;
            let tokens = point.forward(input)?;
            let projected = projection.forward(&tokens);
            values.insert(format!("{name}.resized"), resized);
            values.insert(format!("{name}.tokens"), tokens);
            values.insert(format!("{name}.projected"), projected);
        }
        Ok(values)
    }
}
