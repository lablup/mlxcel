// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Checked live-allocation accounting, not a physical-host admission estimate.
//! The external process guard owns host pressure. Geometry is never truncated.
use anyhow::{Context, Result, ensure};

pub(super) fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("FlexiCubes cardinality overflow")
}
pub(super) fn bytes<T>(count: usize) -> Result<usize> {
    product(count, std::mem::size_of::<T>())
}
pub(super) fn sum(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .context("FlexiCubes live-byte sum overflow")
    })
}
/// Account all retained arrays before the next reservation. The only ceiling
/// here is Rust's addressable allocation size, not an assumed free-RAM budget.
pub(super) fn phase(name: &str, retained: usize, additions: &[usize]) -> Result<usize> {
    let live = sum(&[retained, sum(additions)?])?;
    ensure!(
        live <= isize::MAX as usize,
        "FlexiCubes {name} live allocations exceed addressable size: {live} bytes"
    );
    tracing::debug!(phase = name, live_bytes = live, "FlexiCubes host workspace");
    Ok(live)
}
pub(super) fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let size = bytes::<T>(count)?;
    ensure!(
        size <= isize::MAX as usize,
        "FlexiCubes allocation exceeds addressable size"
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .with_context(|| format!("reserve FlexiCubes {size} bytes"))?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_objects_mesh_live_bytes_reject_overflow_before_allocation() {
        assert!(product(usize::MAX, 8).is_err());
        assert!(bytes::<[f32; 101]>(usize::MAX).is_err());
        assert!(sum(&[usize::MAX, 1]).is_err());
        assert!(phase("test", isize::MAX as usize, &[1]).is_err());
        assert!(reserve::<u64>(usize::MAX).is_err());
    }
    #[test]
    fn sam3d_objects_mesh_phase_accounts_retained_capacity() -> Result<()> {
        // Deduplication reduces length, never the allocation charged here.
        let mut values = reserve::<u64>(64)?;
        values.extend([1, 1]);
        values.dedup();
        assert_eq!(values.len(), 1);
        let retained = bytes::<u64>(values.capacity())?;
        assert!(retained >= 64 * 8);
        assert_eq!(
            phase("test", 101 * 4, &[retained, 40 * 8])?,
            101 * 4 + retained + 40 * 8
        );
        Ok(())
    }
    #[test]
    fn sam3d_objects_mesh_full_occupancy_plan_has_no_arbitrary_cap() -> Result<()> {
        use super::super::{Cube, Vertex};
        let m = product(42_000, 64)?;
        let c = 9_072_000;
        let u = 14_406_000;
        let d = product(c, 4)?;
        let e = product(c, 12)?;
        let external = sum(&[product(m, 101 * 2)?, product(m, 2 * 16 + 27 * 4)?])?;
        let edges_phase = phase(
            "max occupancy crossing edges",
            external,
            &[
                bytes::<Vertex>(u)?,
                bytes::<Cube>(c)?,
                bytes::<[f32; 3]>(d)?,
                bytes::<[f32; 6]>(d)?,
                bytes::<f32>(d)?,
                bytes::<[u32; 12]>(c)?,
                bytes::<(usize, usize, usize, usize)>(e)?,
            ],
        )?;
        assert!(edges_phase > 4usize * 1024 * 1024 * 1024);
        // This checks addressability and formulas; it does not allocate the
        // arrays or claim the physical host has this much memory available.
        Ok(())
    }
}
