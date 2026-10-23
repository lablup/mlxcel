// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Scoped, test-only MoGe observations. The sink controls evaluation and storage.
use std::cell::RefCell;
use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use mlxcel_core::MlxArray;

type Sink = Box<dyn FnMut(&str, &MlxArray) -> Result<()>>;
struct Capture {
    sink: Sink,
    series: BTreeMap<String, usize>,
}
thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        CAPTURE.with(|capture| *capture.borrow_mut() = None);
    }
}

/// Observe one synchronous inference, clearing the observer on error or unwind.
/// The callback must not recursively invoke MoGe; nested scopes are rejected.
pub(crate) fn with_capture<T>(
    sink: impl FnMut(&str, &MlxArray) -> Result<()> + 'static,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    CAPTURE.with(|capture| {
        let mut capture = capture.borrow_mut();
        ensure!(capture.is_none(), "nested MoGe diagnostic capture");
        *capture = Some(Capture {
            sink: Box::new(sink),
            series: BTreeMap::new(),
        });
        Ok(())
    })?;
    let _reset = Reset;
    run()
}
pub(super) fn active() -> bool {
    CAPTURE.with(|capture| capture.borrow().is_some())
}
pub(super) fn emit(name: &str, value: &MlxArray) -> Result<()> {
    CAPTURE.with(|capture| match &mut *capture.borrow_mut() {
        Some(capture) => (capture.sink)(name, value),
        None => Ok(()),
    })
}
pub(super) fn emit_many(values: &[(&str, &MlxArray)]) -> Result<()> {
    for (name, value) in values {
        emit(name, value)?;
    }
    Ok(())
}
pub(super) fn refinement(name: &str, value: &MlxArray) -> Result<()> {
    CAPTURE.with(|capture| {
        if let Some(capture) = &mut *capture.borrow_mut() {
            let index = capture.series.entry(name.into()).or_default();
            let key = format!("moge.refiner.{index}.{name}");
            *index += 1;
            (capture.sink)(&key, value)?;
        }
        Ok(())
    })
}

/// Emit a small configured pyramid, never every transformer hidden state.
pub(super) fn pyramid(prefix: &str, values: &[mlxcel_core::UniquePtr<MlxArray>]) -> Result<()> {
    if active() {
        for (index, value) in values.iter().enumerate() {
            emit(&format!("{prefix}{index}"), value)?;
        }
    }
    Ok(())
}

/// Reproduce only adapter preprocessing with its exact existing helpers. The
/// real encoder still runs once through its production `call` implementation.
pub(super) fn encoder_input(
    image: &MlxArray,
    rows: usize,
    cols: usize,
    patch: usize,
) -> Result<()> {
    if active() {
        let height = rows
            .checked_mul(patch)
            .ok_or_else(|| anyhow::anyhow!("capture height overflow"))?;
        let width = cols
            .checked_mul(patch)
            .ok_or_else(|| anyhow::anyhow!("capture width overflow"))?;
        let resized = crate::vision::dense::interpolate::resize_bilinear_nhwc(
            image,
            (height, width),
            false,
            true,
        )?;
        emit("moge.encoder.resized", &resized)?;
        let normalized = crate::vision::dense::preprocess::normalize_imagenet(&resized)?;
        emit("moge.encoder.normalized", &normalized)?;
    }
    Ok(())
}

#[test]
fn moge3_capture_scope_cleans_up_and_rejects_nesting() -> Result<()> {
    assert!(!active());
    with_capture(
        |_, _| Ok(()),
        || {
            assert!(active());
            assert!(with_capture(|_, _| Ok(()), || Ok(())).is_err());
            assert!(active());
            Ok(())
        },
    )?;
    assert!(!active());
    let result: Result<()> = with_capture(|_, _| Ok(()), || anyhow::bail!("injected failure"));
    assert!(result.is_err());
    assert!(!active());
    assert!(
        std::panic::catch_unwind(|| {
            let _: Result<()> = with_capture(|_, _| Ok(()), || panic!("injected unwind"));
        })
        .is_err()
    );
    assert!(!active());
    Ok(())
}
