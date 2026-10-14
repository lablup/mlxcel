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

//! Thread-local, one-shot failure injection at decode evaluation boundaries.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaultPoint {
    Submit,
    Draw,
    Readback,
    SamplerRead,
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) fn check(_point: FaultPoint) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
thread_local! {
    static PLAN: std::cell::Cell<Option<(FaultPoint, usize)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn check(point: FaultPoint) -> Result<(), String> {
    PLAN.with(|plan| {
        if let Some((armed, remaining)) = plan.get()
            && armed == point
        {
            if remaining == 0 {
                plan.set(None);
                return Err("injected eval failure".into());
            }
            plan.set(Some((armed, remaining - 1)));
        }
        Ok(())
    })
}

#[cfg(test)]
pub(crate) struct FaultGuard(std::marker::PhantomData<std::rc::Rc<()>>);

#[cfg(test)]
impl FaultGuard {
    pub(crate) fn fired(&self) -> bool {
        PLAN.with(|plan| plan.get().is_none())
    }

    pub(crate) fn arm(point: FaultPoint, nth: usize) -> Self {
        PLAN.with(|plan| {
            assert!(plan.get().is_none(), "fault plan already armed");
            plan.set(Some((point, nth)));
        });
        Self(std::marker::PhantomData)
    }
}

#[cfg(test)]
impl Drop for FaultGuard {
    fn drop(&mut self) {
        PLAN.with(|plan| plan.set(None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_are_one_shot_scoped_and_thread_local() {
        {
            let _guard = FaultGuard::arm(FaultPoint::Submit, 1);
            assert!(check(FaultPoint::Draw).is_ok());
            std::thread::spawn(|| {
                assert!(check(FaultPoint::Submit).is_ok());
                let _guard = FaultGuard::arm(FaultPoint::Readback, 0);
                assert!(check(FaultPoint::Readback).is_err());
            })
            .join()
            .unwrap();
            assert!(check(FaultPoint::Submit).is_ok());
            assert_eq!(
                check(FaultPoint::Submit),
                Err("injected eval failure".into())
            );
            assert!(check(FaultPoint::Submit).is_ok());
        }
        {
            let _guard = FaultGuard::arm(FaultPoint::Draw, 0);
        }
        assert!(check(FaultPoint::Draw).is_ok());
    }
}
