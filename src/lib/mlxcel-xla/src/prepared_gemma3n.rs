//! Owned dense per-layer embeddings for Gemma3n prepared prefill.
//!
//! The scheduler keeps this value in the pending request.  Its `Vec<f32>`
//! ownership makes normal completion, admission rejection, and cancellation all
//! release the potentially large PLE allocation through the same drop path.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gemma3nDensePleError {
    ZeroDimension(&'static str),
    ElementCountOverflow,
    LengthMismatch { actual: usize, expected: usize },
}

impl fmt::Display for Gemma3nDensePleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "Gemma3n dense PLE {name} must be nonzero"),
            Self::ElementCountOverflow => write!(f, "Gemma3n dense PLE element count overflows"),
            Self::LengthMismatch { actual, expected } => write!(
                f,
                "Gemma3n dense PLE has {actual} f32 elements; expected exactly {expected}"
            ),
        }
    }
}

impl std::error::Error for Gemma3nDensePleError {}

/// Dense projected per-layer embeddings with canonical `[capacity, layers,
/// hidden_per_layer]` row-major layout.
///
/// Construction is exact: callers must explicitly pad to `capacity`; this type
/// never truncates a longer payload or silently adds zero rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma3nDensePle {
    values: Vec<f32>,
    capacity: usize,
    layers: usize,
    hidden_per_layer: usize,
}

impl Gemma3nDensePle {
    pub fn new(
        values: Vec<f32>,
        capacity: usize,
        layers: usize,
        hidden_per_layer: usize,
    ) -> Result<Self, Gemma3nDensePleError> {
        for (name, value) in [
            ("capacity", capacity),
            ("layer count", layers),
            ("hidden width", hidden_per_layer),
        ] {
            if value == 0 {
                return Err(Gemma3nDensePleError::ZeroDimension(name));
            }
        }
        let expected = capacity
            .checked_mul(layers)
            .and_then(|v| v.checked_mul(hidden_per_layer))
            .ok_or(Gemma3nDensePleError::ElementCountOverflow)?;
        if values.len() != expected {
            return Err(Gemma3nDensePleError::LengthMismatch {
                actual: values.len(),
                expected,
            });
        }
        Ok(Self {
            values,
            capacity,
            layers,
            hidden_per_layer,
        })
    }

    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        [self.capacity, self.layers, self.hidden_per_layer]
    }

    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Weak};

    #[test]
    fn dense_ple_rejects_truncation_and_implicit_padding() {
        let err = Gemma3nDensePle::new(vec![0.0; 23], 2, 3, 4).unwrap_err();
        assert_eq!(
            err,
            Gemma3nDensePleError::LengthMismatch {
                actual: 23,
                expected: 24
            }
        );
        let ple = Gemma3nDensePle::new(vec![0.0; 24], 2, 3, 4).unwrap();
        assert_eq!(ple.shape(), [2, 3, 4]);
        assert_eq!(ple.byte_len(), 96);
    }

    #[test]
    fn pending_owner_drop_releases_dense_ple_on_cancel() {
        struct Pending {
            _ple: Gemma3nDensePle,
            _lifetime: Arc<()>,
        }
        let lifetime = Arc::new(());
        let weak: Weak<()> = Arc::downgrade(&lifetime);
        let pending = Pending {
            _ple: Gemma3nDensePle::new(vec![0.0; 8], 1, 2, 4).unwrap(),
            _lifetime: lifetime,
        };
        drop(pending);
        assert!(
            weak.upgrade().is_none(),
            "cancel must drop pending PLE owner"
        );
    }
}
