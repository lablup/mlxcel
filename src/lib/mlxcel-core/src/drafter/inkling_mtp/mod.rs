//! Native Inkling chained multi-token-prediction drafter.

pub mod config;
pub mod model;
mod sanitize;

pub use config::InklingMtpConfig;
pub use model::{InklingMtpDraftModel, has_inkling_mtp_tensors};

#[cfg(test)]
mod tests;
