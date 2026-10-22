// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Backbone and task-head composition for Sapiens2.

use std::path::Path;

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::backbone::Sapiens2Backbone;
use super::config::{Sapiens2Config, Sapiens2Task};
use super::heads::Sapiens2Head;
use super::weights::load_sanitized_weights;

pub enum ModelOutput {
    Backbone {
        last_hidden_state: UniquePtr<MlxArray>,
        pooler_output: UniquePtr<MlxArray>,
    },
    Task {
        prediction: UniquePtr<MlxArray>,
        scale: Option<UniquePtr<MlxArray>>,
    },
}

pub struct Sapiens2Model {
    pub config: Sapiens2Config,
    backbone: Sapiens2Backbone,
    head: Option<Sapiens2Head>,
}

impl Sapiens2Model {
    pub fn load(path: &Path) -> Result<Self> {
        let config = Sapiens2Config::from_path(path)?;
        let weights = load_sanitized_weights(path, &config)?;
        let backbone = Sapiens2Backbone::from_weights(&weights, &config)?;
        let head = Sapiens2Head::from_weights(&weights, &config)?;
        Ok(Self {
            config,
            backbone,
            head,
        })
    }

    pub fn forward(&self, pixel_values: &MlxArray) -> Result<ModelOutput> {
        let output = self.backbone.forward(pixel_values)?;
        if self.config.task == Sapiens2Task::Backbone {
            let shape = mlxcel_core::array_shape(&output.tokens);
            let pooler = mlxcel_core::slice(&output.tokens, &[0, 0, 0], &[shape[0], 1, shape[2]]);
            let pooler = mlxcel_core::reshape(&pooler, &[shape[0], shape[2]]);
            return Ok(ModelOutput::Backbone {
                last_hidden_state: output.tokens,
                pooler_output: pooler,
            });
        }
        let head = self
            .head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Sapiens2 task model has no decode head"))?;
        let shape = mlxcel_core::array_shape(&output.tokens);
        ensure!(
            shape[1] >= self.backbone.prefix_tokens() as i32
                && shape[1] - self.backbone.prefix_tokens() as i32
                    == (output.height * output.width) as i32,
            "Sapiens2 patch-token count does not match its grid"
        );
        let patches = mlxcel_core::slice(
            &output.tokens,
            &[0, self.backbone.prefix_tokens() as i32, 0],
            &[shape[0], shape[1], shape[2]],
        );
        let features = mlxcel_core::reshape(
            &patches,
            &[
                shape[0],
                output.height as i32,
                output.width as i32,
                self.config.hidden_size as i32,
            ],
        );
        let output = head.forward(&features);
        Ok(ModelOutput::Task {
            prediction: output.prediction,
            scale: output.scale,
        })
    }
}
