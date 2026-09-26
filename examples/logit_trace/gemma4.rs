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

//! Opt-in teacher-forced Gemma 4 block, corrected verify, and decode-chain
//! arms. Prefill uses the real target adapter so verify arms also exercise
//! its buffered rotating-cache layout past the sliding window.
use anyhow::{Result, bail};
use mlxcel::{
    LanguageModel, LoadedModel,
    models::{Gemma4SpeculativeSinks, Gemma4Wrapper, gemma4_mtp_target::Gemma4MtpTargetAdapter},
};
use mlxcel_core::{
    MlxArray, UniquePtr, generate::SamplingConfig, sampling::LogprobsConfig,
    speculative::mtp::target::MtpTarget, utils::slice_axis,
};

#[derive(Clone, Copy)]
pub enum Mode {
    Block,
    Chain,
    Verify,
}

impl Mode {
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var("MLXCEL_TRACE_GEMMA4").as_deref() {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok("block") => Ok(Some(Self::Block)),
            Ok("chain") => Ok(Some(Self::Chain)),
            Ok("verify") => Ok(Some(Self::Verify)),
            _ => bail!("MLXCEL_TRACE_GEMMA4 must be block|chain|verify"),
        }
    }

    pub fn prefill(self, model: &LoadedModel, tokens: &[i32], width: usize) -> Result<()> {
        let wrapper = wrapper(model)?;
        if matches!(self, Self::Chain) {
            let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
            mlxcel_core::eval(&wrapper.forward(&input, &mut [], None));
        } else {
            let adapter = Gemma4MtpTargetAdapter::new_with_block_size(wrapper, None, width)
                .with_prefill_chunk_size(0);
            let sampling = SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            };
            let _ = adapter.prefill_and_seed(tokens, &sampling, &[], &LogprobsConfig::default());
        }
        Ok(())
    }

    pub fn forward(self, model: &LoadedModel, input: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let wrapper = wrapper(model)?;
        if matches!(self, Self::Chain) {
            let width = mlxcel_core::array_shape(input)[1];
            let mut result = wrapper.forward(&slice_axis(input, 1, 0, 1), &mut [], None);
            mlxcel_core::eval(&result);
            for row in 1..width {
                let logits = wrapper.forward(&slice_axis(input, 1, row, row + 1), &mut [], None);
                mlxcel_core::eval(&logits);
                result = mlxcel_core::concatenate(&result, &logits, 1);
            }
            Ok(result)
        } else {
            let mut sinks = Gemma4SpeculativeSinks {
                mtp_verify: matches!(self, Self::Verify),
                ..Default::default()
            };
            Ok(wrapper.forward_with_speculative_sinks(
                input,
                None,
                None,
                None,
                None,
                None,
                Some(&mut sinks),
                None,
            ))
        }
    }
}

fn wrapper(model: &LoadedModel) -> Result<&Gemma4Wrapper> {
    match model {
        LoadedModel::Gemma4(model) => Ok(model),
        LoadedModel::Gemma4VLM(model) => Ok(&model.text_model),
        LoadedModel::Gemma4Unified(model) => Ok(&model.text_model),
        _ => bail!("MLXCEL_TRACE_GEMMA4 requires a Gemma 4 target"),
    }
}
