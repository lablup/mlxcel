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

//! Verify-round timing on a real Inkling 2 Mini checkpoint (issue #3359).
//!
//! Plain decode never takes the grouped expert kernel, so its benchmark is a
//! verify forward: `width` tokens appended inside a
//! `DecodeLookaheadAppendScope` (what `Engine::verify` opens), evaluated, then
//! trimmed off again. Ignored by default; run it once per arm, with and
//! without `MLXCEL_INKLING2_KERNELS=0`, under the GPU lock:
//!
//! ```text
//! cargo test --profile test-fast --features metal,accelerate --lib \
//!     models::inkling2::verify_bench_tests -- --ignored --nocapture
//! ```
//!
//! It reads `models/inkling-2-mini` (override with `INKLING2_VERIFY_MODEL`)
//! and skips when the checkpoint is absent. Each width is also timed as a
//! plain multi-token forward (no speculative scope), which runs the other two
//! kernels but not the grouped one, so the kernels-on arm isolates the
//! grouped kernel's share. The printed argmax digest of every forward must
//! agree between the arms.

use std::time::Instant;

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::generate::LanguageModel;

use super::Inkling2Model;

const WIDTHS: [i32; 2] = [4, 8];
const WARMUP: usize = 3;
const ROUNDS: usize = 30;

#[test]
#[ignore = "needs the Inkling 2 Mini checkpoint and the GPU lock"]
fn verify_round_timing() {
    let path =
        std::env::var("INKLING2_VERIFY_MODEL").unwrap_or_else(|_| "models/inkling-2-mini".into());
    if !std::path::Path::new(&path).join("config.json").exists() {
        eprintln!("skipping: {path} is not present");
        return;
    }
    let (model, _) = Inkling2Model::load(&path).unwrap();
    let mut caches = model.make_caches();
    let prompt: Vec<i32> = (0..96).map(|i| 1000 + (i * 37) % 5000).collect();
    let input = mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32]);
    mlxcel_core::eval(&model.forward(&input, &mut caches, None));
    for (width, verify) in WIDTHS.iter().flat_map(|&w| [(w, true), (w, false)]) {
        let block: Vec<i32> = (0..width).map(|i| 2000 + i * 11).collect();
        let input = mlxcel_core::from_slice_i32(&block, &[1, width]);
        let mut times = Vec::with_capacity(ROUNDS);
        let mut digest = 0u64;
        for round in 0..WARMUP + ROUNDS {
            let start = Instant::now();
            let argmax = {
                let _speculative = verify.then(DecodeLookaheadAppendScope::enter);
                let logits = model.forward(&input, &mut caches, None);
                let argmax = mlxcel_core::argmax(&logits, -1, false);
                mlxcel_core::eval(&argmax);
                argmax
            };
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            model.trim_state(None, width).unwrap();
            if round >= WARMUP {
                times.push(elapsed);
            }
            if round == 0 {
                let ids = mlxcel_core::utils::array_to_vec_f32(&argmax);
                digest = ids.iter().fold(0u64, |acc, id| {
                    acc.wrapping_mul(1_000_003).wrapping_add(*id as u64)
                });
            }
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "{} width {width}: median {median:.2} ms, mean {mean:.2} ms, min {:.2} ms over \
             {ROUNDS} rounds; argmax digest {digest}",
            if verify { "verify" } else { "plain" },
            times[0]
        );
    }
}
