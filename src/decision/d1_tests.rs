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

//! Gates for the LiquidAI d1 decision scorer (`d1.rs`, `d1_prompt.rs`).
//!
//! - Detection: the published config routes to [`DecisionKind::D1`], the
//!   same config without `auto_map` is refused, and generation detection
//!   still reads both as LFM2-VL.
//! - Prompt and readout, over a scripted tokenizer: option codes (single
//!   letters, `A`..., two-digit, fallback pool), normalization errors with
//!   upstream's messages, the exact question text, and branch batching.
//! - Trunk/branch equivalence on a synthetic float32 LFM2 backbone: several
//!   questions over one trunk give the log-probabilities each gives alone,
//!   with trunk prefill chunking and branch batch boundaries forced.
//! - The real tokenizer reproduces mlx-vlm's trunk, branch and readout ids
//!   (`tests/fixtures/d1/reference.json`); no GPU needed.
//! - Parity against mlx-vlm on `LiquidAI/d1-3B-MLX-8bit`: every fixture case,
//!   text and image, within 2e-3. It soft-skips when the checkpoint is
//!   absent.

use std::collections::HashMap;
use std::path::PathBuf;

use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use super::d1::{D1DecisionModel, answer_logprobs, is_d1_config};
use super::d1_prompt::{
    D1Question, TokenEncoder, aliases, as_question, branch_chunks, cap_pixels, option_codes,
    prompt_ids, question_block, readout, readout_ids, state_block,
};
use super::*;
use crate::models::embedding_test_support::{local_checkpoint, max_abs_diff, mlx_test_guard};
use crate::models::lfm2::Lfm2Model;

const D1_CHECKPOINT: &str = "LiquidAI/d1-3B-MLX-8bit";
const D1_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];
const D1_MEDIA: &[MediaKind] = &[MediaKind::Image];
/// Probability tolerance against mlx-vlm on the same checkpoint.
const PARITY_TOLERANCE: f64 = 2e-3;
/// Log-probability tolerance between trunk/branch and single-pass reads on
/// the float32 synthetic backbone.
const EQUIVALENCE_TOLERANCE: f32 = 1e-4;

// --- helpers -------------------------------------------------------------

/// A scripted tokenizer: listed texts encode to their listed ids, anything
/// else to one token per character (ids from 5000 up), so a single
/// character is one token unless listed otherwise.
struct ScriptedEncoder(HashMap<String, Vec<u32>>);

impl ScriptedEncoder {
    fn new(entries: &[(&str, &[u32])]) -> Self {
        Self(
            entries
                .iter()
                .map(|(text, ids)| (text.to_string(), ids.to_vec()))
                .collect(),
        )
    }
}

impl TokenEncoder for ScriptedEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        if let Some(ids) = self.0.get(text) {
            return Ok(ids.clone());
        }
        Ok(text.chars().map(|c| 5000 + c as u32).collect())
    }
}

fn validated(state: Option<&Value>, questions: &Value, images: bool) -> Vec<(String, Question)> {
    validate_questions(
        D1_TYPES,
        D1_MEDIA,
        state,
        questions,
        MediaPresence {
            images,
            ..MediaPresence::default()
        },
    )
    .expect("questions pass the shared validation")
}

fn d1_question(question: Value) -> Result<D1Question, DecisionError> {
    let questions = json!({ "q": question });
    let state = json!("state");
    let validated = validate_questions(
        D1_TYPES,
        D1_MEDIA,
        Some(&state),
        &questions,
        MediaPresence::default(),
    )?;
    as_question(&validated[0].1)
}

fn invalid_message(result: Result<D1Question, DecisionError>) -> String {
    match result {
        Err(DecisionError::InvalidInput(message)) => message,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/d1/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture is committed"))
        .expect("fixture parses")
}

fn fixture_case(name: &str) -> Value {
    fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .cloned()
        .unwrap_or_else(|| panic!("fixture case {name}"))
}

fn ids(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_i64().unwrap() as i32)
        .collect()
}

fn case_state(case: &Value) -> Option<Value> {
    Some(case["state"].clone()).filter(|state| !state.is_null())
}

fn cats_image(doubled: bool) -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sam3/cats.png");
    let image = image::open(path).expect("cats fixture decodes").to_rgb8();
    if !doubled {
        return image::DynamicImage::ImageRgb8(image);
    }
    // Pixel replication, the exact 2x upscale the fixture generator uses.
    let (w, h) = image.dimensions();
    let big = image::RgbImage::from_fn(w * 2, h * 2, |x, y| *image.get_pixel(x / 2, y / 2));
    image::DynamicImage::ImageRgb8(big)
}

// --- detection -----------------------------------------------------------

/// The routing keys of `LiquidAI/d1-3B` and its MLX conversions.
fn published_config() -> Value {
    json!({
        "architectures": ["Lfm2VlForConditionalGeneration"],
        "auto_map": {"AutoModel": "modeling_d1.D1Model"},
        "model_type": "lfm2_vl",
    })
}

fn without_auto_map() -> Value {
    let mut plain = published_config();
    plain.as_object_mut().unwrap().remove("auto_map");
    plain
}

#[test]
fn published_config_is_a_d1_decision_checkpoint() {
    assert_eq!(
        detect_decision_kind(&published_config()).unwrap(),
        DecisionKind::D1
    );
    assert_eq!(DecisionKind::D1.as_str(), "d1");
    let mut dashed = published_config();
    dashed["model_type"] = json!("lfm2-vl");
    assert!(is_d1_config(&dashed));
    if let Some(dir) = local_checkpoint(D1_CHECKPOINT) {
        let config = crate::embeddings::loader::read_embedding_config(&dir).unwrap();
        assert_eq!(detect_decision_kind(&config).unwrap(), DecisionKind::D1);
    }
}

#[test]
fn lfm2_vl_without_d1_auto_map_is_refused() {
    assert!(detect_decision_kind(&without_auto_map()).is_err());
    let mut other = published_config();
    other["auto_map"] = json!({"AutoModel": "modeling_lfm2_vl.Lfm2VlModel"});
    assert!(detect_decision_kind(&other).is_err());
    // A D1Model auto_map on another backbone is not this scorer's either.
    let mut other_backbone = published_config();
    other_backbone["model_type"] = json!("lfm3_vl");
    assert!(detect_decision_kind(&other_backbone).is_err());
}

#[test]
fn plain_lfm2_vl_decision_load_names_the_path() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        without_auto_map().to_string(),
    )
    .unwrap();
    let err = load_decision_model(dir.path()).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("{} is not a decision checkpoint", dir.path().display())
    );
}

#[test]
fn generation_detection_still_reads_d1_as_lfm2_vl() {
    for config in [published_config(), without_auto_map()] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        assert_eq!(
            crate::models::get_model_type(dir.path()).unwrap(),
            crate::models::ModelType::Lfm2VL
        );
    }
}

// --- normalization -------------------------------------------------------

#[test]
fn as_question_requires_instructions() {
    let message = invalid_message(d1_question(json!({"type": "bool"})));
    assert_eq!(message, "a question needs `instructions`");
}

#[test]
fn as_question_bounds_score_levels() {
    let levels: Vec<String> = (0..11).map(|i| format!("level {i}")).collect();
    let message = invalid_message(d1_question(
        json!({"type": "score", "instructions": "How?", "criteria": levels}),
    ));
    assert_eq!(
        message,
        "a score needs criteria: a list of 2 to 10 level descriptions, lowest first"
    );
    let ten: Vec<String> = (0..10).map(|i| format!("level {i}")).collect();
    assert!(d1_question(json!({"type": "score", "instructions": "How?", "criteria": ten})).is_ok());
}

#[test]
fn as_question_refuses_bool_keys_other_than_true_and_false() {
    let message = invalid_message(d1_question(json!({
        "type": "bool",
        "instructions": "Refund?",
        "criteria": {"true": "yes", "maybe": "?", "always": "!"},
    })));
    assert_eq!(
        message,
        "bool criteria describe \"true\" and \"false\", got keys 'always', 'maybe', 'true'"
    );
}

#[test]
fn as_question_reads_bool_as_noul_and_list_choices_as_labels() {
    let noul = d1_question(json!({"type": "bool", "instructions": "Refund?"})).unwrap();
    assert_eq!(
        noul,
        D1Question::Noul {
            instructions: "Refund?".to_string(),
            descriptions: None
        }
    );
    let alias = d1_question(json!({"type": "noul", "instructions": "Refund?"})).unwrap();
    assert_eq!(alias, noul);
    // A non-empty map renders both lines; a missing side prints Python's None.
    let half = d1_question(json!({
        "type": "bool", "instructions": "Refund?", "criteria": {"true": "asks for money back"},
    }))
    .unwrap();
    assert_eq!(
        half,
        D1Question::Noul {
            instructions: "Refund?".to_string(),
            descriptions: Some(("asks for money back".to_string(), "None".to_string())),
        }
    );
    let choice =
        d1_question(json!({"instructions": "Team?", "criteria": ["billing", "tech"]})).unwrap();
    assert_eq!(
        choice,
        D1Question::Choice {
            instructions: "Team?".to_string(),
            options: vec![("billing".to_string(), None), ("tech".to_string(), None)],
        }
    );
}

// --- option codes --------------------------------------------------------

#[test]
fn option_codes_follow_upstream_branches() {
    assert_eq!(
        option_codes(&strings(&["x", "Y", " z "])),
        strings(&["x", "Y", "z"])
    );
    assert_eq!(
        option_codes(&strings(&["billing", "technical", "c"])),
        strings(&["A", "B", "C"])
    );
    let many: Vec<String> = (0..27).map(|i| format!("label {i}")).collect();
    let codes = option_codes(&many);
    assert_eq!(codes.len(), 27);
    assert_eq!(codes[0], "00");
    assert_eq!(codes[9], "09");
    assert_eq!(codes[26], "26");
    let twenty_six: Vec<String> = (0..26).map(|i| format!("label {i}")).collect();
    assert_eq!(option_codes(&twenty_six)[25], "Z");
}

#[test]
fn aliases_fall_back_to_the_first_unused_single_token() {
    // `B` encodes to the same id as `A`, and `C` is two tokens, so both fall
    // back to the pool, whose first unused single-token entries are `D`, `E`.
    let encoder = ScriptedEncoder::new(&[("A", &[1]), ("B", &[1]), ("C", &[3, 4])]);
    let codes = aliases(&encoder, &strings(&["one", "two", "three"])).unwrap();
    let expected = vec![
        ("A".to_string(), 1),
        ("D".to_string(), 5000 + 'D' as u32),
        ("E".to_string(), 5000 + 'E' as u32),
    ];
    assert_eq!(codes, expected);
}

#[test]
fn alias_fallback_encodes_each_pool_entry_at_most_once() {
    // Every code and pool entry is two tokens except the pool's lowercase
    // letters, so options fall back deep into the pool. A rescan from the
    // start per option would encode ~150 entries per option; resuming the
    // scan keeps the total near one pool sweep plus one call per option.
    struct Counting(std::cell::Cell<usize>);
    impl TokenEncoder for Counting {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
            self.0.set(self.0.get() + 1);
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_lowercase() => Ok(vec![c as u32]),
                _ => Ok(vec![1, 2]),
            }
        }
    }
    let labels: Vec<String> = (0..40).map(|i| format!("label {i}")).collect();
    let encoder = Counting(std::cell::Cell::new(0));
    let err = aliases(&encoder, &labels).unwrap_err();
    assert_eq!(
        err,
        DecisionError::InvalidInput("no single-token alias left for 40 options".to_string())
    );
    let pool_len = 26 + 100 + 26 + 200 + 26 * 26;
    assert!(
        encoder.0.get() <= labels.len() + pool_len,
        "{} encoder calls",
        encoder.0.get()
    );
    // The resumed scan still hands out the first unused entries in order.
    let six: Vec<String> = (0..6).map(|i| format!("label {i}")).collect();
    let codes: Vec<String> = aliases(&encoder, &six)
        .unwrap()
        .into_iter()
        .map(|(code, _)| code)
        .collect();
    assert_eq!(codes, strings(&["a", "b", "c", "d", "e", "f"]));
}

#[test]
fn prompts_longer_than_the_context_are_invalid_input() {
    use super::d1::{check_prompt_length, max_prompt_tokens_from_config};
    assert_eq!(
        max_prompt_tokens_from_config(&json!({"text_config": {"max_position_embeddings": 128000}})),
        Some(128_000)
    );
    assert_eq!(
        max_prompt_tokens_from_config(&json!({"text_config": {}})),
        None
    );
    // The trunk counts once and every question counts: 60 + 10 + 30 fits.
    let branches = vec![vec![1; 10], vec![1; 30]];
    assert!(check_prompt_length(60, &branches, 100).is_ok());
    assert_eq!(
        check_prompt_length(61, &branches, 100).unwrap_err(),
        DecisionError::InvalidInput(
            "the decision prompt (state and all questions) is 101 tokens, more than the model's 100-token context"
                .to_string()
        )
    );
    // Many short questions on a short state are refused by their sum.
    assert!(check_prompt_length(1, &vec![vec![1; 2]; 49], 100).is_ok());
    assert!(check_prompt_length(1, &vec![vec![1; 2]; 50], 100).is_err());
    assert!(check_prompt_length(101, &[], 100).is_err());
}

#[test]
fn aliases_fail_when_the_pool_is_exhausted() {
    // Every text is the same single token, so only the first option gets one.
    struct OneToken;
    impl TokenEncoder for OneToken {
        fn encode_text(&self, _text: &str) -> Result<Vec<u32>, DecisionError> {
            Ok(vec![7])
        }
    }
    let err = aliases(&OneToken, &strings(&["one", "two"])).unwrap_err();
    assert_eq!(
        err,
        DecisionError::InvalidInput("no single-token alias left for 2 options".to_string())
    );
}

// --- prompt text ---------------------------------------------------------

#[test]
fn choice_block_lists_codes_with_descriptions() {
    let encoder = ScriptedEncoder::new(&[]);
    let question = d1_question(json!({
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds", "tech_support": null, "sales": ""},
    }))
    .unwrap();
    assert_eq!(
        question_block(&encoder, &question).unwrap(),
        "Which team should handle this?\n\nOptions:\nA Charges, refunds\nB tech support\n\
         C sales\n\nReply with the option code only."
    );
}

#[test]
fn bool_block_with_criteria_names_both_answers() {
    let encoder = ScriptedEncoder::new(&[]);
    let question = d1_question(json!({
        "type": "bool",
        "instructions": "Does this need a reply today?",
        "criteria": {"true": "the customer is blocked", "false": "it can wait"},
    }))
    .unwrap();
    assert_eq!(
        question_block(&encoder, &question).unwrap(),
        "Does this need a reply today?\nYes: the customer is blocked\nNo: it can wait\n\n\
         Reply with yes or no only."
    );
    let bare =
        d1_question(json!({"type": "bool", "instructions": "Refund?", "criteria": {}})).unwrap();
    assert_eq!(
        question_block(&encoder, &bare).unwrap(),
        "Refund?\n\nReply with yes or no only."
    );
}

#[test]
fn score_block_numbers_levels_lowest_first() {
    let encoder = ScriptedEncoder::new(&[]);
    let question = d1_question(json!({
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    }))
    .unwrap();
    assert_eq!(
        question_block(&encoder, &question).unwrap(),
        "How urgent is this?\n\n0 Can wait\n1 Today\n2 Blocking the customer now\n\n\
         Reply with a single digit 0-2 only."
    );
}

#[test]
fn state_block_renders_strings_raw_and_values_as_indented_json() {
    assert_eq!(state_block(&json!("charged twice")), "charged twice\n\n");
    assert_eq!(
        state_block(&json!({"ticket": "환불", "items": [1, 2]})),
        "{\n  \"ticket\": \"환불\",\n  \"items\": [\n    1,\n    2\n  ]\n}\n\n"
    );
}

#[test]
fn one_question_is_one_pass_and_several_share_the_trunk() {
    let encoder = ScriptedEncoder::new(&[]);
    let state = json!("s");
    let refund = d1_question(json!({"type": "bool", "instructions": "R?"})).unwrap();
    let urgency =
        d1_question(json!({"type": "score", "instructions": "U?", "criteria": ["a", "b"]}))
            .unwrap();
    let text = |ids: &[i32]| -> String {
        ids.iter()
            .map(|&id| char::from_u32(id as u32 - 5000).unwrap())
            .collect()
    };
    let (trunk, branches) = prompt_ids(
        &encoder,
        "<bos>",
        Some(&state),
        std::slice::from_ref(&refund),
        0,
    )
    .unwrap();
    assert!(branches.is_empty());
    assert_eq!(
        text(&trunk),
        "<bos><|im_start|>user\ns\n\n\nQUESTION:\nR?\n\nReply with yes or no only.\
         <|im_end|>\n<|im_start|>assistant\n"
    );
    let (trunk, branches) = prompt_ids(&encoder, "", None, &[refund, urgency], 2).unwrap();
    assert_eq!(text(&trunk), "<|im_start|>user\n<image><image>");
    assert_eq!(branches.len(), 2);
    assert_eq!(
        text(&branches[0]),
        "R?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

// --- readout -------------------------------------------------------------

#[test]
fn readout_softmaxes_each_options_best_token() {
    let groups = vec![vec![0, 1], vec![2]];
    let logz = [-3.0f32, -1.0, -2.0];
    let p = readout(&groups, &logz).unwrap();
    let e = (-1.0f64).exp() + (-2.0f64).exp();
    assert!((p[0] - (-1.0f64).exp() / e).abs() < 1e-12);
    assert!((p[1] - (-2.0f64).exp() / e).abs() < 1e-12);
    assert!(readout(&[vec![9]], &logz).is_err());
}

#[test]
fn readout_ids_without_single_token_forms_are_internal() {
    struct TwoTokens;
    impl TokenEncoder for TwoTokens {
        fn encode_text(&self, _text: &str) -> Result<Vec<u32>, DecisionError> {
            Ok(vec![1, 2])
        }
    }
    let noul = d1_question(json!({"type": "bool", "instructions": "R?"})).unwrap();
    assert!(matches!(
        readout_ids(&TwoTokens, &noul),
        Err(DecisionError::Internal(_))
    ));
    let score = d1_question(json!({"type": "score", "instructions": "U?", "criteria": ["a", "b"]}))
        .unwrap();
    assert!(matches!(
        readout_ids(&TwoTokens, &score),
        Err(DecisionError::Internal(_))
    ));
}

#[test]
fn choice_readout_adds_the_space_prefixed_code_when_distinct() {
    let encoder = ScriptedEncoder::new(&[("A", &[10]), (" A", &[11]), ("B", &[12]), (" B", &[12])]);
    let question = d1_question(json!({"instructions": "T?", "criteria": ["x1", "x2"]})).unwrap();
    assert_eq!(
        readout_ids(&encoder, &question).unwrap(),
        vec![vec![10, 11], vec![12]]
    );
}

// --- branch batching -----------------------------------------------------

#[test]
fn branch_chunks_go_longest_first_within_budget_and_length_ratio() {
    // Everything fits: one batch, listed in request order.
    assert_eq!(branch_chunks(&[5, 8, 6], 10, 1000), vec![vec![0, 1, 2]]);
    // Budget for exactly two rows of trunk + longest (2 * (10 + 8) = 36).
    assert_eq!(branch_chunks(&[5, 8, 6], 10, 36), vec![vec![1, 2], vec![0]]);
    // A row under half the longest starts its own batch.
    assert_eq!(
        branch_chunks(&[3, 8, 4], 10, 1000),
        vec![vec![1, 2], vec![0]]
    );
    // A budget below one row still runs every row, one at a time.
    assert_eq!(branch_chunks(&[2, 2], 10, 5), vec![vec![0], vec![1]]);
}

// --- image cap -----------------------------------------------------------

#[test]
fn cap_pixels_downscales_by_area_and_keeps_small_images() {
    let small = image::DynamicImage::new_rgba8(640, 480);
    let kept = cap_pixels(&small);
    assert_eq!((kept.width(), kept.height()), (640, 480));
    assert!(matches!(kept, image::DynamicImage::ImageRgb8(_)));
    let big = image::DynamicImage::new_rgb8(1280, 960);
    let capped = cap_pixels(&big);
    // scale = sqrt(1048576 / 1228800); int(1280 * scale), int(960 * scale).
    assert_eq!((capped.width(), capped.height()), (1182, 886));
}

// --- trunk/branch equivalence on a synthetic float32 backbone -------------

fn tiny_backbone() -> Lfm2Model {
    use crate::models::lfm2_embedding::lfm2_embedding_tests::{tiny_args, tiny_weights};
    let args = tiny_args();
    Lfm2Model::from_weights(args.clone(), tiny_weights(&args)).expect("synthetic LFM2 builds")
}

fn embed(model: &Lfm2Model, ids: &[i32]) -> UniquePtr<MlxArray> {
    let array = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    model.input_embeddings(&array)
}

#[test]
fn trunk_and_branches_match_single_passes_in_f32() {
    let _guard = mlx_test_guard();
    let model = tiny_backbone();
    let trunk: Vec<i32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
    let branches: Vec<Vec<i32>> = vec![
        vec![5, 9, 11, 2],
        vec![1, 30, 4, 4, 17, 8, 21],
        vec![12, 6, 25, 3, 19],
    ];
    let singles: Vec<Vec<f32>> = branches
        .iter()
        .map(|branch| {
            let full: Vec<i32> = trunk.iter().chain(branch).copied().collect();
            answer_logprobs(&model, &embed(&model, &full), &[], 1 << 16)
                .unwrap()
                .remove(0)
        })
        .collect();
    let trunk_embeds = embed(&model, &trunk);
    // One batch; then a budget that splits the rows across a batch boundary;
    // then one so small the trunk is read in 7-token chunks and every row
    // runs alone; then one under the branch width, so each row is also read
    // in 3-token pieces against its tiled cache.
    for budget in [1 << 16, 2 * (trunk.len() + 7), 7, 3] {
        let together = answer_logprobs(&model, &trunk_embeds, &branches, budget).unwrap();
        assert_eq!(together.len(), branches.len());
        for (row, (got, want)) in together.iter().zip(&singles).enumerate() {
            let diff = max_abs_diff(got, want);
            assert!(
                diff <= EQUIVALENCE_TOLERANCE,
                "budget {budget}, question {row}: trunk/branch log-probs differ by {diff}"
            );
        }
    }
    // A single pass over a prompt longer than the step is chunked too.
    let full: Vec<i32> = trunk.iter().chain(&branches[1]).copied().collect();
    let chunked = answer_logprobs(&model, &embed(&model, &full), &[], 7)
        .unwrap()
        .remove(0);
    assert!(max_abs_diff(&chunked, &singles[1]) <= EQUIVALENCE_TOLERANCE);
}

// --- real tokenizer vs mlx-vlm ids (no GPU) --------------------------------

#[test]
fn real_tokenizer_reproduces_mlx_vlm_text_prompts_and_readout_ids() {
    let Some(dir) = local_checkpoint(D1_CHECKPOINT) else {
        return;
    };
    let tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("tokenizer loads");
    for name in [
        "text_all",
        "text_refund",
        "text_team",
        "text_urgency",
        "options_27",
        "bool_criteria",
    ] {
        let case = fixture_case(name);
        let state = case_state(&case);
        let questions = validated(state.as_ref(), &case["questions"], false);
        let normalized: Vec<D1Question> = questions
            .iter()
            .map(|(_, q)| as_question(q).unwrap())
            .collect();
        let (trunk, branches) = prompt_ids(
            &tokenizer,
            "<|startoftext|>",
            state.as_ref(),
            &normalized,
            0,
        )
        .unwrap();
        assert_eq!(trunk, ids(&case["trunk_ids"]), "{name}: trunk ids");
        let want: Vec<Vec<i32>> = case["branch_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(ids)
            .collect();
        assert_eq!(branches, want, "{name}: branch ids");
        for (question, want) in normalized
            .iter()
            .zip(case["readout_ids"].as_array().unwrap())
        {
            let groups = readout_ids(&tokenizer, question).unwrap();
            let want: Vec<Vec<u32>> = want
                .as_array()
                .unwrap()
                .iter()
                .map(|g| ids(g).into_iter().map(|id| id as u32).collect())
                .collect();
            assert_eq!(groups, want, "{name}: readout ids");
        }
    }
}

// --- parity against mlx-vlm on the real checkpoint --------------------------

fn assert_close(a: &Value, b: &Value, tolerance: f64, what: &str) {
    let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
    assert!(
        (a - b).abs() <= tolerance,
        "{what}: {a} vs mlx-vlm {b} (tolerance {tolerance})"
    );
}

fn assert_answers_match(name: &str, got: &Value, want: &Value) {
    let got = got.as_object().unwrap();
    let want = want.as_object().unwrap();
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{name}: answer names"
    );
    for (question, want) in want {
        let got = &got[question];
        let at = format!("{name}/{question}");
        assert_eq!(got["type"], want["type"], "{at}: type");
        if want["type"] == "bool" {
            assert_close(
                &got["probability"],
                &want["probability"],
                PARITY_TOLERANCE,
                &at,
            );
            assert_eq!(got["value"], want["value"], "{at}: value");
            continue;
        }
        let (gp, wp) = (
            got["probabilities"].as_object().unwrap(),
            want["probabilities"].as_object().unwrap(),
        );
        assert_eq!(gp.keys().collect::<Vec<_>>(), wp.keys().collect::<Vec<_>>());
        for (label, p) in wp {
            assert_close(&gp[label], p, PARITY_TOLERANCE, &format!("{at}/{label}"));
        }
        assert_close(
            &got["metadata"]["confidence"],
            &want["metadata"]["confidence"],
            PARITY_TOLERANCE,
            &format!("{at}: confidence"),
        );
        if want["type"] == "choice" {
            assert_eq!(got["value"], want["value"], "{at}: value");
        } else {
            // The expected level moves by at most (n - 1) * tolerance.
            let spread = (wp.len() as f64 - 1.0) * PARITY_TOLERANCE;
            assert_close(
                &got["value"],
                &want["value"],
                spread,
                &format!("{at}: value"),
            );
            assert_eq!(got["metadata"]["legend"], want["metadata"]["legend"]);
        }
    }
}

#[test]
fn d1_3b_matches_mlx_vlm_on_text_and_images() {
    let Some(dir) = local_checkpoint(D1_CHECKPOINT) else {
        return;
    };
    let _guard = mlx_test_guard();
    let mut model = D1DecisionModel::load(&dir).expect("d1-3B loads");
    for case in fixture()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let images = match name {
            "cats" | "cats_multi" => vec![cats_image(false)],
            "cats_2x" => vec![cats_image(true)],
            _ => Vec::new(),
        };
        let state = case_state(case);
        let questions = validated(state.as_ref(), &case["questions"], !images.is_empty());
        let (trunk, branches) = model
            .prompt_ids(state.as_ref(), &questions, &images)
            .unwrap();
        assert_eq!(trunk, ids(&case["trunk_ids"]), "{name}: trunk ids");
        assert_eq!(branches.len(), case["branch_ids"].as_array().unwrap().len());
        let result = super::predict(
            &mut model,
            state.as_ref(),
            &case["questions"],
            DecisionMedia {
                images,
                ..DecisionMedia::default()
            },
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        let response = result.to_response_json("d1");
        eprintln!("{name}: {}", response["answers"]);
        assert_answers_match(name, &response["answers"], &case["result"]["answers"]);
        assert_eq!(response["usage"], case["result"]["usage"], "{name}: usage");
    }
}

#[test]
fn d1_3b_refuses_audio_through_the_shared_gate() {
    let Some(dir) = local_checkpoint(D1_CHECKPOINT) else {
        return;
    };
    let _guard = mlx_test_guard();
    let mut model = D1DecisionModel::load(&dir).expect("d1-3B loads");
    let err = super::predict(
        &mut model,
        None,
        &json!({"refund": {"type": "bool", "instructions": "Refund?"}}),
        DecisionMedia {
            audio: Some(vec![0.0; 16_000]),
            ..DecisionMedia::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        err,
        DecisionError::InvalidInput("This model does not support audio input".to_string())
    );
}
