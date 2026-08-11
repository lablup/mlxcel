use super::scheduler_muse_glimmer_support as muse;
use super::*;
use mlxcel_core::cache::{SequenceId, SequenceStateBackend};

use crate::models::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
};
use crate::server::batch::sequence::{FinishReason, SequenceState};

#[derive(Clone)]
struct ImageCase {
    image: Vec<u8>,
    max_tokens: usize,
    expected_prompt: Vec<i32>,
}

#[derive(Debug, PartialEq, Eq)]
struct RequestSummary {
    prompt_tokens: usize,
    completion_tokens: usize,
    text: String,
    streamed: Vec<String>,
    finish_reason: String,
}

fn cases() -> (ImageCase, ImageCase) {
    (
        ImageCase {
            image: muse::image_bytes(2, 2, 7),
            max_tokens: 2,
            expected_prompt: expanded_prompt(1),
        },
        ImageCase {
            image: muse::image_bytes(4, 2, 79),
            max_tokens: 3,
            expected_prompt: expanded_prompt(2),
        },
    )
}

fn expanded_prompt(patch_tokens: usize) -> Vec<i32> {
    let mut out = vec![DEFAULT_IMAGE_START_TOKEN_ID];
    out.extend(std::iter::repeat_n(DEFAULT_IMAGE_TOKEN_ID, patch_tokens));
    out.push(DEFAULT_IMAGE_END_TOKEN_ID);
    out.push(2);
    out
}

fn request_prompt() -> Vec<i32> {
    vec![DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, 2]
}

fn summarize(stream: muse::StreamSummary) -> RequestSummary {
    RequestSummary {
        prompt_tokens: stream.result.prompt_tokens,
        completion_tokens: stream.result.completion_tokens,
        text: stream.result.text,
        streamed: stream.tokens,
        finish_reason: stream.result.finish_reason,
    }
}

fn execute_prefill_tick(sched: &mut BatchScheduler) {
    match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            if sched.max_batch_prefill > 1
                && sched.prefill_queue.len() >= 2
                && sched.chunked_prefill_seq.is_none()
                && sched.batched_prefill_admits_head()
            {
                sched.execute_batched_prefill();
            } else {
                sched.execute_prefill(id);
            }
        }
        other => panic!("expected prefill action, got {other:?}"),
    }
}

fn execute_decode_tick(sched: &mut BatchScheduler, expected_ids: &[SequenceId]) {
    match sched.decide_action() {
        BatchSchedulerAction::Decode(ids) => {
            assert_same_ids(&ids, expected_ids);
            sched.execute_decode_step(&ids);
        }
        other => panic!("expected decode action, got {other:?}"),
    }
}

fn assert_same_ids(left: &[SequenceId], right: &[SequenceId]) {
    let mut left = left.iter().map(|id| id.as_u64()).collect::<Vec<_>>();
    let mut right = right.iter().map(|id| id.as_u64()).collect::<Vec<_>>();
    left.sort_unstable();
    right.sort_unstable();
    assert_eq!(left, right);
}

fn assert_zero_state(sched: &BatchScheduler, seq_id: SequenceId) {
    assert_eq!(
        muse::muse_state_offsets(sched, seq_id),
        Some(vec![(true, 0, 0), (false, 0, 0)])
    );
}

fn assert_active_prefilled(
    sched: &BatchScheduler,
    seq_id: SequenceId,
    case: &ImageCase,
    generated: usize,
) {
    let Some(seq) = sched.active_batch.get(seq_id) else {
        panic!("sequence {seq_id} was not active");
    };
    assert_eq!(seq.prompt_tokens, case.expected_prompt);
    assert_eq!(seq.images, vec![case.image.clone()]);
    assert_eq!(
        seq.generated_tokens,
        vec![muse::GENERATED_TOKEN_ID; generated]
    );
    let Some(embeddings) = seq.vlm_embeddings.as_ref() else {
        panic!("sequence {seq_id} lost prepared Muse embeddings");
    };
    assert_eq!(
        mlxcel_core::array_shape(&embeddings.inputs_embeds),
        vec![1, case.expected_prompt.len() as i32, 8]
    );
}

fn assert_offsets_after_tokens(sched: &BatchScheduler, seq_id: SequenceId, prompt_len: usize) {
    let offset = prompt_len as i32 + 1;
    assert_eq!(
        muse::muse_state_offsets(sched, seq_id),
        Some(vec![(true, offset, offset), (false, offset, offset)])
    );
}

fn assert_finished_length(sched: &BatchScheduler, seq_id: SequenceId) {
    let Some(seq) = sched.active_batch.get(seq_id) else {
        panic!("sequence {seq_id} was not active");
    };
    assert!(matches!(
        seq.state,
        SequenceState::Finished(FinishReason::Length)
    ));
}

fn assert_decoding(sched: &BatchScheduler, seq_id: SequenceId) {
    let Some(seq) = sched.active_batch.get(seq_id) else {
        panic!("sequence {seq_id} was not active");
    };
    assert!(matches!(seq.state, SequenceState::Decoding));
}

fn run_isolated(case: &ImageCase, collect_tokens: bool) -> RequestSummary {
    let mut sched = muse::scheduler(1);
    let rx = muse::enqueue(
        &mut sched,
        request_prompt(),
        case.image.clone(),
        case.max_tokens,
    );
    let seq_id = SequenceId::from_raw(0);
    assert_zero_state(&sched, seq_id);
    execute_prefill_tick(&mut sched);
    assert_active_prefilled(&sched, seq_id, case, 1);
    while sched.active_batch.get(seq_id).is_some() {
        execute_decode_tick(&mut sched, &[seq_id]);
        sched.finalize_completed();
    }
    assert_eq!(muse::muse_state_offsets(&sched, seq_id), None);
    if collect_tokens {
        summarize(muse::collect_stream(&rx))
    } else {
        let result = muse::collect_result_only(&rx);
        RequestSummary {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            text: result.text,
            streamed: Vec::new(),
            finish_reason: result.finish_reason,
        }
    }
}

fn assert_reuse_after_release(sched: &mut BatchScheduler) {
    let image = muse::image_bytes(2, 2, 17);
    let rx = muse::enqueue(sched, request_prompt(), image, 1);
    let seq_id = SequenceId::from_raw(2);
    assert_zero_state(sched, seq_id);
    execute_prefill_tick(sched);
    let summary = summarize(muse::collect_stream(&rx));
    assert_eq!(summary.prompt_tokens, 4);
    assert_eq!(summary.completion_tokens, 1);
    assert_eq!(summary.text, "a");
    assert_eq!(summary.streamed, vec!["a"]);
    assert_eq!(muse::muse_state_offsets(sched, seq_id), None);
}

fn run_two_concurrent(parallelism: usize) -> (RequestSummary, RequestSummary) {
    let (small, wide) = cases();
    let mut sched = muse::scheduler(parallelism);
    assert_eq!(
        sched.model.sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned
    );

    let small_rx = muse::enqueue(
        &mut sched,
        request_prompt(),
        small.image.clone(),
        small.max_tokens,
    );
    let wide_rx = muse::enqueue(
        &mut sched,
        request_prompt(),
        wide.image.clone(),
        wide.max_tokens,
    );
    let small_id = SequenceId::from_raw(0);
    let wide_id = SequenceId::from_raw(1);
    assert_eq!(sched.prefill_queue.len(), 2);
    assert_zero_state(&sched, small_id);
    assert_zero_state(&sched, wide_id);

    if parallelism == 1 {
        execute_prefill_tick(&mut sched);
        assert_active_prefilled(&sched, small_id, &small, 1);
        assert_zero_state(&sched, wide_id);
        execute_decode_tick(&mut sched, &[small_id]);
        assert_offsets_after_tokens(&sched, small_id, small.expected_prompt.len());
        assert_finished_length(&sched, small_id);
        sched.finalize_completed();
        assert_eq!(muse::muse_state_offsets(&sched, small_id), None);
        assert_zero_state(&sched, wide_id);

        execute_prefill_tick(&mut sched);
        assert_active_prefilled(&sched, wide_id, &wide, 1);
        execute_decode_tick(&mut sched, &[wide_id]);
        assert_offsets_after_tokens(&sched, wide_id, wide.expected_prompt.len());
        assert_decoding(&sched, wide_id);
        execute_decode_tick(&mut sched, &[wide_id]);
        assert_finished_length(&sched, wide_id);
        sched.finalize_completed();
    } else {
        execute_prefill_tick(&mut sched);
        assert_eq!(sched.prefill_queue.len(), 0);
        assert_active_prefilled(&sched, small_id, &small, 1);
        assert_active_prefilled(&sched, wide_id, &wide, 1);
        execute_decode_tick(&mut sched, &[small_id, wide_id]);
        assert_offsets_after_tokens(&sched, small_id, small.expected_prompt.len());
        assert_offsets_after_tokens(&sched, wide_id, wide.expected_prompt.len());
        assert_finished_length(&sched, small_id);
        assert_decoding(&sched, wide_id);
        sched.finalize_completed();
        assert_eq!(muse::muse_state_offsets(&sched, small_id), None);
        assert!(sched.active_batch.get(wide_id).is_some());

        execute_decode_tick(&mut sched, &[wide_id]);
        assert_finished_length(&sched, wide_id);
        sched.finalize_completed();
    }

    assert!(sched.active_batch.is_empty());
    assert_eq!(muse::muse_state_offsets(&sched, small_id), None);
    assert_eq!(muse::muse_state_offsets(&sched, wide_id), None);
    assert_reuse_after_release(&mut sched);
    (
        summarize(muse::collect_stream(&small_rx)),
        summarize(muse::collect_stream(&wide_rx)),
    )
}

#[test]
fn muse_scheduler_two_image_requests_parallelism_one_match_isolated_runs() {
    let (small, wide) = cases();
    let isolated_small = run_isolated(&small, true);
    let isolated_wide = run_isolated(&wide, true);
    let (small_result, wide_result) = run_two_concurrent(1);

    assert_eq!(small_result, isolated_small);
    assert_eq!(wide_result, isolated_wide);
    assert_eq!(small_result.prompt_tokens, 4);
    assert_eq!(wide_result.prompt_tokens, 5);
    assert_eq!(small_result.streamed, vec!["a", "a"]);
    assert_eq!(wide_result.streamed, vec!["a", "a", "a"]);
}

#[test]
fn muse_scheduler_two_image_requests_parallelism_two_match_isolated_runs() {
    let (small, wide) = cases();
    let isolated_small = run_isolated(&small, true);
    let isolated_wide = run_isolated(&wide, true);
    let (small_result, wide_result) = run_two_concurrent(2);

    assert_eq!(small_result, isolated_small);
    assert_eq!(wide_result, isolated_wide);
    assert_eq!(small_result.prompt_tokens, 4);
    assert_eq!(wide_result.prompt_tokens, 5);
    assert_eq!(small_result.streamed, vec!["a", "a"]);
    assert_eq!(wide_result.streamed, vec!["a", "a", "a"]);
}

#[test]
fn muse_scheduler_streaming_and_result_only_collection_are_equivalent() {
    let (small, _) = cases();
    let streaming = run_isolated(&small, true);
    let result_only = run_isolated(&small, false);

    assert_eq!(streaming.prompt_tokens, result_only.prompt_tokens);
    assert_eq!(streaming.completion_tokens, result_only.completion_tokens);
    assert_eq!(streaming.finish_reason, result_only.finish_reason);
    assert_eq!(streaming.text, result_only.text);
    assert_eq!(streaming.streamed.concat(), streaming.text);
}
